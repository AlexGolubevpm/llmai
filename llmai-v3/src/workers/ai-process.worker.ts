import { type Job as BullJob } from "bullmq";
import { prisma } from "@/lib/db";
import {
  postprocessLLMResponse,
  cleanText,
  type StopWordEntry,
} from "@/lib/text-processing";
import { publishProgress } from "@/lib/queue";
import * as fs from "fs";
import Papa from "papaparse";
import type { JobConfig } from "@/types";

const MAX_RETRIES = 3;
const MIN_DELAY_MS = 300;

function sleep(ms: number) {
  return new Promise((r) => setTimeout(r, ms));
}

const DEFAULT_PROMPT = `Analyze this image and the provided context. Return a JSON object with these fields:

1. "tags": comma-separated list of up to 15 descriptive tags (actions, body types, positions, clothing, setting, hair color, ethnicity)
2. "scene": 1-2 sentence description of what is happening in the scene
3. "type": content type (hentai/anime/3D/real/CGI/cartoon), number of people, art style
4. "title": SEO-optimized title, max 90 characters, English, engaging, search-optimized for 2026
5. "description": SEO meta description, max 160 characters, complements the title

Context:
Original title: {title}
Existing tags: {existing_tags}
Categories: {categories}

Return ONLY valid JSON, no markdown, no explanation:
{"tags":"...","scene":"...","type":"...","title":"...","description":"..."}`;

/**
 * Download image → base64 data URL.
 */
async function downloadImageBase64(url: string): Promise<string | null> {
  if (!url) return null;
  try {
    const resp = await fetch(url, {
      headers: { "User-Agent": "Mozilla/5.0 (compatible; LLMAI/3.0)" },
      redirect: "follow",
      signal: AbortSignal.timeout(30000),
    });
    if (!resp.ok) {
      console.warn(`[AI Process] Image download ${resp.status}: ${url}`);
      return null;
    }
    const contentType = resp.headers.get("content-type") || "image/jpeg";
    const buffer = await resp.arrayBuffer();
    return `data:${contentType};base64,${Buffer.from(buffer).toString("base64")}`;
  } catch (err) {
    console.warn(`[AI Process] Image error: ${(err as Error).message}`);
    return null;
  }
}

/**
 * Single vision+text call — one model, one prompt, one API call per row.
 * Returns all fields: tags, scene, type, title, description.
 */
async function processWithVision(
  apiKey: string,
  model: string,
  prompt: string,
  imageData: string | null,
  temperature: number,
  maxTokens: number
): Promise<string> {
  const BASE_URL = process.env.OPENROUTER_BASE_URL || "https://openrouter.ai/api/v1";

  const messages: Record<string, unknown>[] = [];

  if (imageData) {
    messages.push({
      role: "user",
      content: [
        { type: "image_url", image_url: { url: imageData } },
        { type: "text", text: prompt },
      ],
    });
  } else {
    messages.push({ role: "user", content: prompt });
  }

  await sleep(MIN_DELAY_MS);

  const resp = await fetch(`${BASE_URL}/chat/completions`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${apiKey}`,
      "HTTP-Referer": process.env.NEXT_PUBLIC_APP_URL || "http://localhost:3000",
      "X-Title": "LLMAI v3.0",
    },
    body: JSON.stringify({
      model,
      messages,
      max_tokens: maxTokens,
      temperature,
      // Route through non-Russian providers if needed
      provider: { allow_fallbacks: true },
    }),
  });

  if (resp.status === 429) {
    const wait = parseInt(resp.headers.get("retry-after") || "5") * 1000;
    await sleep(wait);
    throw new Error("Rate limited (429)");
  }

  if (!resp.ok) {
    const text = await resp.text();
    throw new Error(`OpenRouter ${resp.status}: ${text.slice(0, 300)}`);
  }

  const data = await resp.json();
  return data.choices?.[0]?.message?.content || "";
}

async function retry<T>(fn: () => Promise<T>, n: number, ctx: string): Promise<T> {
  let err: Error | null = null;
  for (let i = 0; i < n; i++) {
    try {
      return await fn();
    } catch (e) {
      err = e as Error;
      console.error(`[AI Process] ${ctx} attempt ${i + 1}/${n}: ${err.message}`);
      if (i < n - 1) await sleep(Math.pow(2, i) * 1000);
    }
  }
  throw err!;
}

function parseJsonResponse(raw: string): Record<string, string> {
  try {
    const cleaned = raw.replace(/```json\s*/g, "").replace(/```\s*/g, "").trim();
    return JSON.parse(cleaned);
  } catch {
    const match = raw.match(/\{[\s\S]*\}/);
    if (match) {
      try { return JSON.parse(match[0]); } catch { /* fall through */ }
    }
  }
  return {};
}

export async function aiProcessProcessor(job: BullJob) {
  const { jobId } = job.data;
  const dbJob = await prisma.job.findUniqueOrThrow({ where: { id: jobId } });
  const config = dbJob.config as unknown as JobConfig;
  const apiKey = process.env.OPENROUTER_API_KEY!;

  // Single model for everything
  const model = config.visionModel || config.model || "google/gemini-2.5-flash-preview-05-20";
  const prompt = config.visionPrompt || DEFAULT_PROMPT;
  const temperature = config.temperature || 0.7;
  const maxTokens = config.maxTokens || 500;
  const maxWorkers = Math.min(config.maxWorkers || 3, 10);

  console.log(`[AI Process] Job ${jobId} | Model: ${model} | Workers: ${maxWorkers}`);
  console.log(`[AI Process] Config:`, JSON.stringify(config));

  await prisma.job.update({
    where: { id: jobId },
    data: { status: "RUNNING", startedAt: new Date() },
  });

  const fileContent = fs.readFileSync(dbJob.inputFileUrl, "utf-8");
  const parsed = Papa.parse(fileContent, { header: true });
  const rows = parsed.data as Record<string, string>[];
  const totalRows = rows.length;

  const allowedTags = await prisma.allowedTag.findMany();
  const allowedCategories = await prisma.allowedCategory.findMany();
  const allowedTagNames = allowedTags.map((t) => t.name).join(", ");
  const allowedCatNames = allowedCategories.map((c) => c.name).join(", ");
  const stopWords: StopWordEntry[] = (
    await prisma.stopWord.findMany({ where: { isActive: true } })
  ).map((w) => ({ word: w.word, replacement: w.replacement }));

  await prisma.job.update({
    where: { id: jobId },
    data: { totalRows, totalPasses: 1 },
  });

  const errorLog: { row: number; step: string; error: string; retries: number }[] = [];
  const startTime = Date.now();

  // Process all rows — ONE API call per row
  for (let i = 0; i < rows.length; i += maxWorkers) {
    // Check cancellation
    if (i > 0 && i % (maxWorkers * 5) === 0) {
      const fresh = await prisma.job.findUnique({ where: { id: jobId }, select: { status: true } });
      if (fresh?.status === "CANCELLED") {
        console.log(`[AI Process] Job ${jobId} cancelled at row ${i}`);
        break;
      }
    }

    const chunk = rows.slice(i, Math.min(i + maxWorkers, rows.length));

    await Promise.all(chunk.map(async (row, idx) => {
      const globalIdx = i + idx;
      const thumbUrl = row["thumbnail_url"] || row["thumb_url"] || "";
      const title = row["title"] || "";
      row["original_title"] = title;

      // Download image
      const imageData = await downloadImageBase64(thumbUrl);

      // Build prompt with context
      const finalPrompt = prompt
        .replace("{title}", title)
        .replace("{existing_tags}", row["tags"] || "")
        .replace("{categories}", row["categories"] || "")
        + (allowedTagNames ? `\nAllowed tags: [${allowedTagNames}]` : "")
        + (allowedCatNames ? `\nAllowed categories: [${allowedCatNames}]` : "");

      try {
        const raw = await retry(
          () => processWithVision(apiKey, model, finalPrompt, imageData, temperature, maxTokens),
          MAX_RETRIES, `row${globalIdx}`
        );

        const result = parseJsonResponse(postprocessLLMResponse(raw));

        row["ai_tags"] = (result.tags || "").slice(0, 500);
        row["scene_description"] = (result.scene || "").slice(0, 500);
        row["content_type"] = (result.type || "").slice(0, 200);

        let seoTitle = postprocessLLMResponse(result.title || title);
        let seoDesc = postprocessLLMResponse(result.description || "");

        if (stopWords.length > 0) {
          seoTitle = cleanText(seoTitle, stopWords);
          seoDesc = cleanText(seoDesc, stopWords);
        }

        row["seo_title"] = seoTitle.slice(0, 90);
        row["seo_description"] = seoDesc.slice(0, 160);
      } catch (err) {
        row["ai_tags"] = "";
        row["scene_description"] = "";
        row["content_type"] = "";
        row["seo_title"] = title;
        row["seo_description"] = "";
        errorLog.push({ row: globalIdx, step: "process", error: (err as Error).message, retries: MAX_RETRIES });
      }
    }));

    const processed = Math.min(i + maxWorkers, totalRows);
    const elapsed = (Date.now() - startTime) / 1000;
    const speed = processed / elapsed;

    await prisma.job.update({
      where: { id: jobId },
      data: { processedRows: processed, failedRows: errorLog.length },
    });
    await publishProgress(jobId, {
      status: "RUNNING",
      processedRows: processed,
      totalRows,
      failedRows: errorLog.length,
      currentPass: 1,
      totalPasses: 1,
      eta: Math.round((totalRows - processed) / speed) || 0,
      speed: Math.round(speed * 10) / 10,
    });
  }

  // Write output
  const outputPath = dbJob.inputFileUrl.replace(/(\.\w+)$/, `_ai_processed$1`);
  fs.writeFileSync(outputPath, Papa.unparse(rows), "utf-8");

  const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
  console.log(`[AI Process] Job ${jobId} done in ${elapsed}s. Rows: ${totalRows}, Errors: ${errorLog.length}`);

  await prisma.job.update({
    where: { id: jobId },
    data: {
      status: "COMPLETED",
      outputFileUrl: outputPath,
      completedAt: new Date(),
      processedRows: totalRows,
      errorLog: errorLog.length > 0 ? errorLog : undefined,
    },
  });

  await publishProgress(jobId, { status: "COMPLETED", processedRows: totalRows, totalRows });
}
