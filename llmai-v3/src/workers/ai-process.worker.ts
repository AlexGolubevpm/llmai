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

const DEFAULT_PROMPT = `Look at this image carefully. Describe what you see.

Return ONLY a valid JSON object with these 3 fields:
{
  "tags": "comma-separated list of up to 15 descriptive tags",
  "scene": "1-2 sentence description of what is happening",
  "type": "content type: hentai, anime, 3D, real, CGI, or cartoon"
}

For tags include: actions, body types, positions, clothing, setting, hair color.
Do not include any markdown, explanation, or text outside the JSON.`;

/**
 * Download image from URL and convert to base64.
 */
async function downloadImage(url: string): Promise<string | null> {
  if (!url) return null;
  try {
    const resp = await fetch(url, {
      headers: { "User-Agent": "Mozilla/5.0 (compatible; LLMAI/3.0)" },
      redirect: "follow",
      signal: AbortSignal.timeout(30000),
    });
    if (!resp.ok) {
      console.warn(`[AI] Image ${resp.status}: ${url.slice(0, 80)}`);
      return null;
    }
    const ct = resp.headers.get("content-type") || "image/jpeg";
    const buf = await resp.arrayBuffer();
    return `data:${ct};base64,${Buffer.from(buf).toString("base64")}`;
  } catch (err) {
    console.warn(`[AI] Image error: ${(err as Error).message}`);
    return null;
  }
}

/**
 * Call OpenRouter vision API.
 */
async function callVision(
  apiKey: string,
  model: string,
  prompt: string,
  imageBase64: string | null,
  maxTokens: number,
  temperature: number
): Promise<string> {
  const BASE_URL = process.env.OPENROUTER_BASE_URL || "https://openrouter.ai/api/v1";

  const content: unknown[] = [];
  if (imageBase64) {
    content.push({ type: "image_url", image_url: { url: imageBase64 } });
  }
  content.push({ type: "text", text: prompt });

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
      messages: [{ role: "user", content }],
      max_tokens: maxTokens,
      temperature,
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
    throw new Error(`OpenRouter ${resp.status}: ${text.slice(0, 200)}`);
  }

  const data = await resp.json();
  return data.choices?.[0]?.message?.content || "";
}

async function retry<T>(fn: () => Promise<T>, n: number, ctx: string): Promise<T> {
  let err: Error | null = null;
  for (let i = 0; i < n; i++) {
    try { return await fn(); } catch (e) {
      err = e as Error;
      console.error(`[AI] ${ctx} attempt ${i + 1}/${n}: ${err.message}`);
      if (i < n - 1) await sleep(Math.pow(2, i) * 1000);
    }
  }
  throw err!;
}

function parseJson(raw: string): Record<string, string> {
  // Clean up the response
  let cleaned = raw
    .replace(/```json\s*/gi, "")
    .replace(/```\s*/g, "")
    .replace(/^\s*json\s*/i, "")
    .trim();

  // Try direct parse
  try { return JSON.parse(cleaned); } catch {}

  // Try extracting JSON from the response
  const match = cleaned.match(/\{[\s\S]*\}/);
  if (match) {
    try { return JSON.parse(match[0]); } catch {}

    // Try fixing common JSON issues (trailing commas, single quotes)
    let fixed = match[0]
      .replace(/,\s*}/g, "}")
      .replace(/,\s*]/g, "]")
      .replace(/'/g, '"');
    try { return JSON.parse(fixed); } catch {}
  }

  console.warn(`[AI] Failed to parse JSON from response: ${raw.slice(0, 200)}`);
  return {};
}

export async function aiProcessProcessor(job: BullJob) {
  const { jobId } = job.data;
  const dbJob = await prisma.job.findUniqueOrThrow({ where: { id: jobId } });
  const config = dbJob.config as unknown as JobConfig;
  const apiKey = process.env.OPENROUTER_API_KEY!;

  const model = config.visionModel || config.model || "google/gemini-2.5-flash-preview-05-20";
  const prompt = config.visionPrompt || DEFAULT_PROMPT;
  const temperature = config.temperature || 0.7;
  const maxTokens = config.maxTokens || 500;
  const maxWorkers = Math.min(config.maxWorkers || 3, 10);

  console.log(`[AI] Job ${jobId} | Model: ${model} | Workers: ${maxWorkers} | MaxTokens: ${maxTokens}`);

  await prisma.job.update({
    where: { id: jobId },
    data: { status: "RUNNING", startedAt: new Date() },
  });

  const fileContent = fs.readFileSync(dbJob.inputFileUrl, "utf-8");
  const parsed = Papa.parse(fileContent, { header: true });
  const rows = parsed.data as Record<string, string>[];
  const totalRows = rows.length;

  const stopWords: StopWordEntry[] = config.applyStopWords
    ? (await prisma.stopWord.findMany({ where: { isActive: true } }))
        .map((w) => ({ word: w.word, replacement: w.replacement }))
    : [];

  await prisma.job.update({
    where: { id: jobId },
    data: { totalRows, totalPasses: 1 },
  });

  const errorLog: { row: number; step: string; error: string; retries: number }[] = [];
  const startTime = Date.now();

  for (let i = 0; i < rows.length; i += maxWorkers) {
    // Check cancellation
    if (i > 0 && i % (maxWorkers * 5) === 0) {
      const fresh = await prisma.job.findUnique({ where: { id: jobId }, select: { status: true } });
      if (fresh?.status === "CANCELLED") {
        console.log(`[AI] Job ${jobId} cancelled at row ${i}`);
        break;
      }
    }

    const chunk = rows.slice(i, Math.min(i + maxWorkers, rows.length));

    await Promise.all(chunk.map(async (row, idx) => {
      const globalIdx = i + idx;
      const thumbUrl = row["thumbnail_url"] || row["thumb_url"] || "";
      row["original_title"] = row["title"] || "";

      // Download image on our server
      const img = await downloadImage(thumbUrl);

      if (!img) {
        row["ai_tags"] = "";
        row["scene_description"] = "";
        row["content_type"] = "";
        errorLog.push({ row: globalIdx, step: "download", error: `Failed to download: ${thumbUrl.slice(0, 80)}`, retries: 0 });
        return;
      }

      try {
        const raw = await retry(
          () => callVision(apiKey, model, prompt, img, maxTokens, temperature),
          MAX_RETRIES, `row${globalIdx}`
        );

        // Log first row's raw response for debugging
        if (globalIdx === 0) {
          console.log(`[AI] Row 0 raw response (first 500 chars): ${raw.slice(0, 500)}`);
        }

        const result = parseJson(raw);

        let tags = result.tags || "";
        let scene = result.scene || result.scene_description || "";
        let type = result.type || result.content_type || "";

        if (stopWords.length > 0) {
          tags = cleanText(tags, stopWords);
          scene = cleanText(scene, stopWords);
        }

        row["ai_tags"] = tags.slice(0, 500);
        row["scene_description"] = scene.slice(0, 500);
        row["content_type"] = type.slice(0, 200);
      } catch (err) {
        row["ai_tags"] = "";
        row["scene_description"] = "";
        row["content_type"] = "";
        errorLog.push({ row: globalIdx, step: "vision", error: (err as Error).message, retries: MAX_RETRIES });
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
  console.log(`[AI] Job ${jobId} done in ${elapsed}s. ${totalRows} rows, ${errorLog.length} errors`);

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
