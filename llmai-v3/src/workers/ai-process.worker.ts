import { type Job as BullJob } from "bullmq";
import { prisma } from "@/lib/db";
import { chatCompletion } from "@/lib/openrouter-client";
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

// ---- Default prompts ----

const DEFAULT_VISION_PROMPT = `Analyze this image and return a JSON object with exactly these fields:
1. "tags": comma-separated list of up to 15 descriptive tags (actions, body types, positions, clothing, setting, hair color, ethnicity)
2. "scene": 1-2 sentence description of what is happening in the scene
3. "type": content type and style, format: "<type> | <count> people | <style>" where type is one of: hentai, anime, 3D, real, CGI, cartoon

Return ONLY valid JSON, no markdown, no explanation:
{"tags":"tag1, tag2, ...","scene":"...","type":"..."}`;

const DEFAULT_SEO_PROMPT = `Based on the context below, generate SEO-optimized title and description.

Context:
Original title: {title}
Tags: {tags}
Scene: {scene}
Content type: {type}
Existing tags: {existing_tags}
Categories: {categories}

Requirements:
- title: max 90 characters, English, natural, engaging, search-optimized for 2026
- description: max 160 characters, complements the title with secondary keywords

Return ONLY valid JSON, no markdown:
{"title":"...","description":"..."}`;

// ---- Image download ----

/**
 * Download image from URL → base64 data URL.
 * Our server downloads because vision model providers often can't reach external sites.
 * No caching — each image used once then GC'd.
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
    const base64 = Buffer.from(buffer).toString("base64");
    return `data:${contentType};base64,${base64}`;
  } catch (err) {
    console.warn(`[AI Process] Image error: ${(err as Error).message} — ${url}`);
    return null;
  }
}

// ---- Vision call ----

async function visionCall(
  apiKey: string,
  model: string,
  systemPrompt: string,
  userPrompt: string,
  imageUrl: string | null,
  temperature: number,
  maxTokens: number
): Promise<string> {
  const BASE_URL = process.env.OPENROUTER_BASE_URL || "https://openrouter.ai/api/v1";

  const messages: Record<string, unknown>[] = [
    { role: "system", content: systemPrompt },
  ];

  if (imageUrl) {
    messages.push({
      role: "user",
      content: [
        { type: "image_url", image_url: { url: imageUrl } },
        { type: "text", text: userPrompt },
      ],
    });
  } else {
    messages.push({ role: "user", content: userPrompt });
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
    body: JSON.stringify({ model, messages, max_tokens: maxTokens, temperature }),
  });

  if (resp.status === 429) {
    const wait = parseInt(resp.headers.get("retry-after") || "5") * 1000;
    console.warn(`[AI Process] 429, waiting ${wait}ms`);
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

// ---- Retry helper ----

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

// ---- JSON parser with fallback ----

function parseJsonResponse(raw: string): Record<string, string> {
  // Try direct JSON parse
  try {
    const cleaned = raw.replace(/```json\s*/g, "").replace(/```\s*/g, "").trim();
    return JSON.parse(cleaned);
  } catch {
    // Fallback: extract JSON from response
    const match = raw.match(/\{[\s\S]*\}/);
    if (match) {
      try {
        return JSON.parse(match[0]);
      } catch { /* fall through */ }
    }
  }
  return {};
}

// ---- Main processor ----

export async function aiProcessProcessor(job: BullJob) {
  const { jobId } = job.data;
  const dbJob = await prisma.job.findUniqueOrThrow({ where: { id: jobId } });
  const config = dbJob.config as unknown as JobConfig;
  const apiKey = process.env.OPENROUTER_API_KEY!;

  console.log(`[AI Process] Raw config:`, JSON.stringify(config));
  const visionModel = config.visionModel || config.model || "xiaomi/mimo-v2-omni";
  const textModel = config.textModel || "openai/gpt-4o-mini";
  const visionPrompt = config.visionPrompt || DEFAULT_VISION_PROMPT;
  const seoPromptTemplate = config.seoPrompt || DEFAULT_SEO_PROMPT;
  const temperature = config.temperature || 0.7;
  const maxWorkers = Math.min(config.maxWorkers || 3, 10);

  console.log(`[AI Process] Job ${jobId} | Vision: ${visionModel} | Text: ${textModel} | Workers: ${maxWorkers}`);

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
  const stopWords: StopWordEntry[] = (
    await prisma.stopWord.findMany({ where: { isActive: true } })
  ).map((w) => ({ word: w.word, replacement: w.replacement }));

  await prisma.job.update({
    where: { id: jobId },
    data: { totalRows, totalPasses: 2 },
  });

  const errorLog: { row: number; step: string; error: string; retries: number }[] = [];
  const startTime = Date.now();

  // ==========================================
  // STEP 1: Vision — tags + scene + type
  // ==========================================
  console.log(`[AI Process] Step 1: Vision analysis (${totalRows} rows)`);
  await prisma.job.update({ where: { id: jobId }, data: { currentPass: 1 } });

  let cancelled = false;
  for (let i = 0; i < rows.length; i += maxWorkers) {
    // Check cancellation every chunk
    if (i > 0 && i % (maxWorkers * 5) === 0) {
      const fresh = await prisma.job.findUnique({ where: { id: jobId }, select: { status: true } });
      if (fresh?.status === "CANCELLED") {
        console.log(`[AI Process] Job ${jobId} cancelled at step 1, row ${i}`);
        cancelled = true;
        break;
      }
    }

    const chunk = rows.slice(i, Math.min(i + maxWorkers, rows.length));

    await Promise.all(chunk.map(async (row, idx) => {
      const globalIdx = i + idx;
      const thumbUrl = row["thumbnail_url"] || row["thumb_url"] || "";

      // Download image on our server → base64 (providers can't always reach external URLs)
      const imageData = await downloadImageBase64(thumbUrl);

      try {
        const raw = await retry(
          () => visionCall(apiKey, visionModel, "You analyze images and return structured JSON.", visionPrompt, imageData, temperature, 300),
          MAX_RETRIES, `S1 row${globalIdx}`
        );

        const parsed = parseJsonResponse(postprocessLLMResponse(raw));
        row["ai_tags"] = (parsed.tags || "").slice(0, 500);
        row["scene_description"] = (parsed.scene || "").slice(0, 500);
        row["content_type"] = (parsed.type || "").slice(0, 200);
      } catch (err) {
        row["ai_tags"] = "";
        row["scene_description"] = "";
        row["content_type"] = "";
        errorLog.push({ row: globalIdx, step: "1-vision", error: (err as Error).message, retries: MAX_RETRIES });
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
      status: "RUNNING", processedRows: processed, totalRows,
      failedRows: errorLog.length, currentPass: 1, totalPasses: 2,
      eta: Math.round((totalRows * 2 - processed) / speed) || 0,
      speed: Math.round(speed * 10) / 10,
    });
  }

  // Save step 1 intermediate file
  const step1Path = dbJob.inputFileUrl.replace(/(\.\w+)$/, `_step1$1`);
  fs.writeFileSync(step1Path, Papa.unparse(rows), "utf-8");
  console.log(`[AI Process] Step 1 complete, saved: ${step1Path}`);

  // Update job with step1 file (downloadable immediately)
  await prisma.job.update({
    where: { id: jobId },
    data: { outputFileUrl: step1Path },
  });

  // ==========================================
  // STEP 2: SEO — title + description
  // ==========================================
  if (!cancelled) {
  console.log(`[AI Process] Step 2: SEO generation (${totalRows} rows)`);
  await prisma.job.update({ where: { id: jobId }, data: { currentPass: 2 } });

  const allowedTagNames = allowedTags.map((t) => t.name).join(", ");
  const allowedCatNames = allowedCategories.map((c) => c.name).join(", ");

  for (let i = 0; i < rows.length; i += maxWorkers) {
    // Check cancellation
    if (i > 0 && i % (maxWorkers * 5) === 0) {
      const fresh = await prisma.job.findUnique({ where: { id: jobId }, select: { status: true } });
      if (fresh?.status === "CANCELLED") {
        console.log(`[AI Process] Job ${jobId} cancelled at step 2, row ${i}`);
        cancelled = true;
        break;
      }
    }

    const chunk = rows.slice(i, Math.min(i + maxWorkers, rows.length));

    await Promise.all(chunk.map(async (row, idx) => {
      const globalIdx = i + idx;
      const title = row["title"] || "";
      row["original_title"] = title;

      // Build SEO prompt with context
      const prompt = seoPromptTemplate
        .replace("{title}", title)
        .replace("{tags}", row["ai_tags"] || "")
        .replace("{scene}", row["scene_description"] || "")
        .replace("{type}", row["content_type"] || "")
        .replace("{existing_tags}", row["tags"] || "")
        .replace("{categories}", row["categories"] || "")
        + (allowedTagNames ? `\nAllowed tags: [${allowedTagNames}]` : "")
        + (allowedCatNames ? `\nAllowed categories: [${allowedCatNames}]` : "");

      try {
        const raw = await retry(
          () => chatCompletion(apiKey, {
            model: textModel,
            systemPrompt: "Expert SEO writer for adult content. Return valid JSON only.",
            userPrompt: prompt,
            maxTokens: config.maxTokens || 200,
            temperature,
            topP: config.topP || 1.0,
            minP: config.minP || 0.0,
            topK: config.topK || 40,
            presencePenalty: config.presencePenalty || 0.2,
            frequencyPenalty: config.frequencyPenalty || 0.4,
            repetitionPenalty: config.repetitionPenalty || 1.2,
          }),
          MAX_RETRIES, `S2 row${globalIdx}`
        );

        const parsed = parseJsonResponse(raw);
        let seoTitle = postprocessLLMResponse(parsed.title || title);
        let seoDesc = postprocessLLMResponse(parsed.description || "");

        if (stopWords.length > 0) {
          seoTitle = cleanText(seoTitle, stopWords);
          seoDesc = cleanText(seoDesc, stopWords);
        }

        row["seo_title"] = seoTitle.slice(0, 90);
        row["seo_description"] = seoDesc.slice(0, 160);
      } catch (err) {
        row["seo_title"] = title;
        row["seo_description"] = "";
        errorLog.push({ row: globalIdx, step: "2-seo", error: (err as Error).message, retries: MAX_RETRIES });
      }
    }));

    const processed = Math.min(i + maxWorkers, totalRows);
    const elapsed = (Date.now() - startTime) / 1000;
    const totalDone = totalRows + processed;
    const speed = totalDone / elapsed;

    await prisma.job.update({
      where: { id: jobId },
      data: { processedRows: processed, failedRows: errorLog.length },
    });
    await publishProgress(jobId, {
      status: "RUNNING", processedRows: processed, totalRows,
      failedRows: errorLog.length, currentPass: 2, totalPasses: 2,
      eta: Math.round((totalRows * 2 - totalDone) / speed) || 0,
      speed: Math.round(speed * 10) / 10,
    });
  }
  } // end if (!cancelled) for step 2

  // Save final file (all columns) — even if cancelled, save partial results
  const finalPath = dbJob.inputFileUrl.replace(/(\.\w+)$/, `_final$1`);
  fs.writeFileSync(finalPath, Papa.unparse(rows), "utf-8");

  const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
  const finalStatus = cancelled ? "CANCELLED" : "COMPLETED";
  console.log(`[AI Process] Job ${jobId} ${finalStatus} in ${elapsed}s. Rows: ${totalRows}, Errors: ${errorLog.length}`);

  await prisma.job.update({
    where: { id: jobId },
    data: {
      status: finalStatus,
      outputFileUrl: finalPath,
      completedAt: new Date(),
      processedRows: totalRows,
      errorLog: errorLog.length > 0 ? errorLog : undefined,
    },
  });

  await publishProgress(jobId, { status: finalStatus, processedRows: totalRows, totalRows });
}
