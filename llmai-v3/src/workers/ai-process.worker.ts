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

const MAX_ROW_RETRIES = 3;

/**
 * Download image → base64. No caching — used once per row then GC'd.
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

/**
 * Send vision request with base64 image (or text-only fallback).
 */
async function visionCall(
  apiKey: string,
  model: string,
  systemPrompt: string,
  userPrompt: string,
  base64Image: string | null,
  maxTokens: number
): Promise<string> {
  const BASE_URL = process.env.OPENROUTER_BASE_URL || "https://openrouter.ai/api/v1";

  const messages: Record<string, unknown>[] = [
    { role: "system", content: systemPrompt },
  ];

  if (base64Image) {
    messages.push({
      role: "user",
      content: [
        { type: "image_url", image_url: { url: base64Image } },
        { type: "text", text: userPrompt },
      ],
    });
  } else {
    messages.push({ role: "user", content: userPrompt });
  }

  const resp = await fetch(`${BASE_URL}/chat/completions`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${apiKey}`,
      "HTTP-Referer": process.env.NEXT_PUBLIC_APP_URL || "http://localhost:3000",
      "X-Title": "LLMAI v3.0",
    },
    body: JSON.stringify({ model, messages, max_tokens: maxTokens, temperature: 0.7 }),
  });

  if (!resp.ok) {
    const text = await resp.text();
    throw new Error(`OpenRouter ${resp.status}: ${text.slice(0, 300)}`);
  }

  const data = await resp.json();
  return data.choices?.[0]?.message?.content || "";
}

async function retry<T>(fn: () => Promise<T>, retries: number, ctx: string): Promise<T> {
  let lastErr: Error | null = null;
  for (let i = 0; i < retries; i++) {
    try {
      return await fn();
    } catch (err) {
      lastErr = err as Error;
      console.error(`[AI Process] ${ctx} attempt ${i + 1}/${retries}: ${(err as Error).message}`);
      if (i < retries - 1) await new Promise((r) => setTimeout(r, Math.pow(2, i) * 1000));
    }
  }
  throw lastErr!;
}

/**
 * Process ALL 5 steps for a single row.
 * Downloads image once, runs steps 1-3 (vision), frees image,
 * then runs steps 4-5 (text-only).
 */
async function processRow(
  row: Record<string, string>,
  rowIdx: number,
  apiKey: string,
  visionModel: string,
  config: JobConfig,
  allowedTagNames: string,
  allowedCatNames: string,
  stopWords: StopWordEntry[],
  errorLog: { row: number; step: string; error: string; retries: number }[]
): Promise<void> {
  const thumbUrl = row["thumbnail_url"] || row["thumb_url"] || "";
  const title = row["title"] || "";
  row["original_title"] = title;

  // Download image ONCE for steps 1-3
  const base64Image = await downloadImageBase64(thumbUrl);

  // STEP 1: Tagging
  try {
    const result = await retry(
      () => visionCall(apiKey, visionModel,
        "You are an expert image tagger for adult content.",
        "Analyze this image. Return ONLY a comma-separated list of descriptive tags (max 15). Tags: actions, body types, positions, clothing, setting, hair color, ethnicity. No explanations.",
        base64Image, 200),
      MAX_ROW_RETRIES, `S1 row${rowIdx}`
    );
    row["ai_tags"] = postprocessLLMResponse(result).replace(/\n/g, ", ");
  } catch (err) {
    row["ai_tags"] = "";
    errorLog.push({ row: rowIdx, step: "1-tagging", error: (err as Error).message, retries: MAX_ROW_RETRIES });
  }

  // STEP 2: Scene Description
  try {
    const result = await retry(
      () => visionCall(apiKey, visionModel,
        "You describe adult content scenes concisely for SEO.",
        "Describe this scene in 1-2 sentences. Be specific about what is happening, who is involved, the setting.",
        base64Image, 150),
      MAX_ROW_RETRIES, `S2 row${rowIdx}`
    );
    row["scene_description"] = postprocessLLMResponse(result);
  } catch (err) {
    row["scene_description"] = "";
    errorLog.push({ row: rowIdx, step: "2-scene", error: (err as Error).message, retries: MAX_ROW_RETRIES });
  }

  // STEP 3: Content Type Detection
  try {
    const result = await retry(
      () => visionCall(apiKey, visionModel,
        "You identify content type and style from thumbnails.",
        "Identify:\n1. Type (hentai/anime/3D/real/CGI/cartoon)\n2. Number of people\n3. Art style\n\nReturn ONLY:\nTYPE: <type>\nCOUNT: <number>\nSTYLE: <style or real>",
        base64Image, 100),
      MAX_ROW_RETRIES, `S3 row${rowIdx}`
    );
    row["content_type"] = postprocessLLMResponse(result);
  } catch (err) {
    row["content_type"] = "";
    errorLog.push({ row: rowIdx, step: "3-type", error: (err as Error).message, retries: MAX_ROW_RETRIES });
  }

  // base64Image is now eligible for GC — no reference kept

  // STEP 4: SEO Title (text-only LLM)
  try {
    const prompt = `Generate SEO NSFW title:\nOriginal: ${title}\nTags: ${row["ai_tags"]}\nScene: ${row["scene_description"]}\nType: ${row["content_type"]}\nExisting tags: ${row["tags"] || ""}\nCategories: ${row["categories"] || ""}${allowedTagNames ? `\nAllowed tags: [${allowedTagNames}]` : ""}${allowedCatNames ? `\nAllowed categories: [${allowedCatNames}]` : ""}\n\nMax 90 chars, English, natural, search-optimized. Return ONLY the title.`;

    const result = await retry(
      () => chatCompletion(apiKey, {
        model: config.model || "openai/gpt-4o-mini",
        systemPrompt: "Expert SEO title writer for adult content.",
        userPrompt: prompt,
        maxTokens: config.maxTokens || 100,
        temperature: config.temperature || 0.7,
        topP: config.topP || 1.0,
        minP: config.minP || 0.0,
        topK: config.topK || 40,
        presencePenalty: config.presencePenalty || 0.2,
        frequencyPenalty: config.frequencyPenalty || 0.4,
        repetitionPenalty: config.repetitionPenalty || 1.2,
      }),
      MAX_ROW_RETRIES, `S4 row${rowIdx}`
    );
    let seoTitle = postprocessLLMResponse(result);
    if (stopWords.length > 0) seoTitle = cleanText(seoTitle, stopWords);
    row["seo_title"] = seoTitle.slice(0, 90);
  } catch (err) {
    row["seo_title"] = title;
    errorLog.push({ row: rowIdx, step: "4-title", error: (err as Error).message, retries: MAX_ROW_RETRIES });
  }

  // STEP 5: SEO Description (text-only LLM)
  try {
    const prompt = `Generate SEO meta description:\nTitle: ${row["seo_title"]}\nTags: ${row["ai_tags"]}\nScene: ${row["scene_description"]}\nType: ${row["content_type"]}\n\nMax 160 chars, complements title, English, natural. Return ONLY the description.`;

    const result = await retry(
      () => chatCompletion(apiKey, {
        model: config.model || "openai/gpt-4o-mini",
        systemPrompt: "Expert SEO description writer.",
        userPrompt: prompt,
        maxTokens: 100,
        temperature: config.temperature || 0.7,
        topP: config.topP || 1.0,
        minP: config.minP || 0.0,
        topK: config.topK || 40,
        presencePenalty: config.presencePenalty || 0.2,
        frequencyPenalty: config.frequencyPenalty || 0.4,
        repetitionPenalty: config.repetitionPenalty || 1.0,
      }),
      MAX_ROW_RETRIES, `S5 row${rowIdx}`
    );
    let seoDesc = postprocessLLMResponse(result);
    if (stopWords.length > 0) seoDesc = cleanText(seoDesc, stopWords);
    row["seo_description"] = seoDesc.slice(0, 160);
  } catch (err) {
    row["seo_description"] = "";
    errorLog.push({ row: rowIdx, step: "5-desc", error: (err as Error).message, retries: MAX_ROW_RETRIES });
  }
}

export async function aiProcessProcessor(job: BullJob) {
  const { jobId } = job.data;
  const dbJob = await prisma.job.findUniqueOrThrow({ where: { id: jobId } });
  const config = dbJob.config as unknown as JobConfig;
  const apiKey = process.env.OPENROUTER_API_KEY!;
  const visionModel = config.model || "xiaomi/mimo-v2-omni";

  console.log(`[AI Process] Starting job ${jobId}, vision: ${visionModel}, rows: TBD`);

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
    data: { totalRows, totalPasses: 5 },
  });

  const errorLog: { row: number; step: string; error: string; retries: number }[] = [];
  const startTime = Date.now();
  const maxWorkers = config.maxWorkers || 3;

  console.log(`[AI Process] Processing ${totalRows} rows, concurrency: ${maxWorkers}`);

  // Process rows in chunks with concurrency limit.
  // Each row: download image → steps 1-3 (vision) → steps 4-5 (text) → done.
  // Image freed after each row — RAM usage: maxWorkers * ~500KB max.
  for (let i = 0; i < rows.length; i += maxWorkers) {
    const chunk = rows.slice(i, Math.min(i + maxWorkers, rows.length));

    await Promise.all(
      chunk.map((row, idx) =>
        processRow(row, i + idx, apiKey, visionModel, config, allowedTagNames, allowedCatNames, stopWords, errorLog)
      )
    );

    const processed = Math.min(i + maxWorkers, totalRows);
    const elapsed = (Date.now() - startTime) / 1000;
    const speed = processed / elapsed;
    const eta = Math.round((totalRows - processed) / speed) || 0;

    // Determine which step we're conceptually on (for UI)
    const currentStep = Math.min(5, Math.ceil((processed / totalRows) * 5) || 1);

    await prisma.job.update({
      where: { id: jobId },
      data: { processedRows: processed, failedRows: errorLog.length, currentPass: currentStep },
    });

    await publishProgress(jobId, {
      status: "RUNNING",
      processedRows: processed,
      totalRows,
      failedRows: errorLog.length,
      currentPass: currentStep,
      totalPasses: 5,
      eta,
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
