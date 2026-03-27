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
 * Vision-capable chat completion — sends image URL in the message.
 */
async function visionCompletion(
  apiKey: string,
  model: string,
  systemPrompt: string,
  userPrompt: string,
  imageUrl: string,
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
      temperature: 0.7,
    }),
  });

  if (!resp.ok) {
    const text = await resp.text();
    throw new Error(`OpenRouter API ${resp.status}: ${text}`);
  }

  const data = await resp.json();
  return data.choices?.[0]?.message?.content || "";
}

async function retryCall<T>(
  fn: () => Promise<T>,
  retries: number,
  context: string
): Promise<T> {
  let lastError: Error | null = null;
  for (let attempt = 0; attempt < retries; attempt++) {
    try {
      return await fn();
    } catch (err) {
      lastError = err as Error;
      console.error(`[AI Process] ${context} attempt ${attempt + 1}/${retries} failed: ${(err as Error).message}`);
      if (attempt < retries - 1) {
        await new Promise((r) => setTimeout(r, Math.pow(2, attempt) * 1000));
      }
    }
  }
  throw lastError!;
}

export async function aiProcessProcessor(job: BullJob) {
  const { jobId } = job.data;
  const dbJob = await prisma.job.findUniqueOrThrow({ where: { id: jobId } });
  const config = dbJob.config as unknown as JobConfig;
  const apiKey = process.env.OPENROUTER_API_KEY!;
  const visionModel = config.model || "xiaomi/mimo-v2-omni";

  console.log(`[AI Process] Starting job ${jobId}, model: ${visionModel}`);

  await prisma.job.update({
    where: { id: jobId },
    data: { status: "RUNNING", startedAt: new Date() },
  });

  const fileContent = fs.readFileSync(dbJob.inputFileUrl, "utf-8");
  const parsed = Papa.parse(fileContent, { header: true });
  const rows = parsed.data as Record<string, string>[];

  const allowedTags = await prisma.allowedTag.findMany();
  const allowedCategories = await prisma.allowedCategory.findMany();
  const stopWords: StopWordEntry[] = (
    await prisma.stopWord.findMany({ where: { isActive: true } })
  ).map((w) => ({ word: w.word, replacement: w.replacement }));

  const totalRows = rows.length;
  await prisma.job.update({
    where: { id: jobId },
    data: { totalRows, totalPasses: 5 },
  });

  const errorLog: { row: number; step: string; error: string; retries: number }[] = [];
  const startTime = Date.now();
  const maxWorkers = config.maxWorkers || 3;
  const chunkSize = config.chunkSize || 5;

  // ====== STEP 1: Tagging (from thumbnail) ======
  console.log(`[AI Process] Step 1: Tagging ${totalRows} thumbnails`);
  await prisma.job.update({ where: { id: jobId }, data: { currentPass: 1 } });

  for (let i = 0; i < rows.length; i += chunkSize) {
    const chunk = rows.slice(i, Math.min(i + chunkSize, rows.length));
    const promises = chunk.map(async (row, idx) => {
      const globalIdx = i + idx;
      const thumbUrl = row["thumbnail_url"] || row["thumb_url"] || "";
      try {
        const result = await retryCall(
          () => visionCompletion(
            apiKey, visionModel,
            "You are an expert image tagger for adult content. Analyze the image and return relevant tags.",
            "Analyze this image. Return ONLY a comma-separated list of descriptive tags (max 15 tags). Tags should describe: actions, body types, positions, clothing, setting, hair color, ethnicity. Do NOT include explanations.",
            thumbUrl, 200
          ),
          MAX_ROW_RETRIES,
          `Step1 row ${globalIdx}`
        );
        row["ai_tags"] = postprocessLLMResponse(result).replace(/\n/g, ", ");
      } catch (err) {
        row["ai_tags"] = "";
        errorLog.push({ row: globalIdx, step: "1-tagging", error: (err as Error).message, retries: MAX_ROW_RETRIES });
      }
    });
    for (let b = 0; b < promises.length; b += maxWorkers) {
      await Promise.all(promises.slice(b, b + maxWorkers));
    }
    await updateProgress(jobId, i + chunk.length, totalRows, 1, 5, errorLog.length, startTime);
  }

  // ====== STEP 2: Scene Description (from thumbnail) ======
  console.log(`[AI Process] Step 2: Scene descriptions`);
  await prisma.job.update({ where: { id: jobId }, data: { currentPass: 2 } });

  for (let i = 0; i < rows.length; i += chunkSize) {
    const chunk = rows.slice(i, Math.min(i + chunkSize, rows.length));
    const promises = chunk.map(async (row, idx) => {
      const globalIdx = i + idx;
      const thumbUrl = row["thumbnail_url"] || row["thumb_url"] || "";
      try {
        const result = await retryCall(
          () => visionCompletion(
            apiKey, visionModel,
            "You are an expert at describing adult content scenes concisely.",
            "Describe this scene in 1-2 sentences. Be specific about what is happening, who is involved, and the setting. Keep it concise and descriptive for SEO purposes.",
            thumbUrl, 150
          ),
          MAX_ROW_RETRIES,
          `Step2 row ${globalIdx}`
        );
        row["scene_description"] = postprocessLLMResponse(result);
      } catch (err) {
        row["scene_description"] = "";
        errorLog.push({ row: globalIdx, step: "2-scene", error: (err as Error).message, retries: MAX_ROW_RETRIES });
      }
    });
    for (let b = 0; b < promises.length; b += maxWorkers) {
      await Promise.all(promises.slice(b, b + maxWorkers));
    }
    await updateProgress(jobId, i + chunk.length, totalRows, 2, 5, errorLog.length, startTime);
  }

  // ====== STEP 3: Model/Character Detection (from thumbnail) ======
  console.log(`[AI Process] Step 3: Model detection`);
  await prisma.job.update({ where: { id: jobId }, data: { currentPass: 3 } });

  for (let i = 0; i < rows.length; i += chunkSize) {
    const chunk = rows.slice(i, Math.min(i + chunkSize, rows.length));
    const promises = chunk.map(async (row, idx) => {
      const globalIdx = i + idx;
      const thumbUrl = row["thumbnail_url"] || row["thumb_url"] || "";
      try {
        const result = await retryCall(
          () => visionCompletion(
            apiKey, visionModel,
            "You are an expert at identifying content type and style from thumbnails.",
            "Based on this image, identify:\n1. Content type (hentai/anime/3D/real/CGI/cartoon)\n2. Number of people/characters\n3. Art style if animated\n\nReturn ONLY in format:\nTYPE: <type>\nCOUNT: <number>\nSTYLE: <style or 'real'>",
            thumbUrl, 100
          ),
          MAX_ROW_RETRIES,
          `Step3 row ${globalIdx}`
        );
        row["content_type"] = postprocessLLMResponse(result);
      } catch (err) {
        row["content_type"] = "";
        errorLog.push({ row: globalIdx, step: "3-model", error: (err as Error).message, retries: MAX_ROW_RETRIES });
      }
    });
    for (let b = 0; b < promises.length; b += maxWorkers) {
      await Promise.all(promises.slice(b, b + maxWorkers));
    }
    await updateProgress(jobId, i + chunk.length, totalRows, 3, 5, errorLog.length, startTime);
  }

  // ====== STEP 4: SEO Title Generation (LLM, text-only) ======
  console.log(`[AI Process] Step 4: SEO title generation`);
  await prisma.job.update({ where: { id: jobId }, data: { currentPass: 4 } });

  const allowedTagNames = allowedTags.map((t) => t.name).join(", ");
  const allowedCatNames = allowedCategories.map((c) => c.name).join(", ");

  for (let i = 0; i < rows.length; i += chunkSize) {
    const chunk = rows.slice(i, Math.min(i + chunkSize, rows.length));
    const promises = chunk.map(async (row, idx) => {
      const globalIdx = i + idx;
      const title = row["title"] || "";
      row["original_title"] = title;

      const prompt = `Generate an SEO-optimized NSFW title based on:

Original title: ${title}
AI tags: ${row["ai_tags"] || ""}
Scene: ${row["scene_description"] || ""}
Content type: ${row["content_type"] || ""}
Existing tags: ${row["tags"] || ""}
Existing categories: ${row["categories"] || ""}
${allowedTagNames ? `Allowed tags: [${allowedTagNames}]` : ""}
${allowedCatNames ? `Allowed categories: [${allowedCatNames}]` : ""}

Requirements:
- Max 90 characters
- Include relevant keywords
- Natural, engaging, search-optimized for 2026
- English only

Return ONLY the title, nothing else.`;

      try {
        const result = await retryCall(
          () => chatCompletion(apiKey, {
            model: config.model || "openai/gpt-4o-mini",
            systemPrompt: "You are an expert SEO title writer for adult content. Generate compelling, search-optimized titles.",
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
          MAX_ROW_RETRIES,
          `Step4 row ${globalIdx}`
        );

        let seoTitle = postprocessLLMResponse(result);
        if (stopWords.length > 0) seoTitle = cleanText(seoTitle, stopWords);
        row["seo_title"] = seoTitle.slice(0, 90);
      } catch (err) {
        row["seo_title"] = title;
        errorLog.push({ row: globalIdx, step: "4-title", error: (err as Error).message, retries: MAX_ROW_RETRIES });
      }
    });
    for (let b = 0; b < promises.length; b += maxWorkers) {
      await Promise.all(promises.slice(b, b + maxWorkers));
    }
    await updateProgress(jobId, i + chunk.length, totalRows, 4, 5, errorLog.length, startTime);
  }

  // ====== STEP 5: SEO Description Generation (LLM, text-only) ======
  console.log(`[AI Process] Step 5: SEO description generation`);
  await prisma.job.update({ where: { id: jobId }, data: { currentPass: 5 } });

  for (let i = 0; i < rows.length; i += chunkSize) {
    const chunk = rows.slice(i, Math.min(i + chunkSize, rows.length));
    const promises = chunk.map(async (row, idx) => {
      const globalIdx = i + idx;

      const prompt = `Generate an SEO meta description based on:

Title: ${row["seo_title"] || row["title"] || ""}
Tags: ${row["ai_tags"] || ""}
Scene: ${row["scene_description"] || ""}
Content type: ${row["content_type"] || ""}

Requirements:
- Max 160 characters
- Complement the title with secondary keywords
- Natural, engaging, optimized for search in 2026

Return ONLY the description, nothing else.`;

      try {
        const result = await retryCall(
          () => chatCompletion(apiKey, {
            model: config.model || "openai/gpt-4o-mini",
            systemPrompt: "You are an expert SEO description writer. Generate concise, compelling meta descriptions.",
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
          MAX_ROW_RETRIES,
          `Step5 row ${globalIdx}`
        );

        let seoDesc = postprocessLLMResponse(result);
        if (stopWords.length > 0) seoDesc = cleanText(seoDesc, stopWords);
        row["seo_description"] = seoDesc.slice(0, 160);
      } catch (err) {
        row["seo_description"] = "";
        errorLog.push({ row: globalIdx, step: "5-description", error: (err as Error).message, retries: MAX_ROW_RETRIES });
      }
    });
    for (let b = 0; b < promises.length; b += maxWorkers) {
      await Promise.all(promises.slice(b, b + maxWorkers));
    }
    await updateProgress(jobId, i + chunk.length, totalRows, 5, 5, errorLog.length, startTime);
  }

  // Write output
  const outputPath = dbJob.inputFileUrl.replace(/(\.\w+)$/, `_ai_processed$1`);
  fs.writeFileSync(outputPath, Papa.unparse(rows), "utf-8");

  const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
  console.log(`[AI Process] Job ${jobId} completed in ${elapsed}s. Rows: ${totalRows}, Errors: ${errorLog.length}`);

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

async function updateProgress(
  jobId: string,
  processed: number,
  total: number,
  step: number,
  totalSteps: number,
  errors: number,
  startTime: number
) {
  const elapsed = (Date.now() - startTime) / 1000;
  const totalDone = (step - 1) * total + processed;
  const totalWork = totalSteps * total;
  const speed = totalDone / elapsed;
  const eta = Math.round((totalWork - totalDone) / speed) || 0;

  await prisma.job.update({
    where: { id: jobId },
    data: { processedRows: processed, failedRows: errors },
  });

  await publishProgress(jobId, {
    status: "RUNNING",
    processedRows: processed,
    totalRows: total,
    failedRows: errors,
    currentPass: step,
    totalPasses: totalSteps,
    eta,
    speed: Math.round(speed * 10) / 10,
  });
}
