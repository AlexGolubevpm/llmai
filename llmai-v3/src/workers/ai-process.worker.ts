import { type Job as BullJob } from "bullmq";
import { prisma } from "@/lib/db";
import { chatCompletion } from "@/lib/openrouter-client";
import { analyzeThumbnailsBatch } from "@/lib/wd-tagger-client";
import {
  postprocessLLMResponse,
  cleanText,
  type StopWordEntry,
} from "@/lib/text-processing";
import { publishProgress } from "@/lib/queue";
import * as fs from "fs";
import Papa from "papaparse";
import type { JobConfig, WDTaggerResult } from "@/types";

const MAX_ROW_RETRIES = 3;
const TAGGER_BATCH_SIZE = 10; // Process thumbnails in batches of 10

export async function aiProcessProcessor(job: BullJob) {
  const { jobId } = job.data;
  const dbJob = await prisma.job.findUniqueOrThrow({ where: { id: jobId } });
  const config = dbJob.config as unknown as JobConfig;
  const apiKey = process.env.OPENROUTER_API_KEY!;

  await prisma.job.update({
    where: { id: jobId },
    data: { status: "RUNNING", startedAt: new Date() },
  });

  // Read input
  const fileContent = fs.readFileSync(dbJob.inputFileUrl, "utf-8");
  const parsed = Papa.parse(fileContent, { header: true });
  const rows = parsed.data as Record<string, string>[];

  // Load allowed tags and categories from DB
  const allowedTags = await prisma.allowedTag.findMany();
  const allowedCategories = await prisma.allowedCategory.findMany();
  const stopWords: StopWordEntry[] = (
    await prisma.stopWord.findMany({ where: { isActive: true } })
  ).map((w) => ({ word: w.word, replacement: w.replacement }));

  const totalRows = rows.length;

  await prisma.job.update({
    where: { id: jobId },
    data: { totalRows, totalPasses: 3 },
  });

  const errorLog: { row: number; error: string; retries: number }[] = [];
  const startTime = Date.now();

  // ====== STEP 1: WD Tagger — Batch analyze all thumbnails ======
  await prisma.job.update({ where: { id: jobId }, data: { currentPass: 1 } });

  const thumbnailUrls = rows.map(
    (r) => r["thumbnail_url"] || r["thumb_url"] || ""
  );
  const taggerResults: (WDTaggerResult | null)[] = [];

  // Process in batches to avoid overwhelming HuggingFace
  for (let i = 0; i < thumbnailUrls.length; i += TAGGER_BATCH_SIZE) {
    const batchUrls = thumbnailUrls.slice(i, i + TAGGER_BATCH_SIZE);
    const batchResults = await analyzeThumbnailsBatch(batchUrls, {
      onProgress: (done) => {
        const totalDone = i + done;
        publishProgress(jobId, {
          status: "RUNNING",
          processedRows: totalDone,
          totalRows,
          failedRows: errorLog.length,
          currentPass: 1,
          totalPasses: 3,
          eta: estimateEta(startTime, totalDone, totalRows * 3),
          speed: calcSpeed(startTime, totalDone),
        });
      },
    });

    taggerResults.push(...batchResults);

    // Update DB progress
    await prisma.job.update({
      where: { id: jobId },
      data: { processedRows: Math.min(i + TAGGER_BATCH_SIZE, totalRows) },
    });
  }

  // Store Step 1 results into rows
  for (let i = 0; i < rows.length; i++) {
    const result = taggerResults[i];
    if (result) {
      rows[i]["ai_raw_tags"] = result.tags.join(", ");
      rows[i]["rating"] = result.rating[0]?.label || "unknown";
    } else {
      rows[i]["ai_raw_tags"] = "";
      rows[i]["rating"] = "unknown";
      if (thumbnailUrls[i]) {
        errorLog.push({
          row: i,
          error: "WD Tagger: failed after retries",
          retries: 3,
        });
      }
    }
  }

  // ====== STEP 2: Tag & Category Mapping (LLM) — parallel in chunks ======
  await prisma.job.update({ where: { id: jobId }, data: { currentPass: 2 } });

  const allowedTagNames = allowedTags.map((t) => t.name).join(", ");
  const allowedCatNames = allowedCategories.map((c) => c.name).join(", ");
  const maxWorkers = config.maxWorkers || 5;
  const chunkSize = config.chunkSize || 10;

  for (let i = 0; i < rows.length; i += chunkSize) {
    const chunk = rows.slice(i, Math.min(i + chunkSize, rows.length));

    const promises = chunk.map(async (row, idx) => {
      const globalIdx = i + idx;
      const detectedTags = row["ai_raw_tags"] || "";

      const mappingPrompt = `Given these detected tags from image analysis: [${detectedTags}]
Existing tags from feed: [${row["tags"] || ""}]
Existing categories from feed: [${row["categories"] || ""}]

Allowed tags: [${allowedTagNames}]
Allowed categories: [${allowedCatNames}]

Task: Map the detected and existing tags to the closest matches from the Allowed tags list. Select exactly 5 tags and 1-3 categories.
Return ONLY in this format:
TAGS: tag1, tag2, tag3, tag4, tag5
CATEGORIES: cat1, cat2`;

      for (let retry = 0; retry < MAX_ROW_RETRIES; retry++) {
        try {
          const result = await chatCompletion(apiKey, {
            model: config.model || "openai/gpt-4o-mini",
            systemPrompt:
              "You are an expert tag and category mapper. Follow instructions exactly.",
            userPrompt: mappingPrompt,
            maxTokens: 200,
            temperature: 0.3,
            topP: 1.0,
            minP: 0.0,
            topK: 40,
            presencePenalty: 0.0,
            frequencyPenalty: 0.0,
            repetitionPenalty: 1.0,
          });

          const tagsMatch = result.match(/TAGS:\s*(.+)/i);
          const catsMatch = result.match(/CATEGORIES:\s*(.+)/i);
          row["mapped_tags"] = tagsMatch ? tagsMatch[1].trim() : "";
          row["mapped_categories"] = catsMatch ? catsMatch[1].trim() : "";
          return;
        } catch (err) {
          if (retry === MAX_ROW_RETRIES - 1) {
            errorLog.push({
              row: globalIdx,
              error: `Step 2: ${(err as Error).message}`,
              retries: retry + 1,
            });
            row["mapped_tags"] = "";
            row["mapped_categories"] = "";
          }
          await new Promise((r) =>
            setTimeout(r, Math.pow(2, retry) * 1000)
          );
        }
      }
    });

    // Process with concurrency limit
    for (let b = 0; b < promises.length; b += maxWorkers) {
      await Promise.all(promises.slice(b, b + maxWorkers));
    }

    const step2Done = Math.min(i + chunkSize, totalRows);
    await prisma.job.update({
      where: { id: jobId },
      data: { processedRows: step2Done },
    });
    await publishProgress(jobId, {
      status: "RUNNING",
      processedRows: step2Done,
      totalRows,
      failedRows: errorLog.length,
      currentPass: 2,
      totalPasses: 3,
      eta: estimateEta(startTime, totalRows + step2Done, totalRows * 3),
      speed: calcSpeed(startTime, totalRows + step2Done),
    });
  }

  // ====== STEP 3: SEO Title & Description Generation (LLM) — parallel in chunks ======
  await prisma.job.update({ where: { id: jobId }, data: { currentPass: 3 } });

  for (let i = 0; i < rows.length; i += chunkSize) {
    const chunk = rows.slice(i, Math.min(i + chunkSize, rows.length));

    const promises = chunk.map(async (row, idx) => {
      const globalIdx = i + idx;
      const title = row["title"] || "";
      row["original_title"] = title;

      const seoPrompt = `Based on the following information, generate an SEO-optimized NSFW title and description.

Original title: ${title}
Tags: ${row["mapped_tags"] || ""}
Categories: ${row["mapped_categories"] || ""}
Content rating: ${row["rating"] || ""}

Requirements:
- SEO title: max 90 characters, include relevant keywords, engaging and descriptive
- SEO description: max 160 characters, complement the title, include secondary keywords
- Both should be in English, natural-sounding, and optimized for search engines in 2026

Return ONLY in this format:
TITLE: your seo title here
DESCRIPTION: your seo description here`;

      for (let retry = 0; retry < MAX_ROW_RETRIES; retry++) {
        try {
          const result = await chatCompletion(apiKey, {
            model: config.model || "openai/gpt-4o-mini",
            systemPrompt:
              "You are an expert SEO content writer specializing in adult content. Generate compelling, search-optimized titles and descriptions.",
            userPrompt: seoPrompt,
            maxTokens: config.maxTokens || 300,
            temperature: config.temperature || 0.7,
            topP: config.topP || 1.0,
            minP: config.minP || 0.0,
            topK: config.topK || 40,
            presencePenalty: config.presencePenalty || 0.2,
            frequencyPenalty: config.frequencyPenalty || 0.4,
            repetitionPenalty: config.repetitionPenalty || 1.2,
          });

          const titleMatch = result.match(/TITLE:\s*(.+)/i);
          const descMatch = result.match(/DESCRIPTION:\s*(.+)/i);

          let seoTitle = titleMatch ? titleMatch[1].trim() : title;
          let seoDesc = descMatch ? descMatch[1].trim() : "";

          seoTitle = postprocessLLMResponse(seoTitle);
          seoDesc = postprocessLLMResponse(seoDesc);
          if (stopWords.length > 0) {
            seoTitle = cleanText(seoTitle, stopWords);
            seoDesc = cleanText(seoDesc, stopWords);
          }

          row["seo_title"] = seoTitle.slice(0, 90);
          row["seo_description"] = seoDesc.slice(0, 160);
          return;
        } catch (err) {
          if (retry === MAX_ROW_RETRIES - 1) {
            errorLog.push({
              row: globalIdx,
              error: `Step 3: ${(err as Error).message}`,
              retries: retry + 1,
            });
            row["seo_title"] = title;
            row["seo_description"] = "";
          }
          await new Promise((r) =>
            setTimeout(r, Math.pow(2, retry) * 1000)
          );
        }
      }
    });

    for (let b = 0; b < promises.length; b += maxWorkers) {
      await Promise.all(promises.slice(b, b + maxWorkers));
    }

    const step3Done = Math.min(i + chunkSize, totalRows);
    await prisma.job.update({
      where: { id: jobId },
      data: { processedRows: step3Done },
    });
    await publishProgress(jobId, {
      status: "RUNNING",
      processedRows: step3Done,
      totalRows,
      failedRows: errorLog.length,
      currentPass: 3,
      totalPasses: 3,
      eta: estimateEta(startTime, totalRows * 2 + step3Done, totalRows * 3),
      speed: calcSpeed(startTime, totalRows * 2 + step3Done),
    });
  }

  // Write output
  const outputPath = dbJob.inputFileUrl.replace(
    /(\.\w+)$/,
    `_ai_processed$1`
  );
  fs.writeFileSync(outputPath, Papa.unparse(rows), "utf-8");

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

  await publishProgress(jobId, {
    status: "COMPLETED",
    processedRows: totalRows,
    totalRows,
  });
}

function estimateEta(
  startTime: number,
  completedSteps: number,
  totalSteps: number
): number {
  const elapsed = (Date.now() - startTime) / 1000;
  if (completedSteps === 0) return 0;
  const speed = completedSteps / elapsed;
  return Math.round((totalSteps - completedSteps) / speed);
}

function calcSpeed(startTime: number, completedSteps: number): number {
  const elapsed = (Date.now() - startTime) / 1000;
  if (elapsed === 0) return 0;
  return Math.round((completedSteps / elapsed) * 10) / 10;
}
