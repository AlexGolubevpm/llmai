import { type Job as BullJob } from "bullmq";
import { prisma } from "@/lib/db";
import { chatCompletion } from "@/lib/novita-client";
import { analyzeThumbnail } from "@/lib/wd-tagger-client";
import { postprocessLLMResponse, cleanText, type StopWordEntry } from "@/lib/text-processing";
import { publishProgress } from "@/lib/queue";
import * as fs from "fs";
import Papa from "papaparse";
import type { JobConfig } from "@/types";

const MAX_ROW_RETRIES = 3;

export async function aiProcessProcessor(job: BullJob) {
  const { jobId } = job.data;
  const dbJob = await prisma.job.findUniqueOrThrow({ where: { id: jobId } });
  const config = dbJob.config as unknown as JobConfig;
  const apiKey = process.env.NOVITA_API_KEY!;

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
  // 3 steps per row
  const totalSteps = totalRows * 3;
  let completedSteps = 0;

  await prisma.job.update({
    where: { id: jobId },
    data: { totalRows, totalPasses: 3 },
  });

  const errorLog: { row: number; error: string; retries: number }[] = [];
  const startTime = Date.now();

  for (let i = 0; i < rows.length; i++) {
    const row = rows[i];
    const thumbnailUrl = row["thumbnail_url"] || row["thumb_url"] || "";
    const title = row["title"] || "";

    // ====== STEP 1: WD Tagger — Analyze thumbnail ======
    let aiRawTags = "";
    let rating = "";
    let detectedTags: string[] = [];

    try {
      if (thumbnailUrl) {
        let taggerResult = null;
        for (let retry = 0; retry < MAX_ROW_RETRIES; retry++) {
          try {
            taggerResult = await analyzeThumbnail(thumbnailUrl);
            break;
          } catch (err) {
            if (retry === MAX_ROW_RETRIES - 1) {
              errorLog.push({ row: i, error: `WD Tagger: ${(err as Error).message}`, retries: retry + 1 });
            }
            await new Promise((r) => setTimeout(r, Math.pow(2, retry) * 1000));
          }
        }

        if (taggerResult) {
          detectedTags = taggerResult.tags;
          aiRawTags = taggerResult.tags.join(", ");
          rating = taggerResult.rating[0]?.label || "unknown";
        }
      }
    } catch (err) {
      errorLog.push({ row: i, error: `Step 1: ${(err as Error).message}`, retries: 0 });
    }

    row["ai_raw_tags"] = aiRawTags;
    row["rating"] = rating;
    completedSteps++;

    // ====== STEP 2: Tag & Category Mapping (LLM) ======
    let mappedTags = "";
    let mappedCategories = "";

    try {
      const allowedTagNames = allowedTags.map((t) => t.name).join(", ");
      const allowedCatNames = allowedCategories.map((c) => c.name).join(", ");

      const mappingPrompt = `Given these detected tags from image analysis: [${detectedTags.join(", ")}]
Existing tags from feed: [${row["tags"] || ""}]
Existing categories from feed: [${row["categories"] || ""}]

Allowed tags: [${allowedTagNames}]
Allowed categories: [${allowedCatNames}]

Task: Map the detected and existing tags to the closest matches from the Allowed tags list. Select exactly 5 tags and 1-3 categories.
Return ONLY in this format:
TAGS: tag1, tag2, tag3, tag4, tag5
CATEGORIES: cat1, cat2`;

      let mappingResult = "";
      for (let retry = 0; retry < MAX_ROW_RETRIES; retry++) {
        try {
          mappingResult = await chatCompletion(apiKey, {
            model: config.model || "meta-llama/llama-3.1-8b-instruct",
            systemPrompt: "You are an expert tag and category mapper. Follow instructions exactly.",
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
          break;
        } catch (err) {
          if (retry === MAX_ROW_RETRIES - 1) {
            errorLog.push({ row: i, error: `Step 2: ${(err as Error).message}`, retries: retry + 1 });
          }
          await new Promise((r) => setTimeout(r, Math.pow(2, retry) * 1000));
        }
      }

      // Parse mapping result
      const tagsMatch = mappingResult.match(/TAGS:\s*(.+)/i);
      const catsMatch = mappingResult.match(/CATEGORIES:\s*(.+)/i);
      mappedTags = tagsMatch ? tagsMatch[1].trim() : "";
      mappedCategories = catsMatch ? catsMatch[1].trim() : "";
    } catch (err) {
      errorLog.push({ row: i, error: `Step 2: ${(err as Error).message}`, retries: 0 });
    }

    row["mapped_tags"] = mappedTags;
    row["mapped_categories"] = mappedCategories;
    completedSteps++;

    // ====== STEP 3: SEO Title & Description Generation (LLM) ======
    try {
      const seoPrompt = `Based on the following information, generate an SEO-optimized NSFW title and description.

Original title: ${title}
Tags: ${mappedTags}
Categories: ${mappedCategories}
Content rating: ${rating}

Requirements:
- SEO title: max 90 characters, include relevant keywords, engaging and descriptive
- SEO description: max 160 characters, complement the title, include secondary keywords
- Both should be in English, natural-sounding, and optimized for search engines in 2026

Return ONLY in this format:
TITLE: your seo title here
DESCRIPTION: your seo description here`;

      let seoResult = "";
      for (let retry = 0; retry < MAX_ROW_RETRIES; retry++) {
        try {
          seoResult = await chatCompletion(apiKey, {
            model: config.model || "meta-llama/llama-3.1-8b-instruct",
            systemPrompt: "You are an expert SEO content writer specializing in adult content. Generate compelling, search-optimized titles and descriptions.",
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
          break;
        } catch (err) {
          if (retry === MAX_ROW_RETRIES - 1) {
            errorLog.push({ row: i, error: `Step 3: ${(err as Error).message}`, retries: retry + 1 });
          }
          await new Promise((r) => setTimeout(r, Math.pow(2, retry) * 1000));
        }
      }

      // Parse SEO result
      const titleMatch = seoResult.match(/TITLE:\s*(.+)/i);
      const descMatch = seoResult.match(/DESCRIPTION:\s*(.+)/i);

      let seoTitle = titleMatch ? titleMatch[1].trim() : title;
      let seoDesc = descMatch ? descMatch[1].trim() : "";

      // Post-process
      seoTitle = postprocessLLMResponse(seoTitle);
      seoDesc = postprocessLLMResponse(seoDesc);
      if (stopWords.length > 0) {
        seoTitle = cleanText(seoTitle, stopWords);
        seoDesc = cleanText(seoDesc, stopWords);
      }

      row["seo_title"] = seoTitle.slice(0, 90);
      row["seo_description"] = seoDesc.slice(0, 160);
    } catch (err) {
      errorLog.push({ row: i, error: `Step 3: ${(err as Error).message}`, retries: 0 });
      row["seo_title"] = title;
      row["seo_description"] = "";
    }

    completedSteps++;
    row["original_title"] = title;

    // Update progress
    const elapsed = (Date.now() - startTime) / 1000;
    const speed = completedSteps / elapsed;
    const remaining = (totalSteps - completedSteps) / speed;

    await prisma.job.update({
      where: { id: jobId },
      data: {
        processedRows: i + 1,
        failedRows: errorLog.length,
      },
    });

    await publishProgress(jobId, {
      status: "RUNNING",
      processedRows: i + 1,
      totalRows,
      failedRows: errorLog.length,
      currentPass: Math.ceil(completedSteps / totalRows),
      totalPasses: 3,
      eta: Math.round(remaining),
      speed: Math.round(speed * 10) / 10,
    });
  }

  // Write output
  const outputPath = dbJob.inputFileUrl.replace(/(\.\w+)$/, `_ai_processed$1`);
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

  await publishProgress(jobId, { status: "COMPLETED", processedRows: totalRows, totalRows });
}
