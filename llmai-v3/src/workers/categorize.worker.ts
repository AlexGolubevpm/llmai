import { type Job as BullJob } from "bullmq";
import { prisma } from "@/lib/db";
import { chatCompletion } from "@/lib/openrouter-client";
import { publishProgress } from "@/lib/queue";
import * as fs from "fs";
import Papa from "papaparse";
import type { JobConfig } from "@/types";

const MAX_RETRIES = 3;
const MIN_DELAY_MS = 300;

function sleep(ms: number) {
  return new Promise((r) => setTimeout(r, ms));
}

function parseJson(raw: string): Record<string, string> {
  const cleaned = raw.replace(/```json\s*/gi, "").replace(/```\s*/g, "").trim();
  try { return JSON.parse(cleaned); } catch {}
  const match = cleaned.match(/\{[\s\S]*\}/);
  if (match) {
    try { return JSON.parse(match[0]); } catch {}
    try { return JSON.parse(match[0].replace(/,\s*}/g, "}").replace(/'/g, '"')); } catch {}
  }
  return {};
}

export async function categorizeProcessor(job: BullJob) {
  const { jobId } = job.data;
  const dbJob = await prisma.job.findUniqueOrThrow({ where: { id: jobId } });
  const config = dbJob.config as unknown as JobConfig & {
    bundleTags?: string;
    bundleCategories?: string;
    bundleName?: string;
    numCategories?: number;
    numTags?: number;
  };
  const apiKey = process.env.OPENROUTER_API_KEY!;

  const model = config.model || "openai/gpt-4o-mini";
  const temperature = config.temperature || 0.3; // Low temp for consistent categorization
  const maxTokens = config.maxTokens || 200;
  const maxWorkers = Math.min(config.maxWorkers || 5, 10);
  const bundleTags = config.bundleTags || "";
  const bundleCategories = config.bundleCategories || "";
  const numCategories = config.numCategories || 3;
  const numTags = config.numTags || 8;

  console.log(`[Categorize] Job ${jobId} | Model: ${model} | Bundle: ${config.bundleName || "none"} | Workers: ${maxWorkers}`);
  console.log(`[Categorize] Tags: ${bundleTags.split(",").length}, Categories: ${bundleCategories.split(",").length}`);

  await prisma.job.update({
    where: { id: jobId },
    data: { status: "RUNNING", startedAt: new Date() },
  });

  const fileContent = fs.readFileSync(dbJob.inputFileUrl, "utf-8");
  const parsed = Papa.parse(fileContent, { header: true });
  const rows = parsed.data as Record<string, string>[];
  const totalRows = rows.length;

  await prisma.job.update({
    where: { id: jobId },
    data: { totalRows, totalPasses: 1 },
  });

  const errorLog: { row: number; step: string; error: string; retries: number }[] = [];
  const startTime = Date.now();

  const prompt = `You are a content categorization engine for an adult tube site.

Available categories (pick exactly ${numCategories}):
[${bundleCategories}]

Available tags (pick exactly ${numTags}):
[${bundleTags}]

Based on the video title, existing tags, and existing categories below, select the BEST matching categories and tags from the lists above.

Title: {title}
Current tags: {tags}
Current categories: {categories}

Rules:
- Pick EXACTLY ${numCategories} categories from the available list
- Pick EXACTLY ${numTags} tags from the available list
- Match based on content relevance to the title and existing metadata
- Prefer specific tags over generic ones
- Categories and tags must come from the provided lists ONLY

Return ONLY valid JSON:
{"categories":"cat1, cat2, cat3","tags":"tag1, tag2, tag3, tag4, tag5, tag6, tag7, tag8"}`;

  for (let i = 0; i < rows.length; i += maxWorkers) {
    // Check cancellation
    if (i > 0 && i % (maxWorkers * 5) === 0) {
      const fresh = await prisma.job.findUnique({ where: { id: jobId }, select: { status: true } });
      if (fresh?.status === "CANCELLED") {
        console.log(`[Categorize] Job ${jobId} cancelled at row ${i}`);
        break;
      }
    }

    const chunk = rows.slice(i, Math.min(i + maxWorkers, rows.length));

    await Promise.all(chunk.map(async (row, idx) => {
      const globalIdx = i + idx;
      const title = row["title"] || "";
      const existingTags = row["tags"] || "";
      const existingCats = row["categories"] || "";

      const rowPrompt = prompt
        .replace("{title}", title)
        .replace("{tags}", existingTags)
        .replace("{categories}", existingCats);

      for (let retry = 0; retry < MAX_RETRIES; retry++) {
        try {
          await sleep(MIN_DELAY_MS);
          const raw = await chatCompletion(apiKey, {
            model,
            systemPrompt: "You categorize content. Return ONLY valid JSON.",
            userPrompt: rowPrompt,
            maxTokens,
            temperature,
            topP: 1.0,
            minP: 0.0,
            topK: 40,
            presencePenalty: 0.0,
            frequencyPenalty: 0.0,
            repetitionPenalty: 1.0,
          });

          if (globalIdx === 0) console.log(`[Categorize] Row 0 raw: ${raw.slice(0, 300)}`);

          const result = parseJson(raw);
          row["new_categories"] = (result.categories || "").slice(0, 300);
          row["new_tags"] = (result.tags || "").slice(0, 500);
          break;
        } catch (err) {
          if (retry === MAX_RETRIES - 1) {
            row["new_categories"] = existingCats;
            row["new_tags"] = existingTags;
            errorLog.push({ row: globalIdx, step: "categorize", error: (err as Error).message, retries: MAX_RETRIES });
          }
          await sleep(Math.pow(2, retry) * 1000);
        }
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

  const outputPath = dbJob.inputFileUrl.replace(/(\.\w+)$/, `_categorized$1`);
  fs.writeFileSync(outputPath, Papa.unparse(rows), "utf-8");

  console.log(`[Categorize] Job ${jobId} done. ${totalRows} rows, ${errorLog.length} errors`);

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
