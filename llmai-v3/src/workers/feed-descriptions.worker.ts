import { type Job as BullJob } from "bullmq";
import { prisma } from "@/lib/db";
import { chatCompletion } from "@/lib/openrouter-client";
import { postprocessLLMResponse, cleanText, type StopWordEntry } from "@/lib/text-processing";
import { publishProgress } from "@/lib/queue";
import * as fs from "fs";
import Papa from "papaparse";
import type { JobConfig } from "@/types";

const MAX_RETRIES = 3;
const MIN_DELAY_MS = 300;

function sleep(ms: number) {
  return new Promise((r) => setTimeout(r, ms));
}

const DEFAULT_PROMPT = `Write an SEO-optimized meta description for an adult video page.

Video title: {title}
Categories: {categories}
Tags: {tags}

Requirements:
- 120-160 characters
- Must mention the most relevant keywords from tags/categories
- Compelling, drives clicks
- Natural English, no keyword stuffing
- Specific to this video, not generic

Return ONLY the description text, nothing else.`;

export async function feedDescriptionsProcessor(job: BullJob) {
  const { jobId } = job.data;
  const dbJob = await prisma.job.findUniqueOrThrow({ where: { id: jobId } });
  const config = dbJob.config as unknown as JobConfig & { customPrompt?: string };
  const apiKey = process.env.OPENROUTER_API_KEY!;

  const model = config.model || "openai/gpt-4o-mini";
  const prompt = config.customPrompt || config.userPrompt || DEFAULT_PROMPT;
  const temperature = config.temperature || 0.7;
  const maxTokens = config.maxTokens || 200;
  const maxWorkers = Math.min(config.maxWorkers || 5, 10);

  console.log(`[FeedDesc] Job ${jobId} | Model: ${model} | Workers: ${maxWorkers}`);

  await prisma.job.update({
    where: { id: jobId },
    data: { status: "RUNNING", startedAt: new Date() },
  });

  const fileContent = fs.readFileSync(dbJob.inputFileUrl, "utf-8");
  const parsed = Papa.parse(fileContent, { header: true });
  const rows = parsed.data as Record<string, string>[];
  const totalRows = rows.length;

  const stopWords: StopWordEntry[] = config.applyStopWords
    ? (await prisma.stopWord.findMany({ where: { isActive: true } })).map((w) => ({ word: w.word, replacement: w.replacement }))
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
        console.log(`[FeedDesc] Job ${jobId} cancelled at row ${i}`);
        break;
      }
    }

    const chunk = rows.slice(i, Math.min(i + maxWorkers, rows.length));

    await Promise.all(chunk.map(async (row, idx) => {
      const globalIdx = i + idx;
      const title = row["title"] || row["Название"] || "";
      const categories = row["categories"] || row["Категории"] || "";
      const tags = row["tags"] || row["Тэги"] || "";

      const rowPrompt = prompt
        .replace(/\{title\}/g, title)
        .replace(/\{categories\}/g, categories)
        .replace(/\{tags\}/g, tags);

      for (let retry = 0; retry < MAX_RETRIES; retry++) {
        try {
          await sleep(MIN_DELAY_MS);
          const raw = await chatCompletion(apiKey, {
            model,
            systemPrompt: "You write SEO descriptions for adult video pages. Return ONLY the description text.",
            userPrompt: rowPrompt,
            maxTokens,
            temperature,
            topP: 1.0,
            minP: 0.0,
            topK: 40,
            presencePenalty: 0.2,
            frequencyPenalty: 0.4,
            repetitionPenalty: 1.0,
          });

          let desc = postprocessLLMResponse(raw);
          if (stopWords.length > 0) desc = cleanText(desc, stopWords);
          row["description"] = desc.slice(0, 200);

          if (globalIdx === 0) console.log(`[FeedDesc] Row 0: "${desc.slice(0, 100)}"`);
          break;
        } catch (err) {
          if (retry === MAX_RETRIES - 1) {
            row["description"] = "";
            errorLog.push({ row: globalIdx, step: "generate", error: (err as Error).message, retries: MAX_RETRIES });
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

  const outputPath = dbJob.inputFileUrl.replace(/(\.\w+)$/, `_descriptions$1`);
  fs.writeFileSync(outputPath, Papa.unparse(rows), "utf-8");

  console.log(`[FeedDesc] Job ${jobId} done. ${totalRows} rows, ${errorLog.length} errors`);

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
