import { type Job as BullJob } from "bullmq";
import { prisma } from "@/lib/db";
import { chatCompletion } from "@/lib/openrouter-client";
import { postprocessLLMResponse } from "@/lib/text-processing";
import { publishProgress } from "@/lib/queue";
import * as fs from "fs";
import Papa from "papaparse";
import type { JobConfig } from "@/types";

const MAX_ROW_RETRIES = 3;

export async function translateProcessor(job: BullJob) {
  const { jobId } = job.data;
  const dbJob = await prisma.job.findUniqueOrThrow({ where: { id: jobId } });
  const config = dbJob.config as unknown as JobConfig;
  const apiKey = process.env.OPENROUTER_API_KEY!;

  await prisma.job.update({
    where: { id: jobId },
    data: { status: "RUNNING", startedAt: new Date() },
  });

  const fileContent = fs.readFileSync(dbJob.inputFileUrl, "utf-8");
  const parsed = Papa.parse(fileContent, { header: true });
  const rows = parsed.data as Record<string, string>[];
  const titleCol = config.titleCol || "title";
  const chunkSize = config.chunkSize || 10;
  const maxWorkers = config.maxWorkers || 5;

  const userPrompt = `Translate the following text from ${config.sourceLanguage || "English"} to ${config.targetLanguage || "Chinese"}:`;

  await prisma.job.update({
    where: { id: jobId },
    data: { totalRows: rows.length },
  });

  const results: string[] = new Array(rows.length).fill("");
  const errorLog: { row: number; error: string; retries: number }[] = [];
  let processed = 0;
  const startTime = Date.now();

  for (let i = 0; i < rows.length; i += chunkSize) {
    // Check cancellation
    if (i > 0 && i % (chunkSize * 5) === 0) {
      const fresh = await prisma.job.findUnique({ where: { id: jobId }, select: { status: true } });
      if (fresh?.status === "CANCELLED") {
        console.log(`[Translate] Job ${jobId} cancelled at row ${i}`);
        break;
      }
    }
    const chunk = rows.slice(i, Math.min(i + chunkSize, rows.length));

    const promises = chunk.map(async (row, idx) => {
      const globalIdx = i + idx;
      const text = row[titleCol] || "";

      for (let retry = 0; retry < MAX_ROW_RETRIES; retry++) {
        try {
          const raw = await chatCompletion(apiKey, {
            model: config.model || "openai/gpt-4o-mini",
            systemPrompt: config.systemPrompt || "You are a professional translator.",
            userPrompt: `${userPrompt}\n${text}`,
            maxTokens: config.maxTokens || 512,
            temperature: config.temperature || 0.7,
            topP: config.topP || 1.0,
            minP: config.minP || 0.0,
            topK: config.topK || 40,
            presencePenalty: config.presencePenalty || 0.0,
            frequencyPenalty: config.frequencyPenalty || 0.0,
            repetitionPenalty: config.repetitionPenalty || 1.0,
          });
          results[globalIdx] = postprocessLLMResponse(raw);
          return;
        } catch (err) {
          if (retry === MAX_ROW_RETRIES - 1) {
            errorLog.push({ row: globalIdx, error: (err as Error).message, retries: retry + 1 });
            results[globalIdx] = text; // Keep original on failure
          }
          await new Promise((r) => setTimeout(r, Math.pow(2, retry) * 1000));
        }
      }
    });

    // Process with concurrency limit
    for (let b = 0; b < promises.length; b += maxWorkers) {
      await Promise.all(promises.slice(b, b + maxWorkers));
    }

    processed += chunk.length;
    const elapsed = (Date.now() - startTime) / 1000;
    const speed = processed / elapsed;

    await prisma.job.update({
      where: { id: jobId },
      data: { processedRows: processed, failedRows: errorLog.length },
    });

    await publishProgress(jobId, {
      status: "RUNNING",
      processedRows: processed,
      totalRows: rows.length,
      failedRows: errorLog.length,
      currentPass: 1,
      totalPasses: 1,
      eta: Math.round((rows.length - processed) / speed),
      speed: Math.round(speed * 10) / 10,
    });
  }

  // Write output
  for (let idx = 0; idx < rows.length; idx++) {
    rows[idx]["translated_title"] = results[idx];
  }

  const outputPath = dbJob.inputFileUrl.replace(/(\.\w+)$/, `_translated$1`);
  fs.writeFileSync(outputPath, Papa.unparse(rows), "utf-8");

  await prisma.job.update({
    where: { id: jobId },
    data: {
      status: "COMPLETED",
      outputFileUrl: outputPath,
      completedAt: new Date(),
      errorLog: errorLog.length > 0 ? errorLog : undefined,
    },
  });

  await publishProgress(jobId, { status: "COMPLETED", processedRows: rows.length, totalRows: rows.length });
}
