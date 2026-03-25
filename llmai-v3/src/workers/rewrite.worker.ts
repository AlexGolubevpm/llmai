import { type Job as BullJob } from "bullmq";
import { prisma } from "@/lib/db";
import { chatCompletion } from "@/lib/novita-client";
import { postprocessLLMResponse, cleanText, type StopWordEntry } from "@/lib/text-processing";
import { publishProgress } from "@/lib/queue";
import * as fs from "fs";
import Papa from "papaparse";
import type { JobConfig } from "@/types";

const MAX_ROW_RETRIES = 3;
const FAILED_ROWS_EXTRA_CYCLES = 2;

interface RowResult {
  index: number;
  value: string;
  error?: string;
}

async function processRow(
  apiKey: string,
  config: JobConfig,
  text: string
): Promise<string> {
  const raw = await chatCompletion(apiKey, {
    model: config.model || "meta-llama/llama-3.1-8b-instruct",
    systemPrompt: config.systemPrompt || "You are a helpful assistant.",
    userPrompt: `${config.userPrompt || ""}\n${text}`,
    maxTokens: config.maxTokens || 512,
    temperature: config.temperature || 0.7,
    topP: config.topP || 1.0,
    minP: config.minP || 0.0,
    topK: config.topK || 40,
    presencePenalty: config.presencePenalty || 0.0,
    frequencyPenalty: config.frequencyPenalty || 0.0,
    repetitionPenalty: config.repetitionPenalty || 1.0,
  });
  return postprocessLLMResponse(raw);
}

async function loadStopWords(): Promise<StopWordEntry[]> {
  const words = await prisma.stopWord.findMany({ where: { isActive: true } });
  return words.map((w) => ({ word: w.word, replacement: w.replacement }));
}

export async function rewriteProcessor(job: BullJob) {
  const { jobId } = job.data;
  const dbJob = await prisma.job.findUniqueOrThrow({ where: { id: jobId } });
  const config = dbJob.config as unknown as JobConfig;
  const apiKey = process.env.NOVITA_API_KEY!;

  await prisma.job.update({
    where: { id: jobId },
    data: { status: "RUNNING", startedAt: new Date() },
  });

  // Read input file
  const fileContent = fs.readFileSync(dbJob.inputFileUrl, "utf-8");
  const parsed = Papa.parse(fileContent, { header: true });
  const rows = parsed.data as Record<string, string>[];
  const titleCol = config.titleCol || "title";
  const totalPasses = config.multiplier || 1;
  const chunkSize = config.chunkSize || 10;
  const maxWorkers = config.maxWorkers || 5;

  await prisma.job.update({
    where: { id: jobId },
    data: { totalRows: rows.length, totalPasses },
  });

  const stopWords = config.applyStopWords ? await loadStopWords() : [];
  const errorLog: { row: number; error: string; retries: number }[] = [];
  let currentData = rows.map((r) => r[titleCol] || "");

  // Process each pass
  for (let pass = 1; pass <= totalPasses; pass++) {
    await prisma.job.update({
      where: { id: jobId },
      data: { currentPass: pass, processedRows: 0 },
    });

    const results: string[] = new Array(currentData.length).fill("");
    const failedIndices: Set<number> = new Set();
    let processedInPass = 0;
    const startTime = Date.now();

    // Process in chunks
    for (let i = 0; i < currentData.length; i += chunkSize) {
      const chunk = currentData.slice(i, Math.min(i + chunkSize, currentData.length));
      const chunkPromises: Promise<RowResult>[] = chunk.map((text, idx) => {
        const globalIdx = i + idx;
        return (async (): Promise<RowResult> => {
          for (let retry = 0; retry < MAX_ROW_RETRIES; retry++) {
            try {
              const result = await processRow(apiKey, config, text);
              return { index: globalIdx, value: result };
            } catch (err) {
              if (retry === MAX_ROW_RETRIES - 1) {
                const errMsg = (err as Error).message;
                errorLog.push({ row: globalIdx, error: errMsg, retries: retry + 1 });
                failedIndices.add(globalIdx);
                return { index: globalIdx, value: text, error: errMsg };
              }
              // Wait before retry
              await new Promise((r) => setTimeout(r, Math.pow(2, retry) * 1000));
            }
          }
          return { index: i, value: text };
        })();
      });

      // Limit concurrency manually
      const batchSize = maxWorkers;
      for (let b = 0; b < chunkPromises.length; b += batchSize) {
        const batch = chunkPromises.slice(b, b + batchSize);
        const batchResults = await Promise.all(batch);
        for (const r of batchResults) {
          results[r.index] = r.value;
        }
      }

      processedInPass += chunk.length;
      const elapsed = (Date.now() - startTime) / 1000;
      const speed = processedInPass / elapsed;
      const remaining = ((currentData.length - processedInPass) / speed) || 0;

      await prisma.job.update({
        where: { id: jobId },
        data: {
          processedRows: processedInPass,
          failedRows: failedIndices.size,
          errorLog: errorLog.length > 0 ? errorLog : undefined,
        },
      });

      await publishProgress(jobId, {
        status: "RUNNING",
        processedRows: processedInPass,
        totalRows: currentData.length,
        failedRows: failedIndices.size,
        currentPass: pass,
        totalPasses,
        eta: Math.round(remaining),
        speed: Math.round(speed * 10) / 10,
      });
    }

    // Retry failed rows (up to FAILED_ROWS_EXTRA_CYCLES times)
    for (let cycle = 0; cycle < FAILED_ROWS_EXTRA_CYCLES && failedIndices.size > 0; cycle++) {
      const toRetry = [...failedIndices];
      for (const idx of toRetry) {
        try {
          results[idx] = await processRow(apiKey, config, currentData[idx]);
          failedIndices.delete(idx);
        } catch {
          // Still failed, keep in set
        }
      }
    }

    // Apply stop words if enabled
    if (stopWords.length > 0) {
      for (let idx = 0; idx < results.length; idx++) {
        results[idx] = cleanText(results[idx], stopWords);
      }
    }

    // Add results as new column
    const colName = `rewrite_${pass}`;
    for (let idx = 0; idx < rows.length; idx++) {
      rows[idx][colName] = results[idx];
    }

    // Next pass uses this pass's results
    currentData = results;
  }

  // Write output file
  const outputPath = dbJob.inputFileUrl.replace(/(\.\w+)$/, `_result$1`);
  const output = Papa.unparse(rows);
  fs.writeFileSync(outputPath, output, "utf-8");

  await prisma.job.update({
    where: { id: jobId },
    data: {
      status: "COMPLETED",
      outputFileUrl: outputPath,
      completedAt: new Date(),
      errorLog: errorLog.length > 0 ? errorLog : undefined,
      failedRows: 0, // Reset after final retries
    },
  });

  await publishProgress(jobId, {
    status: "COMPLETED",
    processedRows: rows.length,
    totalRows: rows.length,
    currentPass: totalPasses,
    totalPasses,
  });
}
