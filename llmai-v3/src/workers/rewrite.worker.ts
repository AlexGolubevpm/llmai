import { type Job as BullJob } from "bullmq";
import { prisma } from "@/lib/db";
import { chatCompletion } from "@/lib/novita-client";
import {
  submitBatchAndWait,
  type BatchRequest,
  type BatchJob,
} from "@/lib/novita-batch-client";
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
const FAILED_ROWS_EXTRA_CYCLES = 2;
const BATCH_API_THRESHOLD = 50; // Use batch API for 50+ rows

interface RowResult {
  index: number;
  value: string;
  error?: string;
}

function buildChatParams(config: JobConfig, text: string) {
  return {
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
  };
}

async function processRowRealtime(
  apiKey: string,
  config: JobConfig,
  text: string
): Promise<string> {
  const raw = await chatCompletion(apiKey, buildChatParams(config, text));
  return postprocessLLMResponse(raw);
}

async function loadStopWords(): Promise<StopWordEntry[]> {
  const words = await prisma.stopWord.findMany({ where: { isActive: true } });
  return words.map((w) => ({ word: w.word, replacement: w.replacement }));
}

/**
 * Process a single pass using the Novita Batch API.
 * Submits all rows as a single batch, polls for completion, parses results.
 */
async function processPassBatch(
  apiKey: string,
  config: JobConfig,
  data: string[],
  jobId: string,
  pass: number,
  totalPasses: number,
  startTime: number
): Promise<{ results: string[]; failedIndices: Set<number>; errors: { row: number; error: string; retries: number }[] }> {
  const results: string[] = new Array(data.length).fill("");
  const failedIndices = new Set<number>();
  const errors: { row: number; error: string; retries: number }[] = [];

  // Build batch requests
  const batchRequests: BatchRequest[] = data.map((text, idx) => ({
    customId: `row-${idx}`,
    params: buildChatParams(config, text),
  }));

  try {
    const { results: batchResults, errors: batchErrors } =
      await submitBatchAndWait(apiKey, batchRequests, {
        pollIntervalMs: 5000,
        timeoutMs: 7200000, // 2 hours
        onProgress: (batch: BatchJob) => {
          const completed = batch.request_counts?.completed || 0;
          const total = batch.request_counts?.total || data.length;
          publishProgress(jobId, {
            status: "RUNNING",
            processedRows: completed,
            totalRows: total,
            failedRows: batch.request_counts?.failed || 0,
            currentPass: pass,
            totalPasses,
            eta: estimateEta(startTime, completed, total),
            speed: calcSpeed(startTime, completed),
          });
        },
      });

    // Map results back
    for (let idx = 0; idx < data.length; idx++) {
      const customId = `row-${idx}`;
      const content = batchResults.get(customId);
      if (content) {
        results[idx] = postprocessLLMResponse(content);
      } else {
        const errorMsg = batchErrors.get(customId) || "No response from batch";
        errors.push({ row: idx, error: errorMsg, retries: 1 });
        failedIndices.add(idx);
        results[idx] = data[idx]; // Keep original on failure
      }
    }
  } catch (err) {
    // Batch API failed entirely — fall back to realtime
    console.error(`Batch API failed: ${(err as Error).message}, falling back to realtime`);
    return processPassRealtime(apiKey, config, data, jobId, pass, totalPasses, startTime);
  }

  return { results, failedIndices, errors };
}

/**
 * Process a single pass using realtime chat completions (original approach).
 */
async function processPassRealtime(
  apiKey: string,
  config: JobConfig,
  data: string[],
  jobId: string,
  pass: number,
  totalPasses: number,
  startTime: number
): Promise<{ results: string[]; failedIndices: Set<number>; errors: { row: number; error: string; retries: number }[] }> {
  const results: string[] = new Array(data.length).fill("");
  const failedIndices = new Set<number>();
  const errors: { row: number; error: string; retries: number }[] = [];
  const chunkSize = config.chunkSize || 10;
  const maxWorkers = config.maxWorkers || 5;
  let processedInPass = 0;

  for (let i = 0; i < data.length; i += chunkSize) {
    const chunk = data.slice(i, Math.min(i + chunkSize, data.length));
    const chunkPromises: Promise<RowResult>[] = chunk.map((text, idx) => {
      const globalIdx = i + idx;
      return (async (): Promise<RowResult> => {
        for (let retry = 0; retry < MAX_ROW_RETRIES; retry++) {
          try {
            const result = await processRowRealtime(apiKey, config, text);
            return { index: globalIdx, value: result };
          } catch (err) {
            if (retry === MAX_ROW_RETRIES - 1) {
              const errMsg = (err as Error).message;
              errors.push({ row: globalIdx, error: errMsg, retries: retry + 1 });
              failedIndices.add(globalIdx);
              return { index: globalIdx, value: text, error: errMsg };
            }
            await new Promise((r) =>
              setTimeout(r, Math.pow(2, retry) * 1000)
            );
          }
        }
        return { index: i, value: text };
      })();
    });

    for (let b = 0; b < chunkPromises.length; b += maxWorkers) {
      const batch = chunkPromises.slice(b, b + maxWorkers);
      const batchResults = await Promise.all(batch);
      for (const r of batchResults) {
        results[r.index] = r.value;
      }
    }

    processedInPass += chunk.length;
    const elapsed = (Date.now() - startTime) / 1000;
    const speed = processedInPass / elapsed;
    const remaining = (data.length - processedInPass) / speed || 0;

    await prisma.job.update({
      where: { id: jobId },
      data: {
        processedRows: processedInPass,
        failedRows: failedIndices.size,
        errorLog: errors.length > 0 ? errors : undefined,
      },
    });

    await publishProgress(jobId, {
      status: "RUNNING",
      processedRows: processedInPass,
      totalRows: data.length,
      failedRows: failedIndices.size,
      currentPass: pass,
      totalPasses,
      eta: Math.round(remaining),
      speed: Math.round(speed * 10) / 10,
    });
  }

  return { results, failedIndices, errors };
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
  const useBatchApi = rows.length >= BATCH_API_THRESHOLD;

  await prisma.job.update({
    where: { id: jobId },
    data: { totalRows: rows.length, totalPasses },
  });

  const stopWords = config.applyStopWords ? await loadStopWords() : [];
  const allErrors: { row: number; error: string; retries: number }[] = [];
  let currentData = rows.map((r) => r[titleCol] || "");

  // Process each pass
  for (let pass = 1; pass <= totalPasses; pass++) {
    await prisma.job.update({
      where: { id: jobId },
      data: { currentPass: pass, processedRows: 0 },
    });

    const startTime = Date.now();

    // Choose batch or realtime based on row count
    const { results, failedIndices, errors } = useBatchApi
      ? await processPassBatch(apiKey, config, currentData, jobId, pass, totalPasses, startTime)
      : await processPassRealtime(apiKey, config, currentData, jobId, pass, totalPasses, startTime);

    allErrors.push(...errors);

    // Retry failed rows (realtime fallback for batch failures too)
    for (
      let cycle = 0;
      cycle < FAILED_ROWS_EXTRA_CYCLES && failedIndices.size > 0;
      cycle++
    ) {
      const toRetry = [...failedIndices];
      for (const idx of toRetry) {
        try {
          results[idx] = await processRowRealtime(
            apiKey,
            config,
            currentData[idx]
          );
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
      errorLog: allErrors.length > 0 ? allErrors : undefined,
      failedRows: 0,
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

function estimateEta(startTime: number, done: number, total: number): number {
  const elapsed = (Date.now() - startTime) / 1000;
  if (done === 0) return 0;
  return Math.round(((total - done) / (done / elapsed)));
}

function calcSpeed(startTime: number, done: number): number {
  const elapsed = (Date.now() - startTime) / 1000;
  if (elapsed === 0) return 0;
  return Math.round((done / elapsed) * 10) / 10;
}
