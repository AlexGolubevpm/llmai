/**
 * Novita AI Batch API client.
 *
 * Flow:
 *   1. Build a .jsonl file — each line: {"custom_id":"row-0","body":{chat completion params}}
 *   2. Upload the file via POST /v1/files (purpose: "batch")
 *   3. Create a batch via POST /v1/batches
 *   4. Poll batch status via GET /v1/batches/{id}
 *   5. Download results via GET /v1/files/{output_file_id}/content
 *
 * Limits: max 50,000 requests/batch, 100MB file, 48h window, same model per batch.
 * Input files retained 15 days, output files 30 days.
 */

import type { ChatCompletionParams } from "@/types";

const BASE_URL = process.env.NOVITA_BASE_URL || "https://api.novita.ai/openai";
// Batch endpoints use /v1/ prefix
const BATCH_BASE = BASE_URL.replace(/\/openai\/?$/, "/openai/v1");

function sleep(ms: number) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

// ---- JSONL Builder ----

export interface BatchRequest {
  customId: string;
  params: ChatCompletionParams;
}

/**
 * Build JSONL for Novita Batch API.
 * Format per line: {"custom_id":"...","body":{...}}
 * All requests MUST use the same model.
 */
export function buildBatchJsonl(requests: BatchRequest[]): string {
  return requests
    .map((req) => {
      const line = {
        custom_id: req.customId,
        body: {
          model: req.params.model,
          messages: [
            { role: "system", content: req.params.systemPrompt },
            { role: "user", content: req.params.userPrompt },
          ],
          max_tokens: req.params.maxTokens,
          temperature: req.params.temperature,
          top_p: req.params.topP,
          min_p: req.params.minP,
          top_k: req.params.topK,
          presence_penalty: req.params.presencePenalty,
          frequency_penalty: req.params.frequencyPenalty,
          repetition_penalty: req.params.repetitionPenalty,
        },
      };
      return JSON.stringify(line);
    })
    .join("\n");
}

// ---- File Upload ----

export async function uploadBatchFile(
  apiKey: string,
  jsonlContent: string
): Promise<string> {
  const blob = new Blob([jsonlContent], { type: "application/jsonl" });
  const formData = new FormData();
  formData.append("file", blob, "batch_input.jsonl");
  formData.append("purpose", "batch");

  const resp = await fetch(`${BATCH_BASE}/files`, {
    method: "POST",
    headers: { Authorization: `Bearer ${apiKey}` },
    body: formData,
  });

  if (!resp.ok) {
    const text = await resp.text();
    throw new Error(`Batch file upload failed ${resp.status}: ${text}`);
  }

  const data = await resp.json();
  return data.id; // e.g. "file_d2cor0es1cas73c0cj60"
}

// ---- Batch Create ----

export interface BatchJob {
  id: string;
  status: string; // VALIDATING, PROGRESS, COMPLETED, FAILED, EXPIRED, CANCELLING, CANCELLED
  input_file_id: string;
  output_file_id: string;
  error_file_id: string;
  total: number;
  completed: number;
  failed: number;
  request_counts: {
    total: number;
    completed: number;
    failed: number;
  } | null;
}

export async function createBatch(
  apiKey: string,
  inputFileId: string
): Promise<BatchJob> {
  const resp = await fetch(`${BATCH_BASE}/batches`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${apiKey}`,
    },
    body: JSON.stringify({
      input_file_id: inputFileId,
      endpoint: "/v1/chat/completions",
      completion_window: "48h",
    }),
  });

  if (!resp.ok) {
    const text = await resp.text();
    throw new Error(`Batch create failed ${resp.status}: ${text}`);
  }

  return resp.json();
}

// ---- Batch Status ----

export async function getBatchStatus(
  apiKey: string,
  batchId: string
): Promise<BatchJob> {
  const resp = await fetch(`${BATCH_BASE}/batches/${batchId}`, {
    method: "GET",
    headers: { Authorization: `Bearer ${apiKey}` },
  });

  if (!resp.ok) {
    const text = await resp.text();
    throw new Error(`Batch status failed ${resp.status}: ${text}`);
  }

  return resp.json();
}

// ---- Cancel Batch ----

export async function cancelBatch(
  apiKey: string,
  batchId: string
): Promise<BatchJob> {
  const resp = await fetch(`${BATCH_BASE}/batches/${batchId}/cancel`, {
    method: "POST",
    headers: { Authorization: `Bearer ${apiKey}` },
  });

  if (!resp.ok) {
    const text = await resp.text();
    throw new Error(`Batch cancel failed ${resp.status}: ${text}`);
  }

  return resp.json();
}

// ---- Poll Until Complete ----

const TERMINAL_STATUSES = [
  "COMPLETED",
  "FAILED",
  "EXPIRED",
  "CANCELLED",
  // Lowercase fallback in case API returns mixed case
  "completed",
  "failed",
  "expired",
  "cancelled",
];

export async function pollBatchUntilDone(
  apiKey: string,
  batchId: string,
  options?: {
    pollIntervalMs?: number;
    timeoutMs?: number;
    onProgress?: (batch: BatchJob) => void;
  }
): Promise<BatchJob> {
  const pollInterval = options?.pollIntervalMs || 10000; // 10 seconds
  const timeout = options?.timeoutMs || 7200000; // 2 hours
  const startTime = Date.now();

  while (true) {
    const batch = await getBatchStatus(apiKey, batchId);
    options?.onProgress?.(batch);

    const status = batch.status.toUpperCase();
    if (TERMINAL_STATUSES.includes(status)) {
      return batch;
    }

    if (Date.now() - startTime > timeout) {
      // Try to cancel the batch before throwing
      try {
        await cancelBatch(apiKey, batchId);
      } catch { /* ignore */ }
      throw new Error(`Batch ${batchId} timed out after ${timeout}ms (status: ${batch.status})`);
    }

    await sleep(pollInterval);
  }
}

// ---- Download Results ----

export interface BatchResultLine {
  custom_id: string;
  id?: string;
  response: {
    status_code: number;
    body?: {
      choices?: Array<{
        message?: {
          role: string;
          content: string;
        };
      }>;
    };
  } | null;
  error: { message: string; code?: string } | null;
  request_id?: string;
}

export async function downloadBatchResults(
  apiKey: string,
  fileId: string
): Promise<BatchResultLine[]> {
  const resp = await fetch(`${BATCH_BASE}/files/${fileId}/content`, {
    method: "GET",
    headers: { Authorization: `Bearer ${apiKey}` },
  });

  if (!resp.ok) {
    const text = await resp.text();
    throw new Error(`File download failed ${resp.status}: ${text}`);
  }

  const text = await resp.text();
  return text
    .trim()
    .split("\n")
    .filter(Boolean)
    .map((line) => JSON.parse(line) as BatchResultLine);
}

// ---- High-Level: Submit and Wait ----

/**
 * Full batch pipeline: build JSONL → upload → create batch → poll → download results.
 * Returns a Map<customId, content> for successful results.
 */
export async function submitBatchAndWait(
  apiKey: string,
  requests: BatchRequest[],
  options?: {
    onProgress?: (batch: BatchJob) => void;
    pollIntervalMs?: number;
    timeoutMs?: number;
  }
): Promise<{
  results: Map<string, string>;
  errors: Map<string, string>;
}> {
  if (requests.length === 0) {
    return { results: new Map(), errors: new Map() };
  }

  if (requests.length > 50000) {
    throw new Error(`Batch too large: ${requests.length} requests (max 50,000)`);
  }

  // 1. Build JSONL
  const jsonl = buildBatchJsonl(requests);
  console.log(`[Batch] Built JSONL: ${requests.length} requests, ${jsonl.length} bytes`);

  // 2. Upload
  const fileId = await uploadBatchFile(apiKey, jsonl);
  console.log(`[Batch] Uploaded file: ${fileId}`);

  // 3. Create batch
  const batch = await createBatch(apiKey, fileId);
  console.log(`[Batch] Created batch: ${batch.id} (status: ${batch.status})`);

  // 4. Poll until done
  const finalBatch = await pollBatchUntilDone(apiKey, batch.id, {
    pollIntervalMs: options?.pollIntervalMs,
    timeoutMs: options?.timeoutMs,
    onProgress: options?.onProgress,
  });
  console.log(`[Batch] Final status: ${finalBatch.status}, completed: ${finalBatch.completed}, failed: ${finalBatch.failed}`);

  // 5. Download results
  const results = new Map<string, string>();
  const errors = new Map<string, string>();

  if (finalBatch.status.toUpperCase() === "FAILED") {
    throw new Error(`Batch ${batch.id} failed`);
  }

  if (finalBatch.output_file_id) {
    const lines = await downloadBatchResults(apiKey, finalBatch.output_file_id);
    for (const line of lines) {
      if (line.response && line.response.status_code === 200) {
        const content =
          line.response.body?.choices?.[0]?.message?.content || "";
        results.set(line.custom_id, content);
      } else if (line.error) {
        errors.set(line.custom_id, line.error.message);
      }
    }
  }

  // Download errors file if present
  if (finalBatch.error_file_id) {
    try {
      const errorLines = await downloadBatchResults(
        apiKey,
        finalBatch.error_file_id
      );
      for (const line of errorLines) {
        if (line.error) {
          errors.set(line.custom_id, line.error.message);
        }
      }
    } catch {
      // Error file might not always be available
    }
  }

  return { results, errors };
}
