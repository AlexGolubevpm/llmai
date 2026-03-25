/**
 * Novita AI Batch API client.
 *
 * Flow:
 *   1. Build a .jsonl file with chat completion requests
 *   2. Upload the file via POST /files
 *   3. Create a batch via POST /batches with the file ID
 *   4. Poll batch status via GET /batches/{id}
 *   5. Download results via GET /files/{output_file_id}/content
 *
 * Each line in the .jsonl is:
 * {"custom_id":"row-0","method":"POST","url":"/chat/completions","body":{...}}
 *
 * Novita API is OpenAI-compatible, so batch format matches OpenAI's.
 */

import type { ChatCompletionParams } from "@/types";

const BASE_URL = process.env.NOVITA_BASE_URL || "https://api.novita.ai/openai";

function getHeaders(apiKey: string) {
  return {
    "Content-Type": "application/json",
    Authorization: `Bearer ${apiKey}`,
  };
}

function sleep(ms: number) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

// ---- JSONL Builder ----

export interface BatchRequest {
  customId: string;
  params: ChatCompletionParams;
}

export function buildBatchJsonl(requests: BatchRequest[]): string {
  return requests
    .map((req) => {
      const line = {
        custom_id: req.customId,
        method: "POST",
        url: "/chat/completions",
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

  const resp = await fetch(`${BASE_URL}/files`, {
    method: "POST",
    headers: { Authorization: `Bearer ${apiKey}` },
    body: formData,
  });

  if (!resp.ok) {
    const text = await resp.text();
    throw new Error(`File upload failed ${resp.status}: ${text}`);
  }

  const data = await resp.json();
  return data.id; // file ID
}

// ---- Batch Create ----

export interface BatchJob {
  id: string;
  status: string; // validating, in_progress, completed, failed, expired, cancelled
  input_file_id: string;
  output_file_id: string | null;
  error_file_id: string | null;
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
  const resp = await fetch(`${BASE_URL}/batches`, {
    method: "POST",
    headers: getHeaders(apiKey),
    body: JSON.stringify({
      input_file_id: inputFileId,
      endpoint: "/chat/completions",
      completion_window: "24h",
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
  const resp = await fetch(`${BASE_URL}/batches/${batchId}`, {
    method: "GET",
    headers: getHeaders(apiKey),
  });

  if (!resp.ok) {
    const text = await resp.text();
    throw new Error(`Batch status failed ${resp.status}: ${text}`);
  }

  return resp.json();
}

// ---- Poll Until Complete ----

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
  const timeout = options?.timeoutMs || 3600000; // 1 hour
  const startTime = Date.now();

  while (true) {
    const batch = await getBatchStatus(apiKey, batchId);
    options?.onProgress?.(batch);

    if (
      batch.status === "completed" ||
      batch.status === "failed" ||
      batch.status === "expired" ||
      batch.status === "cancelled"
    ) {
      return batch;
    }

    if (Date.now() - startTime > timeout) {
      throw new Error(`Batch ${batchId} timed out after ${timeout}ms`);
    }

    await sleep(pollInterval);
  }
}

// ---- Download Results ----

export interface BatchResultLine {
  custom_id: string;
  response: {
    status_code: number;
    body: {
      choices: Array<{
        message: {
          role: string;
          content: string;
        };
      }>;
    };
  } | null;
  error: { message: string; code: string } | null;
}

export async function downloadBatchResults(
  apiKey: string,
  outputFileId: string
): Promise<BatchResultLine[]> {
  const resp = await fetch(`${BASE_URL}/files/${outputFileId}/content`, {
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
  // 1. Build JSONL
  const jsonl = buildBatchJsonl(requests);

  // 2. Upload
  const fileId = await uploadBatchFile(apiKey, jsonl);

  // 3. Create batch
  const batch = await createBatch(apiKey, fileId);

  // 4. Poll until done
  const finalBatch = await pollBatchUntilDone(apiKey, batch.id, {
    pollIntervalMs: options?.pollIntervalMs,
    timeoutMs: options?.timeoutMs,
    onProgress: options?.onProgress,
  });

  // 5. Download results
  const results = new Map<string, string>();
  const errors = new Map<string, string>();

  if (finalBatch.output_file_id) {
    const lines = await downloadBatchResults(apiKey, finalBatch.output_file_id);
    for (const line of lines) {
      if (line.response && line.response.status_code === 200) {
        const content =
          line.response.body.choices?.[0]?.message?.content || "";
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
