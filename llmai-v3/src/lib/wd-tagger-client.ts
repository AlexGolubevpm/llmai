import type { WDTaggerResult } from "@/types";

const WD_TAGGER_URL =
  process.env.WD_TAGGER_URL ||
  "https://deepghs-wd-tagger-heatmap-more-models.hf.space";

const DEFAULT_MODEL = "SmilingWolf/wd-vit-tagger-v3";
const DEFAULT_THRESHOLD = 0.35;
const MAX_RETRIES = 4;
const MAX_CONCURRENT = 2; // Conservative for queue-based API
const MIN_REQUEST_INTERVAL_MS = 1000;

let lastRequestTime = 0;
let activeRequests = 0;
const requestQueue: Array<{
  resolve: (v: void) => void;
  reject: (e: Error) => void;
}> = [];

function sleep(ms: number) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function randomSessionHash(): string {
  return Math.random().toString(36).substring(2, 12);
}

async function acquireSlot(): Promise<void> {
  if (activeRequests < MAX_CONCURRENT) {
    activeRequests++;
    const now = Date.now();
    const timeSinceLastRequest = now - lastRequestTime;
    if (timeSinceLastRequest < MIN_REQUEST_INTERVAL_MS) {
      await sleep(MIN_REQUEST_INTERVAL_MS - timeSinceLastRequest);
    }
    lastRequestTime = Date.now();
    return;
  }
  return new Promise((resolve, reject) => {
    requestQueue.push({ resolve, reject });
  });
}

function releaseSlot(): void {
  activeRequests--;
  if (requestQueue.length > 0) {
    const next = requestQueue.shift()!;
    activeRequests++;
    lastRequestTime = Date.now();
    next.resolve();
  }
}

/**
 * Call WD Tagger via Gradio Queue API.
 *
 * Flow:
 *   1. POST /queue/join — submit job to queue
 *   2. GET /queue/data?session_hash=xxx — SSE stream for result
 */
async function callGradioQueue(
  imageUrl: string,
  model: string,
  threshold: number
): Promise<WDTaggerResult> {
  const sessionHash = randomSessionHash();

  // Step 1: Join the queue
  const joinResp = await fetch(`${WD_TAGGER_URL}/queue/join`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      data: [
        {
          path: imageUrl,
          meta: { _type: "gradio.FileData" },
          orig_name: "image.jpg",
        },
        model,
        threshold,
      ],
      fn_index: 0,
      session_hash: sessionHash,
    }),
  });

  if (!joinResp.ok) {
    const text = await joinResp.text();
    throw new Error(`WD Tagger queue/join error ${joinResp.status}: ${text}`);
  }

  // Step 2: Listen for result via SSE
  const dataResp = await fetch(
    `${WD_TAGGER_URL}/queue/data?session_hash=${sessionHash}`,
    {
      method: "GET",
      headers: { Accept: "text/event-stream" },
    }
  );

  if (!dataResp.ok) {
    const text = await dataResp.text();
    throw new Error(`WD Tagger queue/data error ${dataResp.status}: ${text}`);
  }

  const body = await dataResp.text();

  // Parse SSE events to find the "process_completed" event
  const events = body.split("\n\n").filter(Boolean);
  for (const event of events) {
    const dataLine = event
      .split("\n")
      .find((l) => l.startsWith("data: "));
    if (!dataLine) continue;

    const json = dataLine.slice(6); // Remove "data: "
    let parsed;
    try {
      parsed = JSON.parse(json);
    } catch {
      continue;
    }

    if (parsed.msg === "process_completed" && parsed.output?.data) {
      return parseGradioOutput(parsed.output.data);
    }

    if (parsed.msg === "process_completed" && parsed.output?.error) {
      throw new Error(`WD Tagger error: ${parsed.output.error}`);
    }
  }

  throw new Error("WD Tagger: no result received from queue");
}

function parseGradioOutput(data: unknown[]): WDTaggerResult {
  const [
    , // gallery
    , // combined heatmap
    caption,
    tagsStr,
    ratingData,
    characterData,
    generalData,
  ] = data as [
    unknown,
    unknown,
    string,
    string,
    { label: string; confidences: { label: string; confidence: number }[] },
    { label: string; confidences: { label: string; confidence: number }[] },
    { label: string; confidences: { label: string; confidence: number }[] },
  ];

  return {
    caption: caption || "",
    tags: tagsStr
      ? tagsStr
          .split(",")
          .map((t: string) => t.trim())
          .filter(Boolean)
      : [],
    rating: (ratingData?.confidences || []).map((c) => ({
      label: c.label,
      confidence: c.confidence,
    })),
    characters: (characterData?.confidences || []).map((c) => c.label),
    generalTags: (generalData?.confidences || []).map((c) => ({
      tag: c.label,
      confidence: c.confidence,
    })),
  };
}

/**
 * Analyze a single thumbnail with retry and exponential backoff.
 */
export async function analyzeThumbnail(
  imageUrl: string,
  options?: { model?: string; threshold?: number }
): Promise<WDTaggerResult> {
  const model = options?.model || DEFAULT_MODEL;
  const threshold = options?.threshold || DEFAULT_THRESHOLD;

  await acquireSlot();
  try {
    return await analyzeThumbnailWithRetry(imageUrl, model, threshold);
  } finally {
    releaseSlot();
  }
}

async function analyzeThumbnailWithRetry(
  imageUrl: string,
  model: string,
  threshold: number
): Promise<WDTaggerResult> {
  let lastError: Error | null = null;

  for (let attempt = 0; attempt < MAX_RETRIES; attempt++) {
    try {
      return await callGradioQueue(imageUrl, model, threshold);
    } catch (err) {
      lastError = err as Error;
      const msg = (err as Error).message || "";

      console.warn(
        `WD Tagger attempt ${attempt + 1}/${MAX_RETRIES} failed: ${msg}`
      );

      // Wait before retry
      const backoff = Math.pow(2, attempt) * 2000;
      await sleep(backoff);
    }
  }

  throw lastError || new Error("WD Tagger: max retries exceeded");
}

/**
 * Analyze multiple thumbnails with concurrency control.
 */
export async function analyzeThumbnailsBatch(
  imageUrls: string[],
  options?: {
    model?: string;
    threshold?: number;
    onProgress?: (done: number, total: number) => void;
  }
): Promise<(WDTaggerResult | null)[]> {
  let completed = 0;
  const total = imageUrls.length;

  const promises = imageUrls.map(async (url) => {
    if (!url) return null;
    try {
      const result = await analyzeThumbnail(url, options);
      completed++;
      options?.onProgress?.(completed, total);
      return result;
    } catch (err) {
      completed++;
      options?.onProgress?.(completed, total);
      console.error(`WD Tagger failed for ${url}: ${(err as Error).message}`);
      return null;
    }
  });

  return Promise.all(promises);
}
