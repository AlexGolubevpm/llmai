import type { WDTaggerResult } from "@/types";

const WD_TAGGER_URL =
  process.env.WD_TAGGER_URL ||
  "https://deepghs-wd-tagger-heatmap-more-models.hf.space";

const DEFAULT_MODEL = "SmilingWolf/wd-vit-tagger-v3";
const DEFAULT_THRESHOLD = 0.35;
const MAX_RETRIES = 4;
const MAX_CONCURRENT = 3; // HuggingFace Spaces are limited
const MIN_REQUEST_INTERVAL_MS = 500; // Rate limit: max 2 req/sec

let lastRequestTime = 0;
let activeRequests = 0;
const requestQueue: Array<{
  resolve: (v: void) => void;
  reject: (e: Error) => void;
}> = [];

interface GradioResponse {
  data: [
    unknown, // gallery
    unknown, // combined heatmap
    string, // caption
    string, // tags (comma-separated)
    { label: string; confidences: { label: string; confidence: number }[] }, // rating
    { label: string; confidences: { label: string; confidence: number }[] }, // character
    { label: string; confidences: { label: string; confidence: number }[] }, // general
  ];
}

function sleep(ms: number) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/**
 * Simple concurrency limiter for HuggingFace Space.
 * Limits to MAX_CONCURRENT parallel requests and MIN_REQUEST_INTERVAL_MS between requests.
 */
async function acquireSlot(): Promise<void> {
  if (activeRequests < MAX_CONCURRENT) {
    activeRequests++;
    // Rate limit: ensure minimum interval between requests
    const now = Date.now();
    const timeSinceLastRequest = now - lastRequestTime;
    if (timeSinceLastRequest < MIN_REQUEST_INTERVAL_MS) {
      await sleep(MIN_REQUEST_INTERVAL_MS - timeSinceLastRequest);
    }
    lastRequestTime = Date.now();
    return;
  }
  // Queue the request
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
 * Wake up the HuggingFace Space if it's sleeping.
 * Sends a lightweight request to trigger cold start.
 */
async function wakeUpSpace(): Promise<void> {
  try {
    const resp = await fetch(`${WD_TAGGER_URL}/info`, {
      method: "GET",
      signal: AbortSignal.timeout(10000),
    });
    if (resp.ok) return;
  } catch {
    // Space might be waking up, wait and continue
  }
  // Wait for cold start (typically 20-30 seconds)
  await sleep(15000);
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
    return await analyzeThumbnailInternal(imageUrl, model, threshold);
  } finally {
    releaseSlot();
  }
}

async function analyzeThumbnailInternal(
  imageUrl: string,
  model: string,
  threshold: number
): Promise<WDTaggerResult> {
  let lastError: Error | null = null;

  for (let attempt = 0; attempt < MAX_RETRIES; attempt++) {
    try {
      const resp = await fetch(`${WD_TAGGER_URL}/api/predict`, {
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
        }),
        signal: AbortSignal.timeout(60000), // 60s timeout for cold starts
      });

      if (resp.ok) {
        const result: GradioResponse = await resp.json();
        return parseGradioResponse(result);
      }

      // 503 = Space is sleeping/loading
      if (resp.status === 503) {
        console.warn(
          `WD Tagger Space is sleeping, waking up (attempt ${attempt + 1}/${MAX_RETRIES})`
        );
        await wakeUpSpace();
        continue;
      }

      // 429 = Rate limited
      if (resp.status === 429) {
        const backoff = Math.pow(2, attempt) * 2000; // 2s, 4s, 8s, 16s
        console.warn(`WD Tagger rate limited, waiting ${backoff}ms`);
        await sleep(backoff);
        continue;
      }

      const text = await resp.text();
      throw new Error(`WD Tagger API error ${resp.status}: ${text}`);
    } catch (err) {
      lastError = err as Error;
      const msg = (err as Error).message || "";

      // Timeout or network error — retry with backoff
      if (
        msg.includes("timeout") ||
        msg.includes("abort") ||
        msg.includes("fetch") ||
        msg.includes("ECONNREFUSED")
      ) {
        const backoff = Math.pow(2, attempt) * 2000;
        console.warn(
          `WD Tagger network error: ${msg}, retry ${attempt + 1}/${MAX_RETRIES} in ${backoff}ms`
        );
        if (attempt === 0) {
          // First failure might mean space is sleeping
          await wakeUpSpace();
        } else {
          await sleep(backoff);
        }
        continue;
      }

      // Non-retryable error
      throw err;
    }
  }

  throw lastError || new Error("WD Tagger: max retries exceeded");
}

function parseGradioResponse(result: GradioResponse): WDTaggerResult {
  const [, , caption, tagsStr, ratingData, characterData, generalData] =
    result.data;

  return {
    caption: caption || "",
    tags: tagsStr
      ? tagsStr
          .split(",")
          .map((t: string) => t.trim())
          .filter(Boolean)
      : [],
    rating: (ratingData?.confidences || []).map(
      (c: { label: string; confidence: number }) => ({
        label: c.label,
        confidence: c.confidence,
      })
    ),
    characters: (characterData?.confidences || []).map(
      (c: { label: string }) => c.label
    ),
    generalTags: (generalData?.confidences || []).map(
      (c: { label: string; confidence: number }) => ({
        tag: c.label,
        confidence: c.confidence,
      })
    ),
  };
}

/**
 * Analyze multiple thumbnails in parallel with concurrency control.
 * Returns results in the same order as input URLs.
 */
export async function analyzeThumbnailsBatch(
  imageUrls: string[],
  options?: { model?: string; threshold?: number; onProgress?: (done: number, total: number) => void }
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
