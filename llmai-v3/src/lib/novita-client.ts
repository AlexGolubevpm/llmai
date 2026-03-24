import type { ChatCompletionParams, NovitaModel } from "@/types";
import { redis } from "./redis";

const BASE_URL = process.env.NOVITA_BASE_URL || "https://api.novita.ai/openai";
const MAX_RETRIES = 5;
const MODELS_CACHE_KEY = "novita:models";
const MODELS_CACHE_TTL = 3600; // 1 hour

function sleep(ms: number) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function fetchWithRetry(
  url: string,
  options: RequestInit,
  retries = MAX_RETRIES
): Promise<Response> {
  let lastError: Error | null = null;

  for (let attempt = 0; attempt < retries; attempt++) {
    try {
      const resp = await fetch(url, options);

      if (resp.ok) return resp;

      if (resp.status === 429 || resp.status >= 500) {
        const backoff = Math.pow(2, attempt) * 1000; // 1s, 2s, 4s, 8s, 16s
        console.warn(
          `Novita API ${resp.status}, retry ${attempt + 1}/${retries} in ${backoff}ms`
        );
        await sleep(backoff);
        continue;
      }

      // Non-retryable error
      const text = await resp.text();
      throw new Error(`Novita API error ${resp.status}: ${text}`);
    } catch (err) {
      lastError = err as Error;
      if (attempt < retries - 1 && (err as Error).message?.includes("fetch")) {
        const backoff = Math.pow(2, attempt) * 1000;
        await sleep(backoff);
        continue;
      }
    }
  }

  throw lastError || new Error("Max retries exceeded");
}

function getHeaders(apiKey: string) {
  return {
    "Content-Type": "application/json",
    Authorization: `Bearer ${apiKey}`,
  };
}

export async function listModels(apiKey: string): Promise<NovitaModel[]> {
  // Check cache first
  const cached = await redis.get(MODELS_CACHE_KEY);
  if (cached) {
    return JSON.parse(cached);
  }

  const resp = await fetchWithRetry(`${BASE_URL}/models`, {
    method: "GET",
    headers: getHeaders(apiKey),
  });

  const data = await resp.json();
  const models: NovitaModel[] = (data.data || []).map(
    (m: { id: string; object?: string }) => ({
      id: m.id,
      object: m.object || "model",
    })
  );

  // Cache for 1 hour
  await redis.setex(MODELS_CACHE_KEY, MODELS_CACHE_TTL, JSON.stringify(models));

  return models;
}

export async function chatCompletion(
  apiKey: string,
  params: ChatCompletionParams
): Promise<string> {
  const payload = {
    model: params.model,
    messages: [
      { role: "system", content: params.systemPrompt },
      { role: "user", content: params.userPrompt },
    ],
    max_tokens: params.maxTokens,
    temperature: params.temperature,
    top_p: params.topP,
    min_p: params.minP,
    top_k: params.topK,
    presence_penalty: params.presencePenalty,
    frequency_penalty: params.frequencyPenalty,
    repetition_penalty: params.repetitionPenalty,
  };

  const resp = await fetchWithRetry(`${BASE_URL}/chat/completions`, {
    method: "POST",
    headers: getHeaders(apiKey),
    body: JSON.stringify(payload),
  });

  const data = await resp.json();
  return data.choices?.[0]?.message?.content || "";
}
