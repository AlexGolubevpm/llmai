/**
 * OpenRouter API client.
 *
 * OpenRouter is OpenAI-compatible:
 * - Base URL: https://openrouter.ai/api/v1
 * - Auth: Authorization: Bearer <key>
 * - Endpoints: /chat/completions, /models
 * - Extra header: HTTP-Referer, X-Title for rankings
 */

import type { ChatCompletionParams, LLMModel } from "@/types";
import { redis } from "./redis";

const BASE_URL = process.env.OPENROUTER_BASE_URL || "https://openrouter.ai/api/v1";
const MAX_RETRIES = 5;
const MODELS_CACHE_KEY = "openrouter:models";
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
        // Check for Retry-After header
        const retryAfter = resp.headers.get("retry-after");
        const backoff = retryAfter
          ? parseInt(retryAfter) * 1000
          : Math.pow(2, attempt) * 1000;
        console.warn(
          `OpenRouter API ${resp.status}, retry ${attempt + 1}/${retries} in ${backoff}ms`
        );
        await sleep(backoff);
        continue;
      }

      // Non-retryable error
      const text = await resp.text();
      throw new Error(`OpenRouter API error ${resp.status}: ${text}`);
    } catch (err) {
      lastError = err as Error;
      if (
        attempt < retries - 1 &&
        (err as Error).message?.includes("fetch")
      ) {
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
    "HTTP-Referer": process.env.NEXT_PUBLIC_APP_URL || "http://localhost:3000",
    "X-Title": "LLMAI v3.0",
  };
}

export async function listModels(apiKey: string): Promise<LLMModel[]> {
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
  const models: LLMModel[] = (data.data || []).map(
    (m: { id: string; name?: string; context_length?: number; pricing?: { prompt?: string; completion?: string } }) => ({
      id: m.id,
      object: "model",
      name: m.name || m.id,
      contextLength: m.context_length || 0,
      promptPrice: m.pricing?.prompt || "0",
      completionPrice: m.pricing?.completion || "0",
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
  const payload: Record<string, unknown> = {
    model: params.model,
    messages: [
      { role: "system", content: params.systemPrompt },
      { role: "user", content: params.userPrompt },
    ],
    max_tokens: params.maxTokens,
    temperature: params.temperature,
    top_p: params.topP,
    frequency_penalty: params.frequencyPenalty,
    presence_penalty: params.presencePenalty,
    repetition_penalty: params.repetitionPenalty,
  };

  // OpenRouter supports top_k and min_p via provider routing
  if (params.topK && params.topK > 0) {
    payload.top_k = params.topK;
  }
  if (params.minP && params.minP > 0) {
    payload.min_p = params.minP;
  }

  const resp = await fetchWithRetry(`${BASE_URL}/chat/completions`, {
    method: "POST",
    headers: getHeaders(apiKey),
    body: JSON.stringify(payload),
  });

  const data = await resp.json();
  return data.choices?.[0]?.message?.content || "";
}
