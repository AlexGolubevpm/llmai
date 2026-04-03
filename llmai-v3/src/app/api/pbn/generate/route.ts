import { NextRequest, NextResponse } from "next/server";

const BASE_URL = process.env.OPENROUTER_BASE_URL || "https://openrouter.ai/api/v1";
const MIN_DELAY_MS = 300;

function sleep(ms: number) {
  return new Promise((r) => setTimeout(r, ms));
}

export async function POST(req: NextRequest) {
  try {
    const apiKey = process.env.OPENROUTER_API_KEY;
    if (!apiKey) {
      return NextResponse.json({ error: "OPENROUTER_API_KEY not set" }, { status: 500 });
    }

    const body = await req.json();
    const { model, prompt, temperature, maxTokens } = body;

    if (!prompt) {
      return NextResponse.json({ error: "Промпт обязателен" }, { status: 400 });
    }

    await sleep(MIN_DELAY_MS);

    const resp = await fetch(`${BASE_URL}/chat/completions`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${apiKey}`,
        "HTTP-Referer": process.env.NEXT_PUBLIC_APP_URL || "http://localhost:3000",
        "X-Title": "LLMAI v3.0",
      },
      body: JSON.stringify({
        model: model || "openai/gpt-4o-mini",
        messages: [{ role: "user", content: prompt }],
        max_tokens: maxTokens || 500,
        temperature: temperature || 0.9,
        provider: { allow_fallbacks: true },
      }),
    });

    if (resp.status === 429) {
      const wait = parseInt(resp.headers.get("retry-after") || "5") * 1000;
      await sleep(wait);
      return NextResponse.json({ error: "Rate limited, попробуйте позже" }, { status: 429 });
    }

    if (!resp.ok) {
      const text = await resp.text();
      return NextResponse.json({ error: `OpenRouter ${resp.status}: ${text.slice(0, 200)}` }, { status: resp.status });
    }

    const data = await resp.json();
    const text = data.choices?.[0]?.message?.content || "";

    return NextResponse.json({ text });
  } catch (err) {
    console.error("PBN generate error:", err);
    return NextResponse.json({ error: (err as Error).message }, { status: 500 });
  }
}
