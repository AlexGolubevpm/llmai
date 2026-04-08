import { NextRequest, NextResponse } from "next/server";
import { prisma } from "@/lib/db";
import { chatCompletion } from "@/lib/openrouter-client";
import { postprocessLLMResponse, cleanText, type StopWordEntry } from "@/lib/text-processing";
import { publishProgress } from "@/lib/queue";
import { getQueueByType } from "@/lib/queue";
import * as fs from "fs";
import { writeFile, mkdir } from "fs/promises";
import { join } from "path";
import { v4 as uuidv4 } from "uuid";
import Papa from "papaparse";

const UPLOAD_DIR = process.env.UPLOAD_DIR || "./uploads";

export async function POST(req: NextRequest) {
  try {
    const apiKey = process.env.OPENROUTER_API_KEY;
    if (!apiKey) {
      return NextResponse.json({ error: "OPENROUTER_API_KEY not set" }, { status: 500 });
    }

    const body = await req.json();
    const { model, prompt, temperature, maxTokens, rows, applyStopWords } = body;

    if (!rows || !Array.isArray(rows) || rows.length === 0) {
      return NextResponse.json({ error: "Нет данных для обработки" }, { status: 400 });
    }

    // Load stop words if needed
    let stopWords: StopWordEntry[] = [];
    if (applyStopWords) {
      stopWords = (await prisma.stopWord.findMany({ where: { isActive: true } }))
        .map((w) => ({ word: w.word, replacement: w.replacement }));
    }

    const results: { title: string; tags: string; categories: string; seo_title: string; seo_description: string }[] = [];
    const errors: string[] = [];

    for (let i = 0; i < rows.length; i++) {
      const row = rows[i];
      try {
        const rowPrompt = (prompt || "")
          .replace(/\{title\}/g, row.title || "")
          .replace(/\{tags\}/g, row.tags || "")
          .replace(/\{categories\}/g, row.categories || "");

        const raw = await chatCompletion(apiKey, {
          model: model || "openai/gpt-4o-mini",
          systemPrompt: "You generate SEO titles and descriptions. Return ONLY valid JSON.",
          userPrompt: rowPrompt,
          maxTokens: maxTokens || 300,
          temperature: temperature || 0.7,
          topP: 1.0,
          minP: 0.0,
          topK: 40,
          presencePenalty: 0.2,
          frequencyPenalty: 0.4,
          repetitionPenalty: 1.2,
        });

        // Parse JSON
        let parsed: Record<string, string> = {};
        try {
          const cleaned = raw.replace(/```json\s*/gi, "").replace(/```\s*/g, "").trim();
          parsed = JSON.parse(cleaned);
        } catch {
          const match = raw.match(/\{[\s\S]*\}/);
          if (match) {
            try { parsed = JSON.parse(match[0]); } catch {}
          }
        }

        let seoTitle = postprocessLLMResponse(parsed.title || parsed.seo_title || row.title || "");
        let seoDesc = postprocessLLMResponse(parsed.description || parsed.seo_description || "");

        if (stopWords.length > 0) {
          seoTitle = cleanText(seoTitle, stopWords);
          seoDesc = cleanText(seoDesc, stopWords);
        }

        results.push({
          title: row.title || "",
          tags: row.tags || "",
          categories: row.categories || "",
          seo_title: seoTitle.slice(0, 90),
          seo_description: seoDesc.slice(0, 160),
        });
      } catch (err) {
        errors.push(`Row ${i}: ${(err as Error).message}`);
        results.push({
          title: row.title || "",
          tags: row.tags || "",
          categories: row.categories || "",
          seo_title: row.title || "",
          seo_description: "",
        });
      }
    }

    return NextResponse.json({ results, errors });
  } catch (err) {
    console.error("SEO generator error:", err);
    return NextResponse.json({ error: (err as Error).message }, { status: 500 });
  }
}
