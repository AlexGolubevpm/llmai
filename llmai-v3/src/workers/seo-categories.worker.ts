import { type Job as BullJob } from "bullmq";
import { prisma } from "@/lib/db";
import { chatCompletion } from "@/lib/openrouter-client";
import { postprocessLLMResponse, cleanText, type StopWordEntry } from "@/lib/text-processing";
import { publishProgress } from "@/lib/queue";
import * as fs from "fs";
import Papa from "papaparse";
import type { JobConfig } from "@/types";

const MAX_RETRIES = 3;
const MIN_DELAY_MS = 300;

function sleep(ms: number) {
  return new Promise((r) => setTimeout(r, ms));
}

const DEFAULT_PROMPT = `You are an SEO expert for adult tube sites. Generate a unique SEO title and meta description for a specific tag/category page.

The tag/category name is: "{name}"

This is a page that lists all videos tagged with "{name}" on an adult tube site.

Requirements for title:
- 50-65 characters
- MUST include the exact tag/category name "{name}"
- Include power words: Free, HD, Best, Watch, Hot
- Must be unique and specific to this tag

Requirements for description:
- 120-155 characters
- MUST mention "{name}" at least once
- Describe what visitors will find on this specific tag page
- Include call-to-action: watch, explore, browse, discover

Return ONLY valid JSON:
{"title":"...","description":"..."}`;

function parseJson(raw: string): Record<string, string> {
  const cleaned = raw.replace(/```json\s*/gi, "").replace(/```\s*/g, "").replace(/^\s*json\s*/i, "").trim();
  try { return JSON.parse(cleaned); } catch {}
  const match = cleaned.match(/\{[\s\S]*\}/);
  if (match) {
    try { return JSON.parse(match[0]); } catch {}
    try { return JSON.parse(match[0].replace(/,\s*}/g, "}").replace(/'/g, '"')); } catch {}
  }
  console.warn(`[SEO Cat] Failed to parse: ${raw.slice(0, 200)}`);
  return {};
}

export async function seoCategoriesProcessor(job: BullJob) {
  const { jobId } = job.data;
  const dbJob = await prisma.job.findUniqueOrThrow({ where: { id: jobId } });
  const config = dbJob.config as unknown as JobConfig & { names?: string[]; customPrompt?: string };
  const apiKey = process.env.OPENROUTER_API_KEY!;

  const model = config.model || "openai/gpt-4o-mini";
  const prompt = config.customPrompt || config.visionPrompt || DEFAULT_PROMPT;
  const temperature = config.temperature || 0.7;
  const maxTokens = config.maxTokens || 300;
  const names: string[] = config.names || [];

  console.log(`[SEO Cat] Job ${jobId} | Model: ${model} | ${names.length} tags/categories`);

  await prisma.job.update({
    where: { id: jobId },
    data: { status: "RUNNING", startedAt: new Date(), totalRows: names.length, totalPasses: 1 },
  });

  const stopWords: StopWordEntry[] = config.applyStopWords
    ? (await prisma.stopWord.findMany({ where: { isActive: true } })).map((w) => ({ word: w.word, replacement: w.replacement }))
    : [];

  const rows: Record<string, string>[] = [];
  const errorLog: { row: number; step: string; error: string; retries: number }[] = [];
  const startTime = Date.now();

  for (let i = 0; i < names.length; i++) {
    const name = names[i];

    // Check cancellation
    if (i > 0 && i % 20 === 0) {
      const fresh = await prisma.job.findUnique({ where: { id: jobId }, select: { status: true } });
      if (fresh?.status === "CANCELLED") {
        console.log(`[SEO Cat] Job ${jobId} cancelled at ${i}`);
        break;
      }
    }

    const rowPrompt = prompt.replace(/\{name\}/g, name);

    let seoTitle = "";
    let seoDesc = "";

    for (let retry = 0; retry < MAX_RETRIES; retry++) {
      try {
        await sleep(MIN_DELAY_MS);
        const raw = await chatCompletion(apiKey, {
          model,
          systemPrompt: "You generate SEO titles and descriptions. Return ONLY valid JSON.",
          userPrompt: rowPrompt,
          maxTokens,
          temperature,
          topP: 1.0,
          minP: 0.0,
          topK: 40,
          presencePenalty: 0.2,
          frequencyPenalty: 0.4,
          repetitionPenalty: 1.2,
        });

        if (i === 0) console.log(`[SEO Cat] Row 0 "${name}" raw: ${raw.slice(0, 300)}`);

        const parsed = parseJson(raw);
        seoTitle = postprocessLLMResponse(parsed.title || parsed.seo_title || "");
        seoDesc = postprocessLLMResponse(parsed.description || parsed.seo_description || "");

        if (stopWords.length > 0) {
          seoTitle = cleanText(seoTitle, stopWords);
          seoDesc = cleanText(seoDesc, stopWords);
        }
        break;
      } catch (err) {
        if (retry === MAX_RETRIES - 1) {
          errorLog.push({ row: i, step: "generate", error: (err as Error).message, retries: MAX_RETRIES });
        }
        await sleep(Math.pow(2, retry) * 1000);
      }
    }

    rows.push({
      tag_category: name,
      seo_title: seoTitle.slice(0, 90),
      seo_description: seoDesc.slice(0, 160),
    });

    const processed = i + 1;
    const elapsed = (Date.now() - startTime) / 1000;
    const speed = processed / elapsed;

    await prisma.job.update({
      where: { id: jobId },
      data: { processedRows: processed, failedRows: errorLog.length },
    });
    await publishProgress(jobId, {
      status: "RUNNING",
      processedRows: processed,
      totalRows: names.length,
      failedRows: errorLog.length,
      currentPass: 1,
      totalPasses: 1,
      eta: Math.round((names.length - processed) / speed) || 0,
      speed: Math.round(speed * 10) / 10,
    });
  }

  // Write output CSV
  const outputPath = (dbJob.inputFileUrl || `./uploads/${jobId}`).replace(/(\.\w+)?$/, `_seo_categories.csv`);
  fs.writeFileSync(outputPath, Papa.unparse(rows), "utf-8");

  console.log(`[SEO Cat] Job ${jobId} done. ${rows.length} rows, ${errorLog.length} errors`);

  await prisma.job.update({
    where: { id: jobId },
    data: {
      status: "COMPLETED",
      outputFileUrl: outputPath,
      completedAt: new Date(),
      processedRows: rows.length,
      errorLog: errorLog.length > 0 ? errorLog : undefined,
    },
  });

  await publishProgress(jobId, { status: "COMPLETED", processedRows: rows.length, totalRows: names.length });
}
