import { type Job as BullJob } from "bullmq";
import { prisma } from "@/lib/db";
import { chatCompletion } from "@/lib/openrouter-client";
import { publishProgress } from "@/lib/queue";
import * as fs from "fs";
import Papa from "papaparse";
import type { JobConfig } from "@/types";

const MAX_RETRIES = 3;
const MIN_DELAY_MS = 300;

function sleep(ms: number) {
  return new Promise((r) => setTimeout(r, ms));
}

export async function pbnProcessor(job: BullJob) {
  const { jobId } = job.data;
  const dbJob = await prisma.job.findUniqueOrThrow({ where: { id: jobId } });
  const config = dbJob.config as unknown as JobConfig & { sites?: string[]; quantity?: number; customPrompt?: string };
  const apiKey = process.env.OPENROUTER_API_KEY!;

  const model = config.model || "openai/gpt-4o-mini";
  const prompt = config.customPrompt || "";
  const temperature = config.temperature || 0.9;
  const maxTokens = config.maxTokens || 500;
  const sites = config.sites || [];
  const quantity = config.quantity || 100;

  console.log(`[PBN] Job ${jobId} | Model: ${model} | ${quantity} texts | Sites: ${sites.join(", ")}`);

  await prisma.job.update({
    where: { id: jobId },
    data: { status: "RUNNING", startedAt: new Date(), totalRows: quantity, totalPasses: 1 },
  });

  const rows: Record<string, string>[] = [];
  const errorLog: { row: number; step: string; error: string; retries: number }[] = [];
  const startTime = Date.now();
  const sitesStr = sites.map((s, i) => `${i + 1}. ${s}`).join("\n");

  for (let i = 0; i < quantity; i++) {
    // Check cancellation
    if (i > 0 && i % 20 === 0) {
      const fresh = await prisma.job.findUnique({ where: { id: jobId }, select: { status: true } });
      if (fresh?.status === "CANCELLED") {
        console.log(`[PBN] Job ${jobId} cancelled at ${i}`);
        break;
      }
    }

    const textPrompt = prompt
      .replace(/\{sites\}/g, sitesStr)
      .replace(/\{number\}/g, String(i + 1))
      .replace(/\{total\}/g, String(quantity));

    let text = "";
    for (let retry = 0; retry < MAX_RETRIES; retry++) {
      try {
        await sleep(MIN_DELAY_MS);
        text = await chatCompletion(apiKey, {
          model,
          systemPrompt: "You are an expert SEO copywriter.",
          userPrompt: textPrompt,
          maxTokens,
          temperature,
          topP: 1.0,
          minP: 0.0,
          topK: 40,
          presencePenalty: 0.3,
          frequencyPenalty: 0.5,
          repetitionPenalty: 1.3,
        });
        break;
      } catch (err) {
        if (retry === MAX_RETRIES - 1) {
          errorLog.push({ row: i, step: "generate", error: (err as Error).message, retries: MAX_RETRIES });
        }
        await sleep(Math.pow(2, retry) * 1000);
      }
    }

    rows.push({ id: String(i + 1), text, sites: sites.join("; ") });

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
      totalRows: quantity,
      failedRows: errorLog.length,
      currentPass: 1,
      totalPasses: 1,
      eta: Math.round((quantity - processed) / speed) || 0,
      speed: Math.round(speed * 10) / 10,
    });
  }

  // Write output
  const outputPath = (dbJob.inputFileUrl || `./uploads/${jobId}`).replace(/(\.\w+)?$/, `_pbn.csv`);
  fs.writeFileSync(outputPath, Papa.unparse(rows), "utf-8");

  console.log(`[PBN] Job ${jobId} done. ${rows.length} texts, ${errorLog.length} errors`);

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

  await publishProgress(jobId, { status: "COMPLETED", processedRows: rows.length, totalRows: quantity });
}
