import { type Job as BullJob } from "bullmq";
import { prisma } from "@/lib/db";
import { cleanText, type StopWordEntry } from "@/lib/text-processing";
import { publishProgress } from "@/lib/queue";
import * as fs from "fs";
import Papa from "papaparse";
import type { JobConfig } from "@/types";

export async function postprocessProcessor(job: BullJob) {
  const { jobId } = job.data;
  const dbJob = await prisma.job.findUniqueOrThrow({ where: { id: jobId } });
  const config = dbJob.config as unknown as JobConfig;

  await prisma.job.update({
    where: { id: jobId },
    data: { status: "RUNNING", startedAt: new Date() },
  });

  const fileContent = fs.readFileSync(dbJob.inputFileUrl, "utf-8");
  const parsed = Papa.parse(fileContent, { header: true });
  const rows = parsed.data as Record<string, string>[];
  const titleCol = config.titleCol || "title";

  await prisma.job.update({
    where: { id: jobId },
    data: { totalRows: rows.length },
  });

  // Load stop words from DB
  const dbStopWords = await prisma.stopWord.findMany({ where: { isActive: true } });
  const stopWords: StopWordEntry[] = dbStopWords.map((w) => ({
    word: w.word,
    replacement: w.replacement,
  }));

  const harmfulPatterns = config.harmfulPatterns || [];

  for (let i = 0; i < rows.length; i++) {
    const text = rows[i][titleCol] || "";
    rows[i]["cleaned"] = cleanText(text, stopWords, harmfulPatterns);

    if (i % 100 === 0 || i === rows.length - 1) {
      await prisma.job.update({
        where: { id: jobId },
        data: { processedRows: i + 1 },
      });
      await publishProgress(jobId, {
        status: "RUNNING",
        processedRows: i + 1,
        totalRows: rows.length,
        currentPass: 1,
        totalPasses: 1,
      });
    }
  }

  const outputPath = dbJob.inputFileUrl.replace(/(\.\w+)$/, `_cleaned$1`);
  fs.writeFileSync(outputPath, Papa.unparse(rows), "utf-8");

  await prisma.job.update({
    where: { id: jobId },
    data: {
      status: "COMPLETED",
      outputFileUrl: outputPath,
      completedAt: new Date(),
      processedRows: rows.length,
    },
  });

  await publishProgress(jobId, { status: "COMPLETED", processedRows: rows.length, totalRows: rows.length });
}
