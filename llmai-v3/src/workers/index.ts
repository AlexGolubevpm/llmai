import { createWorker, REWRITE_QUEUE, TRANSLATE_QUEUE, POSTPROCESS_QUEUE, AI_PROCESS_QUEUE } from "@/lib/queue";
import { rewriteProcessor } from "./rewrite.worker";
import { translateProcessor } from "./translate.worker";
import { postprocessProcessor } from "./postprocess.worker";
import { aiProcessProcessor } from "./ai-process.worker";

console.log("Starting workers...");

const rewriteWorker = createWorker(REWRITE_QUEUE, rewriteProcessor);
const translateWorker = createWorker(TRANSLATE_QUEUE, translateProcessor);
const postprocessWorker = createWorker(POSTPROCESS_QUEUE, postprocessProcessor);
const aiProcessWorker = createWorker(AI_PROCESS_QUEUE, aiProcessProcessor);

const workers = [rewriteWorker, translateWorker, postprocessWorker, aiProcessWorker];

for (const worker of workers) {
  worker.on("completed", (job) => {
    console.log(`Job ${job.id} completed on queue ${job.queueName}`);
  });
  worker.on("failed", (job, err) => {
    console.error(`Job ${job?.id} failed on queue ${job?.queueName}:`, err.message);
  });
}

console.log("All workers started successfully.");

// Graceful shutdown
async function shutdown() {
  console.log("Shutting down workers...");
  await Promise.all(workers.map((w) => w.close()));
  process.exit(0);
}

process.on("SIGTERM", shutdown);
process.on("SIGINT", shutdown);
