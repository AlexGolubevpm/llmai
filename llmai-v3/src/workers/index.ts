import { createWorker, REWRITE_QUEUE, TRANSLATE_QUEUE, POSTPROCESS_QUEUE, AI_PROCESS_QUEUE, SEO_CATEGORIES_QUEUE, PBN_QUEUE, CATEGORIZE_QUEUE, FEED_DESCRIPTIONS_QUEUE } from "@/lib/queue";
import { rewriteProcessor } from "./rewrite.worker";
import { translateProcessor } from "./translate.worker";
import { postprocessProcessor } from "./postprocess.worker";
import { aiProcessProcessor } from "./ai-process.worker";
import { seoCategoriesProcessor } from "./seo-categories.worker";
import { pbnProcessor } from "./pbn.worker";
import { categorizeProcessor } from "./categorize.worker";
import { feedDescriptionsProcessor } from "./feed-descriptions.worker";

console.log("Starting workers...");

const rewriteWorker = createWorker(REWRITE_QUEUE, rewriteProcessor);
const translateWorker = createWorker(TRANSLATE_QUEUE, translateProcessor);
const postprocessWorker = createWorker(POSTPROCESS_QUEUE, postprocessProcessor);
const aiProcessWorker = createWorker(AI_PROCESS_QUEUE, aiProcessProcessor);
const seoCategoriesWorker = createWorker(SEO_CATEGORIES_QUEUE, seoCategoriesProcessor);
const pbnWorker = createWorker(PBN_QUEUE, pbnProcessor);
const categorizeWorker = createWorker(CATEGORIZE_QUEUE, categorizeProcessor);
const feedDescWorker = createWorker(FEED_DESCRIPTIONS_QUEUE, feedDescriptionsProcessor);

const workers = [rewriteWorker, translateWorker, postprocessWorker, aiProcessWorker, seoCategoriesWorker, pbnWorker, categorizeWorker, feedDescWorker];

for (const worker of workers) {
  worker.on("completed", (job) => {
    console.log(`Job ${job.id} completed on queue ${job.queueName}`);
  });
  worker.on("failed", (job, err) => {
    console.error(`Job ${job?.id} failed on queue ${job?.queueName}:`, err.message);
  });
}

console.log("All workers started successfully.");

async function shutdown() {
  console.log("Shutting down workers...");
  await Promise.all(workers.map((w) => w.close()));
  process.exit(0);
}

process.on("SIGTERM", shutdown);
process.on("SIGINT", shutdown);
