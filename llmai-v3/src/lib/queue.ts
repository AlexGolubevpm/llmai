import { Queue, Worker, type Processor, type WorkerOptions } from "bullmq";
import { redis } from "./redis";

const connection = {
  host: new URL(process.env.REDIS_URL || "redis://localhost:6379").hostname,
  port: parseInt(
    new URL(process.env.REDIS_URL || "redis://localhost:6379").port || "6379"
  ),
};

// Queue names
export const REWRITE_QUEUE = "rewrite";
export const TRANSLATE_QUEUE = "translate";
export const POSTPROCESS_QUEUE = "postprocess";
export const AI_PROCESS_QUEUE = "ai-process";
export const SEO_CATEGORIES_QUEUE = "seo-categories";
export const PBN_QUEUE = "pbn";

// Create queues
export const rewriteQueue = new Queue(REWRITE_QUEUE, { connection });
export const translateQueue = new Queue(TRANSLATE_QUEUE, { connection });
export const postprocessQueue = new Queue(POSTPROCESS_QUEUE, { connection });
export const aiProcessQueue = new Queue(AI_PROCESS_QUEUE, { connection });
export const seoCategoriesQueue = new Queue(SEO_CATEGORIES_QUEUE, { connection });
export const pbnQueue = new Queue(PBN_QUEUE, { connection });

// Helper to create workers
export function createWorker(
  queueName: string,
  processor: Processor,
  opts?: Partial<WorkerOptions>
) {
  return new Worker(queueName, processor, {
    connection,
    concurrency: 1,
    ...opts,
  });
}

// Publish job progress to Redis pub/sub
export async function publishProgress(
  jobId: string,
  data: Record<string, unknown>
) {
  await redis.publish(
    `job:progress:${jobId}`,
    JSON.stringify({ jobId, ...data })
  );
}

// Get queue by job type
export function getQueueByType(type: string) {
  switch (type) {
    case "REWRITE":
      return rewriteQueue;
    case "TRANSLATE":
      return translateQueue;
    case "POSTPROCESS":
      return postprocessQueue;
    case "AI_PROCESS":
      return aiProcessQueue;
    case "SEO_CATEGORIES":
      return seoCategoriesQueue;
    case "PBN":
      return pbnQueue;
    default:
      throw new Error(`Unknown job type: ${type}`);
  }
}
