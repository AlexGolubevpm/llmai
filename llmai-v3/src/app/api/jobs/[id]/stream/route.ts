import { NextRequest } from "next/server";
import { getRedisSubscriber } from "@/lib/redis";

// GET /api/jobs/:id/stream — SSE endpoint for real-time progress
export async function GET(
  _req: NextRequest,
  { params }: { params: Promise<{ id: string }> }
) {
  const { id } = await params;

  const encoder = new TextEncoder();
  const subscriber = getRedisSubscriber();
  const channel = `job:progress:${id}`;

  const stream = new ReadableStream({
    start(controller) {
      subscriber.subscribe(channel);

      subscriber.on("message", (_ch: string, message: string) => {
        controller.enqueue(encoder.encode(`data: ${message}\n\n`));

        // Close stream when job is done
        try {
          const data = JSON.parse(message);
          if (data.status === "COMPLETED" || data.status === "FAILED" || data.status === "CANCELLED") {
            setTimeout(() => {
              subscriber.unsubscribe(channel);
              subscriber.quit();
              controller.close();
            }, 500);
          }
        } catch {
          // ignore parse errors
        }
      });

      // Send initial keepalive
      controller.enqueue(encoder.encode(`: keepalive\n\n`));
    },
    cancel() {
      subscriber.unsubscribe(channel);
      subscriber.quit();
    },
  });

  return new Response(stream, {
    headers: {
      "Content-Type": "text/event-stream",
      "Cache-Control": "no-cache",
      Connection: "keep-alive",
    },
  });
}
