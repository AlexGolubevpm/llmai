import { NextRequest, NextResponse } from "next/server";
import { prisma } from "@/lib/db";
import { getQueueByType } from "@/lib/queue";
import type { JobType } from "@/types";

// GET /api/jobs — list all jobs
export async function GET(req: NextRequest) {
  const { searchParams } = new URL(req.url);
  const type = searchParams.get("type") as JobType | null;
  const status = searchParams.get("status");
  const limit = parseInt(searchParams.get("limit") || "50");
  const offset = parseInt(searchParams.get("offset") || "0");

  const where: Record<string, unknown> = {};
  if (type) where.type = type;
  if (status) where.status = status;

  const [jobs, total] = await Promise.all([
    prisma.job.findMany({
      where,
      orderBy: { createdAt: "desc" },
      take: limit,
      skip: offset,
    }),
    prisma.job.count({ where }),
  ]);

  return NextResponse.json({ jobs, total });
}

// POST /api/jobs — create a new job
export async function POST(req: NextRequest) {
  const body = await req.json();
  const { type, config, inputFileUrl } = body;

  if (!type || !inputFileUrl) {
    return NextResponse.json(
      { error: "type and inputFileUrl are required" },
      { status: 400 }
    );
  }

  const job = await prisma.job.create({
    data: {
      type,
      config: config || {},
      inputFileUrl,
      totalPasses: config?.multiplier || 1,
    },
  });

  // Add to queue
  const queue = getQueueByType(type);
  await queue.add(`${type}-${job.id}`, { jobId: job.id });

  return NextResponse.json({ job }, { status: 201 });
}
