import { NextRequest, NextResponse } from "next/server";
import { prisma } from "@/lib/db";
import { getQueueByType } from "@/lib/queue";
import type { JobType } from "@/types";

export async function GET(req: NextRequest) {
  try {
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
  } catch (err) {
    console.error("GET /api/jobs error:", err);
    return NextResponse.json({ jobs: [], total: 0, error: (err as Error).message }, { status: 500 });
  }
}

export async function POST(req: NextRequest) {
  try {
    const body = await req.json();
    const { type, config, inputFileUrl } = body;

    if (!type || !inputFileUrl) {
      return NextResponse.json(
        { error: "type и inputFileUrl обязательны" },
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
  } catch (err) {
    console.error("POST /api/jobs error:", err);
    return NextResponse.json(
      { error: `Ошибка создания задачи: ${(err as Error).message}` },
      { status: 500 }
    );
  }
}
