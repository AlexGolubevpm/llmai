import { NextRequest, NextResponse } from "next/server";
import { readFile } from "fs/promises";
import { prisma } from "@/lib/db";
import { basename } from "path";

// GET /api/files/:id — download result file for a job
export async function GET(
  _req: NextRequest,
  { params }: { params: Promise<{ id: string }> }
) {
  const { id } = await params;
  const job = await prisma.job.findUnique({ where: { id } });

  if (!job || !job.outputFileUrl) {
    return NextResponse.json({ error: "File not found" }, { status: 404 });
  }

  try {
    const content = await readFile(job.outputFileUrl);
    const filename = basename(job.outputFileUrl);

    return new NextResponse(content, {
      headers: {
        "Content-Type": "text/csv",
        "Content-Disposition": `attachment; filename="${filename}"`,
      },
    });
  } catch {
    return NextResponse.json({ error: "File not found on disk" }, { status: 404 });
  }
}
