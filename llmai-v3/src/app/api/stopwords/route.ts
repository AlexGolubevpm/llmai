import { NextRequest, NextResponse } from "next/server";
import { prisma } from "@/lib/db";

// GET /api/stopwords
export async function GET() {
  const stopwords = await prisma.stopWord.findMany({
    orderBy: { createdAt: "desc" },
  });
  return NextResponse.json({ stopwords });
}

// POST /api/stopwords — create or bulk create
export async function POST(req: NextRequest) {
  const body = await req.json();

  // Bulk create
  if (Array.isArray(body)) {
    const created = await prisma.$transaction(
      body.map((item: { word: string; replacement?: string }) =>
        prisma.stopWord.upsert({
          where: { word: item.word },
          update: { replacement: item.replacement || null },
          create: { word: item.word, replacement: item.replacement || null },
        })
      )
    );
    return NextResponse.json({ stopwords: created }, { status: 201 });
  }

  // Single create
  const { word, replacement } = body;
  if (!word) {
    return NextResponse.json({ error: "word is required" }, { status: 400 });
  }

  const stopword = await prisma.stopWord.upsert({
    where: { word },
    update: { replacement: replacement || null },
    create: { word, replacement: replacement || null },
  });

  return NextResponse.json({ stopword }, { status: 201 });
}

// DELETE /api/stopwords?id=xxx
export async function DELETE(req: NextRequest) {
  const { searchParams } = new URL(req.url);
  const id = searchParams.get("id");

  if (!id) {
    return NextResponse.json({ error: "id is required" }, { status: 400 });
  }

  await prisma.stopWord.delete({ where: { id } });
  return NextResponse.json({ success: true });
}

// PATCH /api/stopwords — toggle active status
export async function PATCH(req: NextRequest) {
  const { id, isActive, word, replacement } = await req.json();

  if (!id) {
    return NextResponse.json({ error: "id is required" }, { status: 400 });
  }

  const data: Record<string, unknown> = {};
  if (typeof isActive === "boolean") data.isActive = isActive;
  if (word !== undefined) data.word = word;
  if (replacement !== undefined) data.replacement = replacement;

  const stopword = await prisma.stopWord.update({ where: { id }, data });
  return NextResponse.json({ stopword });
}
