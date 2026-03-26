import { NextRequest, NextResponse } from "next/server";
import { prisma } from "@/lib/db";

export async function GET() {
  try {
    const stopwords = await prisma.stopWord.findMany({
      orderBy: { createdAt: "desc" },
    });
    return NextResponse.json({ stopwords });
  } catch (err) {
    console.error("GET /api/stopwords error:", err);
    return NextResponse.json({ stopwords: [], error: (err as Error).message }, { status: 500 });
  }
}

export async function POST(req: NextRequest) {
  try {
    const body = await req.json();

    if (Array.isArray(body)) {
      // Filter out empty words
      const validItems = body.filter(
        (item: { word?: string }) => item.word && item.word.trim()
      );
      if (validItems.length === 0) {
        return NextResponse.json({ error: "Нет валидных слов" }, { status: 400 });
      }

      const created = await prisma.$transaction(
        validItems.map((item: { word: string; replacement?: string | null }) =>
          prisma.stopWord.upsert({
            where: { word: item.word.trim() },
            update: { replacement: item.replacement || null },
            create: { word: item.word.trim(), replacement: item.replacement || null },
          })
        )
      );
      return NextResponse.json({ stopwords: created }, { status: 201 });
    }

    const { word, replacement } = body;
    if (!word || !word.trim()) {
      return NextResponse.json({ error: "Слово обязательно" }, { status: 400 });
    }

    const stopword = await prisma.stopWord.upsert({
      where: { word: word.trim() },
      update: { replacement: replacement || null },
      create: { word: word.trim(), replacement: replacement || null },
    });

    return NextResponse.json({ stopword }, { status: 201 });
  } catch (err) {
    console.error("POST /api/stopwords error:", err);
    return NextResponse.json({ error: `Ошибка: ${(err as Error).message}` }, { status: 500 });
  }
}

export async function DELETE(req: NextRequest) {
  try {
    const id = new URL(req.url).searchParams.get("id");
    if (!id) return NextResponse.json({ error: "id обязателен" }, { status: 400 });
    await prisma.stopWord.delete({ where: { id } });
    return NextResponse.json({ success: true });
  } catch (err) {
    console.error("DELETE /api/stopwords error:", err);
    return NextResponse.json({ error: (err as Error).message }, { status: 500 });
  }
}

export async function PATCH(req: NextRequest) {
  try {
    const { id, isActive, word, replacement } = await req.json();
    if (!id) return NextResponse.json({ error: "id обязателен" }, { status: 400 });

    const data: Record<string, unknown> = {};
    if (typeof isActive === "boolean") data.isActive = isActive;
    if (word !== undefined) data.word = word;
    if (replacement !== undefined) data.replacement = replacement;

    const stopword = await prisma.stopWord.update({ where: { id }, data });
    return NextResponse.json({ stopword });
  } catch (err) {
    console.error("PATCH /api/stopwords error:", err);
    return NextResponse.json({ error: (err as Error).message }, { status: 500 });
  }
}
