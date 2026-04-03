import { NextRequest, NextResponse } from "next/server";
import { prisma } from "@/lib/db";

export async function GET() {
  try {
    const bundles = await prisma.bundle.findMany({ orderBy: { createdAt: "desc" } });
    return NextResponse.json({ bundles });
  } catch (err) {
    console.error("GET /api/bundles error:", err);
    return NextResponse.json({ bundles: [], error: (err as Error).message }, { status: 500 });
  }
}

export async function POST(req: NextRequest) {
  try {
    const body = await req.json();
    const { name, description, tags, categories, prompt, isDefault } = body;

    if (!name) {
      return NextResponse.json({ error: "Название обязательно" }, { status: 400 });
    }

    if (isDefault) {
      await prisma.bundle.updateMany({ data: { isDefault: false } });
    }

    const bundle = await prisma.bundle.upsert({
      where: { name },
      update: {
        description: description || null,
        tags: tags || "",
        categories: categories || "",
        prompt: prompt || null,
        isDefault: isDefault || false,
      },
      create: {
        name,
        description: description || null,
        tags: tags || "",
        categories: categories || "",
        prompt: prompt || null,
        isDefault: isDefault || false,
      },
    });

    return NextResponse.json({ bundle }, { status: 201 });
  } catch (err) {
    console.error("POST /api/bundles error:", err);
    return NextResponse.json({ error: (err as Error).message }, { status: 500 });
  }
}

export async function DELETE(req: NextRequest) {
  try {
    const id = new URL(req.url).searchParams.get("id");
    if (!id) return NextResponse.json({ error: "id обязателен" }, { status: 400 });
    await prisma.bundle.delete({ where: { id } });
    return NextResponse.json({ success: true });
  } catch (err) {
    console.error("DELETE /api/bundles error:", err);
    return NextResponse.json({ error: (err as Error).message }, { status: 500 });
  }
}
