import { NextRequest, NextResponse } from "next/server";
import { prisma } from "@/lib/db";

// GET /api/tags?type=tags|categories
export async function GET(req: NextRequest) {
  const type = new URL(req.url).searchParams.get("type") || "tags";

  if (type === "categories") {
    const categories = await prisma.allowedCategory.findMany({ orderBy: { name: "asc" } });
    return NextResponse.json({ categories });
  }

  const tags = await prisma.allowedTag.findMany({ orderBy: { name: "asc" } });
  return NextResponse.json({ tags });
}

// POST /api/tags
export async function POST(req: NextRequest) {
  const body = await req.json();
  const { type, items } = body; // items: [{name, category/slug}]

  if (!Array.isArray(items) || items.length === 0) {
    return NextResponse.json({ error: "items array required" }, { status: 400 });
  }

  if (type === "categories") {
    const created = await prisma.$transaction(
      items.map((item: { name: string; slug?: string }) =>
        prisma.allowedCategory.upsert({
          where: { name: item.name },
          update: { slug: item.slug || item.name.toLowerCase().replace(/\s+/g, "-") },
          create: {
            name: item.name,
            slug: item.slug || item.name.toLowerCase().replace(/\s+/g, "-"),
          },
        })
      )
    );
    return NextResponse.json({ categories: created }, { status: 201 });
  }

  const created = await prisma.$transaction(
    items.map((item: { name: string; category?: string }) =>
      prisma.allowedTag.upsert({
        where: { name: item.name },
        update: { category: item.category || "general" },
        create: { name: item.name, category: item.category || "general" },
      })
    )
  );
  return NextResponse.json({ tags: created }, { status: 201 });
}

// DELETE /api/tags?type=tags|categories&id=xxx
export async function DELETE(req: NextRequest) {
  const { searchParams } = new URL(req.url);
  const type = searchParams.get("type") || "tags";
  const id = searchParams.get("id");
  if (!id) return NextResponse.json({ error: "id required" }, { status: 400 });

  if (type === "categories") {
    await prisma.allowedCategory.delete({ where: { id } });
  } else {
    await prisma.allowedTag.delete({ where: { id } });
  }
  return NextResponse.json({ success: true });
}
