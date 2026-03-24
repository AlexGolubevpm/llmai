import { NextRequest, NextResponse } from "next/server";
import { prisma } from "@/lib/db";

export async function GET() {
  const presets = await prisma.preset.findMany({ orderBy: { createdAt: "desc" } });
  return NextResponse.json({ presets });
}

export async function POST(req: NextRequest) {
  const body = await req.json();
  const {
    name, systemPrompt, maxTokens, temperature, topP, minP, topK,
    presencePenalty, frequencyPenalty, repetitionPenalty, isDefault,
  } = body;

  if (!name) {
    return NextResponse.json({ error: "name is required" }, { status: 400 });
  }

  // If setting as default, unset others
  if (isDefault) {
    await prisma.preset.updateMany({ data: { isDefault: false } });
  }

  const preset = await prisma.preset.upsert({
    where: { name },
    update: {
      systemPrompt, maxTokens, temperature, topP, minP, topK,
      presencePenalty, frequencyPenalty, repetitionPenalty, isDefault: isDefault || false,
    },
    create: {
      name, systemPrompt, maxTokens, temperature, topP, minP, topK,
      presencePenalty, frequencyPenalty, repetitionPenalty, isDefault: isDefault || false,
    },
  });

  return NextResponse.json({ preset }, { status: 201 });
}

export async function DELETE(req: NextRequest) {
  const { searchParams } = new URL(req.url);
  const id = searchParams.get("id");
  if (!id) return NextResponse.json({ error: "id required" }, { status: 400 });
  await prisma.preset.delete({ where: { id } });
  return NextResponse.json({ success: true });
}
