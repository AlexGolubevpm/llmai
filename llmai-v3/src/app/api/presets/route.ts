import { NextRequest, NextResponse } from "next/server";
import { prisma } from "@/lib/db";

export async function GET() {
  try {
    const presets = await prisma.preset.findMany({ orderBy: { createdAt: "desc" } });
    return NextResponse.json({ presets });
  } catch (err) {
    console.error("GET /api/presets error:", err);
    return NextResponse.json({ presets: [], error: (err as Error).message }, { status: 500 });
  }
}

export async function POST(req: NextRequest) {
  try {
    const body = await req.json();
    const {
      name, model, systemPrompt, maxTokens, temperature, topP, minP, topK,
      presencePenalty, frequencyPenalty, repetitionPenalty, isDefault,
    } = body;

    if (!name) {
      return NextResponse.json({ error: "Название обязательно" }, { status: 400 });
    }

    if (isDefault) {
      await prisma.preset.updateMany({ data: { isDefault: false } });
    }

    const data = {
      model: model || "meta-llama/llama-3.1-8b-instruct",
      systemPrompt: systemPrompt || "You are a helpful assistant.",
      maxTokens: maxTokens || 512,
      temperature: temperature ?? 0.7,
      topP: topP ?? 1.0,
      minP: minP ?? 0.0,
      topK: topK || 40,
      presencePenalty: presencePenalty ?? 0.0,
      frequencyPenalty: frequencyPenalty ?? 0.0,
      repetitionPenalty: repetitionPenalty ?? 1.0,
      isDefault: isDefault || false,
    };

    const preset = await prisma.preset.upsert({
      where: { name },
      update: data,
      create: { name, ...data },
    });

    return NextResponse.json({ preset }, { status: 201 });
  } catch (err) {
    console.error("POST /api/presets error:", err);
    return NextResponse.json({ error: `Ошибка сохранения: ${(err as Error).message}` }, { status: 500 });
  }
}

export async function DELETE(req: NextRequest) {
  try {
    const { searchParams } = new URL(req.url);
    const id = searchParams.get("id");
    if (!id) return NextResponse.json({ error: "id обязателен" }, { status: 400 });
    await prisma.preset.delete({ where: { id } });
    return NextResponse.json({ success: true });
  } catch (err) {
    console.error("DELETE /api/presets error:", err);
    return NextResponse.json({ error: (err as Error).message }, { status: 500 });
  }
}
