import { NextResponse } from "next/server";
import { listModels } from "@/lib/openrouter-client";

export async function GET() {
  const apiKey = process.env.OPENROUTER_API_KEY;
  if (!apiKey) {
    return NextResponse.json({ error: "OPENROUTER_API_KEY not set" }, { status: 500 });
  }

  try {
    const models = await listModels(apiKey);
    return NextResponse.json({ models });
  } catch (err) {
    console.error("GET /api/models error:", err);
    return NextResponse.json(
      { error: (err as Error).message },
      { status: 500 }
    );
  }
}
