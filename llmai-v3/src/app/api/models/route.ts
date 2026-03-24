import { NextResponse } from "next/server";
import { listModels } from "@/lib/novita-client";

export async function GET() {
  const apiKey = process.env.NOVITA_API_KEY;
  if (!apiKey) {
    return NextResponse.json({ error: "NOVITA_API_KEY not set" }, { status: 500 });
  }

  try {
    const models = await listModels(apiKey);
    return NextResponse.json({ models });
  } catch (err) {
    return NextResponse.json(
      { error: (err as Error).message },
      { status: 500 }
    );
  }
}
