import { NextRequest, NextResponse } from "next/server";
import { writeFile, mkdir } from "fs/promises";
import { join } from "path";
import { v4 as uuidv4 } from "uuid";

const UPLOAD_DIR = process.env.UPLOAD_DIR || "./uploads";

export async function POST(req: NextRequest) {
  const formData = await req.formData();
  const file = formData.get("file") as File | null;

  if (!file) {
    return NextResponse.json({ error: "No file provided" }, { status: 400 });
  }

  // Validate file type
  const name = file.name.toLowerCase();
  if (!name.endsWith(".csv") && !name.endsWith(".txt")) {
    return NextResponse.json(
      { error: "Only CSV and TXT files are supported" },
      { status: 400 }
    );
  }

  // Validate file size (100MB max)
  if (file.size > 100 * 1024 * 1024) {
    return NextResponse.json(
      { error: "File too large (max 100MB)" },
      { status: 400 }
    );
  }

  await mkdir(UPLOAD_DIR, { recursive: true });

  const ext = name.split(".").pop();
  const filename = `${uuidv4()}.${ext}`;
  const filepath = join(UPLOAD_DIR, filename);

  const bytes = await file.arrayBuffer();
  await writeFile(filepath, Buffer.from(bytes));

  // Count rows for preview
  const content = Buffer.from(bytes).toString("utf-8");
  const lineCount = content.split("\n").filter((l) => l.trim()).length;

  return NextResponse.json({
    fileUrl: filepath,
    filename: file.name,
    size: file.size,
    lineCount,
  });
}
