import { NextRequest, NextResponse } from "next/server";
import { writeFile, mkdir } from "fs/promises";
import { join } from "path";
import { v4 as uuidv4 } from "uuid";

const UPLOAD_DIR = process.env.UPLOAD_DIR || "./uploads";

/**
 * POST /api/files/from-text
 * Creates a temporary CSV file from raw text input.
 *
 * Body: { text: string, mode: "rewrite" | "ai-process" }
 *
 * For rewrite: creates CSV with "title" column (one row per line, or single row)
 * For ai-process: creates CSV with pipe-delimited fields or single row with title + thumbnail_url
 */
export async function POST(req: NextRequest) {
  try {
    const body = await req.json();
    const { text, mode, thumbnailUrl } = body;

    if (!text || !text.trim()) {
      return NextResponse.json({ error: "Текст обязателен" }, { status: 400 });
    }

    await mkdir(UPLOAD_DIR, { recursive: true });

    let csvContent: string;
    let lineCount: number;

    if (mode === "ai-process") {
      // AI Process: single row with title + optional thumbnail_url
      const escapedTitle = text.trim().replace(/"/g, '""');
      const escapedThumb = (thumbnailUrl || "").trim().replace(/"/g, '""');
      csvContent = `id,video_url,thumbnail_url,title,tags,categories\n1,,"${escapedThumb}","${escapedTitle}",,`;
      lineCount = 1;
    } else {
      // Rewrite: each line = separate row, or single text = single row
      const lines = text.trim().split("\n").filter((l: string) => l.trim());
      const csvRows = lines.map((line: string) => {
        const escaped = line.trim().replace(/"/g, '""');
        return `"${escaped}"`;
      });
      csvContent = `title\n${csvRows.join("\n")}`;
      lineCount = lines.length;
    }

    const filename = `${uuidv4()}.csv`;
    const filepath = join(UPLOAD_DIR, filename);
    await writeFile(filepath, csvContent, "utf-8");

    return NextResponse.json({
      fileUrl: filepath,
      filename: `text-input-${lineCount}-rows.csv`,
      size: csvContent.length,
      lineCount,
      isTextInput: true,
    });
  } catch (err) {
    console.error("POST /api/files/from-text error:", err);
    return NextResponse.json(
      { error: `Ошибка: ${(err as Error).message}` },
      { status: 500 }
    );
  }
}
