import { NextRequest, NextResponse } from "next/server";
import { writeFile, mkdir } from "fs/promises";
import { join } from "path";
import { v4 as uuidv4 } from "uuid";

const UPLOAD_DIR = process.env.UPLOAD_DIR || "./uploads";

function analyzeTxtContent(content: string): {
  isPipeFeed: boolean;
  headers: string[];
  lineCount: number;
} {
  const lines = content.split("\n").filter((l) => l.trim());
  const lineCount = lines.length;
  if (lineCount === 0) return { isPipeFeed: false, headers: [], lineCount };

  const firstLine = lines[0];
  const pipeCount = (firstLine.match(/\|/g) || []).length;

  if (pipeCount >= 4) {
    return {
      isPipeFeed: true,
      headers: ["id", "video_url", "thumbnail_url", "title", "tags", "categories"],
      lineCount,
    };
  }

  return { isPipeFeed: false, headers: [], lineCount };
}

function convertPipeFeedToCsv(content: string, headers: string[]): string {
  const lines = content.split("\n").filter((l) => l.trim());
  const csvLines = [headers.join(",")];

  for (const line of lines) {
    const parts = line.split("|");
    const csvRow = headers.map((_, i) => {
      const val = (parts[i] || "").trim();
      if (val.includes(",") || val.includes('"') || val.includes("\n")) {
        return `"${val.replace(/"/g, '""')}"`;
      }
      return val;
    });
    csvLines.push(csvRow.join(","));
  }

  return csvLines.join("\n");
}

export async function POST(req: NextRequest) {
  try {
    let formData: FormData;
    try {
      formData = await req.formData();
    } catch (err) {
      console.error("FormData parse error:", err);
      return NextResponse.json(
        { error: "Не удалось прочитать файл. Попробуйте снова." },
        { status: 400 }
      );
    }

    const file = formData.get("file") as File | null;

    if (!file) {
      return NextResponse.json({ error: "Файл не найден" }, { status: 400 });
    }

    const name = file.name.toLowerCase();
    if (!name.endsWith(".csv") && !name.endsWith(".txt")) {
      return NextResponse.json(
        { error: "Поддерживаются только CSV и TXT файлы" },
        { status: 400 }
      );
    }

    if (file.size > 100 * 1024 * 1024) {
      return NextResponse.json(
        { error: "Файл слишком большой (макс 100MB)" },
        { status: 400 }
      );
    }

    await mkdir(UPLOAD_DIR, { recursive: true });

    let bytes: ArrayBuffer;
    try {
      bytes = await file.arrayBuffer();
    } catch (err) {
      console.error("File read error:", err);
      return NextResponse.json(
        { error: "Не удалось прочитать содержимое файла" },
        { status: 400 }
      );
    }

    let content = Buffer.from(bytes).toString("utf-8");
    let finalFilename: string;
    let isPipeFeed = false;
    let feedHeaders: string[] = [];

    if (name.endsWith(".txt")) {
      const analysis = analyzeTxtContent(content);
      if (analysis.isPipeFeed) {
        content = convertPipeFeedToCsv(content, analysis.headers);
        finalFilename = `${uuidv4()}.csv`;
        isPipeFeed = true;
        feedHeaders = analysis.headers;
      } else {
        finalFilename = `${uuidv4()}.txt`;
      }
    } else {
      finalFilename = `${uuidv4()}.csv`;
    }

    const filepath = join(UPLOAD_DIR, finalFilename);
    await writeFile(filepath, content, "utf-8");

    const lineCount = content.split("\n").filter((l) => l.trim()).length;
    const dataRows =
      isPipeFeed || name.endsWith(".csv") ? lineCount - 1 : lineCount;

    return NextResponse.json({
      fileUrl: filepath,
      filename: file.name,
      size: file.size,
      lineCount: Math.max(dataRows, 0),
      isPipeFeed,
      feedHeaders,
    });
  } catch (err) {
    console.error("Upload error:", err);
    return NextResponse.json(
      { error: `Ошибка загрузки: ${(err as Error).message}` },
      { status: 500 }
    );
  }
}
