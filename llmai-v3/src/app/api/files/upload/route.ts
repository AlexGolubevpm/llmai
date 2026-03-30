import { NextRequest, NextResponse } from "next/server";
import { writeFile, mkdir } from "fs/promises";
import { join } from "path";
import { v4 as uuidv4 } from "uuid";

const UPLOAD_DIR = process.env.UPLOAD_DIR || "./uploads";

// ---- Auto-detect column types by content ----

function looksLikeUrl(val: string): boolean {
  return /^https?:\/\//i.test(val.trim());
}

function looksLikeImageUrl(val: string): boolean {
  const v = val.trim().toLowerCase();
  return looksLikeUrl(v) && (/\.(jpg|jpeg|png|gif|webp|bmp)/i.test(v) || /screenshot/i.test(v) || /thumb/i.test(v));
}

function looksLikeVideoUrl(val: string): boolean {
  const v = val.trim().toLowerCase();
  return looksLikeUrl(v) && (/\.(mp4|webm|m3u8|avi|mov)/i.test(v) || /video/i.test(v));
}

function looksLikeId(val: string): boolean {
  return /^\d+$/.test(val.trim()) && val.trim().length < 15;
}

function looksLikeTags(val: string): boolean {
  // Comma-separated short words
  const parts = val.split(",").map((p) => p.trim()).filter(Boolean);
  return parts.length >= 2 && parts.every((p) => p.length < 50);
}

function looksLikeTitle(val: string): boolean {
  // Multiple words, not a URL, not pure numbers, not comma-separated tags
  const trimmed = val.trim();
  if (!trimmed || looksLikeUrl(trimmed) || /^\d+$/.test(trimmed)) return false;
  const words = trimmed.split(/\s+/);
  return words.length >= 2;
}

/**
 * Auto-detect pipe-delimited TXT feed.
 * Analyzes first 5 lines to determine column types.
 * Returns canonical headers: id, video_url, thumbnail_url, title, tags, categories
 * mapped to actual column positions.
 */
function analyzePipeFeed(content: string): {
  isPipeFeed: boolean;
  columnMap: Record<string, number>; // header → column index
  columnCount: number;
  lineCount: number;
} {
  const lines = content.split("\n").filter((l) => l.trim());
  if (lines.length === 0) return { isPipeFeed: false, columnMap: {}, columnCount: 0, lineCount: 0 };

  const firstLine = lines[0];
  const pipeCount = (firstLine.match(/\|/g) || []).length;

  // Need at least 2 pipes (3 columns) to be a feed
  if (pipeCount < 2) return { isPipeFeed: false, columnMap: {}, columnCount: 0, lineCount: lines.length };

  const columnCount = pipeCount + 1;

  // Analyze first 5 rows to score each column
  const sampleLines = lines.slice(0, Math.min(5, lines.length));
  const scores: Record<string, number[]> = {
    id: new Array(columnCount).fill(0),
    title: new Array(columnCount).fill(0),
    tags: new Array(columnCount).fill(0),
    thumbnail_url: new Array(columnCount).fill(0),
    video_url: new Array(columnCount).fill(0),
  };

  for (const line of sampleLines) {
    const parts = line.split("|");
    for (let col = 0; col < Math.min(parts.length, columnCount); col++) {
      const val = parts[col].trim();
      if (!val) continue;

      if (looksLikeId(val)) scores.id[col] += 3;
      if (looksLikeImageUrl(val)) scores.thumbnail_url[col] += 5;
      else if (looksLikeVideoUrl(val)) scores.video_url[col] += 5;
      else if (looksLikeUrl(val)) scores.video_url[col] += 2; // generic URL → probably video
      if (looksLikeTags(val)) scores.tags[col] += 3;
      if (looksLikeTitle(val)) scores.title[col] += 2;
    }
  }

  // Assign columns greedily by highest score
  const columnMap: Record<string, number> = {};
  const usedCols = new Set<number>();
  const headers = ["thumbnail_url", "video_url", "id", "title", "tags"]; // priority order

  for (const header of headers) {
    let bestCol = -1;
    let bestScore = 0;
    for (let col = 0; col < columnCount; col++) {
      if (usedCols.has(col)) continue;
      if (scores[header][col] > bestScore) {
        bestScore = scores[header][col];
        bestCol = col;
      }
    }
    if (bestCol >= 0 && bestScore > 0) {
      columnMap[header] = bestCol;
      usedCols.add(bestCol);
    }
  }

  // Remaining unassigned columns → categories (first unused)
  for (let col = 0; col < columnCount; col++) {
    if (!usedCols.has(col)) {
      if (!columnMap.categories) {
        columnMap.categories = col;
        usedCols.add(col);
      }
    }
  }

  // Ensure we at least found a title or thumbnail
  if (!columnMap.title && !columnMap.thumbnail_url) {
    return { isPipeFeed: false, columnMap: {}, columnCount, lineCount: lines.length };
  }

  return { isPipeFeed: true, columnMap, columnCount, lineCount: lines.length };
}

/**
 * Convert pipe-delimited TXT to CSV using auto-detected column map.
 */
function convertPipeFeedToCsv(
  content: string,
  columnMap: Record<string, number>
): string {
  const lines = content.split("\n").filter((l) => l.trim());

  // Canonical output headers
  const outputHeaders = ["id", "video_url", "thumbnail_url", "title", "tags", "categories"];
  const csvLines = [outputHeaders.join(",")];

  for (const line of lines) {
    const parts = line.split("|");
    const csvRow = outputHeaders.map((header) => {
      const colIdx = columnMap[header];
      const val = colIdx !== undefined && colIdx < parts.length
        ? parts[colIdx].trim()
        : "";
      // Escape for CSV
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
    let detectedMap: Record<string, number> = {};

    if (name.endsWith(".txt")) {
      const analysis = analyzePipeFeed(content);

      if (analysis.isPipeFeed) {
        content = convertPipeFeedToCsv(content, analysis.columnMap);
        finalFilename = `${uuidv4()}.csv`;
        isPipeFeed = true;
        feedHeaders = ["id", "video_url", "thumbnail_url", "title", "tags", "categories"];
        detectedMap = analysis.columnMap;
        console.log(`[Upload] Auto-detected pipe feed (${analysis.columnCount} cols):`, analysis.columnMap);
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
      detectedColumns: detectedMap,
    });
  } catch (err) {
    console.error("Upload error:", err);
    return NextResponse.json(
      { error: `Ошибка загрузки: ${(err as Error).message}` },
      { status: 500 }
    );
  }
}
