import { NextRequest, NextResponse } from "next/server";
import { writeFile, mkdir } from "fs/promises";
import { join } from "path";
import { v4 as uuidv4 } from "uuid";

const UPLOAD_DIR = process.env.UPLOAD_DIR || "./uploads";

/**
 * Detect if a TXT file is pipe-delimited feed format:
 * id|video_url|thumbnail_url|title|tags|categories
 * Returns { isPipeFeed, headers, lineCount }
 */
function analyzeTxtContent(content: string): {
  isPipeFeed: boolean;
  headers: string[];
  lineCount: number;
} {
  const lines = content.split("\n").filter((l) => l.trim());
  const lineCount = lines.length;

  if (lineCount === 0) return { isPipeFeed: false, headers: [], lineCount };

  // Check if first line has pipe separators and looks like data (not headers)
  const firstLine = lines[0];
  const pipeCount = (firstLine.match(/\|/g) || []).length;

  // 5 pipes = 6 fields: id|video_url|thumbnail_url|title|tags|categories
  if (pipeCount >= 4) {
    return {
      isPipeFeed: true,
      headers: ["id", "video_url", "thumbnail_url", "title", "tags", "categories"],
      lineCount,
    };
  }

  return { isPipeFeed: false, headers: [], lineCount };
}

/**
 * Convert pipe-delimited TXT to CSV format for processing.
 */
function convertPipeFeedToCsv(content: string, headers: string[]): string {
  const lines = content.split("\n").filter((l) => l.trim());
  const csvLines = [headers.join(",")];

  for (const line of lines) {
    const parts = line.split("|");
    // Escape CSV fields (wrap in quotes if contains comma)
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

  const bytes = await file.arrayBuffer();
  let content = Buffer.from(bytes).toString("utf-8");
  let finalFilename: string;
  let isPipeFeed = false;
  let feedHeaders: string[] = [];

  // Check if TXT file is a pipe-delimited feed
  if (name.endsWith(".txt")) {
    const analysis = analyzeTxtContent(content);

    if (analysis.isPipeFeed) {
      // Convert pipe-delimited TXT to CSV
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
  // Subtract 1 for header row in CSV
  const dataRows = isPipeFeed || name.endsWith(".csv") ? lineCount - 1 : lineCount;

  return NextResponse.json({
    fileUrl: filepath,
    filename: file.name,
    size: file.size,
    lineCount: Math.max(dataRows, 0),
    isPipeFeed,
    feedHeaders,
  });
}
