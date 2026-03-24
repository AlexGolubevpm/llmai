// Text processing pipeline — ported from Python app.py and enhanced

const EMOJI_PATTERN =
  /[\u{1F600}-\u{1F64F}\u{1F300}-\u{1F5FF}\u{1F680}-\u{1F6FF}\u{1F1E0}-\u{1F1FF}]+/gu;

const HIEROGLYPH_PATTERN = /[\u4e00-\u9fff]+/g;

const FORBIDDEN_SYMBOLS_PATTERN = /[><"№;%*/\\{}]+/g;

const DOMAIN_PATTERN =
  /(?:see\s+full\s+version[\s:,-]*)?(?:https?:\/\/)?[\w\-.]+(?:\.com|\.net)\S*/gi;

const COMMENT_KEYWORDS = [
  "error",
  "note",
  "however",
  "sorry",
  "<tool>",
  "direct link",
  "i can't",
];

export interface StopWordEntry {
  word: string;
  replacement: string | null;
}

export function removeEmojis(text: string): string {
  return text.replace(EMOJI_PATTERN, "");
}

export function removeHieroglyphs(text: string): string {
  return text.replace(HIEROGLYPH_PATTERN, "");
}

export function removeForbiddenSymbols(text: string): string {
  return text.replace(FORBIDDEN_SYMBOLS_PATTERN, " ");
}

export function removeDomains(text: string): string {
  return text.replace(DOMAIN_PATTERN, " ");
}

export function stripCommentaryPhrases(text: string): string {
  for (const keyword of COMMENT_KEYWORDS) {
    const escaped = keyword.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
    const pattern = new RegExp(`\\b${escaped}\\b.*`, "gi");
    text = text.replace(pattern, "");
  }
  return text;
}

export function applyStopWordReplacements(
  text: string,
  stopWords: StopWordEntry[]
): string {
  for (const { word, replacement } of stopWords) {
    const escaped = word.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
    const pattern = new RegExp(`\\b${escaped}\\b`, "gi");
    text = text.replace(pattern, replacement || "");
  }
  return text;
}

export function fixMissingSpaces(text: string): string {
  if (!text) return text;
  // Letter-digit boundaries
  text = text.replace(/(?<=[A-Za-zА-Яа-я])(?=[0-9])/g, " ");
  text = text.replace(/(?<=[0-9])(?=[A-Za-zА-Яа-я])/g, " ");
  // camelCase boundaries
  text = text.replace(/(?<=[a-zа-я])(?=[A-ZА-Я])/g, " ");
  // Long words without spaces
  if (!text.includes(" ") && text.length > 15) {
    text = text.replace(/([A-Za-zА-Яа-я]{3,})([A-Za-zА-Яа-я]{3,})/g, "$1 $2");
  }
  return text;
}

export function truncateTitle(text: string, maxLen = 100): string {
  if (text.length <= maxLen) return text;
  const truncated = text.slice(0, maxLen).trimEnd();
  const lastSpace = truncated.lastIndexOf(" ");
  if (lastSpace > 20) return truncated.slice(0, lastSpace).trim();
  return truncated.trim();
}

export function removeQuotes(text: string): string {
  return text.replace(/"/g, "");
}

/**
 * Full text cleaning pipeline.
 * Applies all transformations in order.
 */
export function cleanText(
  text: string,
  stopWords: StopWordEntry[] = [],
  harmfulPatterns: string[] = []
): string {
  if (!text) return "";
  let result = String(text);

  // Remove harmful patterns
  for (const pattern of harmfulPatterns) {
    if (pattern) {
      const escaped = pattern.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
      result = result.replace(new RegExp(escaped, "gi"), "");
    }
  }

  result = removeDomains(result);
  result = stripCommentaryPhrases(result);
  result = applyStopWordReplacements(result, stopWords);
  result = removeHieroglyphs(result);
  result = removeEmojis(result);
  result = removeForbiddenSymbols(result);
  result = fixMissingSpaces(result);
  result = result.replace(/\s+/g, " ").trim();
  result = truncateTitle(result);

  return result;
}

/**
 * Post-processing for LLM responses (used after rewrite/translate).
 */
export function postprocessLLMResponse(text: string): string {
  let result = text;
  // Remove "Note:" fragments
  result = result.replace(/\s*Note:.*/gi, "");
  // Remove sentence-starting unwanted words
  result = result.replace(
    /(^|(?<=[.!?]\s))\s*(?:fucking|explicit|intense)[\s,:\-]+/gi,
    "$1"
  );
  // Replace censored words
  result = result.replace(/\bF\*+\b/gi, "fuck");
  // Clean
  result = removeHieroglyphs(result);
  result = removeEmojis(result);
  result = removeQuotes(result);
  result = result.replace(/\s+/g, " ").trim();
  return result;
}
