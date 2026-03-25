import type { WDTaggerResult } from "@/types";

const WD_TAGGER_URL =
  process.env.WD_TAGGER_URL ||
  "https://deepghs-wd-tagger-heatmap-more-models.hf.space";

const DEFAULT_MODEL = "SmilingWolf/wd-vit-tagger-v3";
const DEFAULT_THRESHOLD = 0.35;

interface GradioResponse {
  data: [
    unknown, // gallery
    unknown, // combined heatmap
    string, // caption
    string, // tags (comma-separated)
    { label: string; confidences: { label: string; confidence: number }[] }, // rating
    { label: string; confidences: { label: string; confidence: number }[] }, // character
    { label: string; confidences: { label: string; confidence: number }[] }, // general
  ];
}

export async function analyzeThumbnail(
  imageUrl: string,
  options?: { model?: string; threshold?: number }
): Promise<WDTaggerResult> {
  const model = options?.model || DEFAULT_MODEL;
  const threshold = options?.threshold || DEFAULT_THRESHOLD;

  const resp = await fetch(`${WD_TAGGER_URL}/api/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      data: [
        {
          path: imageUrl,
          meta: { _type: "gradio.FileData" },
          orig_name: "image.jpg",
        },
        model,
        threshold,
      ],
    }),
  });

  if (!resp.ok) {
    const text = await resp.text();
    throw new Error(`WD Tagger API error ${resp.status}: ${text}`);
  }

  const result: GradioResponse = await resp.json();
  const [, , caption, tagsStr, ratingData, characterData, generalData] =
    result.data;

  return {
    caption: caption || "",
    tags: tagsStr
      ? tagsStr.split(",").map((t: string) => t.trim()).filter(Boolean)
      : [],
    rating: (ratingData?.confidences || []).map(
      (c: { label: string; confidence: number }) => ({
        label: c.label,
        confidence: c.confidence,
      })
    ),
    characters: (characterData?.confidences || []).map(
      (c: { label: string }) => c.label
    ),
    generalTags: (generalData?.confidences || []).map(
      (c: { label: string; confidence: number }) => ({
        tag: c.label,
        confidence: c.confidence,
      })
    ),
  };
}
