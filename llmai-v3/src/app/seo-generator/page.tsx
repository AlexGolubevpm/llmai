"use client";

import { useState } from "react";
import { motion } from "framer-motion";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Slider } from "@/components/ui/slider";
import { Textarea } from "@/components/ui/textarea";
import { Switch } from "@/components/ui/switch";
import { ModelSelector } from "@/components/model-selector";
import { JobProgress } from "@/components/job-progress";
import { PageHeader } from "@/components/layout/page-header";
import { Play } from "lucide-react";
import { toast } from "sonner";
import { pageVariants } from "@/lib/animations";

const DEFAULT_PROMPT = `You are an SEO expert for adult tube sites. Generate a unique SEO title and meta description for a specific tag/category page.

The tag/category name is: "{name}"

This is a page that lists all videos tagged with "{name}" on an adult tube site.

Requirements for title:
- 50-65 characters
- MUST include the exact tag/category name "{name}"
- Include power words: Free, HD, Best, Watch, Hot
- Must be unique and specific to this tag

Requirements for description:
- 120-155 characters
- MUST mention "{name}" at least once
- Describe what visitors will find on this specific tag page
- Include call-to-action: watch, explore, browse, discover

Return ONLY valid JSON:
{"title":"...","description":"..."}`;

export default function SEOCategoriesPage() {
  const [namesText, setNamesText] = useState("");
  const [model, setModel] = useState("openai/gpt-4o-mini");
  const [prompt, setPrompt] = useState(DEFAULT_PROMPT);
  const [temperature, setTemperature] = useState(0.7);
  const [maxTokens, setMaxTokens] = useState(300);
  const [applyStopWords, setApplyStopWords] = useState(true);
  const [activeJobId, setActiveJobId] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);

  const names = namesText.split("\n").map((l) => l.trim()).filter(Boolean);

  async function startJob() {
    if (names.length === 0) {
      toast.error("Введите теги или категории");
      return;
    }
    setSubmitting(true);
    try {
      // Create a placeholder input file
      const resp = await fetch("/api/files/from-text", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: namesText, mode: "rewrite" }),
      });
      const fileData = await resp.json();
      if (!resp.ok) throw new Error(fileData.error);

      const jobResp = await fetch("/api/jobs", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          type: "SEO_CATEGORIES",
          inputFileUrl: fileData.fileUrl,
          config: {
            model,
            customPrompt: prompt,
            temperature,
            maxTokens,
            applyStopWords,
            names,
          },
        }),
      });
      const jobData = await jobResp.json();
      if (!jobResp.ok) throw new Error(jobData.error);
      setActiveJobId(jobData.job.id);
      toast.success("SEO Categories запущен — результат на Dashboard");
    } catch (err) {
      toast.error((err as Error).message);
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <motion.div {...pageVariants} className="space-y-8">
      <PageHeader
        title="SEO Categories Generator"
        description="Введите теги/категории → AI генерирует SEO title + description для каждого. Результат на Dashboard."
      />

      {activeJobId && <JobProgress jobId={activeJobId} onComplete={() => toast.success("SEO Categories готов! Скачайте на Dashboard.")} />}

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="space-y-6">
          <div className="rounded-xl border bg-[var(--surface)] p-6 space-y-5">
            <h2 className="text-[15px] font-medium">Теги и категории</h2>
            <Textarea
              value={namesText}
              onChange={(e) => setNamesText(e.target.value)}
              rows={12}
              placeholder={"Anal\nBlowjob\nHentai\nMILF\nTeen 18+\nTransgender\nAsian\nAmateur\nPOV\nCreampie"}
              className="font-mono text-sm"
            />
            <p className="text-xs text-[var(--text-muted)]">{names.length} тегов/категорий</p>
          </div>

          <div className="rounded-xl border bg-[var(--surface)] p-6 space-y-5">
            <h2 className="text-[15px] font-medium">Настройки</h2>
            <ModelSelector value={model} onChange={setModel} />
            <div className="grid grid-cols-2 gap-4">
              <div>
                <div className="flex items-baseline justify-between mb-2">
                  <Label className="text-xs font-mono text-[var(--text-muted)]">temperature</Label>
                  <span className="text-xs font-mono font-medium">{temperature.toFixed(2)}</span>
                </div>
                <Slider value={[temperature]} onValueChange={(v) => setTemperature(typeof v === "number" ? v : v[0])} min={0} max={2} step={0.01} />
              </div>
              <div>
                <div className="flex items-baseline justify-between mb-2">
                  <Label className="text-xs font-mono text-[var(--text-muted)]">max_tokens</Label>
                  <span className="text-xs font-mono font-medium">{maxTokens}</span>
                </div>
                <Slider value={[maxTokens]} onValueChange={(v) => setMaxTokens(typeof v === "number" ? v : v[0])} min={100} max={1000} step={50} />
              </div>
            </div>
            <div className="flex items-center justify-between rounded-lg bg-[var(--surface-raised)] px-4 py-3">
              <Label className="text-sm">Стоп-слова</Label>
              <Switch checked={applyStopWords} onCheckedChange={setApplyStopWords} />
            </div>
          </div>
        </div>

        <div className="rounded-xl border bg-[var(--surface)] p-6 space-y-4">
          <div className="flex items-center justify-between">
            <h2 className="text-[15px] font-medium">Промпт</h2>
            <button onClick={() => setPrompt(DEFAULT_PROMPT)} className="text-xs text-[var(--accent-blue)] hover:underline">Сбросить</button>
          </div>
          <Textarea value={prompt} onChange={(e) => setPrompt(e.target.value)} rows={22} className="font-mono text-xs" />
          <div className="rounded-lg bg-[var(--surface-raised)] px-4 py-3 text-xs text-[var(--text-muted)]">
            <p><strong>Переменная:</strong> <code>{"{name}"}</code> — название тега/категории</p>
            <p><strong>CSV выход:</strong> tag_category, seo_title, seo_description</p>
          </div>
        </div>
      </div>

      <Button size="lg" onClick={startJob} disabled={submitting || names.length === 0} className="w-full h-12 text-sm font-medium gap-2">
        <Play className="h-4 w-4" />
        {submitting ? "Создание задачи..." : `Сгенерировать SEO для ${names.length} тегов`}
      </Button>
    </motion.div>
  );
}
