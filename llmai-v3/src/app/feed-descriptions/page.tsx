"use client";

import { useState } from "react";
import { motion } from "framer-motion";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Slider } from "@/components/ui/slider";
import { Textarea } from "@/components/ui/textarea";
import { Switch } from "@/components/ui/switch";
import { FileUpload } from "@/components/file-upload";
import { ModelSelector } from "@/components/model-selector";
import { JobProgress } from "@/components/job-progress";
import { PageHeader } from "@/components/layout/page-header";
import { Play } from "lucide-react";
import { toast } from "sonner";
import { pageVariants } from "@/lib/animations";

const DEFAULT_PROMPT = `Write an SEO-optimized meta description for an adult video page.

Video title: {title}
Categories: {categories}
Tags: {tags}

Requirements:
- 120-160 characters
- Must mention the most relevant keywords from tags/categories
- Compelling, drives clicks
- Natural English, no keyword stuffing
- Specific to this video, not generic

Return ONLY the description text, nothing else.`;

export default function FeedDescriptionsPage() {
  const [fileUrl, setFileUrl] = useState("");
  const [lineCount, setLineCount] = useState(0);
  const [model, setModel] = useState("openai/gpt-4o-mini");
  const [prompt, setPrompt] = useState(DEFAULT_PROMPT);
  const [temperature, setTemperature] = useState(0.7);
  const [maxTokens, setMaxTokens] = useState(200);
  const [maxWorkers, setMaxWorkers] = useState(5);
  const [applyStopWords, setApplyStopWords] = useState(true);
  const [activeJobId, setActiveJobId] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);

  async function startJob() {
    if (!fileUrl) { toast.error("Загрузите файл"); return; }
    setSubmitting(true);
    try {
      const resp = await fetch("/api/jobs", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          type: "FEED_DESCRIPTIONS",
          inputFileUrl: fileUrl,
          config: {
            model,
            customPrompt: prompt,
            temperature,
            maxTokens,
            maxWorkers,
            applyStopWords,
          },
        }),
      });
      const data = await resp.json();
      if (!resp.ok) throw new Error(data.error);
      setActiveJobId(data.job.id);
      toast.success("Генерация описаний запущена — результат на Dashboard");
    } catch (err) { toast.error((err as Error).message); }
    finally { setSubmitting(false); }
  }

  return (
    <motion.div {...pageVariants} className="space-y-8">
      <PageHeader
        title="Описания для фида"
        description="Загрузите TXT/CSV фид → AI напишет description для каждого видео на основе названия, тегов и категорий"
      />

      {activeJobId && <JobProgress jobId={activeJobId} onComplete={() => toast.success("Описания готовы! Скачайте на Dashboard.")} />}

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="space-y-6">
          <div className="rounded-xl border bg-[var(--surface)] p-6 space-y-5">
            <h2 className="text-[15px] font-medium">Фид</h2>
            <FileUpload onUpload={(data) => { setFileUrl(data.fileUrl); setLineCount(data.lineCount); }} />
            {lineCount > 0 && <p className="text-xs text-[var(--text-muted)]">{lineCount.toLocaleString()} строк</p>}
            <div className="rounded-lg bg-[var(--surface-raised)] px-4 py-3 text-xs text-[var(--text-muted)] space-y-1">
              <p><strong>Формат TXT:</strong> <code>#ID|Название|Категории|Тэги</code></p>
              <p><strong>Формат CSV:</strong> колонки <code>title</code>, <code>categories</code>, <code>tags</code></p>
              <p>Авто-парсер определяет формат и порядок колонок</p>
            </div>
          </div>

          <div className="rounded-xl border bg-[var(--surface)] p-6 space-y-5">
            <h2 className="text-[15px] font-medium">Настройки</h2>
            <ModelSelector value={model} onChange={setModel} />
            <div className="grid grid-cols-3 gap-4">
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
                <Slider value={[maxTokens]} onValueChange={(v) => setMaxTokens(typeof v === "number" ? v : v[0])} min={50} max={500} step={10} />
              </div>
              <div>
                <div className="flex items-baseline justify-between mb-2">
                  <Label className="text-xs font-mono text-[var(--text-muted)]">потоки</Label>
                  <span className="text-xs font-mono font-medium">{maxWorkers}</span>
                </div>
                <Slider value={[maxWorkers]} onValueChange={(v) => setMaxWorkers(typeof v === "number" ? v : v[0])} min={1} max={10} step={1} />
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
          <Textarea value={prompt} onChange={(e) => setPrompt(e.target.value)} rows={16} className="font-mono text-xs" />
          <div className="rounded-lg bg-[var(--surface-raised)] px-4 py-3 text-xs text-[var(--text-muted)] space-y-1">
            <p><strong>Переменные:</strong> <code>{"{title}"}</code>, <code>{"{categories}"}</code>, <code>{"{tags}"}</code></p>
            <p><strong>Выход:</strong> колонка <code>description</code> в CSV</p>
          </div>
        </div>
      </div>

      <Button size="lg" onClick={startJob} disabled={submitting || !fileUrl} className="w-full h-12 text-sm font-medium gap-2">
        <Play className="h-4 w-4" />
        {submitting ? "Создание задачи..." : `Сгенерировать описания для ${lineCount ? lineCount.toLocaleString() + " видео" : "фида"}`}
      </Button>
    </motion.div>
  );
}
