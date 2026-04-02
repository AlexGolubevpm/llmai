"use client";

import { useState } from "react";
import { motion } from "framer-motion";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Slider } from "@/components/ui/slider";
import { Textarea } from "@/components/ui/textarea";
import { Input } from "@/components/ui/input";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { FileUpload } from "@/components/file-upload";
import { ModelSelector } from "@/components/model-selector";
import { JobProgress } from "@/components/job-progress";
import { PageHeader } from "@/components/layout/page-header";
import { Play, FileText, Type } from "lucide-react";
import { toast } from "sonner";
import { pageVariants } from "@/lib/animations";

const DEFAULT_PROMPT = `Look at this image carefully. Describe what you see.

Return ONLY a valid JSON object with these 3 fields:
{
  "tags": "comma-separated list of up to 15 descriptive tags",
  "scene": "1-2 sentence description of what is happening",
  "type": "content type: hentai, anime, 3D, real, CGI, or cartoon"
}

For tags include: actions, body types, positions, clothing, setting, hair color.
Do not include any markdown, explanation, or text outside the JSON.`;

export default function AIProcessPage() {
  const [fileUrl, setFileUrl] = useState("");
  const [lineCount, setLineCount] = useState(0);
  const [inputMode, setInputMode] = useState<"file" | "text">("file");
  const [textTitle, setTextTitle] = useState("");
  const [textThumb, setTextThumb] = useState("");

  const [model, setModel] = useState("google/gemini-2.5-flash-preview-05-20");
  const [prompt, setPrompt] = useState(DEFAULT_PROMPT);
  const [temperature, setTemperature] = useState(0.7);
  const [maxTokens, setMaxTokens] = useState(500);
  const [maxWorkers, setMaxWorkers] = useState(3);

  const [activeJobId, setActiveJobId] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);

  async function startJob() {
    if (inputMode === "text" && !textThumb.trim()) { toast.error("Введите URL тумбы"); return; }
    if (inputMode === "file" && !fileUrl) { toast.error("Загрузите файл"); return; }
    setSubmitting(true);
    try {
      let jobFileUrl = fileUrl;
      if (inputMode === "text") {
        const resp = await fetch("/api/files/from-text", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ text: textTitle || "test", mode: "ai-process", thumbnailUrl: textThumb }),
        });
        const data = await resp.json();
        if (!resp.ok) throw new Error(data.error);
        jobFileUrl = data.fileUrl;
      }

      const resp = await fetch("/api/jobs", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          type: "AI_PROCESS",
          inputFileUrl: jobFileUrl,
          config: {
            visionModel: model,
            model: model,
            visionPrompt: prompt,
            temperature,
            maxTokens,
            maxWorkers,
            applyStopWords: true,
          },
        }),
      });
      const data = await resp.json();
      if (!resp.ok) throw new Error(data.error);
      setActiveJobId(data.job.id);
      toast.success("AI Process запущен");
    } catch (err) { toast.error((err as Error).message); }
    finally { setSubmitting(false); }
  }

  return (
    <motion.div {...pageVariants} className="space-y-8">
      <PageHeader
        title="AI Process"
        description="Модель анализирует тумбу и возвращает теги, описание сцены и тип контента"
      />

      {activeJobId && <JobProgress jobId={activeJobId} onComplete={() => toast.success("AI Process завершён!")} />}

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Left column */}
        <div className="space-y-6">
          {/* Source */}
          <div className="rounded-xl border bg-[var(--surface)] p-6 space-y-5">
            <h2 className="text-[15px] font-medium">Источник</h2>
            <Tabs value={inputMode} onValueChange={(v) => v && setInputMode(v as "file" | "text")}>
              <TabsList>
                <TabsTrigger value="file" className="gap-1.5"><FileText className="h-3.5 w-3.5" /> Файл</TabsTrigger>
                <TabsTrigger value="text" className="gap-1.5"><Type className="h-3.5 w-3.5" /> Тест</TabsTrigger>
              </TabsList>
              <TabsContent value="file" className="mt-4 space-y-4">
                <FileUpload onUpload={(data) => { setFileUrl(data.fileUrl); setLineCount(data.lineCount); }} />
                {lineCount > 0 && <p className="text-xs text-[var(--text-muted)]">{lineCount.toLocaleString()} строк</p>}
              </TabsContent>
              <TabsContent value="text" className="mt-4 space-y-4">
                <div>
                  <Label className="text-xs font-medium text-[var(--text-muted)] uppercase tracking-wider">URL тумбы</Label>
                  <Input value={textThumb} onChange={(e) => setTextThumb(e.target.value)} placeholder="https://example.com/thumb.jpg" className="mt-1.5" />
                </div>
                <div>
                  <Label className="text-xs font-medium text-[var(--text-muted)] uppercase tracking-wider">Тайтл (опционально)</Label>
                  <Input value={textTitle} onChange={(e) => setTextTitle(e.target.value)} placeholder="Тайтл для контекста" className="mt-1.5" />
                </div>
              </TabsContent>
            </Tabs>
          </div>

          {/* Model + Settings */}
          <div className="rounded-xl border bg-[var(--surface)] p-6 space-y-5">
            <h2 className="text-[15px] font-medium">Модель и настройки</h2>
            <ModelSelector value={model} onChange={setModel} />

            <div className="grid grid-cols-3 gap-4 pt-2">
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
                <Slider value={[maxTokens]} onValueChange={(v) => setMaxTokens(typeof v === "number" ? v : v[0])} min={100} max={4000} step={50} />
              </div>
              <div>
                <div className="flex items-baseline justify-between mb-2">
                  <Label className="text-xs font-mono text-[var(--text-muted)]">потоки</Label>
                  <span className="text-xs font-mono font-medium">{maxWorkers}</span>
                </div>
                <Slider value={[maxWorkers]} onValueChange={(v) => setMaxWorkers(typeof v === "number" ? v : v[0])} min={1} max={10} step={1} />
              </div>
            </div>
          </div>
        </div>

        {/* Right column: Prompt */}
        <div className="rounded-xl border bg-[var(--surface)] p-6 space-y-4">
          <div className="flex items-center justify-between">
            <h2 className="text-[15px] font-medium">Промпт</h2>
            <button onClick={() => setPrompt(DEFAULT_PROMPT)} className="text-xs text-[var(--accent-blue)] hover:underline">
              Сбросить
            </button>
          </div>
          <Textarea
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            rows={18}
            className="font-mono text-xs"
          />
          <div className="rounded-lg bg-[var(--surface-raised)] px-4 py-3 text-xs text-[var(--text-muted)] space-y-1">
            <p><strong>Выход:</strong> JSON с полями <code>tags</code>, <code>scene</code>, <code>type</code></p>
            <p><strong>Колонки в CSV:</strong> ai_tags, scene_description, content_type</p>
          </div>
        </div>
      </div>

      <Button
        size="lg"
        onClick={startJob}
        disabled={submitting || (inputMode === "file" ? !fileUrl : !textThumb.trim())}
        className="w-full h-12 text-sm font-medium gap-2"
      >
        <Play className="h-4 w-4" />{submitting ? "Запуск..." : "Запустить анализ"}
      </Button>
    </motion.div>
  );
}
