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
import { PresetSelector } from "@/components/preset-selector";
import { JobProgress } from "@/components/job-progress";
import { PageHeader } from "@/components/layout/page-header";
import type { JobConfig } from "@/types";
import { Play, Eye, PenLine, FileText, Type } from "lucide-react";
import { toast } from "sonner";
import { pageVariants } from "@/lib/animations";

const DEFAULT_VISION_PROMPT = `Analyze this image and return a JSON object with exactly these fields:
1. "tags": comma-separated list of up to 15 descriptive tags (actions, body types, positions, clothing, setting, hair color, ethnicity)
2. "scene": 1-2 sentence description of what is happening in the scene
3. "type": content type and style, format: "<type> | <count> people | <style>" where type is one of: hentai, anime, 3D, real, CGI, cartoon

Return ONLY valid JSON, no markdown, no explanation:
{"tags":"tag1, tag2, ...","scene":"...","type":"..."}`;

const DEFAULT_SEO_PROMPT = `Based on the context below, generate SEO-optimized title and description.

Context:
Original title: {title}
Tags: {tags}
Scene: {scene}
Content type: {type}
Existing tags: {existing_tags}
Categories: {categories}

Requirements:
- title: max 90 characters, English, natural, engaging, search-optimized for 2026
- description: max 160 characters, complements the title with secondary keywords

Return ONLY valid JSON, no markdown:
{"title":"...","description":"..."}`;

export default function AIProcessPage() {
  const [fileUrl, setFileUrl] = useState("");
  const [lineCount, setLineCount] = useState(0);
  const [inputMode, setInputMode] = useState<"file" | "text">("file");
  const [textTitle, setTextTitle] = useState("");
  const [textThumb, setTextThumb] = useState("");

  // Step 1: Vision
  const [visionModel, setVisionModel] = useState("xiaomi/mimo-v2-omni");
  const [visionPrompt, setVisionPrompt] = useState(DEFAULT_VISION_PROMPT);

  // Step 2: SEO
  const [textModel, setTextModel] = useState("openai/gpt-4o-mini");
  const [seoPrompt, setSeoPrompt] = useState(DEFAULT_SEO_PROMPT);

  // Shared params
  const [config, setConfig] = useState<JobConfig>({
    systemPrompt: "Expert SEO writer.",
    maxTokens: 300, temperature: 0.7, topP: 1.0, minP: 0.0, topK: 40,
    presencePenalty: 0.2, frequencyPenalty: 0.4, repetitionPenalty: 1.2,
  });
  const [maxWorkers, setMaxWorkers] = useState(3);
  const [activeJobId, setActiveJobId] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);

  async function startJob() {
    if (inputMode === "text" && !textTitle.trim()) { toast.error("Введите текст"); return; }
    if (inputMode === "file" && !fileUrl) { toast.error("Загрузите фид"); return; }
    setSubmitting(true);
    try {
      let jobFileUrl = fileUrl;
      if (inputMode === "text") {
        const resp = await fetch("/api/files/from-text", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ text: textTitle, mode: "ai-process", thumbnailUrl: textThumb }),
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
            ...config,
            visionModel,
            textModel,
            visionPrompt,
            seoPrompt,
            maxWorkers,
            chunkSize: 5,
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
      <PageHeader title="AI Process 3.0" description="2-шаговый pipeline: Vision анализ → SEO генерация. Каждый шаг с отдельной моделью и промптом." />

      {/* Pipeline visualization */}
      <div className="grid grid-cols-2 gap-4">
        <div className="rounded-xl border bg-[var(--surface)] p-5 flex items-center gap-4">
          <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-blue-50 text-blue-600">
            <Eye className="h-5 w-5" />
          </div>
          <div>
            <div className="text-xs font-medium text-[var(--text-muted)]">Шаг 1</div>
            <div className="text-sm font-medium">Vision анализ</div>
            <div className="text-xs text-[var(--text-muted)]">Теги + описание + тип контента</div>
          </div>
        </div>
        <div className="rounded-xl border bg-[var(--surface)] p-5 flex items-center gap-4">
          <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-green-50 text-green-600">
            <PenLine className="h-5 w-5" />
          </div>
          <div>
            <div className="text-xs font-medium text-[var(--text-muted)]">Шаг 2</div>
            <div className="text-sm font-medium">SEO генерация</div>
            <div className="text-xs text-[var(--text-muted)]">Title + Description</div>
          </div>
        </div>
      </div>

      {activeJobId && <JobProgress jobId={activeJobId} onComplete={() => toast.success("AI Process завершён!")} />}

      {/* Source data */}
      <div className="rounded-xl border bg-[var(--surface)] p-6 space-y-5">
        <h2 className="text-[15px] font-medium">Источник данных</h2>
        <Tabs value={inputMode} onValueChange={(v) => v && setInputMode(v as "file" | "text")}>
          <TabsList>
            <TabsTrigger value="file" className="gap-1.5"><FileText className="h-3.5 w-3.5" /> Файл</TabsTrigger>
            <TabsTrigger value="text" className="gap-1.5"><Type className="h-3.5 w-3.5" /> Одиночный тест</TabsTrigger>
          </TabsList>
          <TabsContent value="file" className="mt-4 space-y-4">
            <FileUpload onUpload={(data) => { setFileUrl(data.fileUrl); setLineCount(data.lineCount); }} />
            {lineCount > 0 && <p className="text-xs text-[var(--text-muted)]">{lineCount.toLocaleString()} строк</p>}
          </TabsContent>
          <TabsContent value="text" className="mt-4 space-y-4">
            <div>
              <Label className="text-xs font-medium text-[var(--text-muted)] uppercase tracking-wider">Тайтл</Label>
              <Input value={textTitle} onChange={(e) => setTextTitle(e.target.value)} placeholder="Тайтл для тестирования..." className="mt-1.5" />
            </div>
            <div>
              <Label className="text-xs font-medium text-[var(--text-muted)] uppercase tracking-wider">URL тумбы (опционально)</Label>
              <Input value={textThumb} onChange={(e) => setTextThumb(e.target.value)} placeholder="https://example.com/thumb.jpg" className="mt-1.5" />
            </div>
          </TabsContent>
        </Tabs>
        <div>
          <div className="flex items-baseline justify-between mb-2">
            <Label className="text-xs font-medium text-[var(--text-muted)] uppercase tracking-wider">Потоки</Label>
            <span className="text-xs font-mono text-[var(--text-secondary)]">{maxWorkers}</span>
          </div>
          <Slider value={[maxWorkers]} onValueChange={(v) => setMaxWorkers(typeof v === "number" ? v : v[0])} min={1} max={10} step={1} />
        </div>
      </div>

      {/* Two-step config: side by side */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Step 1: Vision */}
        <div className="rounded-xl border-2 border-blue-100 bg-[var(--surface)] p-6 space-y-5">
          <div className="flex items-center gap-2">
            <div className="flex h-6 w-6 items-center justify-center rounded-lg bg-blue-50 text-blue-600">
              <Eye className="h-3.5 w-3.5" />
            </div>
            <h2 className="text-[15px] font-medium">Шаг 1: Vision модель</h2>
          </div>
          <ModelSelector value={visionModel} onChange={setVisionModel} />
          <div>
            <Label className="text-xs font-medium text-[var(--text-muted)] uppercase tracking-wider">Промпт</Label>
            <Textarea
              value={visionPrompt}
              onChange={(e) => setVisionPrompt(e.target.value)}
              rows={8}
              className="mt-1.5 font-mono text-xs"
            />
            <button onClick={() => setVisionPrompt(DEFAULT_VISION_PROMPT)} className="mt-1 text-xs text-[var(--accent-blue)] hover:underline">
              Сбросить к дефолту
            </button>
          </div>
        </div>

        {/* Step 2: SEO */}
        <div className="rounded-xl border-2 border-green-100 bg-[var(--surface)] p-6 space-y-5">
          <div className="flex items-center gap-2">
            <div className="flex h-6 w-6 items-center justify-center rounded-lg bg-green-50 text-green-600">
              <PenLine className="h-3.5 w-3.5" />
            </div>
            <h2 className="text-[15px] font-medium">Шаг 2: Text модель</h2>
          </div>
          <ModelSelector value={textModel} onChange={setTextModel} />
          <div>
            <Label className="text-xs font-medium text-[var(--text-muted)] uppercase tracking-wider">Промпт</Label>
            <Textarea
              value={seoPrompt}
              onChange={(e) => setSeoPrompt(e.target.value)}
              rows={8}
              className="mt-1.5 font-mono text-xs"
            />
            <p className="mt-1 text-xs text-[var(--text-muted)]">
              Переменные: {"{title}"}, {"{tags}"}, {"{scene}"}, {"{type}"}, {"{existing_tags}"}, {"{categories}"}
            </p>
            <button onClick={() => setSeoPrompt(DEFAULT_SEO_PROMPT)} className="mt-1 text-xs text-[var(--accent-blue)] hover:underline">
              Сбросить к дефолту
            </button>
          </div>
        </div>
      </div>

      {/* Shared generation params */}
      <div className="rounded-xl border bg-[var(--surface)] p-6 space-y-5">
        <h2 className="text-[15px] font-medium">Параметры генерации (шаг 2)</h2>
        <PresetSelector value={config} onChange={setConfig} onModelChange={setTextModel} />
      </div>

      <Button
        size="lg"
        onClick={startJob}
        disabled={submitting || (inputMode === "file" ? !fileUrl : !textTitle.trim())}
        className="w-full h-12 text-sm font-medium gap-2"
      >
        <Play className="h-4 w-4" />{submitting ? "Запуск..." : "Запустить AI Process 3.0"}
      </Button>
    </motion.div>
  );
}
