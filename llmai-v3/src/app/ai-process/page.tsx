"use client";

import { useState } from "react";
import { motion } from "framer-motion";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Slider } from "@/components/ui/slider";
import { FileUpload } from "@/components/file-upload";
import { ModelSelector } from "@/components/model-selector";
import { PresetSelector } from "@/components/preset-selector";
import { JobProgress } from "@/components/job-progress";
import { PageHeader } from "@/components/layout/page-header";
import type { JobConfig } from "@/types";
import { Textarea } from "@/components/ui/textarea";
import { Input } from "@/components/ui/input";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Play, Image, Tags, PenLine, ArrowRight, FileText, Type } from "lucide-react";
import { toast } from "sonner";
import { pageVariants } from "@/lib/animations";
import { cn } from "@/lib/utils";

const STEPS = [
  { icon: Image, label: "WD Tagger", desc: "Анализ тумбы — теги + рейтинг" },
  { icon: Tags, label: "Tag Mapping", desc: "Маппинг на разрешённые теги из БД" },
  { icon: PenLine, label: "SEO Generation", desc: "Title + Description под SEO" },
];

export default function AIProcessPage() {
  const [fileUrl, setFileUrl] = useState("");
  const [lineCount, setLineCount] = useState(0);
  const [inputMode, setInputMode] = useState<"file" | "text">("file");
  const [textTitle, setTextTitle] = useState("");
  const [textThumb, setTextThumb] = useState("");
  const [model, setModel] = useState("openai/gpt-4o-mini");
  const [config, setConfig] = useState<JobConfig>({
    systemPrompt: "You are an expert SEO content writer.",
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
          config: { ...config, model, maxWorkers, chunkSize: 5, applyStopWords: true },
        }),
      });
      const data = await resp.json();
      if (!resp.ok) throw new Error(data.error);
      setActiveJobId(data.job.id);
      toast.success("AI Process 3.0 запущен");
    } catch (err) { toast.error((err as Error).message); }
    finally { setSubmitting(false); }
  }

  return (
    <motion.div {...pageVariants} className="space-y-8">
      <PageHeader title="AI Process 3.0" description="3-шаговый pipeline: анализ изображений → маппинг тегов → SEO-генерация" />

      {/* Pipeline visualization */}
      <div className="rounded-xl border bg-[var(--surface)] p-6">
        <div className="flex items-center justify-between">
          {STEPS.map((step, i) => (
            <div key={step.label} className="flex items-center gap-4 flex-1">
              <div className="flex items-center gap-3 flex-1">
                <div className={cn(
                  "flex h-10 w-10 shrink-0 items-center justify-center rounded-xl",
                  i === 0 ? "bg-blue-50 text-blue-600" :
                  i === 1 ? "bg-purple-50 text-purple-600" :
                  "bg-green-50 text-green-600"
                )}>
                  <step.icon className="h-5 w-5" />
                </div>
                <div className="min-w-0">
                  <div className="text-xs font-medium text-[var(--text-muted)] uppercase tracking-wider">
                    Шаг {i + 1}
                  </div>
                  <div className="text-sm font-medium truncate">{step.label}</div>
                  <div className="text-xs text-[var(--text-muted)] truncate hidden sm:block">{step.desc}</div>
                </div>
              </div>
              {i < STEPS.length - 1 && (
                <ArrowRight className="h-4 w-4 text-[var(--text-muted)] shrink-0 hidden md:block" />
              )}
            </div>
          ))}
        </div>
      </div>

      {activeJobId && <JobProgress jobId={activeJobId} onComplete={() => toast.success("AI Process завершён!")} />}

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="space-y-6">
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
                <div className="rounded-lg bg-[var(--surface-raised)] px-4 py-3">
                  <p className="text-xs text-[var(--text-muted)]">
                    Колонки: <code className="font-mono text-[var(--text-secondary)]">thumbnail_url</code>, <code className="font-mono text-[var(--text-secondary)]">title</code>, <code className="font-mono text-[var(--text-secondary)]">tags</code>, <code className="font-mono text-[var(--text-secondary)]">categories</code>
                  </p>
                </div>
              </TabsContent>
              <TabsContent value="text" className="mt-4 space-y-4">
                <div>
                  <Label className="text-xs font-medium text-[var(--text-muted)] uppercase tracking-wider">Тайтл</Label>
                  <Textarea
                    value={textTitle}
                    onChange={(e) => setTextTitle(e.target.value)}
                    placeholder="Введите тайтл для тестирования AI Process..."
                    rows={3}
                    className="mt-1.5"
                  />
                </div>
                <div>
                  <Label className="text-xs font-medium text-[var(--text-muted)] uppercase tracking-wider">URL тумбы (опционально)</Label>
                  <Input
                    value={textThumb}
                    onChange={(e) => setTextThumb(e.target.value)}
                    placeholder="https://example.com/thumbnail.jpg"
                    className="mt-1.5"
                  />
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
        </div>
        <div className="space-y-6">
          <div className="rounded-xl border bg-[var(--surface)] p-6 space-y-5">
            <h2 className="text-[15px] font-medium">Модель (шаги 2-3)</h2>
            <ModelSelector value={model} onChange={setModel} />
          </div>
          <div className="rounded-xl border bg-[var(--surface)] p-6 space-y-5">
            <h2 className="text-[15px] font-medium">Параметры генерации</h2>
            <PresetSelector value={config} onChange={setConfig} onModelChange={setModel} />
          </div>
        </div>
      </div>

      <Button size="lg" onClick={startJob} disabled={submitting || (inputMode === "file" ? !fileUrl : !textTitle.trim())} className="w-full h-12 text-sm font-medium gap-2">
        <Play className="h-4 w-4" />{submitting ? "Запуск..." : "Запустить AI Process 3.0"}
      </Button>
    </motion.div>
  );
}
