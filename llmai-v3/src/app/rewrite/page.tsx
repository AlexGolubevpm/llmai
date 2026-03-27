"use client";

import { useState } from "react";
import { motion } from "framer-motion";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Slider } from "@/components/ui/slider";
import { Switch } from "@/components/ui/switch";
import { FileUpload } from "@/components/file-upload";
import { ModelSelector } from "@/components/model-selector";
import { PresetSelector } from "@/components/preset-selector";
import { JobProgress } from "@/components/job-progress";
import { PageHeader } from "@/components/layout/page-header";
import type { JobConfig } from "@/types";
import { Play, FileText, Type } from "lucide-react";
import { toast } from "sonner";
import { pageVariants } from "@/lib/animations";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";

export default function RewritePage() {
  const [fileUrl, setFileUrl] = useState("");
  const [lineCount, setLineCount] = useState(0);
  const [inputMode, setInputMode] = useState<"file" | "text">("file");
  const [textInput, setTextInput] = useState("");
  const [model, setModel] = useState("openai/gpt-4o-mini");
  const [config, setConfig] = useState<JobConfig>({
    systemPrompt: "You are a helpful assistant.",
    maxTokens: 512,
    temperature: 0.7,
    topP: 1.0,
    minP: 0.0,
    topK: 40,
    presencePenalty: 0.0,
    frequencyPenalty: 0.0,
    repetitionPenalty: 1.0,
  });
  const [userPrompt, setUserPrompt] = useState("");
  const [multiplier, setMultiplier] = useState(1);
  const [titleCol, setTitleCol] = useState("title");
  const [maxWorkers, setMaxWorkers] = useState(5);
  const [applyStopWords, setApplyStopWords] = useState(true);
  const [activeJobId, setActiveJobId] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);

  async function createFileFromText(): Promise<string | null> {
    try {
      const resp = await fetch("/api/files/from-text", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: textInput, mode: "rewrite" }),
      });
      const data = await resp.json();
      if (!resp.ok) throw new Error(data.error);
      setLineCount(data.lineCount);
      return data.fileUrl;
    } catch (err) {
      toast.error((err as Error).message);
      return null;
    }
  }

  async function startJob() {
    if (inputMode === "text" && !textInput.trim()) {
      toast.error("Введите текст");
      return;
    }
    if (inputMode === "file" && !fileUrl) {
      toast.error("Загрузите файл");
      return;
    }
    setSubmitting(true);
    try {
      let jobFileUrl = fileUrl;
      if (inputMode === "text") {
        const created = await createFileFromText();
        if (!created) { setSubmitting(false); return; }
        jobFileUrl = created;
      }
      const resp = await fetch("/api/jobs", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          type: "REWRITE",
          inputFileUrl: jobFileUrl,
          config: {
            ...config,
            model,
            userPrompt,
            multiplier,
            titleCol,
            maxWorkers,
            applyStopWords,
            chunkSize: 10,
          },
        }),
      });
      const data = await resp.json();
      if (!resp.ok) throw new Error(data.error);
      setActiveJobId(data.job.id);
      toast.success("Задача создана");
    } catch (err) {
      toast.error((err as Error).message);
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <motion.div {...pageVariants} className="space-y-8">
      <PageHeader
        title="Массовый рерайт"
        description="Загрузите файл, настройте параметры и запустите рерайт с множителем"
      />

      {activeJobId && (
        <JobProgress
          jobId={activeJobId}
          onComplete={() => toast.success("Рерайт завершён! Скачайте результат на Dashboard.")}
        />
      )}

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Left column */}
        <div className="space-y-6">
          <div className="rounded-xl border bg-[var(--surface)] p-6 space-y-5">
            <h2 className="text-[15px] font-medium">Источник данных</h2>
            <Tabs value={inputMode} onValueChange={(v) => v && setInputMode(v as "file" | "text")}>
              <TabsList>
                <TabsTrigger value="file" className="gap-1.5"><FileText className="h-3.5 w-3.5" /> Файл</TabsTrigger>
                <TabsTrigger value="text" className="gap-1.5"><Type className="h-3.5 w-3.5" /> Текст</TabsTrigger>
              </TabsList>
              <TabsContent value="file" className="mt-4 space-y-4">
                <FileUpload
                  onUpload={(data) => {
                    setFileUrl(data.fileUrl);
                    setLineCount(data.lineCount);
                  }}
                />
                {lineCount > 0 && (
                  <p className="text-xs text-[var(--text-muted)]">{lineCount.toLocaleString()} строк</p>
                )}
                <div>
                  <Label className="text-xs font-medium text-[var(--text-muted)] uppercase tracking-wider">
                    Колонка для рерайта
                  </Label>
                  <Input value={titleCol} onChange={(e) => setTitleCol(e.target.value)} className="mt-1.5" />
                </div>
              </TabsContent>
              <TabsContent value="text" className="mt-4 space-y-4">
                <div>
                  <Label className="text-xs font-medium text-[var(--text-muted)] uppercase tracking-wider">
                    Текст для рерайта
                  </Label>
                  <Textarea
                    value={textInput}
                    onChange={(e) => setTextInput(e.target.value)}
                    placeholder={"Введите текст для рерайта...\nКаждая строка = отдельная запись"}
                    rows={6}
                    className="mt-1.5"
                  />
                  {textInput.trim() && (
                    <p className="text-xs text-[var(--text-muted)] mt-1.5">
                      {textInput.trim().split("\n").filter((l) => l.trim()).length} строк
                    </p>
                  )}
                </div>
              </TabsContent>
            </Tabs>
          </div>

          <div className="rounded-xl border bg-[var(--surface)] p-6 space-y-5">
            <h2 className="text-[15px] font-medium">Настройки рерайта</h2>
            <div>
              <Label className="text-xs font-medium text-[var(--text-muted)] uppercase tracking-wider">
                Промпт
              </Label>
              <Textarea
                value={userPrompt}
                onChange={(e) => setUserPrompt(e.target.value)}
                placeholder="Введите промпт для рерайта..."
                rows={3}
                className="mt-1.5"
              />
            </div>

            <div>
              <div className="flex items-baseline justify-between mb-2">
                <Label className="text-xs font-medium text-[var(--text-muted)] uppercase tracking-wider">
                  Множитель
                </Label>
                <span className="inline-flex items-center rounded-md bg-[var(--accent-blue-light)] px-2 py-0.5 text-xs font-semibold text-[var(--accent-blue)]">
                  x{multiplier}
                </span>
              </div>
              <Slider
                value={[multiplier]}
                onValueChange={(v) => setMultiplier(typeof v === "number" ? v : v[0])}
                min={1}
                max={10}
                step={1}
              />
              <p className="text-xs text-[var(--text-muted)] mt-1.5">
                {multiplier > 1
                  ? `Файл будет переписан ${multiplier} раз. Каждый проход берёт результат предыдущего.`
                  : "Одиночный рерайт."}
              </p>
            </div>

            <div>
              <div className="flex items-baseline justify-between mb-2">
                <Label className="text-xs font-medium text-[var(--text-muted)] uppercase tracking-wider">
                  Параллельные потоки
                </Label>
                <span className="text-xs font-mono text-[var(--text-secondary)]">{maxWorkers}</span>
              </div>
              <Slider
                value={[maxWorkers]}
                onValueChange={(v) => setMaxWorkers(typeof v === "number" ? v : v[0])}
                min={1}
                max={20}
                step={1}
              />
            </div>

            <div className="flex items-center justify-between rounded-lg bg-[var(--surface-raised)] px-4 py-3">
              <Label className="text-sm">Применять стоп-слова после каждого прохода</Label>
              <Switch checked={applyStopWords} onCheckedChange={setApplyStopWords} />
            </div>
          </div>
        </div>

        {/* Right column */}
        <div className="space-y-6">
          <div className="rounded-xl border bg-[var(--surface)] p-6 space-y-5">
            <h2 className="text-[15px] font-medium">Модель</h2>
            <ModelSelector value={model} onChange={setModel} />
          </div>

          <div className="rounded-xl border bg-[var(--surface)] p-6 space-y-5">
            <h2 className="text-[15px] font-medium">Параметры генерации</h2>
            <PresetSelector value={config} onChange={setConfig} onModelChange={setModel} />
          </div>
        </div>
      </div>

      <Button
        size="lg"
        onClick={startJob}
        disabled={submitting || (inputMode === "file" ? !fileUrl : !textInput.trim())}
        className="w-full h-12 text-sm font-medium gap-2"
      >
        <Play className="h-4 w-4" />
        {submitting ? "Создание задачи..." : `Запустить рерайт x${multiplier}`}
      </Button>
    </motion.div>
  );
}
