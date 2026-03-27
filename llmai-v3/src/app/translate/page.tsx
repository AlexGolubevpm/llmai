"use client";

import { useState } from "react";
import { motion } from "framer-motion";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Input } from "@/components/ui/input";
import { Slider } from "@/components/ui/slider";
import { FileUpload } from "@/components/file-upload";
import { ModelSelector } from "@/components/model-selector";
import { PresetSelector } from "@/components/preset-selector";
import { JobProgress } from "@/components/job-progress";
import { PageHeader } from "@/components/layout/page-header";
import type { JobConfig } from "@/types";
import { Play } from "lucide-react";
import { toast } from "sonner";
import { pageVariants } from "@/lib/animations";
import { cn } from "@/lib/utils";

const LANGUAGES = [
  { code: "English", flag: "EN" },
  { code: "Chinese", flag: "CN" },
  { code: "Japanese", flag: "JP" },
  { code: "Hindi", flag: "HI" },
  { code: "Spanish", flag: "ES" },
  { code: "Russian", flag: "RU" },
  { code: "German", flag: "DE" },
  { code: "French", flag: "FR" },
];

function LanguageGrid({
  value,
  onChange,
  label,
}: {
  value: string;
  onChange: (v: string) => void;
  label: string;
}) {
  return (
    <div>
      <Label className="text-xs font-medium text-[var(--text-muted)] uppercase tracking-wider">
        {label}
      </Label>
      <div className="grid grid-cols-4 gap-2 mt-2">
        {LANGUAGES.map((lang) => (
          <button
            key={lang.code}
            onClick={() => onChange(lang.code)}
            className={cn(
              "flex items-center justify-center gap-1.5 rounded-lg border px-3 py-2 text-xs font-medium transition-all",
              value === lang.code
                ? "border-[var(--accent-blue)] bg-[var(--accent-blue-light)] text-[var(--accent-blue)]"
                : "border-[var(--border)] hover:border-[var(--border-hover)] text-[var(--text-secondary)]"
            )}
          >
            <span className="text-[10px] opacity-50">{lang.flag}</span>
            {lang.code}
          </button>
        ))}
      </div>
    </div>
  );
}

export default function TranslatePage() {
  const [fileUrl, setFileUrl] = useState("");
  const [model, setModel] = useState("openai/gpt-4o-mini");
  const [config, setConfig] = useState<JobConfig>({
    systemPrompt: "You are a professional translator.",
    maxTokens: 512, temperature: 0.7, topP: 1.0, minP: 0.0, topK: 40,
    presencePenalty: 0.0, frequencyPenalty: 0.0, repetitionPenalty: 1.0,
  });
  const [sourceLanguage, setSourceLanguage] = useState("English");
  const [targetLanguage, setTargetLanguage] = useState("Chinese");
  const [titleCol, setTitleCol] = useState("title");
  const [maxWorkers, setMaxWorkers] = useState(5);
  const [activeJobId, setActiveJobId] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);

  async function startJob() {
    if (!fileUrl) { toast.error("Загрузите файл"); return; }
    if (sourceLanguage === targetLanguage) { toast.error("Языки должны отличаться"); return; }
    setSubmitting(true);
    try {
      const resp = await fetch("/api/jobs", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          type: "TRANSLATE",
          inputFileUrl: fileUrl,
          config: { ...config, model, titleCol, maxWorkers, sourceLanguage, targetLanguage, chunkSize: 10 },
        }),
      });
      const data = await resp.json();
      if (!resp.ok) throw new Error(data.error);
      setActiveJobId(data.job.id);
      toast.success("Задача перевода создана");
    } catch (err) { toast.error((err as Error).message); }
    finally { setSubmitting(false); }
  }

  return (
    <motion.div {...pageVariants} className="space-y-8">
      <PageHeader title="Перевод текста" description="Пакетный перевод через LLM" />
      {activeJobId && <JobProgress jobId={activeJobId} onComplete={() => toast.success("Перевод завершён!")} />}

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="space-y-6">
          <div className="rounded-xl border bg-[var(--surface)] p-6 space-y-5">
            <h2 className="text-[15px] font-medium">Файл</h2>
            <FileUpload onUpload={(data) => setFileUrl(data.fileUrl)} />
            <div>
              <Label className="text-xs font-medium text-[var(--text-muted)] uppercase tracking-wider">Колонка для перевода</Label>
              <Input value={titleCol} onChange={(e) => setTitleCol(e.target.value)} className="mt-1.5" />
            </div>
            <LanguageGrid value={sourceLanguage} onChange={setSourceLanguage} label="Исходный язык" />
            <LanguageGrid value={targetLanguage} onChange={setTargetLanguage} label="Целевой язык" />
            <div>
              <div className="flex items-baseline justify-between mb-2">
                <Label className="text-xs font-medium text-[var(--text-muted)] uppercase tracking-wider">Потоки</Label>
                <span className="text-xs font-mono text-[var(--text-secondary)]">{maxWorkers}</span>
              </div>
              <Slider value={[maxWorkers]} onValueChange={(v) => setMaxWorkers(typeof v === "number" ? v : v[0])} min={1} max={20} step={1} />
            </div>
          </div>
        </div>
        <div className="space-y-6">
          <div className="rounded-xl border bg-[var(--surface)] p-6 space-y-5">
            <h2 className="text-[15px] font-medium">Модель</h2>
            <ModelSelector value={model} onChange={setModel} />
          </div>
          <div className="rounded-xl border bg-[var(--surface)] p-6 space-y-5">
            <h2 className="text-[15px] font-medium">Параметры</h2>
            <PresetSelector value={config} onChange={setConfig} onModelChange={setModel} />
          </div>
        </div>
      </div>

      <Button size="lg" onClick={startJob} disabled={submitting || !fileUrl} className="w-full h-12 text-sm font-medium gap-2">
        <Play className="h-4 w-4" />{submitting ? "Создание..." : `Перевести ${sourceLanguage} → ${targetLanguage}`}
      </Button>
    </motion.div>
  );
}
