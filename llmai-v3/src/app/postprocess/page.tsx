"use client";

import { useState } from "react";
import { motion } from "framer-motion";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Switch } from "@/components/ui/switch";
import { FileUpload } from "@/components/file-upload";
import { JobProgress } from "@/components/job-progress";
import { PageHeader } from "@/components/layout/page-header";
import { Play } from "lucide-react";
import { toast } from "sonner";
import { pageVariants } from "@/lib/animations";

const PIPELINE_STEPS = [
  { key: "domains", label: "Удаление доменов и URL", default: true },
  { key: "commentary", label: "Удаление комментариев LLM", default: true },
  { key: "stopwords", label: "Применение стоп-слов из БД", default: true },
  { key: "hieroglyphs", label: "Удаление иероглифов", default: true },
  { key: "emojis", label: "Удаление эмодзи", default: true },
  { key: "symbols", label: "Очистка запрещённых символов", default: true },
  { key: "spaces", label: "Нормализация пробелов", default: true },
  { key: "truncate", label: "Обрезка по длине (100 символов)", default: true },
];

export default function PostprocessPage() {
  const [fileUrl, setFileUrl] = useState("");
  const [titleCol, setTitleCol] = useState("title");
  const [patternsText, setPatternsText] = useState("");
  const [steps, setSteps] = useState<Record<string, boolean>>(
    Object.fromEntries(PIPELINE_STEPS.map((s) => [s.key, s.default]))
  );
  const [activeJobId, setActiveJobId] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);

  async function startJob() {
    if (!fileUrl) { toast.error("Загрузите файл"); return; }
    setSubmitting(true);
    try {
      const harmfulPatterns = patternsText.split("\n").map((l) => l.trim()).filter(Boolean);
      const resp = await fetch("/api/jobs", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          type: "POSTPROCESS",
          inputFileUrl: fileUrl,
          config: { titleCol, harmfulPatterns, applyStopWords: steps.stopwords },
        }),
      });
      const data = await resp.json();
      if (!resp.ok) throw new Error(data.error);
      setActiveJobId(data.job.id);
      toast.success("Постобработка запущена");
    } catch (err) { toast.error((err as Error).message); }
    finally { setSubmitting(false); }
  }

  return (
    <motion.div {...pageVariants} className="space-y-8">
      <PageHeader title="Постобработка" description="Локальная очистка текста без LLM — самая быстрая операция" />
      {activeJobId && <JobProgress jobId={activeJobId} onComplete={() => toast.success("Постобработка завершена!")} />}

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="space-y-6">
          <div className="rounded-xl border bg-[var(--surface)] p-6 space-y-5">
            <h2 className="text-[15px] font-medium">Файл</h2>
            <FileUpload onUpload={(data) => setFileUrl(data.fileUrl)} />
            <div>
              <Label className="text-xs font-medium text-[var(--text-muted)] uppercase tracking-wider">Колонка для очистки</Label>
              <Input value={titleCol} onChange={(e) => setTitleCol(e.target.value)} className="mt-1.5" />
            </div>
          </div>

          <div className="rounded-xl border bg-[var(--surface)] p-6 space-y-5">
            <h2 className="text-[15px] font-medium">Вредные паттерны</h2>
            <Textarea
              value={patternsText}
              onChange={(e) => setPatternsText(e.target.value)}
              rows={5}
              placeholder="По одному на строку — каждый будет удалён из текста"
              className="font-mono text-xs"
            />
          </div>
        </div>

        <div className="rounded-xl border bg-[var(--surface)] p-6 space-y-4">
          <h2 className="text-[15px] font-medium">Pipeline очистки</h2>
          <p className="text-xs text-[var(--text-muted)]">Включите нужные шаги обработки</p>
          <div className="space-y-1">
            {PIPELINE_STEPS.map((step, i) => (
              <div
                key={step.key}
                className="flex items-center justify-between rounded-lg px-4 py-3 hover:bg-[var(--surface-raised)] transition-colors"
              >
                <div className="flex items-center gap-3">
                  <span className="flex h-5 w-5 items-center justify-center rounded-full bg-[var(--surface-raised)] text-[10px] font-medium text-[var(--text-muted)]">
                    {i + 1}
                  </span>
                  <span className="text-sm">{step.label}</span>
                </div>
                <Switch
                  checked={steps[step.key]}
                  onCheckedChange={(v) => setSteps({ ...steps, [step.key]: v })}
                />
              </div>
            ))}
          </div>
        </div>
      </div>

      <Button size="lg" onClick={startJob} disabled={submitting || !fileUrl} className="w-full h-12 text-sm font-medium gap-2">
        <Play className="h-4 w-4" />{submitting ? "Запуск..." : "Запустить очистку"}
      </Button>
    </motion.div>
  );
}
