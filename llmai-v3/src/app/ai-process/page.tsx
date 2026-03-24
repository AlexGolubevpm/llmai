"use client";

import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Slider } from "@/components/ui/slider";
import { FileUpload } from "@/components/file-upload";
import { ModelSelector } from "@/components/model-selector";
import { PresetSelector } from "@/components/preset-selector";
import { JobProgress } from "@/components/job-progress";
import type { JobConfig } from "@/types";
import { Bot, Play } from "lucide-react";
import { toast } from "sonner";

export default function AIProcessPage() {
  const [fileUrl, setFileUrl] = useState("");
  const [lineCount, setLineCount] = useState(0);
  const [model, setModel] = useState("meta-llama/llama-3.1-8b-instruct");
  const [config, setConfig] = useState<JobConfig>({
    systemPrompt: "You are an expert SEO content writer.",
    maxTokens: 300,
    temperature: 0.7,
    topP: 1.0, minP: 0.0, topK: 40,
    presencePenalty: 0.2, frequencyPenalty: 0.4, repetitionPenalty: 1.2,
  });
  const [maxWorkers, setMaxWorkers] = useState(3);
  const [activeJobId, setActiveJobId] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);

  async function startJob() {
    if (!fileUrl) { toast.error("Загрузите фид"); return; }
    setSubmitting(true);
    try {
      const resp = await fetch("/api/jobs", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          type: "AI_PROCESS",
          inputFileUrl: fileUrl,
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
    <div className="space-y-6">
      <div className="flex items-center gap-3">
        <Bot className="h-8 w-8 text-primary" />
        <h1 className="text-3xl font-bold">AI Process 3.0</h1>
      </div>

      <Card>
        <CardContent className="pt-6">
          <div className="grid grid-cols-3 gap-4 text-center text-sm">
            <div className="rounded-lg border p-4">
              <div className="font-bold text-primary mb-1">Шаг 1</div>
              <div className="text-muted-foreground">WD Tagger анализирует тумбу → теги + рейтинг</div>
            </div>
            <div className="rounded-lg border p-4">
              <div className="font-bold text-primary mb-1">Шаг 2</div>
              <div className="text-muted-foreground">LLM маппит теги на разрешённые из БД</div>
            </div>
            <div className="rounded-lg border p-4">
              <div className="font-bold text-primary mb-1">Шаг 3</div>
              <div className="text-muted-foreground">LLM генерирует SEO title + description</div>
            </div>
          </div>
        </CardContent>
      </Card>

      {activeJobId && <JobProgress jobId={activeJobId} onComplete={() => toast.success("AI Process завершён!")} />}

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="space-y-6">
          <Card>
            <CardHeader><CardTitle>Фид (CSV)</CardTitle></CardHeader>
            <CardContent className="space-y-4">
              <FileUpload onUpload={(data) => { setFileUrl(data.fileUrl); setLineCount(data.lineCount); }} />
              {lineCount > 0 && <p className="text-sm text-muted-foreground">Строк: {lineCount}</p>}
              <p className="text-xs text-muted-foreground">
                Ожидаемые колонки: <code>thumbnail_url</code>, <code>title</code>, <code>tags</code>, <code>categories</code>, <code>video_url</code>
              </p>
              <div>
                <Label>Потоки: {maxWorkers}</Label>
                <Slider value={[maxWorkers]} onValueChange={(v) => setMaxWorkers(typeof v === "number" ? v : v[0])} min={1} max={10} step={1} />
              </div>
            </CardContent>
          </Card>
        </div>
        <div className="space-y-6">
          <Card><CardHeader><CardTitle>Модель (для шагов 2-3)</CardTitle></CardHeader><CardContent><ModelSelector value={model} onChange={setModel} /></CardContent></Card>
          <Card><CardHeader><CardTitle>Параметры генерации</CardTitle></CardHeader><CardContent><PresetSelector value={config} onChange={setConfig} /></CardContent></Card>
        </div>
      </div>

      <Button size="lg" onClick={startJob} disabled={submitting || !fileUrl}>
        <Play className="h-4 w-4 mr-2" />{submitting ? "Запуск..." : "Запустить AI Process 3.0"}
      </Button>
    </div>
  );
}
