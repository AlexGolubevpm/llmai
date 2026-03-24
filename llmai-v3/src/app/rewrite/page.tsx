"use client";

import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Slider } from "@/components/ui/slider";
import { Switch } from "@/components/ui/switch";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { FileUpload } from "@/components/file-upload";
import { ModelSelector } from "@/components/model-selector";
import { PresetSelector } from "@/components/preset-selector";
import { JobProgress } from "@/components/job-progress";
import type { JobConfig } from "@/types";
import { Play } from "lucide-react";
import { toast } from "sonner";

export default function RewritePage() {
  const [fileUrl, setFileUrl] = useState("");
  const [lineCount, setLineCount] = useState(0);
  const [model, setModel] = useState("meta-llama/llama-3.1-8b-instruct");
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

  async function startJob() {
    if (!fileUrl) {
      toast.error("Загрузите файл");
      return;
    }
    setSubmitting(true);
    try {
      const resp = await fetch("/api/jobs", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          type: "REWRITE",
          inputFileUrl: fileUrl,
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
    <div className="space-y-6">
      <h1 className="text-3xl font-bold">Массовый рерайт</h1>

      {activeJobId && (
        <JobProgress
          jobId={activeJobId}
          onComplete={() => toast.success("Рерайт завершён! Скачайте результат на Dashboard.")}
        />
      )}

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="space-y-6">
          <Card>
            <CardHeader><CardTitle>Файл</CardTitle></CardHeader>
            <CardContent className="space-y-4">
              <FileUpload
                onUpload={(data) => {
                  setFileUrl(data.fileUrl);
                  setLineCount(data.lineCount);
                }}
              />
              {lineCount > 0 && <p className="text-sm text-muted-foreground">Строк: {lineCount}</p>}
              <div>
                <Label>Колонка для рерайта</Label>
                <Input value={titleCol} onChange={(e) => setTitleCol(e.target.value)} />
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader><CardTitle>Настройки рерайта</CardTitle></CardHeader>
            <CardContent className="space-y-4">
              <div>
                <Label>Промпт</Label>
                <Textarea
                  value={userPrompt}
                  onChange={(e) => setUserPrompt(e.target.value)}
                  placeholder="Введите промпт для рерайта..."
                  rows={3}
                />
              </div>
              <div>
                <Label>Множитель: x{multiplier}</Label>
                <Slider
                  value={[multiplier]}
                  onValueChange={(v) => setMultiplier(typeof v === "number" ? v : v[0])}
                  min={1}
                  max={10}
                  step={1}
                />
                <p className="text-xs text-muted-foreground mt-1">
                  Файл будет переписан {multiplier} раз{multiplier > 1 ? ". Каждый проход берёт результат предыдущего." : "."}
                </p>
              </div>
              <div>
                <Label>Потоки: {maxWorkers}</Label>
                <Slider
                  value={[maxWorkers]}
                  onValueChange={(v) => setMaxWorkers(typeof v === "number" ? v : v[0])}
                  min={1}
                  max={20}
                  step={1}
                />
              </div>
              <div className="flex items-center gap-2">
                <Switch checked={applyStopWords} onCheckedChange={setApplyStopWords} />
                <Label>Применять стоп-слова после каждого прохода</Label>
              </div>
            </CardContent>
          </Card>
        </div>

        <div className="space-y-6">
          <Card>
            <CardHeader><CardTitle>Модель</CardTitle></CardHeader>
            <CardContent>
              <ModelSelector value={model} onChange={setModel} />
            </CardContent>
          </Card>

          <Card>
            <CardHeader><CardTitle>Параметры генерации</CardTitle></CardHeader>
            <CardContent>
              <PresetSelector value={config} onChange={setConfig} />
            </CardContent>
          </Card>
        </div>
      </div>

      <Button size="lg" onClick={startJob} disabled={submitting || !fileUrl}>
        <Play className="h-4 w-4 mr-2" />
        {submitting ? "Создание задачи..." : `Запустить рерайт x${multiplier}`}
      </Button>
    </div>
  );
}
