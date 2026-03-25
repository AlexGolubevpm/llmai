"use client";

import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Input } from "@/components/ui/input";
import { Slider } from "@/components/ui/slider";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { FileUpload } from "@/components/file-upload";
import { ModelSelector } from "@/components/model-selector";
import { PresetSelector } from "@/components/preset-selector";
import { JobProgress } from "@/components/job-progress";
import type { JobConfig } from "@/types";
import { Play } from "lucide-react";
import { toast } from "sonner";

const LANGUAGES = ["English", "Chinese", "Japanese", "Hindi", "Spanish", "Russian", "German", "French"];

export default function TranslatePage() {
  const [fileUrl, setFileUrl] = useState("");
  const [model, setModel] = useState("meta-llama/llama-3.1-8b-instruct");
  const [config, setConfig] = useState<JobConfig>({
    systemPrompt: "You are a professional translator.",
    maxTokens: 512,
    temperature: 0.7,
    topP: 1.0, minP: 0.0, topK: 40,
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
    <div className="space-y-6">
      <h1 className="text-3xl font-bold">Перевод текста</h1>
      {activeJobId && <JobProgress jobId={activeJobId} onComplete={() => toast.success("Перевод завершён!")} />}

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="space-y-6">
          <Card>
            <CardHeader><CardTitle>Файл</CardTitle></CardHeader>
            <CardContent className="space-y-4">
              <FileUpload onUpload={(data) => setFileUrl(data.fileUrl)} />
              <div>
                <Label>Колонка для перевода</Label>
                <Input value={titleCol} onChange={(e) => setTitleCol(e.target.value)} />
              </div>
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <Label>Исходный язык</Label>
                  <Select value={sourceLanguage} onValueChange={(v) => v && setSourceLanguage(v)}>
                    <SelectTrigger><SelectValue /></SelectTrigger>
                    <SelectContent>{LANGUAGES.map((l) => <SelectItem key={l} value={l}>{l}</SelectItem>)}</SelectContent>
                  </Select>
                </div>
                <div>
                  <Label>Целевой язык</Label>
                  <Select value={targetLanguage} onValueChange={(v) => v && setTargetLanguage(v)}>
                    <SelectTrigger><SelectValue /></SelectTrigger>
                    <SelectContent>{LANGUAGES.map((l) => <SelectItem key={l} value={l}>{l}</SelectItem>)}</SelectContent>
                  </Select>
                </div>
              </div>
              <div>
                <Label>Потоки: {maxWorkers}</Label>
                <Slider value={[maxWorkers]} onValueChange={(v) => setMaxWorkers(typeof v === "number" ? v : v[0])} min={1} max={20} step={1} />
              </div>
            </CardContent>
          </Card>
        </div>

        <div className="space-y-6">
          <Card><CardHeader><CardTitle>Модель</CardTitle></CardHeader><CardContent><ModelSelector value={model} onChange={setModel} /></CardContent></Card>
          <Card><CardHeader><CardTitle>Параметры генерации</CardTitle></CardHeader><CardContent><PresetSelector value={config} onChange={setConfig} /></CardContent></Card>
        </div>
      </div>

      <Button size="lg" onClick={startJob} disabled={submitting || !fileUrl}>
        <Play className="h-4 w-4 mr-2" />{submitting ? "Создание..." : "Запустить перевод"}
      </Button>
    </div>
  );
}
