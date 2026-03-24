"use client";

import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { FileUpload } from "@/components/file-upload";
import { JobProgress } from "@/components/job-progress";
import { Play } from "lucide-react";
import { toast } from "sonner";

export default function PostprocessPage() {
  const [fileUrl, setFileUrl] = useState("");
  const [titleCol, setTitleCol] = useState("title");
  const [patternsText, setPatternsText] = useState("");
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
          config: { titleCol, harmfulPatterns, applyStopWords: true },
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
    <div className="space-y-6">
      <h1 className="text-3xl font-bold">Постобработка</h1>
      {activeJobId && <JobProgress jobId={activeJobId} onComplete={() => toast.success("Постобработка завершена!")} />}

      <Card>
        <CardHeader><CardTitle>Файл и настройки</CardTitle></CardHeader>
        <CardContent className="space-y-4">
          <FileUpload onUpload={(data) => setFileUrl(data.fileUrl)} />
          <div>
            <Label>Колонка для очистки</Label>
            <Input value={titleCol} onChange={(e) => setTitleCol(e.target.value)} />
          </div>
          <div>
            <Label>Вредные паттерны (по одному на строку)</Label>
            <Textarea
              value={patternsText}
              onChange={(e) => setPatternsText(e.target.value)}
              rows={5}
              placeholder="Каждая строка — паттерн для удаления из текста"
            />
          </div>
          <p className="text-sm text-muted-foreground">
            Стоп-слова из БД будут применены автоматически. Управлять ими можно на странице &quot;Стоп-слова&quot;.
          </p>
        </CardContent>
      </Card>

      <Button size="lg" onClick={startJob} disabled={submitting || !fileUrl}>
        <Play className="h-4 w-4 mr-2" />{submitting ? "Запуск..." : "Запустить постобработку"}
      </Button>
    </div>
  );
}
