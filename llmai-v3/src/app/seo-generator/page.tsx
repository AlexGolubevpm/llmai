"use client";

import { useState, useCallback } from "react";
import { motion } from "framer-motion";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Input } from "@/components/ui/input";
import { Slider } from "@/components/ui/slider";
import { Textarea } from "@/components/ui/textarea";
import { Switch } from "@/components/ui/switch";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { ModelSelector } from "@/components/model-selector";
import { FileUpload } from "@/components/file-upload";
import { PageHeader } from "@/components/layout/page-header";
import { EmptyState } from "@/components/shared/empty-state";
import { Play, Download, FileText, Type, PenLine } from "lucide-react";
import { toast } from "sonner";
import { pageVariants, staggerItem } from "@/lib/animations";
import Papa from "papaparse";

const DEFAULT_PROMPT = `Generate an SEO-optimized title and meta description for an adult tube video page.

Context:
Title: {title}
Tags: {tags}
Categories: {categories}

Requirements for SEO title (field "title"):
- 60-80 characters max
- Start with action verb or power word
- Include 2-3 high-volume search keywords from the tags
- Use specific niche terms, not generic
- Must read naturally in English
- No ALL CAPS, no emojis, no special characters

Requirements for meta description (field "description"):
- 120-155 characters
- Complement the title — don't repeat it
- Include secondary keywords and long-tail phrases
- Write as compelling preview that drives CTR
- Natural English, no keyword stuffing

Return ONLY valid JSON:
{"title":"...","description":"..."}`;

interface DataRow {
  title: string;
  tags: string;
  categories: string;
}

interface ResultRow extends DataRow {
  seo_title: string;
  seo_description: string;
}

export default function SEOGeneratorPage() {
  const [inputMode, setInputMode] = useState<"file" | "manual">("manual");

  // Manual input
  const [manualRows, setManualRows] = useState<DataRow[]>([{ title: "", tags: "", categories: "" }]);

  // File input
  const [fileRows, setFileRows] = useState<DataRow[]>([]);
  const [fileUrl, setFileUrl] = useState("");

  // Settings
  const [model, setModel] = useState("openai/gpt-4o-mini");
  const [prompt, setPrompt] = useState(DEFAULT_PROMPT);
  const [temperature, setTemperature] = useState(0.7);
  const [maxTokens, setMaxTokens] = useState(300);
  const [applyStopWords, setApplyStopWords] = useState(true);

  // Results
  const [results, setResults] = useState<ResultRow[]>([]);
  const [generating, setGenerating] = useState(false);
  const [progress, setProgress] = useState(0);

  const dataRows = inputMode === "file" ? fileRows : manualRows;

  function addManualRow() {
    setManualRows([...manualRows, { title: "", tags: "", categories: "" }]);
  }

  function updateManualRow(idx: number, field: keyof DataRow, val: string) {
    const updated = [...manualRows];
    updated[idx] = { ...updated[idx], [field]: val };
    setManualRows(updated);
  }

  function removeManualRow(idx: number) {
    if (manualRows.length <= 1) return;
    setManualRows(manualRows.filter((_, i) => i !== idx));
  }

  const handleFileUpload = useCallback((data: { fileUrl: string }) => {
    setFileUrl(data.fileUrl);
    // Read and parse the file via fetch
    fetch(`/api/files/preview?path=${encodeURIComponent(data.fileUrl)}`)
      .catch(() => {});

    // Parse locally from uploaded file content — we'll use the file URL for the job
    // For now just mark that file is ready
    toast.success("Файл загружен. Используйте CSV с колонками: title, tags, categories");
  }, []);

  async function generate() {
    const rows = dataRows.filter((r) => r.title.trim() || r.tags.trim());
    if (rows.length === 0) {
      toast.error("Нет данных для обработки");
      return;
    }

    setGenerating(true);
    setResults([]);
    setProgress(0);

    // Process in batches of 10
    const batchSize = 10;
    const allResults: ResultRow[] = [];

    try {
      for (let i = 0; i < rows.length; i += batchSize) {
        const batch = rows.slice(i, Math.min(i + batchSize, rows.length));

        const resp = await fetch("/api/seo-generator", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            model,
            prompt,
            temperature,
            maxTokens,
            rows: batch,
            applyStopWords,
          }),
        });

        const data = await resp.json();
        if (!resp.ok) throw new Error(data.error);

        allResults.push(...data.results);
        setResults([...allResults]);
        setProgress(Math.round((Math.min(i + batchSize, rows.length) / rows.length) * 100));

        if (data.errors?.length > 0) {
          console.warn("SEO generator errors:", data.errors);
        }
      }

      toast.success(`Сгенерировано ${allResults.length} SEO текстов`);
    } catch (err) {
      toast.error((err as Error).message);
    } finally {
      setGenerating(false);
    }
  }

  function downloadCSV() {
    const csv = Papa.unparse(results);
    const blob = new Blob([csv], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `seo-titles-${results.length}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  }

  function pasteFromClipboard() {
    navigator.clipboard.readText().then((text) => {
      const lines = text.split("\n").filter((l) => l.trim());
      const parsed: DataRow[] = lines.map((line) => {
        const parts = line.split("|").map((p) => p.trim());
        // Try to detect: title|tags|categories or just title
        if (parts.length >= 3) {
          return { title: parts[0], tags: parts[1], categories: parts[2] };
        } else if (parts.length === 2) {
          return { title: parts[0], tags: parts[1], categories: "" };
        }
        return { title: parts[0] || line.trim(), tags: "", categories: "" };
      });
      setManualRows(parsed);
      toast.success(`Вставлено ${parsed.length} строк`);
    }).catch(() => toast.error("Не удалось прочитать буфер обмена"));
  }

  return (
    <motion.div {...pageVariants} className="space-y-8">
      <PageHeader
        title="SEO генератор"
        description="Загрузите теги и категории — AI напишет SEO title и description для каждой строки"
      />

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Left: Data input */}
        <div className="space-y-6">
          <div className="rounded-xl border bg-[var(--surface)] p-6 space-y-5">
            <h2 className="text-[15px] font-medium">Данные</h2>

            <Tabs value={inputMode} onValueChange={(v) => v && setInputMode(v as "file" | "manual")}>
              <TabsList>
                <TabsTrigger value="manual" className="gap-1.5"><Type className="h-3.5 w-3.5" /> Ручной ввод</TabsTrigger>
                <TabsTrigger value="file" className="gap-1.5"><FileText className="h-3.5 w-3.5" /> Файл</TabsTrigger>
              </TabsList>

              <TabsContent value="manual" className="mt-4 space-y-3">
                <div className="flex gap-2 mb-2">
                  <Button variant="outline" size="sm" onClick={addManualRow} className="gap-1.5 text-xs">
                    + Строка
                  </Button>
                  <Button variant="outline" size="sm" onClick={pasteFromClipboard} className="gap-1.5 text-xs">
                    Вставить из буфера
                  </Button>
                </div>

                <div className="max-h-[400px] overflow-auto space-y-2">
                  {manualRows.map((row, idx) => (
                    <div key={idx} className="grid grid-cols-[1fr_1fr_auto] gap-2 items-start">
                      <div>
                        {idx === 0 && <Label className="text-[10px] text-[var(--text-muted)]">Title / Tags</Label>}
                        <Input
                          value={row.title}
                          onChange={(e) => updateManualRow(idx, "title", e.target.value)}
                          placeholder="Title"
                          className="text-xs"
                        />
                      </div>
                      <div>
                        {idx === 0 && <Label className="text-[10px] text-[var(--text-muted)]">Tags / Categories</Label>}
                        <Input
                          value={row.tags}
                          onChange={(e) => updateManualRow(idx, "tags", e.target.value)}
                          placeholder="tag1, tag2, tag3"
                          className="text-xs"
                        />
                      </div>
                      <div className="pt-1">
                        {idx === 0 && <div className="h-3" />}
                        <Button variant="ghost" size="icon" className="h-8 w-8 text-[var(--text-muted)]" onClick={() => removeManualRow(idx)}>
                          <span className="text-xs">×</span>
                        </Button>
                      </div>
                    </div>
                  ))}
                </div>
                <p className="text-xs text-[var(--text-muted)]">{manualRows.filter((r) => r.title.trim()).length} строк</p>
                <p className="text-xs text-[var(--text-muted)]">Формат вставки: <code>title|tags|categories</code> (по строкам)</p>
              </TabsContent>

              <TabsContent value="file" className="mt-4 space-y-4">
                <FileUpload onUpload={handleFileUpload} />
                <p className="text-xs text-[var(--text-muted)]">CSV с колонками: title, tags, categories</p>
              </TabsContent>
            </Tabs>
          </div>

          <div className="rounded-xl border bg-[var(--surface)] p-6 space-y-5">
            <h2 className="text-[15px] font-medium">Настройки</h2>
            <ModelSelector value={model} onChange={setModel} />
            <div className="grid grid-cols-2 gap-4">
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
                <Slider value={[maxTokens]} onValueChange={(v) => setMaxTokens(typeof v === "number" ? v : v[0])} min={100} max={1000} step={50} />
              </div>
            </div>
            <div className="flex items-center justify-between rounded-lg bg-[var(--surface-raised)] px-4 py-3">
              <Label className="text-sm">Применять стоп-слова</Label>
              <Switch checked={applyStopWords} onCheckedChange={setApplyStopWords} />
            </div>
          </div>
        </div>

        {/* Right: Prompt */}
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
            rows={22}
            className="font-mono text-xs"
          />
          <div className="rounded-lg bg-[var(--surface-raised)] px-4 py-3 text-xs text-[var(--text-muted)] space-y-1">
            <p><strong>Переменные:</strong> <code>{"{title}"}</code>, <code>{"{tags}"}</code>, <code>{"{categories}"}</code></p>
            <p><strong>JSON выход:</strong> <code>{`{"title":"...","description":"..."}`}</code></p>
          </div>
        </div>
      </div>

      {/* Progress */}
      {generating && (
        <div className="rounded-xl border bg-[var(--surface)] p-5 space-y-3">
          <div className="flex items-center justify-between">
            <span className="text-sm font-medium">Генерация SEO текстов...</span>
            <span className="text-2xl font-semibold font-mono">{progress}%</span>
          </div>
          <div className="h-2 rounded-full bg-[var(--surface-raised)] overflow-hidden">
            <div className="h-full bg-[var(--accent-blue)] transition-all duration-300 rounded-full" style={{ width: `${progress}%` }} />
          </div>
          <span className="text-xs text-[var(--text-muted)]">{results.length} / {dataRows.filter((r) => r.title.trim() || r.tags.trim()).length}</span>
        </div>
      )}

      {/* Actions */}
      <div className="flex gap-3">
        <Button
          size="lg"
          onClick={generate}
          disabled={generating || dataRows.every((r) => !r.title.trim() && !r.tags.trim())}
          className="flex-1 h-12 text-sm font-medium gap-2"
        >
          <Play className="h-4 w-4" />
          {generating ? `Генерация ${results.length}...` : `Сгенерировать SEO для ${dataRows.filter((r) => r.title.trim() || r.tags.trim()).length} строк`}
        </Button>
        {results.length > 0 && (
          <Button variant="outline" size="lg" onClick={downloadCSV} className="h-12 gap-2">
            <Download className="h-4 w-4" /> CSV
          </Button>
        )}
      </div>

      {/* Results */}
      {results.length > 0 && (
        <div className="rounded-xl border bg-[var(--surface)] overflow-hidden">
          <div className="border-b px-6 py-4">
            <h2 className="text-[15px] font-medium">Результаты ({results.length})</h2>
          </div>
          <div className="max-h-[500px] overflow-auto">
            <Table>
              <TableHeader>
                <TableRow className="hover:bg-transparent">
                  <TableHead className="text-xs font-medium uppercase tracking-wider text-[var(--text-muted)]">Original</TableHead>
                  <TableHead className="text-xs font-medium uppercase tracking-wider text-[var(--text-muted)]">SEO Title</TableHead>
                  <TableHead className="text-xs font-medium uppercase tracking-wider text-[var(--text-muted)]">SEO Description</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {results.map((r, i) => (
                  <motion.tr key={i} variants={staggerItem} initial="initial" animate="animate" transition={{ delay: i * 0.02 }} className="border-b last:border-0 hover:bg-[var(--surface-raised)]">
                    <TableCell className="text-xs max-w-[200px] truncate" title={r.title}>{r.title}</TableCell>
                    <TableCell className="text-xs font-medium text-[var(--accent-blue)] max-w-[250px]">{r.seo_title}</TableCell>
                    <TableCell className="text-xs text-[var(--text-muted)] max-w-[300px]">{r.seo_description}</TableCell>
                  </motion.tr>
                ))}
              </TableBody>
            </Table>
          </div>
        </div>
      )}
    </motion.div>
  );
}
