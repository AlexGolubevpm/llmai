"use client";

import { useState } from "react";
import { motion } from "framer-motion";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Slider } from "@/components/ui/slider";
import { Textarea } from "@/components/ui/textarea";
import { Switch } from "@/components/ui/switch";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { ModelSelector } from "@/components/model-selector";
import { PageHeader } from "@/components/layout/page-header";
import { Play, Download } from "lucide-react";
import { toast } from "sonner";
import { pageVariants, staggerItem } from "@/lib/animations";
import Papa from "papaparse";

const DEFAULT_PROMPT = `You are an SEO expert for adult tube sites. Generate a unique SEO title and meta description for a specific tag/category page.

The tag/category name is: "{name}"

This is a page that lists all videos tagged with "{name}" on an adult tube site.

Requirements for title:
- 50-65 characters
- MUST include the exact tag/category name "{name}"
- Format examples: "Best {name} Porn Videos - Free HD Sex Tubes", "{name} - Watch Free XXX Videos Online"
- Include power words: Free, HD, Best, Watch, Hot
- Must be unique and specific to this tag

Requirements for description:
- 120-155 characters
- MUST mention "{name}" at least once
- Describe what visitors will find on this specific tag page
- Include call-to-action: watch, explore, browse, discover
- Include secondary keywords related to "{name}"

Return ONLY valid JSON:
{"title":"...","description":"..."}`;

interface ResultRow {
  name: string;
  seo_title: string;
  seo_description: string;
}

export default function SEOCategoriesPage() {
  const [namesText, setNamesText] = useState("");
  const [model, setModel] = useState("openai/gpt-4o-mini");
  const [prompt, setPrompt] = useState(DEFAULT_PROMPT);
  const [temperature, setTemperature] = useState(0.7);
  const [maxTokens, setMaxTokens] = useState(300);
  const [applyStopWords, setApplyStopWords] = useState(true);

  const [results, setResults] = useState<ResultRow[]>([]);
  const [generating, setGenerating] = useState(false);
  const [progress, setProgress] = useState(0);

  const names = namesText.split("\n").map((l) => l.trim()).filter(Boolean);

  async function generate() {
    if (names.length === 0) {
      toast.error("Введите теги или категории (по одному на строку)");
      return;
    }

    setGenerating(true);
    setResults([]);
    setProgress(0);

    const allResults: ResultRow[] = [];

    try {
      for (let i = 0; i < names.length; i++) {
        const name = names[i];
        const rowPrompt = prompt.replace(/\{name\}/g, name);

        const resp = await fetch("/api/pbn/generate", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            model,
            prompt: rowPrompt,
            temperature,
            maxTokens,
          }),
        });

        const data = await resp.json();
        if (!resp.ok) throw new Error(data.error);

        // Parse JSON response
        let parsed: Record<string, string> = {};
        try {
          const cleaned = data.text.replace(/```json\s*/gi, "").replace(/```\s*/g, "").trim();
          parsed = JSON.parse(cleaned);
        } catch {
          const match = data.text.match(/\{[\s\S]*\}/);
          if (match) {
            try { parsed = JSON.parse(match[0]); } catch {}
          }
        }

        allResults.push({
          name,
          seo_title: (parsed.title || parsed.seo_title || "").slice(0, 90),
          seo_description: (parsed.description || parsed.seo_description || "").slice(0, 160),
        });

        setResults([...allResults]);
        setProgress(Math.round(((i + 1) / names.length) * 100));
      }

      toast.success(`Сгенерировано ${allResults.length} SEO текстов`);
    } catch (err) {
      toast.error((err as Error).message);
    } finally {
      setGenerating(false);
    }
  }

  function downloadCSV() {
    const csv = Papa.unparse(results.map((r) => ({
      tag_category: r.name,
      seo_title: r.seo_title,
      seo_description: r.seo_description,
    })));
    const blob = new Blob([csv], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `seo-categories-${results.length}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  }

  return (
    <motion.div {...pageVariants} className="space-y-8">
      <PageHeader
        title="SEO Categories Generator"
        description="Введите теги и категории — AI напишет SEO title и description для каждой страницы тега/категории"
      />

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Left: Input + Settings */}
        <div className="space-y-6">
          <div className="rounded-xl border bg-[var(--surface)] p-6 space-y-5">
            <h2 className="text-[15px] font-medium">Теги и категории</h2>
            <div>
              <Label className="text-xs font-medium text-[var(--text-muted)] uppercase tracking-wider">
                Список (по одному на строку)
              </Label>
              <Textarea
                value={namesText}
                onChange={(e) => setNamesText(e.target.value)}
                rows={12}
                placeholder={"Anal\nBlowjob\nHentai\nMILF\nTeen 18+\nTransgender\nAsian\nAmateur\nPOV\nCreampie"}
                className="mt-1.5 font-mono text-sm"
              />
              <p className="text-xs text-[var(--text-muted)] mt-1.5">
                {names.length} тегов/категорий
              </p>
            </div>
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
              <Label className="text-sm">Стоп-слова</Label>
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
            <p><strong>Переменная:</strong> <code>{"{name}"}</code> — подставляется название тега/категории</p>
            <p><strong>JSON:</strong> <code>{`{"title":"...","description":"..."}`}</code></p>
          </div>
        </div>
      </div>

      {/* Progress */}
      {generating && (
        <div className="rounded-xl border bg-[var(--surface)] p-5 space-y-3">
          <div className="flex items-center justify-between">
            <span className="text-sm font-medium">Генерация...</span>
            <span className="text-2xl font-semibold font-mono">{progress}%</span>
          </div>
          <div className="h-2 rounded-full bg-[var(--surface-raised)] overflow-hidden">
            <div className="h-full bg-[var(--accent-blue)] transition-all duration-300 rounded-full" style={{ width: `${progress}%` }} />
          </div>
          <span className="text-xs text-[var(--text-muted)]">{results.length} / {names.length}</span>
        </div>
      )}

      {/* Actions */}
      <div className="flex gap-3">
        <Button
          size="lg"
          onClick={generate}
          disabled={generating || names.length === 0}
          className="flex-1 h-12 text-sm font-medium gap-2"
        >
          <Play className="h-4 w-4" />
          {generating ? `${results.length}/${names.length}...` : `Сгенерировать SEO для ${names.length} тегов`}
        </Button>
        {results.length > 0 && (
          <Button variant="outline" size="lg" onClick={downloadCSV} className="h-12 gap-2">
            <Download className="h-4 w-4" /> CSV
          </Button>
        )}
      </div>

      {/* Results table */}
      {results.length > 0 && (
        <div className="rounded-xl border bg-[var(--surface)] overflow-hidden">
          <div className="border-b px-6 py-4">
            <h2 className="text-[15px] font-medium">Результаты ({results.length})</h2>
          </div>
          <div className="max-h-[600px] overflow-auto">
            <Table>
              <TableHeader>
                <TableRow className="hover:bg-transparent">
                  <TableHead className="text-xs font-medium uppercase tracking-wider text-[var(--text-muted)] w-[120px]">Тег / Категория</TableHead>
                  <TableHead className="text-xs font-medium uppercase tracking-wider text-[var(--text-muted)]">SEO Title</TableHead>
                  <TableHead className="text-xs font-medium uppercase tracking-wider text-[var(--text-muted)]">SEO Description</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {results.map((r, i) => (
                  <motion.tr key={i} variants={staggerItem} initial="initial" animate="animate" transition={{ delay: i * 0.02 }} className="border-b last:border-0 hover:bg-[var(--surface-raised)]">
                    <TableCell className="font-mono text-xs font-medium">{r.name}</TableCell>
                    <TableCell className="text-xs text-[var(--accent-blue)]">{r.seo_title}</TableCell>
                    <TableCell className="text-xs text-[var(--text-muted)]">{r.seo_description}</TableCell>
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
