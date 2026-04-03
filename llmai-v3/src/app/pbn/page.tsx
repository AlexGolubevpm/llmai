"use client";

import { useState } from "react";
import { motion } from "framer-motion";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Input } from "@/components/ui/input";
import { Slider } from "@/components/ui/slider";
import { Textarea } from "@/components/ui/textarea";
import { ModelSelector } from "@/components/model-selector";
import { JobProgress } from "@/components/job-progress";
import { PageHeader } from "@/components/layout/page-header";
import { Play, Plus, X, Download } from "lucide-react";
import { toast } from "sonner";
import { pageVariants } from "@/lib/animations";

const DEFAULT_PROMPT = `You are an expert SEO copywriter specializing in adult content link building for PBN (Private Blog Network) sites.

Write a unique, natural-sounding text block (150-250 words) that organically mentions and links to the following sites:
{sites}

Requirements:
1. Write as a genuine industry expert sharing recommendations — NOT as an ad or review
2. Each site must be mentioned by domain name naturally within the text
3. Describe what makes each site unique: content type, quality, niche focus, user experience
4. Use varied anchor text styles: exact domain, branded mentions, descriptive phrases
5. Include 2-3 relevant long-tail keywords naturally (e.g. "HD Japanese adult videos", "premium hentai archive")
6. Vary the text structure: some texts as paragraphs, some as expert opinions, some as comparisons
7. Tone: professional, knowledgeable, confident — like an industry insider
8. NO promotional language like "best", "top", "amazing", "must-visit"
9. NO lists with bullet points — write flowing prose
10. Each text must be COMPLETELY unique in structure, opening, and approach
11. Include semantic SEO signals: related terms, synonyms, natural language patterns
12. Write in English

Text #{number} of {total}:`;

export default function PBNPage() {
  const [sites, setSites] = useState<string[]>([""]);
  const [quantity, setQuantity] = useState(100);
  const [model, setModel] = useState("openai/gpt-4o-mini");
  const [prompt, setPrompt] = useState(DEFAULT_PROMPT);
  const [temperature, setTemperature] = useState(0.9);
  const [maxTokens, setMaxTokens] = useState(500);

  const [generating, setGenerating] = useState(false);
  const [activeJobId, setActiveJobId] = useState<string | null>(null);
  const [results, setResults] = useState<string[]>([]);
  const [progress, setProgress] = useState(0);

  function addSite() {
    setSites([...sites, ""]);
  }

  function removeSite(idx: number) {
    setSites(sites.filter((_, i) => i !== idx));
  }

  function updateSite(idx: number, val: string) {
    const updated = [...sites];
    updated[idx] = val;
    setSites(updated);
  }

  async function generate() {
    const validSites = sites.filter((s) => s.trim());
    if (validSites.length === 0) {
      toast.error("Добавьте хотя бы один сайт");
      return;
    }

    setGenerating(true);
    setResults([]);
    setProgress(0);

    const sitesStr = validSites.map((s, i) => `${i + 1}. ${s.trim()}`).join("\n");
    const generated: string[] = [];

    try {
      for (let i = 0; i < quantity; i++) {
        const textPrompt = prompt
          .replace("{sites}", sitesStr)
          .replace("{number}", String(i + 1))
          .replace("{total}", String(quantity));

        const resp = await fetch("/api/pbn/generate", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            model,
            prompt: textPrompt,
            temperature,
            maxTokens,
          }),
        });

        const data = await resp.json();
        if (!resp.ok) throw new Error(data.error);

        generated.push(data.text);
        setResults([...generated]);
        setProgress(Math.round(((i + 1) / quantity) * 100));
      }

      toast.success(`Сгенерировано ${generated.length} текстов`);
    } catch (err) {
      toast.error((err as Error).message);
    } finally {
      setGenerating(false);
    }
  }

  function downloadResults() {
    const text = results.map((r, i) => `=== Text #${i + 1} ===\n\n${r}\n`).join("\n\n");
    const blob = new Blob([text], { type: "text/plain" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `pbn-texts-${results.length}.txt`;
    a.click();
    URL.revokeObjectURL(url);
  }

  function downloadCSV() {
    const header = "id,text,sites\n";
    const sitesStr = sites.filter((s) => s.trim()).join("; ");
    const rows = results.map((r, i) => {
      const escaped = r.replace(/"/g, '""').replace(/\n/g, " ");
      return `${i + 1},"${escaped}","${sitesStr}"`;
    });
    const csv = header + rows.join("\n");
    const blob = new Blob([csv], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `pbn-texts-${results.length}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  }

  return (
    <motion.div {...pageVariants} className="space-y-8">
      <PageHeader
        title="PBN SEO тексты"
        description="Генерация уникальных SEO текстов для PBN сетей с органичными ссылками"
      />

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Left: Sites + Settings */}
        <div className="space-y-6">
          <div className="rounded-xl border bg-[var(--surface)] p-6 space-y-5">
            <h2 className="text-[15px] font-medium">Сайты для продвижения</h2>
            <div className="space-y-2">
              {sites.map((site, idx) => (
                <div key={idx} className="flex gap-2">
                  <Input
                    value={site}
                    onChange={(e) => updateSite(idx, e.target.value)}
                    placeholder="example.com"
                    className="flex-1 font-mono text-sm"
                  />
                  {sites.length > 1 && (
                    <Button variant="ghost" size="icon" className="h-9 w-9 shrink-0 text-[var(--error)]" onClick={() => removeSite(idx)} aria-label="Удалить">
                      <X className="h-3.5 w-3.5" />
                    </Button>
                  )}
                </div>
              ))}
            </div>
            <Button variant="outline" size="sm" onClick={addSite} className="gap-1.5">
              <Plus className="h-3.5 w-3.5" /> Добавить сайт
            </Button>
          </div>

          <div className="rounded-xl border bg-[var(--surface)] p-6 space-y-5">
            <h2 className="text-[15px] font-medium">Настройки</h2>
            <ModelSelector value={model} onChange={setModel} />

            <div>
              <div className="flex items-baseline justify-between mb-2">
                <Label className="text-xs font-mono text-[var(--text-muted)]">Количество текстов</Label>
                <span className="text-xs font-mono font-medium">{quantity}</span>
              </div>
              <Slider value={[quantity]} onValueChange={(v) => setQuantity(typeof v === "number" ? v : v[0])} min={1} max={500} step={1} />
            </div>

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
                <Slider value={[maxTokens]} onValueChange={(v) => setMaxTokens(typeof v === "number" ? v : v[0])} min={200} max={2000} step={50} />
              </div>
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
            rows={20}
            className="font-mono text-xs"
          />
          <div className="rounded-lg bg-[var(--surface-raised)] px-4 py-3 text-xs text-[var(--text-muted)] space-y-1">
            <p><strong>Переменные:</strong> <code>{"{sites}"}</code> — список сайтов, <code>{"{number}"}</code> — номер текста, <code>{"{total}"}</code> — всего</p>
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
          <span className="text-xs text-[var(--text-muted)]">{results.length} / {quantity} текстов</span>
        </div>
      )}

      {/* Actions */}
      <div className="flex gap-3">
        <Button
          size="lg"
          onClick={generate}
          disabled={generating || sites.every((s) => !s.trim())}
          className="flex-1 h-12 text-sm font-medium gap-2"
        >
          <Play className="h-4 w-4" />
          {generating ? `Генерация ${results.length}/${quantity}...` : `Сгенерировать ${quantity} текстов`}
        </Button>

        {results.length > 0 && (
          <>
            <Button variant="outline" size="lg" onClick={downloadResults} className="h-12 gap-2">
              <Download className="h-4 w-4" /> TXT
            </Button>
            <Button variant="outline" size="lg" onClick={downloadCSV} className="h-12 gap-2">
              <Download className="h-4 w-4" /> CSV
            </Button>
          </>
        )}
      </div>

      {/* Preview results */}
      {results.length > 0 && (
        <div className="rounded-xl border bg-[var(--surface)] p-6 space-y-4">
          <h2 className="text-[15px] font-medium">Результаты ({results.length} текстов)</h2>
          <div className="space-y-4 max-h-[600px] overflow-auto">
            {results.slice(0, 10).map((text, idx) => (
              <div key={idx} className="rounded-lg border bg-[var(--surface-raised)] p-4">
                <div className="text-[10px] font-medium text-[var(--text-muted)] mb-2">#{idx + 1}</div>
                <p className="text-sm leading-relaxed whitespace-pre-wrap">{text}</p>
              </div>
            ))}
            {results.length > 10 && (
              <p className="text-xs text-[var(--text-muted)] text-center">
                Показано 10 из {results.length}. Скачайте TXT или CSV для полного списка.
              </p>
            )}
          </div>
        </div>
      )}
    </motion.div>
  );
}
