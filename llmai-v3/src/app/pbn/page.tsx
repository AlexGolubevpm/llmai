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
import { Play, Plus, X } from "lucide-react";
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
5. Include 2-3 relevant long-tail keywords naturally
6. Vary the text structure: paragraphs, expert opinions, comparisons
7. Tone: professional, knowledgeable, confident — like an industry insider
8. NO promotional language like "best", "top", "amazing", "must-visit"
9. NO lists with bullet points — write flowing prose
10. Each text must be COMPLETELY unique in structure, opening, and approach

Text #{number} of {total}:`;

export default function PBNPage() {
  const [sites, setSites] = useState<string[]>([""]);
  const [quantity, setQuantity] = useState(100);
  const [model, setModel] = useState("openai/gpt-4o-mini");
  const [prompt, setPrompt] = useState(DEFAULT_PROMPT);
  const [temperature, setTemperature] = useState(0.9);
  const [maxTokens, setMaxTokens] = useState(500);
  const [activeJobId, setActiveJobId] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);

  function addSite() { setSites([...sites, ""]); }
  function removeSite(idx: number) { setSites(sites.filter((_, i) => i !== idx)); }
  function updateSite(idx: number, val: string) {
    const updated = [...sites]; updated[idx] = val; setSites(updated);
  }

  async function startJob() {
    const validSites = sites.filter((s) => s.trim());
    if (validSites.length === 0) { toast.error("Добавьте хотя бы один сайт"); return; }
    setSubmitting(true);
    try {
      // Create placeholder file
      const fileResp = await fetch("/api/files/from-text", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: validSites.join("\n"), mode: "rewrite" }),
      });
      const fileData = await fileResp.json();
      if (!fileResp.ok) throw new Error(fileData.error);

      const jobResp = await fetch("/api/jobs", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          type: "PBN",
          inputFileUrl: fileData.fileUrl,
          config: {
            model,
            customPrompt: prompt,
            temperature,
            maxTokens,
            sites: validSites,
            quantity,
          },
        }),
      });
      const jobData = await jobResp.json();
      if (!jobResp.ok) throw new Error(jobData.error);
      setActiveJobId(jobData.job.id);
      toast.success("PBN генерация запущена — результат на Dashboard");
    } catch (err) { toast.error((err as Error).message); }
    finally { setSubmitting(false); }
  }

  return (
    <motion.div {...pageVariants} className="space-y-8">
      <PageHeader
        title="PBN SEO тексты"
        description="Генерация уникальных SEO текстов для PBN. Результат сохраняется и доступен на Dashboard."
      />

      {activeJobId && <JobProgress jobId={activeJobId} onComplete={() => toast.success("PBN тексты готовы! Скачайте на Dashboard.")} />}

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="space-y-6">
          <div className="rounded-xl border bg-[var(--surface)] p-6 space-y-5">
            <h2 className="text-[15px] font-medium">Сайты для продвижения</h2>
            <div className="space-y-2">
              {sites.map((site, idx) => (
                <div key={idx} className="flex gap-2">
                  <Input value={site} onChange={(e) => updateSite(idx, e.target.value)} placeholder="example.com" className="flex-1 font-mono text-sm" />
                  {sites.length > 1 && (
                    <Button variant="ghost" size="icon" className="h-9 w-9 shrink-0 text-[var(--error)]" onClick={() => removeSite(idx)}>
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

        <div className="rounded-xl border bg-[var(--surface)] p-6 space-y-4">
          <div className="flex items-center justify-between">
            <h2 className="text-[15px] font-medium">Промпт</h2>
            <button onClick={() => setPrompt(DEFAULT_PROMPT)} className="text-xs text-[var(--accent-blue)] hover:underline">Сбросить</button>
          </div>
          <Textarea value={prompt} onChange={(e) => setPrompt(e.target.value)} rows={20} className="font-mono text-xs" />
          <div className="rounded-lg bg-[var(--surface-raised)] px-4 py-3 text-xs text-[var(--text-muted)]">
            <p><strong>Переменные:</strong> <code>{"{sites}"}</code>, <code>{"{number}"}</code>, <code>{"{total}"}</code></p>
            <p><strong>CSV выход:</strong> id, text, sites</p>
          </div>
        </div>
      </div>

      <Button size="lg" onClick={startJob} disabled={submitting || sites.every((s) => !s.trim())} className="w-full h-12 text-sm font-medium gap-2">
        <Play className="h-4 w-4" />
        {submitting ? "Создание задачи..." : `Сгенерировать ${quantity} текстов`}
      </Button>
    </motion.div>
  );
}
