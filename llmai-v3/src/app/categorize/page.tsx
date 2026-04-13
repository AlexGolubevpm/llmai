"use client";

import { useState, useEffect } from "react";
import { motion } from "framer-motion";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Slider } from "@/components/ui/slider";
import { FileUpload } from "@/components/file-upload";
import { ModelSelector } from "@/components/model-selector";
import { JobProgress } from "@/components/job-progress";
import { PageHeader } from "@/components/layout/page-header";
import { Play, Package } from "lucide-react";
import { toast } from "sonner";
import { pageVariants } from "@/lib/animations";
import { cn } from "@/lib/utils";

interface Bundle {
  id: string;
  name: string;
  tags: string;
  categories: string;
  isDefault: boolean;
}

export default function CategorizePage() {
  const [bundles, setBundles] = useState<Bundle[]>([]);
  const [selectedBundle, setSelectedBundle] = useState<Bundle | null>(null);
  const [fileUrl, setFileUrl] = useState("");
  const [lineCount, setLineCount] = useState(0);
  const [model, setModel] = useState("openai/gpt-4o-mini");
  const [numCategories, setNumCategories] = useState(3);
  const [numTags, setNumTags] = useState(8);
  const [maxWorkers, setMaxWorkers] = useState(5);
  const [activeJobId, setActiveJobId] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);

  useEffect(() => {
    fetch("/api/bundles")
      .then((r) => r.json())
      .then((data) => {
        const list = data.bundles || [];
        setBundles(list);
        const def = list.find((b: Bundle) => b.isDefault);
        if (def) setSelectedBundle(def);
      })
      .catch(() => {});
  }, []);

  async function startJob() {
    if (!selectedBundle) { toast.error("Выберите бандл"); return; }
    if (!fileUrl) { toast.error("Загрузите фид"); return; }
    setSubmitting(true);
    try {
      const resp = await fetch("/api/jobs", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          type: "CATEGORIZE",
          inputFileUrl: fileUrl,
          config: {
            model,
            temperature: 0.3,
            maxTokens: 200,
            maxWorkers,
            bundleName: selectedBundle.name,
            bundleTags: selectedBundle.tags,
            bundleCategories: selectedBundle.categories,
            numCategories,
            numTags,
          },
        }),
      });
      const data = await resp.json();
      if (!resp.ok) throw new Error(data.error);
      setActiveJobId(data.job.id);
      toast.success("Категоризация запущена — результат на Dashboard");
    } catch (err) { toast.error((err as Error).message); }
    finally { setSubmitting(false); }
  }

  const tagCount = selectedBundle?.tags.split(",").filter(Boolean).length || 0;
  const catCount = selectedBundle?.categories.split(",").filter(Boolean).length || 0;

  return (
    <motion.div {...pageVariants} className="space-y-8">
      <PageHeader
        title="Категоризация"
        description="Загрузите фид → выберите бандл → AI проставит категории и теги из списка бандла к каждому видео"
      />

      {activeJobId && <JobProgress jobId={activeJobId} onComplete={() => toast.success("Категоризация завершена! Скачайте на Dashboard.")} />}

      {/* Bundle selector */}
      <div className="rounded-xl border bg-[var(--surface)] p-6 space-y-4">
        <h2 className="text-[15px] font-medium">Бандл (набор тегов и категорий)</h2>
        {bundles.length === 0 ? (
          <p className="text-sm text-[var(--text-muted)]">Нет бандлов. <a href="/bundles" className="text-[var(--accent-blue)] hover:underline">Создайте бандл</a> с тегами и категориями.</p>
        ) : (
          <div className="flex flex-wrap gap-2">
            {bundles.map((b) => (
              <button
                key={b.id}
                onClick={() => setSelectedBundle(b)}
                className={cn(
                  "rounded-lg border px-4 py-2.5 text-sm font-medium transition-all",
                  selectedBundle?.id === b.id
                    ? "border-[var(--accent-blue)] bg-[var(--accent-blue-light)] text-[var(--accent-blue)]"
                    : "border-[var(--border)] hover:border-[var(--border-hover)] text-[var(--text-secondary)]"
                )}
              >
                <div className="flex items-center gap-2">
                  <Package className="h-3.5 w-3.5" />
                  {b.name}
                </div>
                <div className="text-[10px] text-[var(--text-muted)] mt-0.5">
                  {b.tags.split(",").filter(Boolean).length} тегов, {b.categories.split(",").filter(Boolean).length} кат.
                </div>
              </button>
            ))}
          </div>
        )}

        {selectedBundle && (
          <div className="rounded-lg bg-[var(--surface-raised)] px-4 py-3 text-xs space-y-1">
            <p><strong>Категории ({catCount}):</strong> {selectedBundle.categories.split(",").slice(0, 8).map((c) => c.trim()).join(", ")}{catCount > 8 ? ` и ещё ${catCount - 8}...` : ""}</p>
            <p><strong>Теги ({tagCount}):</strong> {selectedBundle.tags.split(",").slice(0, 12).map((t) => t.trim()).join(", ")}{tagCount > 12 ? ` и ещё ${tagCount - 12}...` : ""}</p>
          </div>
        )}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Left: File + Settings */}
        <div className="space-y-6">
          <div className="rounded-xl border bg-[var(--surface)] p-6 space-y-5">
            <h2 className="text-[15px] font-medium">Фид</h2>
            <FileUpload onUpload={(data) => { setFileUrl(data.fileUrl); setLineCount(data.lineCount); }} />
            {lineCount > 0 && <p className="text-xs text-[var(--text-muted)]">{lineCount.toLocaleString()} строк</p>}
            <p className="text-xs text-[var(--text-muted)]">CSV/TXT с колонками: id, title, tags, categories</p>
          </div>

          <div className="rounded-xl border bg-[var(--surface)] p-6 space-y-5">
            <h2 className="text-[15px] font-medium">Настройки</h2>
            <ModelSelector value={model} onChange={setModel} />
            <div className="grid grid-cols-3 gap-4">
              <div>
                <div className="flex items-baseline justify-between mb-2">
                  <Label className="text-xs font-mono text-[var(--text-muted)]">категорий</Label>
                  <span className="text-xs font-mono font-medium">{numCategories}</span>
                </div>
                <Slider value={[numCategories]} onValueChange={(v) => setNumCategories(typeof v === "number" ? v : v[0])} min={1} max={10} step={1} />
              </div>
              <div>
                <div className="flex items-baseline justify-between mb-2">
                  <Label className="text-xs font-mono text-[var(--text-muted)]">тегов</Label>
                  <span className="text-xs font-mono font-medium">{numTags}</span>
                </div>
                <Slider value={[numTags]} onValueChange={(v) => setNumTags(typeof v === "number" ? v : v[0])} min={1} max={20} step={1} />
              </div>
              <div>
                <div className="flex items-baseline justify-between mb-2">
                  <Label className="text-xs font-mono text-[var(--text-muted)]">потоки</Label>
                  <span className="text-xs font-mono font-medium">{maxWorkers}</span>
                </div>
                <Slider value={[maxWorkers]} onValueChange={(v) => setMaxWorkers(typeof v === "number" ? v : v[0])} min={1} max={10} step={1} />
              </div>
            </div>
          </div>
        </div>

        {/* Right: How it works */}
        <div className="rounded-xl border bg-[var(--surface)] p-6 space-y-4">
          <h2 className="text-[15px] font-medium">Как работает</h2>
          <div className="space-y-3 text-sm text-[var(--text-secondary)]">
            <div className="flex gap-3">
              <span className="flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-blue-50 text-[10px] font-bold text-blue-600">1</span>
              <p>Загружаете фид с video ID, title, текущими tags и categories</p>
            </div>
            <div className="flex gap-3">
              <span className="flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-blue-50 text-[10px] font-bold text-blue-600">2</span>
              <p>Выбираете бандл — набор разрешённых тегов ({tagCount || "N"} шт) и категорий ({catCount || "N"} шт) для вашего сайта</p>
            </div>
            <div className="flex gap-3">
              <span className="flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-blue-50 text-[10px] font-bold text-blue-600">3</span>
              <p>AI анализирует title + существующие теги/категории каждого видео</p>
            </div>
            <div className="flex gap-3">
              <span className="flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-green-50 text-[10px] font-bold text-green-600">4</span>
              <p>Проставляет {numCategories} категории и {numTags} тегов из списка бандла</p>
            </div>
          </div>
          <div className="rounded-lg bg-[var(--surface-raised)] px-4 py-3 text-xs text-[var(--text-muted)]">
            <p><strong>Вход:</strong> id, title, tags, categories</p>
            <p><strong>Выход:</strong> + new_categories, new_tags</p>
          </div>
        </div>
      </div>

      <Button
        size="lg"
        onClick={startJob}
        disabled={submitting || !selectedBundle || !fileUrl}
        className="w-full h-12 text-sm font-medium gap-2"
      >
        <Play className="h-4 w-4" />
        {submitting ? "Создание задачи..." : `Категоризировать ${lineCount ? lineCount.toLocaleString() + " видео" : "фид"}`}
      </Button>
    </motion.div>
  );
}
