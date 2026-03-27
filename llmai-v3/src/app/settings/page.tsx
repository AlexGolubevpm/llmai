"use client";

import { useEffect, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { Slider } from "@/components/ui/slider";
import { Trash2, Save, Settings as SettingsIcon } from "lucide-react";
import { toast } from "sonner";
import { PageHeader } from "@/components/layout/page-header";
import { EmptyState } from "@/components/shared/empty-state";
import { pageVariants } from "@/lib/animations";
import { cn } from "@/lib/utils";
import type { Preset } from "@/types";

function sv(v: number | readonly number[]): number {
  return typeof v === "number" ? v : v[0];
}

export default function SettingsPage() {
  const [presets, setPresets] = useState<Preset[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [name, setName] = useState("");
  const [modelId, setModelId] = useState("openai/gpt-4o-mini");
  const [systemPrompt, setSystemPrompt] = useState("You are a helpful assistant.");
  const [maxTokens, setMaxTokens] = useState(512);
  const [temperature, setTemperature] = useState(0.7);
  const [topP, setTopP] = useState(1.0);
  const [minP, setMinP] = useState(0.0);
  const [topK, setTopK] = useState(40);
  const [presencePenalty, setPresencePenalty] = useState(0.0);
  const [frequencyPenalty, setFrequencyPenalty] = useState(0.0);
  const [repetitionPenalty, setRepetitionPenalty] = useState(1.0);

  async function fetchPresets() {
    try {
      const resp = await fetch("/api/presets");
      const data = await resp.json();
      setPresets(data.presets || []);
    } catch { toast.error("Не удалось загрузить пресеты"); }
  }

  useEffect(() => { fetchPresets(); }, []);

  function loadPreset(p: Preset & { model?: string }) {
    setSelectedId(p.id);
    setName(p.name);
    setModelId((p as any).model || "openai/gpt-4o-mini");
    setSystemPrompt(p.systemPrompt);
    setMaxTokens(p.maxTokens);
    setTemperature(p.temperature);
    setTopP(p.topP);
    setMinP(p.minP);
    setTopK(p.topK);
    setPresencePenalty(p.presencePenalty);
    setFrequencyPenalty(p.frequencyPenalty);
    setRepetitionPenalty(p.repetitionPenalty);
  }

  function clearEditor() {
    setSelectedId(null);
    setName("");
    setModelId("openai/gpt-4o-mini");
    setSystemPrompt("You are a helpful assistant.");
    setMaxTokens(512);
    setTemperature(0.7);
    setTopP(1.0);
    setMinP(0.0);
    setTopK(40);
    setPresencePenalty(0.0);
    setFrequencyPenalty(0.0);
    setRepetitionPenalty(1.0);
  }

  async function savePreset() {
    if (!name.trim()) { toast.error("Введите название"); return; }
    try {
      const resp = await fetch("/api/presets", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name: name.trim(), model: modelId, systemPrompt, maxTokens, temperature, topP, minP, topK, presencePenalty, frequencyPenalty, repetitionPenalty }),
      });
      const data = await resp.json();
      if (!resp.ok) throw new Error(data.error || "Ошибка сохранения");
      fetchPresets();
      toast.success("Пресет сохранён");
    } catch (err) {
      toast.error((err as Error).message);
    }
  }

  async function deletePreset(id: string) {
    await fetch(`/api/presets?id=${id}`, { method: "DELETE" });
    if (selectedId === id) clearEditor();
    setPresets((p) => p.filter((x) => x.id !== id));
    toast.success("Удалён");
  }

  const sliders = [
    { label: "max_tokens", value: maxTokens, set: setMaxTokens, min: 0, max: 64000, step: 1, format: (v: number) => v.toString() },
    { label: "temperature", value: temperature, set: setTemperature, min: 0, max: 2, step: 0.01, format: (v: number) => v.toFixed(2) },
    { label: "top_p", value: topP, set: setTopP, min: 0, max: 1, step: 0.01, format: (v: number) => v.toFixed(2) },
    { label: "min_p", value: minP, set: setMinP, min: 0, max: 1, step: 0.01, format: (v: number) => v.toFixed(2) },
    { label: "top_k", value: topK, set: setTopK, min: 0, max: 128, step: 1, format: (v: number) => v.toString() },
    { label: "presence_penalty", value: presencePenalty, set: setPresencePenalty, min: 0, max: 2, step: 0.01, format: (v: number) => v.toFixed(2) },
    { label: "frequency_penalty", value: frequencyPenalty, set: setFrequencyPenalty, min: 0, max: 2, step: 0.01, format: (v: number) => v.toFixed(2) },
    { label: "repetition_penalty", value: repetitionPenalty, set: setRepetitionPenalty, min: 0, max: 2, step: 0.01, format: (v: number) => v.toFixed(2) },
  ];

  return (
    <motion.div {...pageVariants} className="space-y-8">
      <PageHeader title="Настройки" description="Управление пресетами параметров LLM" />

      <div className="grid grid-cols-1 lg:grid-cols-[280px_1fr] gap-6">
        {/* Preset list */}
        <div className="rounded-xl border bg-[var(--surface)] overflow-hidden">
          <div className="border-b px-4 py-3 flex items-center justify-between">
            <span className="text-xs font-medium text-[var(--text-muted)] uppercase tracking-wider">Пресеты</span>
            <Button variant="ghost" size="sm" onClick={clearEditor} className="text-xs h-7">+ Новый</Button>
          </div>
          {presets.length === 0 ? (
            <EmptyState icon={<SettingsIcon className="h-5 w-5" />} title="Нет пресетов" className="py-10" />
          ) : (
            <div className="p-2 space-y-1">
              {presets.map((p) => (
                <button
                  key={p.id}
                  onClick={() => loadPreset(p)}
                  className={cn(
                    "flex w-full items-center justify-between rounded-lg px-3 py-2.5 text-sm text-left transition-colors",
                    selectedId === p.id
                      ? "bg-[var(--accent-blue-light)] text-[var(--accent-blue)] font-medium"
                      : "hover:bg-[var(--surface-raised)] text-[var(--text-secondary)]"
                  )}
                >
                  <span className="truncate">{p.name}</span>
                  <Button
                    variant="ghost"
                    size="icon"
                    className="h-6 w-6 text-[var(--error)] opacity-0 group-hover:opacity-100 shrink-0"
                    onClick={(e) => { e.stopPropagation(); deletePreset(p.id); }}
                    aria-label="Удалить пресет"
                  >
                    <Trash2 className="h-3 w-3" />
                  </Button>
                </button>
              ))}
            </div>
          )}
        </div>

        {/* Editor */}
        <div className="rounded-xl border bg-[var(--surface)] p-6 space-y-6">
          <h2 className="text-[15px] font-medium">
            {selectedId ? `Редактирование: ${name}` : "Новый пресет"}
          </h2>

          <div>
            <Label className="text-xs font-medium text-[var(--text-muted)] uppercase tracking-wider">Название</Label>
            <Input value={name} onChange={(e) => setName(e.target.value)} placeholder="My Preset" className="mt-1.5" />
          </div>

          <div>
            <Label className="text-xs font-medium text-[var(--text-muted)] uppercase tracking-wider">Модель</Label>
            <Input value={modelId} onChange={(e) => setModelId(e.target.value)} placeholder="openai/gpt-4o-mini" className="mt-1.5 font-mono text-xs" />
          </div>

          <div>
            <Label className="text-xs font-medium text-[var(--text-muted)] uppercase tracking-wider">System Prompt</Label>
            <Textarea value={systemPrompt} onChange={(e) => setSystemPrompt(e.target.value)} rows={3} className="mt-1.5 font-mono text-xs" />
          </div>

          <div className="grid grid-cols-2 gap-x-6 gap-y-5">
            {sliders.map((s) => (
              <div key={s.label}>
                <div className="flex items-baseline justify-between mb-2">
                  <Label className="text-xs font-mono text-[var(--text-muted)]">{s.label}</Label>
                  <span className="text-xs font-mono font-medium tabular-nums">{s.format(s.value)}</span>
                </div>
                <Slider value={[s.value]} onValueChange={(v) => s.set(sv(v))} min={s.min} max={s.max} step={s.step} />
              </div>
            ))}
          </div>

          <div className="flex gap-3 pt-2">
            <Button onClick={savePreset} className="gap-2">
              <Save className="h-3.5 w-3.5" /> Сохранить
            </Button>
            {selectedId && (
              <Button variant="outline" onClick={() => deletePreset(selectedId)} className="gap-2 text-[var(--error)]">
                <Trash2 className="h-3.5 w-3.5" /> Удалить
              </Button>
            )}
          </div>
        </div>
      </div>
    </motion.div>
  );
}
