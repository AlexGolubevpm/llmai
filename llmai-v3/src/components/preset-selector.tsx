"use client";

import { useEffect, useState } from "react";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Slider } from "@/components/ui/slider";
import { Textarea } from "@/components/ui/textarea";
import type { Preset, JobConfig } from "@/types";

function sv(v: number | readonly number[]): number {
  return typeof v === "number" ? v : v[0];
}

interface Props {
  value: JobConfig;
  onChange: (config: JobConfig) => void;
}

const SLIDERS = [
  { key: "maxTokens" as const, label: "max_tokens", min: 0, max: 64000, step: 1, fmt: (v: number) => v.toString() },
  { key: "temperature" as const, label: "temperature", min: 0, max: 2, step: 0.01, fmt: (v: number) => v.toFixed(2) },
  { key: "topP" as const, label: "top_p", min: 0, max: 1, step: 0.01, fmt: (v: number) => v.toFixed(2) },
  { key: "minP" as const, label: "min_p", min: 0, max: 1, step: 0.01, fmt: (v: number) => v.toFixed(2) },
  { key: "topK" as const, label: "top_k", min: 0, max: 128, step: 1, fmt: (v: number) => v.toString() },
  { key: "presencePenalty" as const, label: "presence_penalty", min: 0, max: 2, step: 0.01, fmt: (v: number) => v.toFixed(2) },
  { key: "frequencyPenalty" as const, label: "frequency_penalty", min: 0, max: 2, step: 0.01, fmt: (v: number) => v.toFixed(2) },
  { key: "repetitionPenalty" as const, label: "repetition_penalty", min: 0, max: 2, step: 0.01, fmt: (v: number) => v.toFixed(2) },
];

const DEFAULTS: Record<string, number> = {
  maxTokens: 512, temperature: 0.7, topP: 1.0, minP: 0.0, topK: 40,
  presencePenalty: 0.0, frequencyPenalty: 0.0, repetitionPenalty: 1.0,
};

export function PresetSelector({ value, onChange }: Props) {
  const [presets, setPresets] = useState<Preset[]>([]);

  useEffect(() => {
    fetch("/api/presets")
      .then((r) => r.json())
      .then((data) => setPresets(data.presets || []))
      .catch(() => {});
  }, []);

  function applyPreset(presetId: string) {
    const preset = presets.find((p) => p.id === presetId);
    if (!preset) return;
    onChange({
      ...value,
      systemPrompt: preset.systemPrompt,
      maxTokens: preset.maxTokens,
      temperature: preset.temperature,
      topP: preset.topP,
      minP: preset.minP,
      topK: preset.topK,
      presencePenalty: preset.presencePenalty,
      frequencyPenalty: preset.frequencyPenalty,
      repetitionPenalty: preset.repetitionPenalty,
    });
  }

  function update(key: keyof JobConfig, val: unknown) {
    onChange({ ...value, [key]: val });
  }

  return (
    <div className="space-y-5">
      {presets.length > 0 && (
        <div>
          <Label className="text-xs font-medium text-[var(--text-muted)] uppercase tracking-wider">
            Пресет
          </Label>
          <Select onValueChange={(v: string | null) => v && applyPreset(String(v))}>
            <SelectTrigger className="mt-1.5">
              <SelectValue placeholder="Выберите пресет" />
            </SelectTrigger>
            <SelectContent>
              {presets.map((p) => (
                <SelectItem key={p.id} value={p.id}>
                  {p.name}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
      )}

      <div>
        <Label className="text-xs font-medium text-[var(--text-muted)] uppercase tracking-wider">
          System Prompt
        </Label>
        <Textarea
          value={value.systemPrompt || ""}
          onChange={(e) => update("systemPrompt", e.target.value)}
          rows={3}
          className="mt-1.5 font-mono text-xs"
        />
      </div>

      <div className="grid grid-cols-2 gap-x-6 gap-y-5">
        {SLIDERS.map((s) => {
          const val = (value[s.key] as number) ?? DEFAULTS[s.key];
          return (
            <div key={s.key}>
              <div className="flex items-baseline justify-between mb-2">
                <span className="text-xs font-mono text-[var(--text-muted)]">{s.label}</span>
                <span className="text-xs font-mono font-medium tabular-nums text-[var(--text-primary)]">
                  {s.fmt(val)}
                </span>
              </div>
              <Slider
                value={[val]}
                onValueChange={(v) => update(s.key, sv(v))}
                min={s.min}
                max={s.max}
                step={s.step}
              />
            </div>
          );
        })}
      </div>
    </div>
  );
}
