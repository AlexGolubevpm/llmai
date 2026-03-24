"use client";

import { useEffect, useState } from "react";
import { Label } from "@/components/ui/label";

function sv(v: number | readonly number[]): number {
  return typeof v === "number" ? v : v[0];
}
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Slider } from "@/components/ui/slider";
import { Textarea } from "@/components/ui/textarea";
import type { Preset, JobConfig } from "@/types";

const DEFAULT_CONFIG: JobConfig = {
  systemPrompt: "You are a helpful assistant.",
  maxTokens: 512,
  temperature: 0.7,
  topP: 1.0,
  minP: 0.0,
  topK: 40,
  presencePenalty: 0.0,
  frequencyPenalty: 0.0,
  repetitionPenalty: 1.0,
};

interface Props {
  value: JobConfig;
  onChange: (config: JobConfig) => void;
}

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
    if (!preset) {
      onChange({ ...value, ...DEFAULT_CONFIG });
      return;
    }
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
    <div className="space-y-4">
      {presets.length > 0 && (
        <div>
          <Label>Пресет</Label>
          <Select onValueChange={(v: string | null) => v && applyPreset(String(v))}>
            <SelectTrigger>
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
        <Label>System Prompt</Label>
        <Textarea
          value={value.systemPrompt || ""}
          onChange={(e) => update("systemPrompt", e.target.value)}
          rows={3}
        />
      </div>

      <div className="grid grid-cols-2 gap-4">
        <div>
          <Label>max_tokens: {value.maxTokens || 512}</Label>
          <Slider
            value={[value.maxTokens || 512]}
            onValueChange={(v) => update("maxTokens", sv(v))}
            min={0}
            max={64000}
            step={1}
          />
        </div>
        <div>
          <Label>temperature: {(value.temperature || 0.7).toFixed(2)}</Label>
          <Slider
            value={[value.temperature || 0.7]}
            onValueChange={(v) => update("temperature", sv(v))}
            min={0}
            max={2}
            step={0.01}
          />
        </div>
        <div>
          <Label>top_p: {(value.topP || 1.0).toFixed(2)}</Label>
          <Slider
            value={[value.topP || 1.0]}
            onValueChange={(v) => update("topP", sv(v))}
            min={0}
            max={1}
            step={0.01}
          />
        </div>
        <div>
          <Label>min_p: {(value.minP || 0).toFixed(2)}</Label>
          <Slider
            value={[value.minP || 0]}
            onValueChange={(v) => update("minP", sv(v))}
            min={0}
            max={1}
            step={0.01}
          />
        </div>
        <div>
          <Label>top_k: {value.topK || 40}</Label>
          <Slider
            value={[value.topK || 40]}
            onValueChange={(v) => update("topK", sv(v))}
            min={0}
            max={128}
            step={1}
          />
        </div>
        <div>
          <Label>presence_penalty: {(value.presencePenalty || 0).toFixed(2)}</Label>
          <Slider
            value={[value.presencePenalty || 0]}
            onValueChange={(v) => update("presencePenalty", sv(v))}
            min={0}
            max={2}
            step={0.01}
          />
        </div>
        <div>
          <Label>frequency_penalty: {(value.frequencyPenalty || 0).toFixed(2)}</Label>
          <Slider
            value={[value.frequencyPenalty || 0]}
            onValueChange={(v) => update("frequencyPenalty", sv(v))}
            min={0}
            max={2}
            step={0.01}
          />
        </div>
        <div>
          <Label>repetition_penalty: {(value.repetitionPenalty || 1).toFixed(2)}</Label>
          <Slider
            value={[value.repetitionPenalty || 1]}
            onValueChange={(v) => update("repetitionPenalty", sv(v))}
            min={0}
            max={2}
            step={0.01}
          />
        </div>
      </div>
    </div>
  );
}
