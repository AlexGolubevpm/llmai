"use client";

import { useEffect, useState } from "react";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { RefreshCw, Search, Check } from "lucide-react";
import { cn } from "@/lib/utils";
import type { NovitaModel } from "@/types";

interface Props {
  value: string;
  onChange: (model: string) => void;
}

const DEFAULT_MODELS = [
  "meta-llama/llama-3.1-8b-instruct",
  "meta-llama/llama-3.1-70b-instruct",
  "Nous-Hermes-2-Mixtral-8x7B-DPO",
  "deepseek/deepseek-v3-0324",
];

export function ModelSelector({ value, onChange }: Props) {
  const [models, setModels] = useState<NovitaModel[]>([]);
  const [loading, setLoading] = useState(false);
  const [search, setSearch] = useState("");

  async function fetchModels() {
    setLoading(true);
    try {
      const resp = await fetch("/api/models");
      const data = await resp.json();
      setModels(data.models || []);
    } catch {
      setModels([]);
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    fetchModels();
  }, []);

  const modelList = models.length > 0 ? models.map((m) => m.id) : DEFAULT_MODELS;

  const filtered = search
    ? modelList.filter((id) => id.toLowerCase().includes(search.toLowerCase()))
    : modelList;

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <Label className="text-xs font-medium text-[var(--text-muted)] uppercase tracking-wider">
          Модель
        </Label>
        <Button
          variant="ghost"
          size="sm"
          className="h-7 gap-1.5 text-xs text-[var(--text-muted)]"
          onClick={fetchModels}
          disabled={loading}
          aria-label="Обновить список моделей"
        >
          <RefreshCw className={cn("h-3 w-3", loading && "animate-spin")} />
          {loading ? "Загрузка..." : "Обновить"}
        </Button>
      </div>

      {/* Current selection */}
      {value && (
        <div className="rounded-lg bg-[var(--accent-blue-light)] px-3 py-2">
          <span className="text-xs font-mono font-medium text-[var(--accent-blue)]">
            {value}
          </span>
        </div>
      )}

      {/* Search */}
      <div className="relative">
        <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-3.5 w-3.5 text-[var(--text-muted)]" />
        <Input
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          placeholder="Поиск модели..."
          className="pl-9 h-9 text-sm"
        />
      </div>

      {/* Model list */}
      <div className="max-h-[200px] overflow-y-auto rounded-lg border divide-y">
        {filtered.length === 0 ? (
          <div className="px-3 py-4 text-center text-xs text-[var(--text-muted)]">
            {search ? `Не найдено: "${search}"` : "Нет доступных моделей"}
          </div>
        ) : (
          filtered.map((id) => (
            <button
              key={id}
              onClick={() => onChange(id)}
              className={cn(
                "flex w-full items-center justify-between px-3 py-2.5 text-left text-xs font-mono transition-colors",
                id === value
                  ? "bg-[var(--accent-blue-light)] text-[var(--accent-blue)]"
                  : "hover:bg-[var(--surface-raised)] text-[var(--text-secondary)]"
              )}
            >
              <span className="truncate">{id}</span>
              {id === value && <Check className="h-3.5 w-3.5 shrink-0 ml-2" />}
            </button>
          ))
        )}
      </div>
    </div>
  );
}
