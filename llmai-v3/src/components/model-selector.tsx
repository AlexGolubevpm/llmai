"use client";

import { useEffect, useState } from "react";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { RefreshCw } from "lucide-react";
import type { NovitaModel } from "@/types";

interface Props {
  value: string;
  onChange: (model: string) => void;
}

const DEFAULT_MODELS = [
  "meta-llama/llama-3.1-8b-instruct",
  "Nous-Hermes-2-Mixtral-8x7B-DPO",
];

export function ModelSelector({ value, onChange }: Props) {
  const [models, setModels] = useState<NovitaModel[]>([]);
  const [loading, setLoading] = useState(false);

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

  const modelList =
    models.length > 0
      ? models.map((m) => m.id)
      : DEFAULT_MODELS;

  return (
    <div className="space-y-2">
      <div className="flex items-center gap-2">
        <Label>Модель</Label>
        <Button
          variant="ghost"
          size="icon"
          className="h-6 w-6"
          onClick={fetchModels}
          disabled={loading}
        >
          <RefreshCw className={`h-3 w-3 ${loading ? "animate-spin" : ""}`} />
        </Button>
      </div>
      <Select value={value} onValueChange={(v) => v && onChange(v)}>
        <SelectTrigger>
          <SelectValue placeholder="Выберите модель" />
        </SelectTrigger>
        <SelectContent>
          {modelList.map((id) => (
            <SelectItem key={id} value={id}>
              {id}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
    </div>
  );
}
