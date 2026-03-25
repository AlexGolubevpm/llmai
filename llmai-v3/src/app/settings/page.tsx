"use client";

import { useEffect, useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { Slider } from "@/components/ui/slider";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Trash2, Plus, Save } from "lucide-react";
import { toast } from "sonner";
import type { Preset } from "@/types";

export default function SettingsPage() {
  const [presets, setPresets] = useState<Preset[]>([]);
  const [name, setName] = useState("");
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
    const resp = await fetch("/api/presets");
    const data = await resp.json();
    setPresets(data.presets || []);
  }

  useEffect(() => { fetchPresets(); }, []);

  async function savePreset() {
    if (!name.trim()) { toast.error("Введите название"); return; }
    await fetch("/api/presets", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        name: name.trim(), systemPrompt, maxTokens, temperature,
        topP, minP, topK, presencePenalty, frequencyPenalty, repetitionPenalty,
      }),
    });
    fetchPresets();
    toast.success("Пресет сохранён");
  }

  async function deletePreset(id: string) {
    await fetch(`/api/presets?id=${id}`, { method: "DELETE" });
    fetchPresets();
  }

  function loadPreset(p: Preset) {
    setName(p.name);
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

  return (
    <div className="space-y-6">
      <h1 className="text-3xl font-bold">Настройки</h1>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card>
          <CardHeader><CardTitle>Пресеты</CardTitle></CardHeader>
          <CardContent>
            <Table>
              <TableHeader>
                <TableRow><TableHead>Название</TableHead><TableHead>Токены</TableHead><TableHead>Темп.</TableHead><TableHead></TableHead></TableRow>
              </TableHeader>
              <TableBody>
                {presets.map((p) => (
                  <TableRow key={p.id} className="cursor-pointer" onClick={() => loadPreset(p)}>
                    <TableCell className="font-medium">{p.name}</TableCell>
                    <TableCell>{p.maxTokens}</TableCell>
                    <TableCell>{p.temperature}</TableCell>
                    <TableCell>
                      <Button variant="ghost" size="icon" className="h-8 w-8 text-red-500" onClick={(e) => { e.stopPropagation(); deletePreset(p.id); }}>
                        <Trash2 className="h-4 w-4" />
                      </Button>
                    </TableCell>
                  </TableRow>
                ))}
                {presets.length === 0 && (
                  <TableRow><TableCell colSpan={4} className="text-center text-muted-foreground py-8">Нет пресетов</TableCell></TableRow>
                )}
              </TableBody>
            </Table>
          </CardContent>
        </Card>

        <Card>
          <CardHeader><CardTitle>Редактор пресета</CardTitle></CardHeader>
          <CardContent className="space-y-4">
            <div><Label>Название</Label><Input value={name} onChange={(e) => setName(e.target.value)} placeholder="My Preset" /></div>
            <div><Label>System Prompt</Label><Textarea value={systemPrompt} onChange={(e) => setSystemPrompt(e.target.value)} rows={3} /></div>
            <div className="grid grid-cols-2 gap-4">
              <div><Label>max_tokens: {maxTokens}</Label><Slider value={[maxTokens]} onValueChange={(v) => setMaxTokens(typeof v === "number" ? v : v[0])} min={0} max={64000} step={1} /></div>
              <div><Label>temperature: {temperature.toFixed(2)}</Label><Slider value={[temperature]} onValueChange={(v) => setTemperature(typeof v === "number" ? v : v[0])} min={0} max={2} step={0.01} /></div>
              <div><Label>top_p: {topP.toFixed(2)}</Label><Slider value={[topP]} onValueChange={(v) => setTopP(typeof v === "number" ? v : v[0])} min={0} max={1} step={0.01} /></div>
              <div><Label>min_p: {minP.toFixed(2)}</Label><Slider value={[minP]} onValueChange={(v) => setMinP(typeof v === "number" ? v : v[0])} min={0} max={1} step={0.01} /></div>
              <div><Label>top_k: {topK}</Label><Slider value={[topK]} onValueChange={(v) => setTopK(typeof v === "number" ? v : v[0])} min={0} max={128} step={1} /></div>
              <div><Label>presence_penalty: {presencePenalty.toFixed(2)}</Label><Slider value={[presencePenalty]} onValueChange={(v) => setPresencePenalty(typeof v === "number" ? v : v[0])} min={0} max={2} step={0.01} /></div>
              <div><Label>frequency_penalty: {frequencyPenalty.toFixed(2)}</Label><Slider value={[frequencyPenalty]} onValueChange={(v) => setFrequencyPenalty(typeof v === "number" ? v : v[0])} min={0} max={2} step={0.01} /></div>
              <div><Label>repetition_penalty: {repetitionPenalty.toFixed(2)}</Label><Slider value={[repetitionPenalty]} onValueChange={(v) => setRepetitionPenalty(typeof v === "number" ? v : v[0])} min={0} max={2} step={0.01} /></div>
            </div>
            <Button onClick={savePreset}><Save className="h-4 w-4 mr-2" /> Сохранить пресет</Button>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
