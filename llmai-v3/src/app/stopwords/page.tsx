"use client";

import { useEffect, useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Trash2, Plus } from "lucide-react";
import { toast } from "sonner";
import type { StopWord } from "@/types";

export default function StopwordsPage() {
  const [stopwords, setStopwords] = useState<StopWord[]>([]);
  const [newWord, setNewWord] = useState("");
  const [newReplacement, setNewReplacement] = useState("");
  const [loading, setLoading] = useState(true);

  async function fetchStopwords() {
    setLoading(true);
    try {
      const resp = await fetch("/api/stopwords");
      const data = await resp.json();
      setStopwords(data.stopwords || []);
    } catch { /* ignore */ }
    finally { setLoading(false); }
  }

  useEffect(() => { fetchStopwords(); }, []);

  async function addStopword() {
    if (!newWord.trim()) { toast.error("Введите слово"); return; }
    try {
      await fetch("/api/stopwords", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ word: newWord.trim(), replacement: newReplacement.trim() || null }),
      });
      setNewWord("");
      setNewReplacement("");
      fetchStopwords();
      toast.success("Стоп-слово добавлено");
    } catch (err) { toast.error((err as Error).message); }
  }

  async function deleteStopword(id: string) {
    await fetch(`/api/stopwords?id=${id}`, { method: "DELETE" });
    fetchStopwords();
  }

  async function toggleStopword(id: string, isActive: boolean) {
    await fetch("/api/stopwords", {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ id, isActive }),
    });
    fetchStopwords();
  }

  return (
    <div className="space-y-6">
      <h1 className="text-3xl font-bold">Стоп-слова</h1>

      <Card>
        <CardHeader><CardTitle>Добавить стоп-слово</CardTitle></CardHeader>
        <CardContent>
          <div className="flex gap-4 items-end">
            <div className="flex-1">
              <Label>Слово</Label>
              <Input value={newWord} onChange={(e) => setNewWord(e.target.value)} placeholder="mother" />
            </div>
            <div className="flex-1">
              <Label>Замена (пусто = удалить)</Label>
              <Input value={newReplacement} onChange={(e) => setNewReplacement(e.target.value)} placeholder="StepMother" />
            </div>
            <Button onClick={addStopword}><Plus className="h-4 w-4 mr-2" /> Добавить</Button>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader><CardTitle>Список стоп-слов ({stopwords.length})</CardTitle></CardHeader>
        <CardContent>
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Активно</TableHead>
                <TableHead>Слово</TableHead>
                <TableHead>Замена</TableHead>
                <TableHead></TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {stopwords.map((sw) => (
                <TableRow key={sw.id}>
                  <TableCell>
                    <Switch checked={sw.isActive} onCheckedChange={(v) => toggleStopword(sw.id, v)} />
                  </TableCell>
                  <TableCell className="font-mono">{sw.word}</TableCell>
                  <TableCell className="font-mono text-muted-foreground">{sw.replacement || "(удалить)"}</TableCell>
                  <TableCell>
                    <Button variant="ghost" size="icon" className="h-8 w-8 text-red-500" onClick={() => deleteStopword(sw.id)}>
                      <Trash2 className="h-4 w-4" />
                    </Button>
                  </TableCell>
                </TableRow>
              ))}
              {stopwords.length === 0 && !loading && (
                <TableRow><TableCell colSpan={4} className="text-center text-muted-foreground py-8">Нет стоп-слов</TableCell></TableRow>
              )}
            </TableBody>
          </Table>
        </CardContent>
      </Card>
    </div>
  );
}
