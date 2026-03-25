"use client";

import { useEffect, useState } from "react";
import { motion } from "framer-motion";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { Textarea } from "@/components/ui/textarea";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Trash2, Plus, Search, Ban, Upload } from "lucide-react";
import { toast } from "sonner";
import { PageHeader } from "@/components/layout/page-header";
import { EmptyState } from "@/components/shared/empty-state";
import { pageVariants, staggerItem } from "@/lib/animations";
import type { StopWord } from "@/types";

export default function StopwordsPage() {
  const [stopwords, setStopwords] = useState<StopWord[]>([]);
  const [loading, setLoading] = useState(true);
  const [search, setSearch] = useState("");
  const [newWord, setNewWord] = useState("");
  const [newReplacement, setNewReplacement] = useState("");
  const [dialogOpen, setDialogOpen] = useState(false);
  const [bulkDialogOpen, setBulkDialogOpen] = useState(false);
  const [bulkText, setBulkText] = useState("");
  const [bulkUploading, setBulkUploading] = useState(false);

  async function fetchStopwords() {
    setLoading(true);
    try {
      const resp = await fetch("/api/stopwords");
      const data = await resp.json();
      setStopwords(data.stopwords || []);
    } catch {
      toast.error("Не удалось загрузить стоп-слова");
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    fetchStopwords();
  }, []);

  async function addStopword() {
    if (!newWord.trim()) {
      toast.error("Введите слово");
      return;
    }
    try {
      await fetch("/api/stopwords", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          word: newWord.trim(),
          replacement: newReplacement.trim() || null,
        }),
      });
      setNewWord("");
      setNewReplacement("");
      setDialogOpen(false);
      fetchStopwords();
      toast.success("Стоп-слово добавлено");
    } catch (err) {
      toast.error((err as Error).message);
    }
  }

  async function bulkAdd() {
    const lines = bulkText
      .split("\n")
      .map((l) => l.trim())
      .filter(Boolean);
    if (lines.length === 0) {
      toast.error("Вставьте хотя бы одно слово");
      return;
    }

    setBulkUploading(true);
    try {
      // Each line: "word" or "word|replacement"
      const items = lines.map((line) => {
        const parts = line.split("|").map((p) => p.trim());
        return {
          word: parts[0],
          replacement: parts[1] || null,
        };
      });

      await fetch("/api/stopwords", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(items),
      });

      setBulkText("");
      setBulkDialogOpen(false);
      fetchStopwords();
      toast.success(`Добавлено ${items.length} стоп-слов`);
    } catch (err) {
      toast.error((err as Error).message);
    } finally {
      setBulkUploading(false);
    }
  }

  async function handleFileUpload(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (!file) return;

    setBulkUploading(true);
    try {
      const text = await file.text();
      const lines = text
        .split("\n")
        .map((l) => l.trim())
        .filter(Boolean);

      const items = lines.map((line) => {
        const parts = line.split("|").map((p) => p.trim());
        return {
          word: parts[0],
          replacement: parts[1] || null,
        };
      });

      await fetch("/api/stopwords", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(items),
      });

      fetchStopwords();
      toast.success(`Загружено ${items.length} стоп-слов из ${file.name}`);
      setBulkDialogOpen(false);
    } catch (err) {
      toast.error((err as Error).message);
    } finally {
      setBulkUploading(false);
    }
  }

  async function deleteStopword(id: string) {
    await fetch(`/api/stopwords?id=${id}`, { method: "DELETE" });
    setStopwords((sw) => sw.filter((s) => s.id !== id));
    toast.success("Удалено");
  }

  async function toggleStopword(id: string, isActive: boolean) {
    setStopwords((sw) =>
      sw.map((s) => (s.id === id ? { ...s, isActive } : s))
    );
    await fetch("/api/stopwords", {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ id, isActive }),
    });
  }

  const filtered = search
    ? stopwords.filter((sw) =>
        sw.word.toLowerCase().includes(search.toLowerCase())
      )
    : stopwords;

  return (
    <motion.div {...pageVariants} className="space-y-8">
      <PageHeader
        title="Стоп-слова"
        description="Управление заменами при обработке текста"
        actions={
          <div className="flex gap-2">
            <Button
              variant="outline"
              size="sm"
              className="gap-2"
              onClick={() => setBulkDialogOpen(true)}
            >
              <Upload className="h-3.5 w-3.5" /> Массовая загрузка
            </Button>
            <Button
              size="sm"
              className="gap-2"
              onClick={() => setDialogOpen(true)}
            >
              <Plus className="h-3.5 w-3.5" /> Добавить
            </Button>
          </div>
        }
      />

      {/* Single add dialog */}
      <Dialog open={dialogOpen} onOpenChange={setDialogOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Добавить стоп-слово</DialogTitle>
          </DialogHeader>
          <div className="space-y-4 pt-4">
            <div>
              <Label>Слово</Label>
              <Input
                value={newWord}
                onChange={(e) => setNewWord(e.target.value)}
                placeholder="mother"
                className="mt-1.5"
              />
            </div>
            <div>
              <Label>Замена (пусто = удалить)</Label>
              <Input
                value={newReplacement}
                onChange={(e) => setNewReplacement(e.target.value)}
                placeholder="StepMother"
                className="mt-1.5"
              />
            </div>
            <Button onClick={addStopword} className="w-full">
              Добавить
            </Button>
          </div>
        </DialogContent>
      </Dialog>

      {/* Bulk add dialog */}
      <Dialog open={bulkDialogOpen} onOpenChange={setBulkDialogOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Массовая загрузка стоп-слов</DialogTitle>
          </DialogHeader>
          <Tabs defaultValue="paste">
            <TabsList className="mb-4">
              <TabsTrigger value="paste">Вставить список</TabsTrigger>
              <TabsTrigger value="file">Загрузить файл</TabsTrigger>
            </TabsList>

            <TabsContent value="paste" className="space-y-4">
              <div>
                <Label>Список стоп-слов (по одному на строку)</Label>
                <Textarea
                  value={bulkText}
                  onChange={(e) => setBulkText(e.target.value)}
                  placeholder={`mother\nfather\nbadword|replacement\nspam phrase`}
                  rows={10}
                  className="mt-1.5 font-mono text-xs"
                />
                <p className="mt-1.5 text-xs text-[var(--text-muted)]">
                  Формат: <code>слово</code> или{" "}
                  <code>слово|замена</code>
                </p>
              </div>
              <Button
                onClick={bulkAdd}
                disabled={bulkUploading || !bulkText.trim()}
                className="w-full"
              >
                {bulkUploading
                  ? "Загрузка..."
                  : `Добавить ${bulkText.split("\n").filter((l) => l.trim()).length} слов`}
              </Button>
            </TabsContent>

            <TabsContent value="file" className="space-y-4">
              <div className="rounded-xl border-2 border-dashed border-[var(--border)] p-8 text-center">
                <Upload className="h-8 w-8 text-[var(--text-muted)] mx-auto mb-3" />
                <p className="text-sm text-[var(--text-secondary)] mb-2">
                  TXT файл (по одному слову на строку)
                </p>
                <label className="cursor-pointer text-sm font-medium text-[var(--accent-blue)] hover:underline">
                  {bulkUploading ? "Загрузка..." : "Выбрать файл"}
                  <input
                    type="file"
                    accept=".txt,.csv"
                    className="hidden"
                    onChange={handleFileUpload}
                    disabled={bulkUploading}
                  />
                </label>
                <p className="mt-2 text-xs text-[var(--text-muted)]">
                  Формат: <code>слово</code> или{" "}
                  <code>слово|замена</code>
                </p>
              </div>
            </TabsContent>
          </Tabs>
        </DialogContent>
      </Dialog>

      <div className="rounded-xl border bg-[var(--surface)] shadow-card overflow-hidden">
        <div className="border-b p-4">
          <div className="relative">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-[var(--text-muted)]" />
            <Input
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              placeholder="Поиск по слову..."
              className="pl-9"
            />
          </div>
        </div>

        {loading ? (
          <div className="p-6 space-y-3">
            {[...Array(5)].map((_, i) => (
              <div key={i} className="h-10 animate-shimmer rounded-lg" />
            ))}
          </div>
        ) : filtered.length === 0 ? (
          <EmptyState
            icon={<Ban className="h-6 w-6" />}
            title={
              search
                ? `Ничего не найдено по "${search}"`
                : "Нет стоп-слов"
            }
            description={
              search
                ? undefined
                : "Добавьте стоп-слова для автоматической очистки текста"
            }
          />
        ) : (
          <Table>
            <TableHeader>
              <TableRow className="hover:bg-transparent">
                <TableHead className="text-xs font-medium uppercase tracking-wider text-[var(--text-muted)] w-16">
                  Вкл
                </TableHead>
                <TableHead className="text-xs font-medium uppercase tracking-wider text-[var(--text-muted)]">
                  Слово
                </TableHead>
                <TableHead className="text-xs font-medium uppercase tracking-wider text-[var(--text-muted)]">
                  Замена
                </TableHead>
                <TableHead className="text-xs font-medium uppercase tracking-wider text-[var(--text-muted)] w-16"></TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {filtered.map((sw, i) => (
                <motion.tr
                  key={sw.id}
                  variants={staggerItem}
                  initial="initial"
                  animate="animate"
                  transition={{ delay: i * 0.02 }}
                  className="border-b last:border-0 hover:bg-[var(--surface-raised)]"
                >
                  <TableCell>
                    <Switch
                      checked={sw.isActive}
                      onCheckedChange={(v) => toggleStopword(sw.id, v)}
                    />
                  </TableCell>
                  <TableCell className="font-mono text-sm">
                    {sw.word}
                  </TableCell>
                  <TableCell className="font-mono text-sm text-[var(--text-muted)]">
                    {sw.replacement || "(удалить)"}
                  </TableCell>
                  <TableCell>
                    <Button
                      variant="ghost"
                      size="icon"
                      className="h-7 w-7 text-[var(--error)]"
                      onClick={() => deleteStopword(sw.id)}
                      aria-label="Удалить"
                    >
                      <Trash2 className="h-3.5 w-3.5" />
                    </Button>
                  </TableCell>
                </motion.tr>
              ))}
            </TableBody>
          </Table>
        )}

        {!loading && stopwords.length > 0 && (
          <div className="border-t px-4 py-3 text-xs text-[var(--text-muted)]">
            {stopwords.length} стоп-слов (
            {stopwords.filter((s) => s.isActive).length} активных)
          </div>
        )}
      </div>
    </motion.div>
  );
}
