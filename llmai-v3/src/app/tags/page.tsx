"use client";

import { useEffect, useState } from "react";
import { motion } from "framer-motion";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Trash2, Plus, Tags } from "lucide-react";
import { toast } from "sonner";
import { PageHeader } from "@/components/layout/page-header";
import { EmptyState } from "@/components/shared/empty-state";
import { pageVariants, staggerItem } from "@/lib/animations";
import type { AllowedTag, AllowedCategory } from "@/types";

export default function TagsPage() {
  const [tags, setTags] = useState<AllowedTag[]>([]);
  const [categories, setCategories] = useState<AllowedCategory[]>([]);
  const [newTag, setNewTag] = useState("");
  const [newTagCategory, setNewTagCategory] = useState("general");
  const [newCategoryName, setNewCategoryName] = useState("");

  async function fetchAll() {
    try {
      const [tagsResp, catsResp] = await Promise.all([
        fetch("/api/tags?type=tags"),
        fetch("/api/tags?type=categories"),
      ]);
      setTags((await tagsResp.json()).tags || []);
      setCategories((await catsResp.json()).categories || []);
    } catch { toast.error("Не удалось загрузить данные"); }
  }

  useEffect(() => { fetchAll(); }, []);

  async function addTag() {
    if (!newTag.trim()) return;
    await fetch("/api/tags", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ type: "tags", items: [{ name: newTag.trim(), category: newTagCategory }] }),
    });
    setNewTag("");
    fetchAll();
    toast.success("Тег добавлен");
  }

  async function addCategory() {
    if (!newCategoryName.trim()) return;
    await fetch("/api/tags", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ type: "categories", items: [{ name: newCategoryName.trim() }] }),
    });
    setNewCategoryName("");
    fetchAll();
    toast.success("Категория добавлена");
  }

  async function deleteTag(id: string) {
    await fetch(`/api/tags?type=tags&id=${id}`, { method: "DELETE" });
    setTags((t) => t.filter((x) => x.id !== id));
  }

  async function deleteCategory(id: string) {
    await fetch(`/api/tags?type=categories&id=${id}`, { method: "DELETE" });
    setCategories((c) => c.filter((x) => x.id !== id));
  }

  return (
    <motion.div {...pageVariants} className="space-y-8">
      <PageHeader title="Теги & Категории" description="Разрешённые теги и категории для AI Process" />

      <Tabs defaultValue="tags">
        <TabsList>
          <TabsTrigger value="tags">Теги ({tags.length})</TabsTrigger>
          <TabsTrigger value="categories">Категории ({categories.length})</TabsTrigger>
        </TabsList>

        <TabsContent value="tags" className="mt-6 space-y-4">
          <div className="rounded-xl border bg-[var(--surface)] p-4">
            <div className="flex gap-3">
              <Input value={newTag} onChange={(e) => setNewTag(e.target.value)} placeholder="Новый тег..." className="flex-1" />
              <Input value={newTagCategory} onChange={(e) => setNewTagCategory(e.target.value)} placeholder="Категория" className="w-40" />
              <Button onClick={addTag} size="sm" className="gap-1.5 shrink-0">
                <Plus className="h-3.5 w-3.5" /> Добавить
              </Button>
            </div>
          </div>
          <div className="rounded-xl border bg-[var(--surface)] shadow-card overflow-hidden">
            {tags.length === 0 ? (
              <EmptyState icon={<Tags className="h-6 w-6" />} title="Нет тегов" description="Добавьте разрешённые теги для AI Process" />
            ) : (
              <Table>
                <TableHeader><TableRow className="hover:bg-transparent">
                  <TableHead className="text-xs font-medium uppercase tracking-wider text-[var(--text-muted)]">Тег</TableHead>
                  <TableHead className="text-xs font-medium uppercase tracking-wider text-[var(--text-muted)]">Категория</TableHead>
                  <TableHead className="w-12"></TableHead>
                </TableRow></TableHeader>
                <TableBody>
                  {tags.map((t, i) => (
                    <motion.tr key={t.id} variants={staggerItem} initial="initial" animate="animate" transition={{ delay: i * 0.02 }} className="border-b last:border-0 hover:bg-[var(--surface-raised)]">
                      <TableCell className="font-mono text-sm">{t.name}</TableCell>
                      <TableCell><span className="rounded-md bg-[var(--surface-raised)] px-2 py-0.5 text-xs">{t.category}</span></TableCell>
                      <TableCell>
                        <Button variant="ghost" size="icon" className="h-7 w-7 text-[var(--error)]" onClick={() => deleteTag(t.id)} aria-label="Удалить тег"><Trash2 className="h-3.5 w-3.5" /></Button>
                      </TableCell>
                    </motion.tr>
                  ))}
                </TableBody>
              </Table>
            )}
          </div>
        </TabsContent>

        <TabsContent value="categories" className="mt-6 space-y-4">
          <div className="rounded-xl border bg-[var(--surface)] p-4">
            <div className="flex gap-3">
              <Input value={newCategoryName} onChange={(e) => setNewCategoryName(e.target.value)} placeholder="Новая категория..." className="flex-1" />
              {newCategoryName && (
                <span className="flex items-center text-xs text-[var(--text-muted)] font-mono">
                  slug: {newCategoryName.toLowerCase().replace(/\s+/g, "-")}
                </span>
              )}
              <Button onClick={addCategory} size="sm" className="gap-1.5 shrink-0">
                <Plus className="h-3.5 w-3.5" /> Добавить
              </Button>
            </div>
          </div>
          <div className="rounded-xl border bg-[var(--surface)] shadow-card overflow-hidden">
            {categories.length === 0 ? (
              <EmptyState icon={<Tags className="h-6 w-6" />} title="Нет категорий" />
            ) : (
              <Table>
                <TableHeader><TableRow className="hover:bg-transparent">
                  <TableHead className="text-xs font-medium uppercase tracking-wider text-[var(--text-muted)]">Категория</TableHead>
                  <TableHead className="text-xs font-medium uppercase tracking-wider text-[var(--text-muted)]">Slug</TableHead>
                  <TableHead className="w-12"></TableHead>
                </TableRow></TableHeader>
                <TableBody>
                  {categories.map((c, i) => (
                    <motion.tr key={c.id} variants={staggerItem} initial="initial" animate="animate" transition={{ delay: i * 0.02 }} className="border-b last:border-0 hover:bg-[var(--surface-raised)]">
                      <TableCell className="text-sm font-medium">{c.name}</TableCell>
                      <TableCell className="font-mono text-xs text-[var(--text-muted)]">{c.slug}</TableCell>
                      <TableCell>
                        <Button variant="ghost" size="icon" className="h-7 w-7 text-[var(--error)]" onClick={() => deleteCategory(c.id)} aria-label="Удалить"><Trash2 className="h-3.5 w-3.5" /></Button>
                      </TableCell>
                    </motion.tr>
                  ))}
                </TableBody>
              </Table>
            )}
          </div>
        </TabsContent>
      </Tabs>
    </motion.div>
  );
}
