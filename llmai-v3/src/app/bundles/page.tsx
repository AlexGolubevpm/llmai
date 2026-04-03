"use client";

import { useEffect, useState } from "react";
import { motion } from "framer-motion";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Trash2, Save, Package, Plus } from "lucide-react";
import { toast } from "sonner";
import { PageHeader } from "@/components/layout/page-header";
import { EmptyState } from "@/components/shared/empty-state";
import { pageVariants } from "@/lib/animations";
import { cn } from "@/lib/utils";

interface Bundle {
  id: string;
  name: string;
  description: string | null;
  tags: string;
  categories: string;
  prompt: string | null;
  isDefault: boolean;
}

export default function BundlesPage() {
  const [bundles, setBundles] = useState<Bundle[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [tags, setTags] = useState("");
  const [categories, setCategories] = useState("");
  const [prompt, setPrompt] = useState("");
  const [isDefault, setIsDefault] = useState(false);

  async function fetchBundles() {
    try {
      const resp = await fetch("/api/bundles");
      const data = await resp.json();
      setBundles(data.bundles || []);
    } catch { toast.error("Не удалось загрузить бандлы"); }
  }

  useEffect(() => { fetchBundles(); }, []);

  function loadBundle(b: Bundle) {
    setSelectedId(b.id);
    setName(b.name);
    setDescription(b.description || "");
    setTags(b.tags);
    setCategories(b.categories);
    setPrompt(b.prompt || "");
    setIsDefault(b.isDefault);
  }

  function clearEditor() {
    setSelectedId(null);
    setName("");
    setDescription("");
    setTags("");
    setCategories("");
    setPrompt("");
    setIsDefault(false);
  }

  async function saveBundle() {
    if (!name.trim()) { toast.error("Введите название"); return; }
    try {
      const resp = await fetch("/api/bundles", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name: name.trim(), description, tags, categories, prompt: prompt || null, isDefault }),
      });
      const data = await resp.json();
      if (!resp.ok) throw new Error(data.error);
      fetchBundles();
      toast.success("Бандл сохранён");
    } catch (err) { toast.error((err as Error).message); }
  }

  async function deleteBundle(id: string) {
    await fetch(`/api/bundles?id=${id}`, { method: "DELETE" });
    if (selectedId === id) clearEditor();
    setBundles((b) => b.filter((x) => x.id !== id));
    toast.success("Удалён");
  }

  return (
    <motion.div {...pageVariants} className="space-y-8">
      <PageHeader
        title="Бандлы"
        description="Наборы тегов и категорий для разных ниш (транс, гей, JAV, хентай и т.д.)"
        actions={
          <Button size="sm" className="gap-2" onClick={clearEditor}>
            <Plus className="h-3.5 w-3.5" /> Новый бандл
          </Button>
        }
      />

      <div className="grid grid-cols-1 lg:grid-cols-[320px_1fr] gap-6">
        {/* Bundle list */}
        <div className="rounded-xl border bg-[var(--surface)] overflow-hidden">
          <div className="border-b px-4 py-3">
            <span className="text-xs font-medium text-[var(--text-muted)] uppercase tracking-wider">Бандлы ({bundles.length})</span>
          </div>
          {bundles.length === 0 ? (
            <EmptyState icon={<Package className="h-5 w-5" />} title="Нет бандлов" description="Создайте бандл для вашей ниши" className="py-10" />
          ) : (
            <div className="p-2 space-y-1">
              {bundles.map((b) => (
                <button
                  key={b.id}
                  onClick={() => loadBundle(b)}
                  className={cn(
                    "flex w-full items-center justify-between rounded-lg px-3 py-2.5 text-sm text-left transition-colors group",
                    selectedId === b.id
                      ? "bg-[var(--accent-blue-light)] text-[var(--accent-blue)] font-medium"
                      : "hover:bg-[var(--surface-raised)] text-[var(--text-secondary)]"
                  )}
                >
                  <div className="min-w-0">
                    <div className="flex items-center gap-2">
                      <span className="truncate">{b.name}</span>
                      {b.isDefault && (
                        <span className="rounded-full bg-blue-100 px-1.5 py-0.5 text-[10px] font-medium text-blue-600">default</span>
                      )}
                    </div>
                    {b.description && (
                      <div className="text-xs text-[var(--text-muted)] truncate mt-0.5">{b.description}</div>
                    )}
                    <div className="text-[10px] text-[var(--text-muted)] mt-0.5">
                      {b.tags.split(",").filter(Boolean).length} тегов, {b.categories.split(",").filter(Boolean).length} категорий
                    </div>
                  </div>
                  <Button
                    variant="ghost"
                    size="icon"
                    className="h-6 w-6 text-[var(--error)] opacity-0 group-hover:opacity-100 shrink-0"
                    onClick={(e) => { e.stopPropagation(); deleteBundle(b.id); }}
                    aria-label="Удалить"
                  >
                    <Trash2 className="h-3 w-3" />
                  </Button>
                </button>
              ))}
            </div>
          )}
        </div>

        {/* Editor */}
        <div className="rounded-xl border bg-[var(--surface)] p-6 space-y-5">
          <h2 className="text-[15px] font-medium">
            {selectedId ? `Редактирование: ${name}` : "Новый бандл"}
          </h2>

          <div className="grid grid-cols-2 gap-4">
            <div>
              <Label className="text-xs font-medium text-[var(--text-muted)] uppercase tracking-wider">Название</Label>
              <Input value={name} onChange={(e) => setName(e.target.value)} placeholder="Транс, Гей, JAV, Хентай..." className="mt-1.5" />
            </div>
            <div>
              <Label className="text-xs font-medium text-[var(--text-muted)] uppercase tracking-wider">Описание</Label>
              <Input value={description} onChange={(e) => setDescription(e.target.value)} placeholder="Transgender content bundle" className="mt-1.5" />
            </div>
          </div>

          <div>
            <Label className="text-xs font-medium text-[var(--text-muted)] uppercase tracking-wider">
              Теги (через запятую)
            </Label>
            <Textarea
              value={tags}
              onChange={(e) => setTags(e.target.value)}
              rows={4}
              placeholder="anal, blowjob, shemale, ladyboy, trans, tgirl, bareback, pov, big-ass, big-tits, petite, interracial, asian, latina, amateur, solo"
              className="mt-1.5 font-mono text-xs"
            />
            <p className="text-xs text-[var(--text-muted)] mt-1">
              {tags.split(",").filter((t) => t.trim()).length} тегов
            </p>
          </div>

          <div>
            <Label className="text-xs font-medium text-[var(--text-muted)] uppercase tracking-wider">
              Категории (через запятую)
            </Label>
            <Textarea
              value={categories}
              onChange={(e) => setCategories(e.target.value)}
              rows={3}
              placeholder="Transgender, Ladyboy, Shemale, TS, Tgirl, Bareback, POV, Amateur, Asian, Latina"
              className="mt-1.5 font-mono text-xs"
            />
            <p className="text-xs text-[var(--text-muted)] mt-1">
              {categories.split(",").filter((c) => c.trim()).length} категорий
            </p>
          </div>

          <div>
            <Label className="text-xs font-medium text-[var(--text-muted)] uppercase tracking-wider">
              Кастомный промпт (опционально)
            </Label>
            <Textarea
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              rows={4}
              placeholder="Оставьте пустым для дефолтного промпта. Сюда можно вписать промпт специфичный для ниши."
              className="mt-1.5 font-mono text-xs"
            />
          </div>

          <div className="flex items-center gap-4 pt-2">
            <Button onClick={saveBundle} className="gap-2">
              <Save className="h-3.5 w-3.5" /> Сохранить
            </Button>
            {selectedId && (
              <Button variant="outline" onClick={() => deleteBundle(selectedId)} className="gap-2 text-[var(--error)]">
                <Trash2 className="h-3.5 w-3.5" /> Удалить
              </Button>
            )}
          </div>
        </div>
      </div>
    </motion.div>
  );
}
