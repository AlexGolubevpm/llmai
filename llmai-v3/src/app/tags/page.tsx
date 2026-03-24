"use client";

import { useEffect, useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Trash2, Plus } from "lucide-react";
import { toast } from "sonner";
import type { AllowedTag, AllowedCategory } from "@/types";

export default function TagsPage() {
  const [tags, setTags] = useState<AllowedTag[]>([]);
  const [categories, setCategories] = useState<AllowedCategory[]>([]);
  const [newTag, setNewTag] = useState("");
  const [newTagCategory, setNewTagCategory] = useState("general");
  const [newCategoryName, setNewCategoryName] = useState("");

  async function fetchAll() {
    const [tagsResp, catsResp] = await Promise.all([
      fetch("/api/tags?type=tags"),
      fetch("/api/tags?type=categories"),
    ]);
    const tagsData = await tagsResp.json();
    const catsData = await catsResp.json();
    setTags(tagsData.tags || []);
    setCategories(catsData.categories || []);
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
    fetchAll();
  }

  async function deleteCategory(id: string) {
    await fetch(`/api/tags?type=categories&id=${id}`, { method: "DELETE" });
    fetchAll();
  }

  return (
    <div className="space-y-6">
      <h1 className="text-3xl font-bold">Теги & Категории</h1>

      <Tabs defaultValue="tags">
        <TabsList>
          <TabsTrigger value="tags">Теги ({tags.length})</TabsTrigger>
          <TabsTrigger value="categories">Категории ({categories.length})</TabsTrigger>
        </TabsList>

        <TabsContent value="tags" className="space-y-4">
          <Card>
            <CardHeader><CardTitle>Добавить тег</CardTitle></CardHeader>
            <CardContent>
              <div className="flex gap-4 items-end">
                <div className="flex-1"><Label>Тег</Label><Input value={newTag} onChange={(e) => setNewTag(e.target.value)} placeholder="blonde" /></div>
                <div className="flex-1"><Label>Категория</Label><Input value={newTagCategory} onChange={(e) => setNewTagCategory(e.target.value)} placeholder="general" /></div>
                <Button onClick={addTag}><Plus className="h-4 w-4 mr-2" /> Добавить</Button>
              </div>
            </CardContent>
          </Card>
          <Card>
            <CardContent className="pt-6">
              <Table>
                <TableHeader><TableRow><TableHead>Тег</TableHead><TableHead>Категория</TableHead><TableHead></TableHead></TableRow></TableHeader>
                <TableBody>
                  {tags.map((t) => (
                    <TableRow key={t.id}>
                      <TableCell className="font-mono">{t.name}</TableCell>
                      <TableCell className="text-muted-foreground">{t.category}</TableCell>
                      <TableCell><Button variant="ghost" size="icon" className="h-8 w-8 text-red-500" onClick={() => deleteTag(t.id)}><Trash2 className="h-4 w-4" /></Button></TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="categories" className="space-y-4">
          <Card>
            <CardHeader><CardTitle>Добавить категорию</CardTitle></CardHeader>
            <CardContent>
              <div className="flex gap-4 items-end">
                <div className="flex-1"><Label>Название</Label><Input value={newCategoryName} onChange={(e) => setNewCategoryName(e.target.value)} placeholder="Anal" /></div>
                <Button onClick={addCategory}><Plus className="h-4 w-4 mr-2" /> Добавить</Button>
              </div>
            </CardContent>
          </Card>
          <Card>
            <CardContent className="pt-6">
              <Table>
                <TableHeader><TableRow><TableHead>Категория</TableHead><TableHead>Slug</TableHead><TableHead></TableHead></TableRow></TableHeader>
                <TableBody>
                  {categories.map((c) => (
                    <TableRow key={c.id}>
                      <TableCell className="font-mono">{c.name}</TableCell>
                      <TableCell className="text-muted-foreground">{c.slug}</TableCell>
                      <TableCell><Button variant="ghost" size="icon" className="h-8 w-8 text-red-500" onClick={() => deleteCategory(c.id)}><Trash2 className="h-4 w-4" /></Button></TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}
