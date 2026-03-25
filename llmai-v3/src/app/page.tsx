"use client";

import { useEffect, useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Download, Trash2, RotateCw } from "lucide-react";
import type { Job } from "@/types";

const statusColors: Record<string, string> = {
  PENDING: "bg-yellow-500",
  RUNNING: "bg-blue-500",
  COMPLETED: "bg-green-500",
  FAILED: "bg-red-500",
  CANCELLED: "bg-gray-500",
};

export default function DashboardPage() {
  const [jobs, setJobs] = useState<Job[]>([]);
  const [loading, setLoading] = useState(true);

  async function fetchJobs() {
    setLoading(true);
    try {
      const resp = await fetch("/api/jobs?limit=50");
      const data = await resp.json();
      setJobs(data.jobs || []);
    } catch {
      /* ignore */
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    fetchJobs();
    const interval = setInterval(fetchJobs, 5000);
    return () => clearInterval(interval);
  }, []);

  async function cancelJob(id: string) {
    await fetch(`/api/jobs/${id}`, { method: "DELETE" });
    fetchJobs();
  }

  const activeJobs = jobs.filter((j) => j.status === "RUNNING" || j.status === "PENDING");
  const completedJobs = jobs.filter((j) => j.status === "COMPLETED");
  const failedJobs = jobs.filter((j) => j.status === "FAILED");

  return (
    <div className="space-y-6">
      <h1 className="text-3xl font-bold">Dashboard</h1>

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-muted-foreground">Всего задач</CardTitle>
          </CardHeader>
          <CardContent><div className="text-2xl font-bold">{jobs.length}</div></CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-muted-foreground">Активные</CardTitle>
          </CardHeader>
          <CardContent><div className="text-2xl font-bold text-blue-500">{activeJobs.length}</div></CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-muted-foreground">Завершённые</CardTitle>
          </CardHeader>
          <CardContent><div className="text-2xl font-bold text-green-500">{completedJobs.length}</div></CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-muted-foreground">С ошибками</CardTitle>
          </CardHeader>
          <CardContent><div className="text-2xl font-bold text-red-500">{failedJobs.length}</div></CardContent>
        </Card>
      </div>

      <Card>
        <CardHeader className="flex flex-row items-center justify-between">
          <CardTitle>Задачи</CardTitle>
          <Button variant="outline" size="sm" onClick={fetchJobs} disabled={loading}>
            <RotateCw className={`h-4 w-4 mr-2 ${loading ? "animate-spin" : ""}`} />
            Обновить
          </Button>
        </CardHeader>
        <CardContent>
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Тип</TableHead>
                <TableHead>Статус</TableHead>
                <TableHead>Прогресс</TableHead>
                <TableHead>Создана</TableHead>
                <TableHead>Действия</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {jobs.map((job) => {
                const pct = job.totalRows > 0 ? Math.round((job.processedRows / job.totalRows) * 100) : 0;
                return (
                  <TableRow key={job.id}>
                    <TableCell><Badge variant="outline">{job.type}</Badge></TableCell>
                    <TableCell><Badge className={statusColors[job.status]}>{job.status}</Badge></TableCell>
                    <TableCell>
                      {job.totalPasses > 1 && <span className="text-xs text-muted-foreground mr-2">Pass {job.currentPass}/{job.totalPasses}</span>}
                      {job.processedRows}/{job.totalRows} ({pct}%)
                    </TableCell>
                    <TableCell className="text-sm text-muted-foreground">{new Date(job.createdAt).toLocaleString("ru")}</TableCell>
                    <TableCell>
                      <div className="flex gap-1">
                        {job.outputFileUrl && (
                          <a href={`/api/files/${job.id}`} download>
                            <Button variant="ghost" size="icon" className="h-8 w-8">
                              <Download className="h-4 w-4" />
                            </Button>
                          </a>
                        )}
                        {(job.status === "RUNNING" || job.status === "PENDING") && (
                          <Button variant="ghost" size="icon" className="h-8 w-8 text-red-500" onClick={() => cancelJob(job.id)}>
                            <Trash2 className="h-4 w-4" />
                          </Button>
                        )}
                      </div>
                    </TableCell>
                  </TableRow>
                );
              })}
              {jobs.length === 0 && (
                <TableRow>
                  <TableCell colSpan={5} className="text-center text-muted-foreground py-8">Нет задач</TableCell>
                </TableRow>
              )}
            </TableBody>
          </Table>
        </CardContent>
      </Card>
    </div>
  );
}
