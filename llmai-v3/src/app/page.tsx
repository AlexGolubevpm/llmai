"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { motion } from "framer-motion";
import { Button } from "@/components/ui/button";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Progress } from "@/components/ui/progress";
import { Download, Trash2, RotateCw, LayoutDashboard } from "lucide-react";
import { toast } from "sonner";
import { PageHeader } from "@/components/layout/page-header";
import { StatCard } from "@/components/shared/stat-card";
import { StatusBadge } from "@/components/shared/status-badge";
import { EmptyState } from "@/components/shared/empty-state";
import { staggerItem } from "@/lib/animations";
import { JOB_TYPE_CONFIG, formatRelativeTime } from "@/lib/constants";
import type { Job } from "@/types";

export default function DashboardPage() {
  const router = useRouter();
  const [jobs, setJobs] = useState<Job[]>([]);
  const [loading, setLoading] = useState(true);

  async function fetchJobs() {
    try {
      const resp = await fetch("/api/jobs?limit=50");
      const data = await resp.json();
      setJobs(data.jobs || []);
    } catch (err) {
      toast.error("Не удалось загрузить задачи");
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
    toast.success("Задача отменена");
    fetchJobs();
  }

  const activeJobs = jobs.filter((j) => j.status === "RUNNING" || j.status === "PENDING");
  const completedJobs = jobs.filter((j) => j.status === "COMPLETED");
  const failedJobs = jobs.filter((j) => j.status === "FAILED");

  return (
    <div className="space-y-8">
      <PageHeader
        title="Dashboard"
        description="Обзор всех задач обработки"
        actions={
          <Button
            variant="outline"
            size="sm"
            onClick={fetchJobs}
            disabled={loading}
            className="gap-2"
          >
            <RotateCw className={`h-3.5 w-3.5 ${loading ? "animate-spin" : ""}`} />
            Обновить
          </Button>
        }
      />

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard label="Всего задач" value={jobs.length} index={0} />
        <StatCard label="Активные" value={activeJobs.length} color="blue" index={1} />
        <StatCard label="Завершённые" value={completedJobs.length} color="green" index={2} />
        <StatCard label="С ошибками" value={failedJobs.length} color="red" index={3} />
      </div>

      <div className="rounded-xl border bg-[var(--surface)] shadow-card overflow-hidden">
        {loading ? (
          <div className="p-6 space-y-4">
            {[...Array(5)].map((_, i) => (
              <div key={i} className="h-12 animate-shimmer rounded-lg" />
            ))}
          </div>
        ) : jobs.length === 0 ? (
          <EmptyState
            icon={<LayoutDashboard className="h-6 w-6" />}
            title="Нет задач"
            description="Создайте первую задачу на странице Рерайт, Перевод или AI Process"
          />
        ) : (
          <Table>
            <TableHeader>
              <TableRow className="border-b hover:bg-transparent">
                <TableHead className="text-xs font-medium uppercase tracking-wider text-[var(--text-muted)]">Тип</TableHead>
                <TableHead className="text-xs font-medium uppercase tracking-wider text-[var(--text-muted)]">Модель</TableHead>
                <TableHead className="text-xs font-medium uppercase tracking-wider text-[var(--text-muted)]">Статус</TableHead>
                <TableHead className="text-xs font-medium uppercase tracking-wider text-[var(--text-muted)]">Прогресс</TableHead>
                <TableHead className="text-xs font-medium uppercase tracking-wider text-[var(--text-muted)]">Дата</TableHead>
                <TableHead className="text-xs font-medium uppercase tracking-wider text-[var(--text-muted)] w-20"></TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {jobs.map((job, i) => {
                const pct = job.totalRows > 0 ? Math.round((job.processedRows / job.totalRows) * 100) : 0;
                const typeConfig = JOB_TYPE_CONFIG[job.type] || { label: job.type, icon: "Bot" };
                return (
                  <motion.tr
                    key={job.id}
                    variants={staggerItem}
                    initial="initial"
                    animate="animate"
                    transition={{ delay: i * 0.03 }}
                    className="border-b last:border-0 hover:bg-[var(--surface-raised)] transition-colors cursor-pointer"
                    onClick={() => router.push(`/jobs/${job.id}`)}
                  >
                    <TableCell>
                      <span className="inline-flex items-center gap-1.5 rounded-md bg-[var(--surface-raised)] px-2 py-0.5 text-xs font-medium">
                        {typeConfig.label}
                      </span>
                    </TableCell>
                    <TableCell>
                      <span className="text-xs font-mono text-[var(--text-muted)] truncate max-w-[150px] block" title={(job.config as any)?.model || "—"}>
                        {((job.config as any)?.model || "—").split("/").pop()}
                      </span>
                    </TableCell>
                    <TableCell>
                      <StatusBadge status={job.status} />
                    </TableCell>
                    <TableCell>
                      <div className="flex items-center gap-3 min-w-[160px]">
                        <Progress value={pct} className="h-1.5 flex-1" />
                        <span className="text-xs font-mono tabular-nums text-[var(--text-muted)] w-16 text-right">
                          {job.processedRows}/{job.totalRows}
                        </span>
                      </div>
                    </TableCell>
                    <TableCell>
                      <span className="text-xs text-[var(--text-muted)]" title={new Date(job.createdAt).toLocaleString("ru")}>
                        {formatRelativeTime(job.createdAt)}
                      </span>
                    </TableCell>
                    <TableCell>
                      <div className="flex gap-1 justify-end">
                        {job.outputFileUrl && (
                          <a href={`/api/files/${job.id}`} download>
                            <Button variant="ghost" size="icon" className="h-7 w-7" aria-label="Скачать результат">
                              <Download className="h-3.5 w-3.5" />
                            </Button>
                          </a>
                        )}
                        {(job.status === "RUNNING" || job.status === "PENDING") && (
                          <Button
                            variant="ghost"
                            size="icon"
                            className="h-7 w-7 text-[var(--error)] hover:text-[var(--error)]"
                            onClick={() => cancelJob(job.id)}
                            aria-label="Отменить задачу"
                          >
                            <Trash2 className="h-3.5 w-3.5" />
                          </Button>
                        )}
                      </div>
                    </TableCell>
                  </motion.tr>
                );
              })}
            </TableBody>
          </Table>
        )}
      </div>
    </div>
  );
}
