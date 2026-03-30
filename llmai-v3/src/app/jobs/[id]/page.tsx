"use client";

import { useEffect, useState, useCallback } from "react";
import { useParams, useRouter } from "next/navigation";
import { motion } from "framer-motion";
import { Button } from "@/components/ui/button";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { PageHeader } from "@/components/layout/page-header";
import { StatusBadge } from "@/components/shared/status-badge";
import { JobProgress } from "@/components/job-progress";
import { JOB_TYPE_CONFIG, formatRelativeTime } from "@/lib/constants";
import { pageVariants } from "@/lib/animations";
import { ArrowLeft, Download, XCircle, RotateCw, AlertTriangle } from "lucide-react";
import { toast } from "sonner";
import type { Job, ErrorLogEntry } from "@/types";

export default function JobDetailPage() {
  const params = useParams();
  const router = useRouter();
  const jobId = params.id as string;
  const [job, setJob] = useState<Job | null>(null);
  const [loading, setLoading] = useState(true);

  const fetchJob = useCallback(async () => {
    try {
      const resp = await fetch(`/api/jobs/${jobId}`);
      const data = await resp.json();
      if (data.job) setJob(data.job);
    } catch {
      toast.error("Не удалось загрузить задачу");
    } finally {
      setLoading(false);
    }
  }, [jobId]);

  useEffect(() => {
    fetchJob();
    const interval = setInterval(fetchJob, 3000);
    return () => clearInterval(interval);
  }, [fetchJob]);

  async function cancelJob() {
    await fetch(`/api/jobs/${jobId}`, { method: "DELETE" });
    toast.success("Задача отменена");
    fetchJob();
  }

  if (loading) {
    return (
      <div className="space-y-6">
        <div className="h-8 w-48 animate-shimmer rounded-lg" />
        <div className="h-40 animate-shimmer rounded-xl" />
        <div className="h-60 animate-shimmer rounded-xl" />
      </div>
    );
  }

  if (!job) {
    return (
      <div className="text-center py-20 text-[var(--text-muted)]">
        Задача не найдена
      </div>
    );
  }

  const typeConfig = JOB_TYPE_CONFIG[job.type] || { label: job.type };
  const pct = job.totalRows > 0 ? Math.round((job.processedRows / job.totalRows) * 100) : 0;
  const isActive = job.status === "RUNNING" || job.status === "PENDING";
  const errorLog = (job.errorLog || []) as ErrorLogEntry[];

  return (
    <motion.div {...pageVariants} className="space-y-6">
      <PageHeader
        title={`${typeConfig.label} — Задача`}
        description={`ID: ${job.id}`}
        actions={
          <div className="flex gap-2">
            <Button variant="outline" size="sm" onClick={() => router.push("/")} className="gap-2">
              <ArrowLeft className="h-3.5 w-3.5" /> Dashboard
            </Button>
            <Button variant="outline" size="sm" onClick={fetchJob} className="gap-2">
              <RotateCw className="h-3.5 w-3.5" /> Обновить
            </Button>
          </div>
        }
      />

      {/* Live progress for active jobs */}
      {isActive && <JobProgress jobId={jobId} onComplete={fetchJob} />}

      {/* Job info card */}
      <div className="rounded-xl border bg-[var(--surface)] p-6">
        <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
          <div>
            <p className="text-xs text-[var(--text-muted)] uppercase tracking-wider mb-1">Статус</p>
            <StatusBadge status={job.status} />
          </div>
          <div>
            <p className="text-xs text-[var(--text-muted)] uppercase tracking-wider mb-1">Тип</p>
            <span className="text-sm font-medium">{typeConfig.label}</span>
          </div>
          <div>
            <p className="text-xs text-[var(--text-muted)] uppercase tracking-wider mb-1">Прогресс</p>
            <span className="text-sm font-mono tabular-nums">
              {job.processedRows}/{job.totalRows} ({pct}%)
            </span>
          </div>
          <div>
            <p className="text-xs text-[var(--text-muted)] uppercase tracking-wider mb-1">Проход</p>
            <span className="text-sm font-mono">{job.currentPass}/{job.totalPasses}</span>
          </div>
          <div>
            <p className="text-xs text-[var(--text-muted)] uppercase tracking-wider mb-1">Создана</p>
            <span className="text-sm">{formatRelativeTime(job.createdAt)}</span>
          </div>
          <div>
            <p className="text-xs text-[var(--text-muted)] uppercase tracking-wider mb-1">Запущена</p>
            <span className="text-sm">{job.startedAt ? formatRelativeTime(job.startedAt) : "—"}</span>
          </div>
          <div>
            <p className="text-xs text-[var(--text-muted)] uppercase tracking-wider mb-1">Завершена</p>
            <span className="text-sm">{job.completedAt ? formatRelativeTime(job.completedAt) : "—"}</span>
          </div>
          <div>
            <p className="text-xs text-[var(--text-muted)] uppercase tracking-wider mb-1">Ошибки</p>
            <span className={`text-sm font-mono ${job.failedRows > 0 ? "text-[var(--error)]" : ""}`}>
              {job.failedRows}
            </span>
          </div>
        </div>
      </div>

      {/* Actions */}
      <div className="flex gap-3">
        {job.outputFileUrl && (
          <a href={`/api/files/${job.id}`} download>
            <Button className="gap-2">
              <Download className="h-4 w-4" /> Скачать результат
            </Button>
          </a>
        )}
        {isActive && (
          <Button variant="outline" onClick={cancelJob} className="gap-2 text-[var(--error)]">
            <XCircle className="h-4 w-4" /> Остановить
          </Button>
        )}
      </div>

      {/* Config */}
      <div className="rounded-xl border bg-[var(--surface)] p-6">
        <h3 className="text-[15px] font-medium mb-4">Параметры задачи</h3>
        <pre className="text-xs font-mono bg-[var(--surface-raised)] p-4 rounded-lg overflow-auto max-h-48">
          {JSON.stringify(job.config, null, 2)}
        </pre>
      </div>

      {/* Error log — deduplicated */}
      {errorLog.length > 0 && (() => {
        // Group errors by message to show unique errors with count
        const grouped = new Map<string, { step: string; error: string; count: number; rows: number[] }>();
        for (const err of errorLog) {
          const key = `${(err as any).step || ""}::${err.error}`;
          const existing = grouped.get(key);
          if (existing) {
            existing.count++;
            if (existing.rows.length < 5) existing.rows.push(err.row);
          } else {
            grouped.set(key, { step: (err as any).step || "", error: err.error, count: 1, rows: [err.row] });
          }
        }
        const uniqueErrors = [...grouped.values()].sort((a, b) => b.count - a.count);

        return (
        <div className="rounded-xl border border-red-200 bg-[var(--error-light)] p-6">
          <div className="flex items-center gap-2 mb-4">
            <AlertTriangle className="h-4 w-4 text-[var(--error)]" />
            <h3 className="text-[15px] font-medium text-[var(--error)]">
              Ошибки: {errorLog.length} всего, {uniqueErrors.length} уникальных
            </h3>
          </div>
          <div className="max-h-96 overflow-auto rounded-lg border bg-[var(--surface)]">
            <Table>
              <TableHeader>
                <TableRow className="hover:bg-transparent">
                  <TableHead className="text-xs w-16">Кол-во</TableHead>
                  {uniqueErrors.some((e) => e.step) && <TableHead className="text-xs w-24">Шаг</TableHead>}
                  <TableHead className="text-xs">Ошибка</TableHead>
                  <TableHead className="text-xs w-28">Строки</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {uniqueErrors.map((err, i) => (
                  <TableRow key={i} className="hover:bg-[var(--surface-raised)]">
                    <TableCell>
                      <span className="inline-flex items-center justify-center rounded-full bg-red-100 px-2 py-0.5 text-xs font-medium text-[var(--error)]">
                        {err.count}x
                      </span>
                    </TableCell>
                    {uniqueErrors.some((e) => e.step) && (
                      <TableCell className="font-mono text-xs text-[var(--text-muted)]">
                        {err.step || "—"}
                      </TableCell>
                    )}
                    <TableCell className="text-xs text-[var(--error)] max-w-lg">
                      <div className="truncate" title={err.error}>{err.error}</div>
                    </TableCell>
                    <TableCell className="font-mono text-xs text-[var(--text-muted)]">
                      {err.rows.join(", ")}{err.count > 5 ? "..." : ""}
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </div>
        </div>
        );
      })()}
    </motion.div>
  );
}
