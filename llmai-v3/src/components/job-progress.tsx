"use client";

import { useEffect, useState } from "react";
import { Progress } from "@/components/ui/progress";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import type { JobProgress as JobProgressType } from "@/types";

interface Props {
  jobId: string;
  onComplete?: () => void;
}

export function JobProgress({ jobId, onComplete }: Props) {
  const [progress, setProgress] = useState<JobProgressType | null>(null);

  useEffect(() => {
    const eventSource = new EventSource(`/api/jobs/${jobId}/stream`);

    eventSource.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data) as JobProgressType;
        setProgress(data);
        if (data.status === "COMPLETED" || data.status === "FAILED") {
          eventSource.close();
          onComplete?.();
        }
      } catch {
        // ignore
      }
    };

    eventSource.onerror = () => {
      eventSource.close();
    };

    return () => eventSource.close();
  }, [jobId, onComplete]);

  if (!progress) {
    return (
      <Card>
        <CardContent className="pt-6">
          <p className="text-sm text-muted-foreground">Подключение к задаче...</p>
        </CardContent>
      </Card>
    );
  }

  const pct =
    progress.totalRows > 0
      ? Math.round((progress.processedRows / progress.totalRows) * 100)
      : 0;

  const statusColors: Record<string, string> = {
    PENDING: "bg-yellow-500",
    RUNNING: "bg-blue-500",
    COMPLETED: "bg-green-500",
    FAILED: "bg-red-500",
    CANCELLED: "bg-gray-500",
  };

  function formatEta(seconds?: number) {
    if (!seconds || seconds <= 0) return "";
    if (seconds < 60) return `~${seconds} сек`;
    return `~${Math.round(seconds / 60)} мин ${Math.round(seconds % 60)} сек`;
  }

  return (
    <Card>
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <CardTitle className="text-sm font-medium">Прогресс задачи</CardTitle>
          <Badge className={statusColors[progress.status]}>{progress.status}</Badge>
        </div>
      </CardHeader>
      <CardContent className="space-y-3">
        <Progress value={pct} className="h-3" />
        <div className="flex flex-wrap gap-4 text-sm text-muted-foreground">
          {progress.totalPasses > 1 && (
            <span>
              Проход {progress.currentPass}/{progress.totalPasses}
            </span>
          )}
          <span>
            Строка {progress.processedRows.toLocaleString()}/
            {progress.totalRows.toLocaleString()}
          </span>
          <span className="font-medium text-foreground">{pct}%</span>
          {progress.eta && progress.eta > 0 && (
            <span>{formatEta(progress.eta)}</span>
          )}
          {progress.speed && progress.speed > 0 && (
            <span>{progress.speed} строк/сек</span>
          )}
          {progress.failedRows > 0 && (
            <span className="text-red-400">
              Ошибки: {progress.failedRows}
            </span>
          )}
        </div>
      </CardContent>
    </Card>
  );
}
