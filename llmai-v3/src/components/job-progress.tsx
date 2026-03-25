"use client";

import { useEffect, useRef, useState, useCallback } from "react";
import { motion } from "framer-motion";
import { Progress } from "@/components/ui/progress";
import { StatusBadge } from "@/components/shared/status-badge";
import { formatEta } from "@/lib/constants";
import type { JobProgress as JobProgressType } from "@/types";

interface Props {
  jobId: string;
  onComplete?: () => void;
}

const MAX_RECONNECT_ATTEMPTS = 5;

export function JobProgress({ jobId, onComplete }: Props) {
  const [progress, setProgress] = useState<JobProgressType | null>(null);
  const [reconnecting, setReconnecting] = useState(false);
  const reconnectAttempts = useRef(0);
  const eventSourceRef = useRef<EventSource | null>(null);

  const connect = useCallback(() => {
    if (eventSourceRef.current) {
      eventSourceRef.current.close();
    }

    const es = new EventSource(`/api/jobs/${jobId}/stream`);
    eventSourceRef.current = es;
    setReconnecting(false);

    es.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data) as JobProgressType;
        setProgress(data);
        reconnectAttempts.current = 0;
        if (data.status === "COMPLETED" || data.status === "FAILED" || data.status === "CANCELLED") {
          es.close();
          onComplete?.();
        }
      } catch {
        // ignore parse errors
      }
    };

    es.onerror = () => {
      es.close();
      if (reconnectAttempts.current < MAX_RECONNECT_ATTEMPTS) {
        reconnectAttempts.current++;
        setReconnecting(true);
        const delay = Math.pow(2, reconnectAttempts.current) * 1000;
        setTimeout(connect, delay);
      }
    };
  }, [jobId, onComplete]);

  useEffect(() => {
    connect();
    return () => eventSourceRef.current?.close();
  }, [connect]);

  if (!progress) {
    return (
      <div className="rounded-xl border bg-[var(--surface)] p-5">
        <div className="flex items-center gap-2 text-sm text-[var(--text-muted)]">
          <motion.div
            className="h-2 w-2 rounded-full bg-blue-400"
            animate={{ opacity: [1, 0.3, 1] }}
            transition={{ duration: 1.5, repeat: Infinity }}
          />
          {reconnecting ? "Переподключение..." : "Подключение к задаче..."}
        </div>
      </div>
    );
  }

  const pct =
    progress.totalRows > 0
      ? Math.round((progress.processedRows / progress.totalRows) * 100)
      : 0;

  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      className="rounded-xl border bg-[var(--surface)] p-5 space-y-4"
    >
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          {progress.totalPasses > 1 && (
            <span className="text-sm font-medium text-[var(--text-secondary)]">
              Проход {progress.currentPass}/{progress.totalPasses}
            </span>
          )}
          <StatusBadge status={progress.status} />
        </div>
        <span className="text-2xl font-semibold font-mono tabular-nums text-[var(--text-primary)]">
          {pct}%
        </span>
      </div>

      <Progress value={pct} className="h-2" />

      <div className="flex flex-wrap gap-x-6 gap-y-1 text-sm">
        <div>
          <span className="text-[var(--text-muted)]">Строк: </span>
          <span className="font-mono tabular-nums font-medium">
            {progress.processedRows.toLocaleString()}/{progress.totalRows.toLocaleString()}
          </span>
        </div>
        {progress.eta != null && progress.eta > 0 && (
          <div>
            <span className="text-[var(--text-muted)]">ETA: </span>
            <span className="font-medium">{formatEta(progress.eta)}</span>
          </div>
        )}
        {progress.speed != null && progress.speed > 0 && (
          <div>
            <span className="text-[var(--text-muted)]">Скорость: </span>
            <span className="font-mono tabular-nums font-medium">{progress.speed} стр/с</span>
          </div>
        )}
        {progress.failedRows > 0 && (
          <div className="text-[var(--error)]">
            {progress.failedRows} ошибок (авто-retry)
          </div>
        )}
      </div>

      {reconnecting && (
        <div className="text-xs text-[var(--warning)] flex items-center gap-1">
          <motion.div
            className="h-1.5 w-1.5 rounded-full bg-orange-400"
            animate={{ opacity: [1, 0.3, 1] }}
            transition={{ duration: 1, repeat: Infinity }}
          />
          Переподключение... ({reconnectAttempts.current}/{MAX_RECONNECT_ATTEMPTS})
        </div>
      )}
    </motion.div>
  );
}
