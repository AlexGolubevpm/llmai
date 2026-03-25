import type { JobStatus, JobType } from "@/types";

export const STATUS_CONFIG: Record<
  string,
  { label: string; color: string; bgColor: string; dotColor: string }
> = {
  PENDING: {
    label: "Ожидание",
    color: "text-stone-600",
    bgColor: "bg-stone-100",
    dotColor: "bg-stone-400",
  },
  RUNNING: {
    label: "В процессе",
    color: "text-blue-700",
    bgColor: "bg-blue-50",
    dotColor: "bg-blue-500",
  },
  COMPLETED: {
    label: "Завершено",
    color: "text-green-700",
    bgColor: "bg-green-50",
    dotColor: "bg-green-500",
  },
  FAILED: {
    label: "Ошибка",
    color: "text-red-700",
    bgColor: "bg-red-50",
    dotColor: "bg-red-500",
  },
  CANCELLED: {
    label: "Отменено",
    color: "text-orange-700",
    bgColor: "bg-orange-50",
    dotColor: "bg-orange-400",
  },
};

export const JOB_TYPE_CONFIG: Record<
  string,
  { label: string; icon: string }
> = {
  REWRITE: { label: "Рерайт", icon: "PenLine" },
  TRANSLATE: { label: "Перевод", icon: "Globe" },
  POSTPROCESS: { label: "Очистка", icon: "Sparkles" },
  AI_PROCESS: { label: "AI Process", icon: "Bot" },
};

export function formatRelativeTime(date: string | Date): string {
  const now = Date.now();
  const then = new Date(date).getTime();
  const diffSec = Math.round((now - then) / 1000);

  if (diffSec < 60) return "только что";
  if (diffSec < 3600) return `${Math.floor(diffSec / 60)} мин назад`;
  if (diffSec < 86400) return `${Math.floor(diffSec / 3600)} ч назад`;
  if (diffSec < 604800) return `${Math.floor(diffSec / 86400)} дн назад`;
  return new Date(date).toLocaleDateString("ru");
}

export function formatEta(seconds?: number): string {
  if (!seconds || seconds <= 0) return "";
  if (seconds < 60) return `~${seconds} сек`;
  const min = Math.floor(seconds / 60);
  const sec = Math.round(seconds % 60);
  return `~${min}:${sec.toString().padStart(2, "0")}`;
}
