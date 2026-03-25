"use client";

import { cn } from "@/lib/utils";
import { STATUS_CONFIG } from "@/lib/constants";

interface Props {
  status: string;
  className?: string;
  pulse?: boolean;
}

export function StatusBadge({ status, className, pulse }: Props) {
  const config = STATUS_CONFIG[status] || STATUS_CONFIG.PENDING;
  const showPulse = pulse || status === "RUNNING";

  return (
    <span
      className={cn(
        "inline-flex items-center gap-1.5 rounded-full px-2.5 py-0.5 text-xs font-medium",
        config.bgColor,
        config.color,
        className
      )}
    >
      <span
        className={cn(
          "h-1.5 w-1.5 rounded-full",
          config.dotColor,
          showPulse && "animate-pulse"
        )}
      />
      {config.label}
    </span>
  );
}
