"use client";

import { motion } from "framer-motion";
import { cn } from "@/lib/utils";
import { cardVariants } from "@/lib/animations";

interface Props {
  label: string;
  value: number;
  color?: "blue" | "green" | "orange" | "red" | "default";
  index?: number;
}

const colorClasses = {
  blue: "text-blue-600",
  green: "text-green-600",
  orange: "text-orange-600",
  red: "text-red-600",
  default: "text-[var(--text-primary)]",
};

const dotColors = {
  blue: "bg-blue-500",
  green: "bg-green-500",
  orange: "bg-orange-500",
  red: "bg-red-500",
  default: "bg-stone-400",
};

export function StatCard({ label, value, color = "default", index = 0 }: Props) {
  return (
    <motion.div
      variants={cardVariants}
      initial="initial"
      animate="animate"
      transition={{ delay: index * 0.05 }}
      className="rounded-xl border bg-[var(--surface)] p-5 shadow-card transition-shadow hover:shadow-card-hover"
    >
      <div className="flex items-center gap-2 mb-3">
        <span className={cn("h-2 w-2 rounded-full", dotColors[color])} />
        <span className="text-xs font-medium text-[var(--text-muted)] uppercase tracking-wider">
          {label}
        </span>
      </div>
      <div className={cn("text-3xl font-semibold font-mono tabular-nums", colorClasses[color])}>
        {value.toLocaleString()}
      </div>
    </motion.div>
  );
}
