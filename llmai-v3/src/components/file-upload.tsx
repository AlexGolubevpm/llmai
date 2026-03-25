"use client";

import { useCallback, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Upload, File, X } from "lucide-react";
import { toast } from "sonner";
import { cn } from "@/lib/utils";

interface UploadResult {
  fileUrl: string;
  filename: string;
  lineCount: number;
  size: number;
}

interface Props {
  onUpload: (result: UploadResult) => void;
  accept?: string;
}

function formatSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

export function FileUpload({ onUpload, accept = ".csv,.txt" }: Props) {
  const [isDragging, setIsDragging] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [uploadedFile, setUploadedFile] = useState<UploadResult | null>(null);

  const handleFile = useCallback(
    async (file: globalThis.File) => {
      setUploading(true);
      try {
        const formData = new FormData();
        formData.append("file", file);
        const resp = await fetch("/api/files/upload", {
          method: "POST",
          body: formData,
        });
        if (!resp.ok) {
          const err = await resp.json();
          throw new Error(err.error || "Upload failed");
        }
        const data = await resp.json();
        setUploadedFile(data);
        onUpload(data);
        toast.success(`${file.name} загружен`);
      } catch (err) {
        toast.error((err as Error).message);
      } finally {
        setUploading(false);
      }
    },
    [onUpload]
  );

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setIsDragging(false);
      const file = e.dataTransfer.files[0];
      if (file) handleFile(file);
    },
    [handleFile]
  );

  function clearFile() {
    setUploadedFile(null);
  }

  return (
    <div
      className={cn(
        "relative rounded-xl border-2 border-dashed transition-all duration-200",
        isDragging
          ? "border-[var(--accent-blue)] bg-[var(--accent-blue-light)] scale-[1.005]"
          : "border-[var(--border)] hover:border-[var(--border-hover)]"
      )}
      onDragOver={(e) => {
        e.preventDefault();
        setIsDragging(true);
      }}
      onDragLeave={() => setIsDragging(false)}
      onDrop={handleDrop}
    >
      <AnimatePresence mode="wait">
        {uploadedFile ? (
          <motion.div
            key="file"
            initial={{ opacity: 0, y: 4 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0 }}
            className="flex items-center gap-3 p-4"
          >
            <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-[var(--accent-blue-light)]">
              <File className="h-5 w-5 text-[var(--accent-blue)]" />
            </div>
            <div className="flex-1 min-w-0">
              <p className="text-sm font-medium truncate">{uploadedFile.filename}</p>
              <p className="text-xs text-[var(--text-muted)]">
                {formatSize(uploadedFile.size)} &middot; {uploadedFile.lineCount.toLocaleString()} строк
              </p>
            </div>
            <button
              onClick={clearFile}
              className="rounded-lg p-1.5 text-[var(--text-muted)] hover:bg-[var(--surface-raised)] transition-colors"
              aria-label="Удалить файл"
            >
              <X className="h-4 w-4" />
            </button>
          </motion.div>
        ) : (
          <motion.div
            key="upload"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="flex flex-col items-center justify-center py-10"
          >
            {uploading ? (
              <div className="flex items-center gap-2 text-sm text-[var(--text-secondary)]">
                <motion.div
                  animate={{ rotate: 360 }}
                  transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
                >
                  <Upload className="h-5 w-5" />
                </motion.div>
                Загрузка...
              </div>
            ) : (
              <>
                <Upload className="h-8 w-8 text-[var(--text-muted)] mb-3" />
                <p className="text-sm text-[var(--text-secondary)] mb-1">
                  Перетащите CSV/TXT файл сюда
                </p>
                <label className="cursor-pointer text-sm font-medium text-[var(--accent-blue)] hover:underline">
                  или нажмите для выбора
                  <input
                    type="file"
                    accept={accept}
                    className="hidden"
                    onChange={(e) => {
                      const file = e.target.files?.[0];
                      if (file) handleFile(file);
                    }}
                  />
                </label>
                <p className="mt-2 text-xs text-[var(--text-muted)]">Макс. 100MB</p>
              </>
            )}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
