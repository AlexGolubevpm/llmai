"use client";

import { useCallback, useState } from "react";
import { Card, CardContent } from "@/components/ui/card";
import { Upload } from "lucide-react";

interface Props {
  onUpload: (result: { fileUrl: string; filename: string; lineCount: number }) => void;
  accept?: string;
}

export function FileUpload({ onUpload, accept = ".csv,.txt" }: Props) {
  const [isDragging, setIsDragging] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [uploadedFile, setUploadedFile] = useState<string | null>(null);

  const handleFile = useCallback(
    async (file: File) => {
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
        setUploadedFile(data.filename);
        onUpload(data);
      } catch (err) {
        alert((err as Error).message);
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

  return (
    <Card
      className={`border-2 border-dashed transition-colors ${
        isDragging ? "border-primary bg-primary/5" : "border-muted"
      }`}
      onDragOver={(e) => {
        e.preventDefault();
        setIsDragging(true);
      }}
      onDragLeave={() => setIsDragging(false)}
      onDrop={handleDrop}
    >
      <CardContent className="flex flex-col items-center justify-center py-8">
        <Upload className="h-10 w-10 text-muted-foreground mb-3" />
        {uploading ? (
          <p className="text-sm text-muted-foreground">Загрузка...</p>
        ) : uploadedFile ? (
          <p className="text-sm text-green-500">Загружен: {uploadedFile}</p>
        ) : (
          <>
            <p className="text-sm text-muted-foreground mb-2">
              Перетащите файл сюда или нажмите для выбора
            </p>
            <label className="cursor-pointer text-sm text-primary hover:underline">
              Выбрать файл
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
          </>
        )}
      </CardContent>
    </Card>
  );
}
