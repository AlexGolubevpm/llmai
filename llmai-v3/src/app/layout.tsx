import type { Metadata } from "next";
import "./globals.css";
import { Sidebar } from "@/components/layout/sidebar";
import { Providers } from "@/components/providers";
import { Toaster } from "@/components/ui/sonner";

export const metadata: Metadata = {
  title: "LLMAI v3.0 — Novita AI Content Platform",
  description: "Professional batch text processing with Novita AI",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="ru" className="h-full">
      <body className="min-h-full flex">
        <Providers>
          <Sidebar />
          <main className="flex-1 overflow-auto">
            <div className="mx-auto max-w-[1400px] px-6 py-8 md:px-8 md:pt-0 pt-16">
              {children}
            </div>
          </main>
          <Toaster position="bottom-right" />
        </Providers>
      </body>
    </html>
  );
}
