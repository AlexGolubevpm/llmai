import type { Metadata } from "next";
import "./globals.css";
import { Sidebar } from "@/components/sidebar";
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
    <html
      lang="ru"
      className="h-full antialiased"
      suppressHydrationWarning
    >
      <body className="min-h-full flex font-sans">
        <Providers>
          <Sidebar />
          <main className="flex-1 overflow-auto">
            <div className="container max-w-6xl mx-auto p-6">{children}</div>
          </main>
          <Toaster />
        </Providers>
      </body>
    </html>
  );
}
