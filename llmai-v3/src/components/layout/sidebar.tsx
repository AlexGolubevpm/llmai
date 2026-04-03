"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { cn } from "@/lib/utils";
import { motion, AnimatePresence } from "framer-motion";
import { useState } from "react";
import {
  LayoutDashboard,
  PenLine,
  Globe,
  Sparkles,
  Bot,
  Ban,
  Tags,
  Settings,
  Package,
  ChevronLeft,
  ChevronRight,
  Menu,
  X,
} from "lucide-react";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Button } from "@/components/ui/button";
import { Sheet, SheetContent } from "@/components/ui/sheet";

const navGroups = [
  {
    label: "Обработка",
    items: [
      { href: "/", label: "Dashboard", icon: LayoutDashboard },
      { href: "/rewrite", label: "Рерайт", icon: PenLine },
      { href: "/translate", label: "Перевод", icon: Globe },
      { href: "/postprocess", label: "Постобработка", icon: Sparkles },
      { href: "/ai-process", label: "AI Process 3.0", icon: Bot },
    ],
  },
  {
    label: "Управление",
    items: [
      { href: "/bundles", label: "Бандлы", icon: Package },
      { href: "/stopwords", label: "Стоп-слова", icon: Ban },
      { href: "/tags", label: "Теги & Категории", icon: Tags },
      { href: "/settings", label: "Настройки", icon: Settings },
    ],
  },
];

function NavItem({
  href,
  label,
  icon: Icon,
  isActive,
  collapsed,
}: {
  href: string;
  label: string;
  icon: React.ComponentType<{ className?: string }>;
  isActive: boolean;
  collapsed: boolean;
}) {
  return (
    <Link
      href={href}
      className={cn(
        "relative flex items-center gap-3 rounded-lg px-3 py-2 text-sm font-medium transition-all duration-150",
        collapsed && "justify-center px-2",
        isActive
          ? "bg-[var(--accent-blue-light)] text-[var(--accent-blue)]"
          : "text-[var(--text-secondary)] hover:bg-[var(--surface-raised)] hover:text-[var(--text-primary)]"
      )}
      title={collapsed ? label : undefined}
      aria-label={label}
    >
      {isActive && (
        <motion.div
          layoutId="activeNav"
          className="absolute left-0 top-1/2 -translate-y-1/2 h-6 w-[3px] rounded-full bg-[var(--accent-blue)]"
          transition={{ type: "spring", stiffness: 300, damping: 30 }}
        />
      )}
      <Icon className={cn("h-[18px] w-[18px] shrink-0", isActive && "text-[var(--accent-blue)]")} />
      <AnimatePresence>
        {!collapsed && (
          <motion.span
            initial={{ opacity: 0, width: 0 }}
            animate={{ opacity: 1, width: "auto" }}
            exit={{ opacity: 0, width: 0 }}
            transition={{ duration: 0.15 }}
            className="overflow-hidden whitespace-nowrap"
          >
            {label}
          </motion.span>
        )}
      </AnimatePresence>
    </Link>
  );
}

function SidebarContent({ collapsed, onCollapse }: { collapsed: boolean; onCollapse?: () => void }) {
  const pathname = usePathname();

  return (
    <div className="flex h-full flex-col">
      <div className={cn("flex h-14 items-center border-b px-4", collapsed && "justify-center px-2")}>
        <Link href="/" className="flex items-center gap-2">
          <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-[var(--accent-blue)] text-white font-bold text-sm">
            L
          </div>
          <AnimatePresence>
            {!collapsed && (
              <motion.span
                initial={{ opacity: 0, width: 0 }}
                animate={{ opacity: 1, width: "auto" }}
                exit={{ opacity: 0, width: 0 }}
                className="font-semibold text-[15px] overflow-hidden whitespace-nowrap"
              >
                LLMAI v3.0
              </motion.span>
            )}
          </AnimatePresence>
        </Link>
      </div>

      <ScrollArea className="flex-1 px-3 py-4">
        <nav className="space-y-6">
          {navGroups.map((group) => (
            <div key={group.label}>
              <AnimatePresence>
                {!collapsed && (
                  <motion.p
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    exit={{ opacity: 0 }}
                    className="mb-2 px-3 text-[11px] font-medium uppercase tracking-wider text-[var(--text-muted)]"
                  >
                    {group.label}
                  </motion.p>
                )}
              </AnimatePresence>
              <div className="space-y-1">
                {group.items.map((item) => {
                  const isActive =
                    pathname === item.href ||
                    (item.href !== "/" && pathname.startsWith(item.href));
                  return (
                    <NavItem
                      key={item.href}
                      {...item}
                      isActive={isActive}
                      collapsed={collapsed}
                    />
                  );
                })}
              </div>
            </div>
          ))}
        </nav>
      </ScrollArea>

      {onCollapse && (
        <div className="border-t p-3">
          <Button
            variant="ghost"
            size="sm"
            onClick={onCollapse}
            className="w-full justify-center text-[var(--text-muted)]"
            aria-label={collapsed ? "Развернуть меню" : "Свернуть меню"}
          >
            {collapsed ? <ChevronRight className="h-4 w-4" /> : <ChevronLeft className="h-4 w-4" />}
          </Button>
        </div>
      )}
    </div>
  );
}

function MobileSheet() {
  const [open, setOpen] = useState(false);
  return (
    <>
      <Button variant="ghost" size="icon" aria-label="Открыть меню" onClick={() => setOpen(true)}>
        <Menu className="h-5 w-5" />
      </Button>
      <Sheet open={open} onOpenChange={setOpen}>
        <SheetContent side="left" className="w-[240px] p-0">
          <SidebarContent collapsed={false} />
        </SheetContent>
      </Sheet>
    </>
  );
}

export function Sidebar() {
  const [collapsed, setCollapsed] = useState(false);

  return (
    <>
      {/* Desktop sidebar */}
      <motion.div
        className="hidden md:flex h-screen sticky top-0 flex-col border-r bg-[var(--surface)]"
        animate={{ width: collapsed ? 64 : 240 }}
        transition={{ type: "spring", stiffness: 300, damping: 30 }}
      >
        <SidebarContent collapsed={collapsed} onCollapse={() => setCollapsed(!collapsed)} />
      </motion.div>

      {/* Mobile hamburger + sheet */}
      <div className="md:hidden fixed top-0 left-0 z-50 flex h-14 w-full items-center border-b bg-[var(--surface)] px-4">
        <MobileSheet />
        <Link href="/" className="ml-3 flex items-center gap-2">
          <div className="flex h-7 w-7 items-center justify-center rounded-lg bg-[var(--accent-blue)] text-white font-bold text-xs">
            L
          </div>
          <span className="font-semibold text-sm">LLMAI</span>
        </Link>
      </div>
    </>
  );
}
