"use client";

import { useEffect, useState } from "react";
import { useRouter, usePathname } from "next/navigation";
import Link from "next/link";
import { useAuth } from "@/lib/auth-context";
import { Button } from "@/components/ui/button";
import { RoleBadge } from "@/components/ui/role-badge";
import { Brain, BookOpen, Home, LogOut, Settings, User, Loader2, Scissors, ChevronLeft, ChevronRight, FlaskConical, Target, Users, Shield, Database, BarChart3, FileText } from "lucide-react";

export default function DashboardLayout({ children }: { children: React.ReactNode }) {
  const router = useRouter();
  const pathname = usePathname();
  const { user, isLoading, logout } = useAuth();
  const [isCollapsed, setIsCollapsed] = useState(() => {
    if (typeof window !== "undefined") {
      const saved = localStorage.getItem("sidebar-collapsed");
      return saved !== null ? JSON.parse(saved) : false;
    }
    return false;
  });

  useEffect(() => {
    if (!isLoading && !user) {
      router.push("/login");
    }
    // Redirect students to student portal
    if (!isLoading && user && user.role === "student") {
      router.push("/student");
    }
  }, [user, isLoading, router]);

  // Sidebar durumunu kaydet
  const toggleSidebar = () => {
    const newState = !isCollapsed;
    setIsCollapsed(newState);
    localStorage.setItem("sidebar-collapsed", JSON.stringify(newState));
  };

  if (isLoading) {
    return (
      <div className="min-h-screen bg-slate-100 flex items-center justify-center">
        <Loader2 className="w-8 h-8 text-slate-600 animate-spin" />
      </div>
    );
  }

  if (!user || user.role === "student") return null;

  const handleLogout = () => {
    logout();
    router.push("/");
  };

  const navItems = [
    { href: "/dashboard", icon: Home, label: "Ana Sayfa", exact: true },
    { href: "/dashboard/courses", icon: BookOpen, label: "Dersler" },
  ];

  const testNavItems = [
    { href: "/dashboard/chunking", icon: Scissors, label: "Chunking" },
    { href: "/dashboard/ragas", icon: FlaskConical, label: "RAGAS" },
    { href: "/dashboard/ragas/test-sets", icon: FileText, label: "Test Setleri" },
    { href: "/dashboard/ragas/test-sets/generate", icon: FileText, label: "Test Sorusu Üretimi" },
    { href: "/dashboard/semantic-similarity", icon: Target, label: "Rouge/BertScore" },
    { href: "/dashboard/giskard", icon: Shield, label: "Giskard" },
    { href: "/dashboard/wandb-runs", icon: Database, label: "W&B Runs" },
    { href: "/dashboard/mteb-benchmark", icon: BarChart3, label: "MTEB Benchmark" },
  ];

  const bottomNavItems = [
    { href: "/dashboard/profile", icon: User, label: "Profil" },
    ...(user.role === "admin" ? [
      { href: "/admin/users", icon: Users, label: "Kullanıcı Yönetimi" },
      { href: "/dashboard/settings", icon: Settings, label: "Ayarlar" }
    ] : []),
  ];

  const isActive = (href: string, exact?: boolean) => {
    if (exact) return pathname === href;
    return pathname === href || pathname.startsWith(href + "/");
  };

  const activeTestHref = testNavItems
    .filter((item) => isActive(item.href))
    .sort((a, b) => b.href.length - a.href.length)[0]?.href;

  return (
    <div className="min-h-screen flex">
      {/* Sidebar - Dark Gradient */}
      <aside className={`${isCollapsed ? "w-16" : "w-64"} bg-gradient-to-b from-slate-900 via-slate-800 to-slate-900 text-white flex flex-col fixed h-full transition-all duration-300 shadow-xl`}>
        {/* Logo */}
        <div className={`p-4 ${isCollapsed ? "px-3" : "p-6"}`}>
          <Link href="/dashboard" className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-lg bg-indigo-600 flex items-center justify-center shrink-0">
              <Brain className="w-6 h-6 text-white" />
            </div>
            {!isCollapsed && (
              <div>
                <h1 className="font-bold text-lg">AkıllıRehber</h1>
                <p className="text-xs text-slate-400">RAG Eğitim Sistemi</p>
              </div>
            )}
          </Link>
        </div>

        {/* Toggle Button */}
        <button
          onClick={toggleSidebar}
          className="absolute -right-3 top-20 w-6 h-6 bg-gradient-to-r from-indigo-600 to-purple-600 hover:from-indigo-500 hover:to-purple-500 rounded-full flex items-center justify-center text-white transition-all shadow-lg shadow-indigo-500/30"
          title={isCollapsed ? "Genişlet" : "Daralt"}
        >
          {isCollapsed ? <ChevronRight className="w-4 h-4" /> : <ChevronLeft className="w-4 h-4" />}
        </button>

        {/* Main Navigation */}
        <nav className="flex-1 px-3 py-4 overflow-hidden">
          <div className="space-y-1">
            {navItems.map((item) => (
              <Link
                key={item.href}
                href={item.href}
                title={isCollapsed ? item.label : undefined}
                className={`flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-colors ${
                  isActive(item.href, item.exact)
                    ? "bg-indigo-600 text-white"
                    : "text-slate-300 hover:bg-slate-800 hover:text-white"
                } ${isCollapsed ? "justify-center" : ""}`}
              >
                <item.icon className="w-5 h-5 shrink-0" />
                {!isCollapsed && item.label}
              </Link>
            ))}
          </div>

          {/* Sistem Testleri Section */}
          <div className="mt-8 pt-6 border-t border-slate-700/50">
            {!isCollapsed && (
              <p className="px-3 text-xs font-medium text-slate-500 uppercase tracking-wider mb-3">
                Sistem Testleri
              </p>
            )}
            <div className="space-y-1">
              {testNavItems.map((item) => (
                <Link
                  key={item.href}
                  href={item.href}
                  title={isCollapsed ? item.label : undefined}
                  className={`flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-colors ${
                    (activeTestHref ? item.href === activeTestHref : isActive(item.href))
                      ? "bg-indigo-600 text-white"
                      : "text-slate-300 hover:bg-slate-800 hover:text-white"
                  } ${isCollapsed ? "justify-center" : ""}`}
                >
                  <item.icon className="w-5 h-5 shrink-0" />
                  {!isCollapsed && item.label}
                </Link>
              ))}
            </div>
          </div>

          <div className="mt-8 pt-6 border-t border-slate-700/50">
            {!isCollapsed && (
              <p className="px-3 text-xs font-medium text-slate-500 uppercase tracking-wider mb-3">
                Hesap
              </p>
            )}
            <div className="space-y-1">
              {bottomNavItems.map((item) => (
                <Link
                  key={item.href}
                  href={item.href}
                  title={isCollapsed ? item.label : undefined}
                  className={`flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-colors ${
                    isActive(item.href)
                      ? "bg-indigo-600 text-white"
                      : "text-slate-300 hover:bg-slate-800 hover:text-white"
                  } ${isCollapsed ? "justify-center" : ""}`}
                >
                  <item.icon className="w-5 h-5 shrink-0" />
                  {!isCollapsed && item.label}
                </Link>
              ))}
            </div>
          </div>
        </nav>

        {/* User Section */}
        <div className="p-4 border-t border-slate-700/50 bg-slate-900/50">
          {isCollapsed ? (
            <div className="flex flex-col items-center gap-2">
              <div className="w-9 h-9 rounded-full bg-indigo-600 flex items-center justify-center text-sm font-semibold">
                {user.full_name.charAt(0).toUpperCase()}
              </div>
              <Button
                variant="ghost"
                size="icon"
                onClick={handleLogout}
                title="Çıkış Yap"
                className="text-slate-400 hover:text-white hover:bg-slate-800"
              >
                <LogOut className="w-4 h-4" />
              </Button>
            </div>
          ) : (
            <>
              <div className="flex items-center gap-3 mb-3">
                <div className="w-9 h-9 rounded-full bg-indigo-600 flex items-center justify-center text-sm font-semibold shrink-0">
                  {user.full_name.charAt(0).toUpperCase()}
                </div>
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2 mb-1">
                    <p className="text-sm font-medium truncate">{user.full_name}</p>
                    <RoleBadge role={user.role} />
                  </div>
                </div>
              </div>
              <Button
                variant="ghost"
                size="sm"
                onClick={handleLogout}
                className="w-full justify-start text-slate-400 hover:text-white hover:bg-slate-800"
              >
                <LogOut className="w-4 h-4 mr-2" />
                Çıkış Yap
              </Button>
            </>
          )}
        </div>
      </aside>

      {/* Main Content - Light */}
      <main className={`flex-1 ${isCollapsed ? "ml-16" : "ml-64"} bg-slate-50 min-h-screen transition-all duration-300`}>
        <div className="p-8">
          {children}
        </div>
      </main>
    </div>
  );
}
