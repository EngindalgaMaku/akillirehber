"use client";

import { useEffect, useState, useCallback } from "react";
import { useRouter, usePathname } from "next/navigation";
import Link from "next/link";
import { useAuth } from "@/lib/auth-context";
import { Button } from "@/components/ui/button";
import { RoleBadge } from "@/components/ui/role-badge";
import { Brain, BookOpen, Home, LogOut, Settings, User, Users, Loader2, ChevronLeft, ChevronRight, FlaskConical, Menu, X } from "lucide-react";

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
  const [isMobileOpen, setIsMobileOpen] = useState(false);

  useEffect(() => {
    if (!isLoading && !user) {
      router.push("/login");
    }
    if (!isLoading && user && user.role === "student") {
      router.push("/student");
    }
  }, [user, isLoading, router]);

  // Mobil menüyü route değişiminde kapat
  useEffect(() => {
    setIsMobileOpen(false);
  }, [pathname]);

  // Body scroll'u kilitle mobil menü açıkken
  useEffect(() => {
    if (isMobileOpen) {
      document.body.style.overflow = "hidden";
    } else {
      document.body.style.overflow = "";
    }
    return () => { document.body.style.overflow = ""; };
  }, [isMobileOpen]);

  const toggleSidebar = () => {
    const newState = !isCollapsed;
    setIsCollapsed(newState);
    localStorage.setItem("sidebar-collapsed", JSON.stringify(newState));
  };

  const closeMobileMenu = useCallback(() => setIsMobileOpen(false), []);

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
    { href: "/dashboard/system-tests", icon: FlaskConical, label: "Sistem Testleri" },
  ];

  const bottomNavItems = [
    { href: "/dashboard/profile", icon: User, label: "Profil" },
    ...(user.role === "admin" ? [
      { href: "/dashboard/users", icon: Users, label: "Kullanıcı Yönetimi" },
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

  const sidebarContent = (
    <>
      {/* Logo */}
      <div className={`p-4 ${isCollapsed && !isMobileOpen ? "px-3" : "p-6"}`}>
        <Link href="/dashboard" className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-lg bg-indigo-600 flex items-center justify-center shrink-0">
            <Brain className="w-6 h-6 text-white" />
          </div>
          {(!isCollapsed || isMobileOpen) && (
            <div>
              <h1 className="font-bold text-lg">AkıllıRehber</h1>
              <p className="text-xs text-slate-400">RAG Eğitim Sistemi</p>
            </div>
          )}
        </Link>
      </div>

      {/* Main Navigation */}
      <nav className="flex-1 px-3 py-4 overflow-y-auto overflow-x-hidden">
        <div className="space-y-1">
          {navItems.map((item) => (
            <Link
              key={item.href}
              href={item.href}
              title={isCollapsed && !isMobileOpen ? item.label : undefined}
              className={`flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-colors ${
                isActive(item.href, item.exact)
                  ? "bg-indigo-600 text-white"
                  : "text-slate-300 hover:bg-slate-800 hover:text-white"
              } ${isCollapsed && !isMobileOpen ? "justify-center" : ""}`}
            >
              <item.icon className="w-5 h-5 shrink-0" />
              {(!isCollapsed || isMobileOpen) && item.label}
            </Link>
          ))}
        </div>

        {/* Sistem Testleri Section */}
        <div className="mt-8 pt-6 border-t border-slate-700/50">
          <div className="space-y-1">
            {testNavItems.map((item) => (
              <Link
                key={item.href}
                href={item.href}
                title={isCollapsed && !isMobileOpen ? item.label : undefined}
                className={`flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-colors ${
                  (activeTestHref ? item.href === activeTestHref : isActive(item.href))
                    ? "bg-indigo-600 text-white"
                    : "text-slate-300 hover:bg-slate-800 hover:text-white"
                } ${isCollapsed && !isMobileOpen ? "justify-center" : ""}`}
              >
                <item.icon className="w-5 h-5 shrink-0" />
                {(!isCollapsed || isMobileOpen) && item.label}
              </Link>
            ))}
          </div>
        </div>

        <div className="mt-8 pt-6 border-t border-slate-700/50">
          {(!isCollapsed || isMobileOpen) && (
            <p className="px-3 text-xs font-medium text-slate-500 uppercase tracking-wider mb-3">
              Hesap
            </p>
          )}
          <div className="space-y-1">
            {bottomNavItems.map((item) => (
              <Link
                key={item.href}
                href={item.href}
                title={isCollapsed && !isMobileOpen ? item.label : undefined}
                className={`flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-colors ${
                  isActive(item.href)
                    ? "bg-indigo-600 text-white"
                    : "text-slate-300 hover:bg-slate-800 hover:text-white"
                } ${isCollapsed && !isMobileOpen ? "justify-center" : ""}`}
              >
                <item.icon className="w-5 h-5 shrink-0" />
                {(!isCollapsed || isMobileOpen) && item.label}
              </Link>
            ))}
          </div>
        </div>
      </nav>

      {/* User Section */}
      <div className="p-4 border-t border-slate-700/50 bg-slate-900/50">
        {isCollapsed && !isMobileOpen ? (
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
    </>
  );

  return (
    <div className="min-h-screen flex overflow-x-hidden">
      {/* Mobil Top Bar */}
      <div className="lg:hidden fixed top-0 left-0 right-0 z-40 h-14 bg-gradient-to-r from-slate-900 via-slate-800 to-slate-900 flex items-center px-4 shadow-lg">
        <button
          onClick={() => setIsMobileOpen(true)}
          className="w-10 h-10 flex items-center justify-center text-white rounded-lg hover:bg-slate-700 transition-colors"
          aria-label="Menüyü aç"
        >
          <Menu className="w-6 h-6" />
        </button>
        <Link href="/dashboard" className="flex items-center gap-2 ml-3">
          <div className="w-8 h-8 rounded-lg bg-indigo-600 flex items-center justify-center">
            <Brain className="w-5 h-5 text-white" />
          </div>
          <span className="font-bold text-white">AkıllıRehber</span>
        </Link>
      </div>

      {/* Mobil Overlay */}
      {isMobileOpen && (
        <div
          className="lg:hidden fixed inset-0 z-50 bg-black/50 backdrop-blur-sm"
          onClick={closeMobileMenu}
          aria-hidden="true"
        />
      )}

      {/* Mobil Sidebar */}
      <aside
        className={`lg:hidden fixed inset-y-0 left-0 z-50 w-72 bg-gradient-to-b from-slate-900 via-slate-800 to-slate-900 text-white flex flex-col transform transition-transform duration-300 ease-in-out ${
          isMobileOpen ? "translate-x-0" : "-translate-x-full"
        }`}
      >
        {/* Mobil Kapat Butonu */}
        <button
          onClick={closeMobileMenu}
          className="absolute top-4 right-4 w-8 h-8 flex items-center justify-center text-slate-400 hover:text-white rounded-lg hover:bg-slate-700 transition-colors"
          aria-label="Menüyü kapat"
        >
          <X className="w-5 h-5" />
        </button>
        {sidebarContent}
      </aside>

      {/* Desktop Sidebar */}
      <aside className={`hidden lg:flex ${isCollapsed ? "w-16" : "w-64"} bg-gradient-to-b from-slate-900 via-slate-800 to-slate-900 text-white flex-col fixed h-full transition-all duration-300 shadow-xl`}>
        {/* Toggle Button */}
        <button
          onClick={toggleSidebar}
          className="absolute -right-3 top-20 w-6 h-6 bg-gradient-to-r from-indigo-600 to-purple-600 hover:from-indigo-500 hover:to-purple-500 rounded-full flex items-center justify-center text-white transition-all shadow-lg shadow-indigo-500/30 z-10"
          title={isCollapsed ? "Genişlet" : "Daralt"}
        >
          {isCollapsed ? <ChevronRight className="w-4 h-4" /> : <ChevronLeft className="w-4 h-4" />}
        </button>
        {sidebarContent}
      </aside>

      {/* Main Content */}
      <main className={`flex-1 bg-slate-50 min-h-screen transition-all duration-300 overflow-x-hidden ${isCollapsed ? "lg:ml-16" : "lg:ml-64"} pt-14 lg:pt-0`}>
        <div className="p-4 sm:p-6 lg:p-8 max-w-full">
          {children}
        </div>
      </main>
    </div>
  );
}
