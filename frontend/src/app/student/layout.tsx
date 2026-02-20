"use client";

import { useEffect, useState } from "react";
import { useRouter, usePathname } from "next/navigation";
import Link from "next/link";
import { useAuth } from "@/lib/auth-context";
import { Button } from "@/components/ui/button";
import { RoleBadge } from "@/components/ui/role-badge";
import { Brain, BookOpen, LogOut, User, Loader2, ChevronLeft, ChevronRight } from "lucide-react";

export default function StudentLayout({ children }: { children: React.ReactNode }) {
  const router = useRouter();
  const pathname = usePathname();
  const { user, isLoading, logout } = useAuth();
  const [isCollapsed, setIsCollapsed] = useState(() => {
    if (typeof window !== "undefined") {
      const saved = localStorage.getItem("student-sidebar-collapsed");
      return saved !== null ? JSON.parse(saved) : false;
    }
    return false;
  });

  useEffect(() => {
    if (!isLoading && !user) {
      router.push("/login");
    }
    // Redirect teachers to dashboard
    if (!isLoading && user && user.role === "teacher") {
      router.push("/dashboard");
    }
  }, [user, isLoading, router]);

  const toggleSidebar = () => {
    const newState = !isCollapsed;
    setIsCollapsed(newState);
    localStorage.setItem("student-sidebar-collapsed", JSON.stringify(newState));
  };

  if (isLoading) {
    return (
      <div className="min-h-screen bg-slate-100 flex items-center justify-center">
        <Loader2 className="w-8 h-8 text-slate-600 animate-spin" />
      </div>
    );
  }

  if (!user || user.role === "teacher") return null;

  const handleLogout = () => {
    logout();
    router.push("/login");
  };

  // Student navigation items - only Dersler, Profil, and Çıkış
  const navItems = [
    { href: "/student", icon: BookOpen, label: "Dersler", exact: true },
    { href: "/student/profile", icon: User, label: "Profil" },
  ];

  const isActive = (href: string, exact?: boolean) => {
    if (exact) return pathname === href;
    return pathname === href || pathname.startsWith(href + "/");
  };

  return (
    <div className="min-h-screen flex">
      {/* Sidebar - Dark */}
      <aside className={`${isCollapsed ? "w-16" : "w-64"} bg-slate-900 text-white flex flex-col fixed h-full transition-all duration-300`}>
        {/* Logo */}
        <div className={`p-4 ${isCollapsed ? "px-3" : "p-6"}`}>
          <Link href="/student" className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-lg bg-indigo-600 flex items-center justify-center shrink-0">
              <Brain className="w-6 h-6 text-white" />
            </div>
            {!isCollapsed && (
              <div>
                <h1 className="font-bold text-lg">AkıllıRehber</h1>
                <p className="text-xs text-slate-400">Öğrenci Portalı</p>
              </div>
            )}
          </Link>
        </div>

        {/* Toggle Button */}
        <button
          onClick={toggleSidebar}
          className="absolute -right-3 top-20 w-6 h-6 bg-slate-700 hover:bg-slate-600 rounded-full flex items-center justify-center text-slate-300 hover:text-white transition-colors shadow-md"
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
        </nav>

        {/* User Section */}
        <div className="p-4 border-t border-slate-700">
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
      <main className={`flex-1 ${isCollapsed ? "ml-16" : "ml-64"} bg-slate-100 min-h-screen transition-all duration-300`}>
        {/* Top Header Bar */}
        <header className="bg-white border-b border-slate-200 px-8 py-4 sticky top-0 z-10">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-slate-500">Hoş geldin,</p>
              <h2 className="text-lg font-semibold text-slate-900">{user.full_name}</h2>
            </div>
            <div className="flex items-center gap-3">
              <div className="text-right">
                <p className="text-xs text-slate-400">Öğrenci Portalı</p>
                <p className="text-sm font-medium text-indigo-600">AkıllıRehber</p>
              </div>
            </div>
          </div>
        </header>
        
        <div className="p-8">
          {children}
        </div>
      </main>
    </div>
  );
}
