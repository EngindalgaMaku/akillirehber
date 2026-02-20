"use client";

import { useEffect, useState, useCallback } from "react";
import { useParams, useRouter, useSearchParams } from "next/navigation";
import { useAuth } from "@/lib/auth-context";
import { api, Course } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { toast } from "sonner";
import { ArrowLeft, Loader2, FileText, Settings, MessageSquare, Cog, BookOpen } from "lucide-react";
import { OverviewTab } from "./components/overview-tab";
import { DocumentsTab } from "./components/documents-tab";
import { ProcessingTab } from "./components/processing-tab";
import { ChatTab } from "./components/chat-tab";
import { SettingsTab } from "./components/settings-tab";

type TabType = "overview" | "documents" | "processing" | "chat" | "settings";

const TABS: { id: TabType; label: string; icon: React.ReactNode }[] = [
  { id: "overview", label: "Özet", icon: <BookOpen className="w-4 h-4" /> },
  { id: "documents", label: "Dokümanlar", icon: <FileText className="w-4 h-4" /> },
  { id: "processing", label: "İşleme", icon: <Settings className="w-4 h-4" /> },
  { id: "chat", label: "Sohbet", icon: <MessageSquare className="w-4 h-4" /> },
  { id: "settings", label: "Ayarlar", icon: <Cog className="w-4 h-4" /> },
];

export default function CourseDetailPage() {
  const params = useParams();
  const router = useRouter();
  const searchParams = useSearchParams();
  const { user } = useAuth();
  const courseId = Number(params.id);

  const [course, setCourse] = useState<Course | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [activeTab, setActiveTab] = useState<TabType>(
    (searchParams.get("tab") as TabType) || "overview"
  );

  const loadCourse = useCallback(async () => {
    try {
      const data = await api.getCourse(courseId);
      setCourse(data);
    } catch {
      toast.error("Ders yüklenirken hata oluştu");
      router.push("/dashboard/courses");
    } finally {
      setIsLoading(false);
    }
  }, [courseId, router]);

  useEffect(() => {
    loadCourse();
  }, [loadCourse]);

  const handleTabChange = (tab: TabType) => {
    setActiveTab(tab);
    router.push(`/dashboard/courses/${courseId}?tab=${tab}`, { scroll: false });
  };

  const isTeacher = user?.role === "teacher";
  const isOwner = user?.role === "admin" || (isTeacher && course?.teacher_id === user?.id);

  if (!user) return null;

  if (isLoading) {
    return (
      <div className="flex justify-center py-12">
        <Loader2 className="w-6 h-6 text-slate-400 animate-spin" />
      </div>
    );
  }

  if (!course) return null;

  return (
    <div className="space-y-6 overflow-x-hidden">
      {/* Header with Gradient */}
      <div className="bg-gradient-to-r from-indigo-600 via-purple-600 to-indigo-700 rounded-xl px-4 sm:px-6 py-4 shadow-lg">
        <div className="flex items-center gap-3 sm:gap-4">
          <Button
            variant="ghost"
            size="icon"
            onClick={() => router.push("/dashboard/courses")}
            className="text-white/80 hover:text-white hover:bg-white/10 shrink-0 sm:hidden"
          >
            <ArrowLeft className="w-5 h-5" />
          </Button>
          <Button
            variant="ghost"
            size="sm"
            onClick={() => router.push("/dashboard/courses")}
            className="text-white/80 hover:text-white hover:bg-white/10 shrink-0 hidden sm:flex"
          >
            <ArrowLeft className="w-4 h-4 mr-1" />
            Geri
          </Button>
          <div className="w-px h-8 bg-white/20 hidden sm:block" />
          <div className="w-9 h-9 sm:w-10 sm:h-10 bg-white/20 backdrop-blur-sm rounded-lg flex items-center justify-center shrink-0">
            <BookOpen className="w-4 h-4 sm:w-5 sm:h-5 text-white" />
          </div>
          <div className="min-w-0">
            <h1 className="text-base sm:text-xl font-bold text-white truncate">{course.name}</h1>
            {course.description && (
              <p className="text-white/70 text-xs sm:text-sm truncate">{course.description}</p>
            )}
          </div>
        </div>
      </div>

      {/* Tabs with Card Style */}
      <div className="bg-white rounded-xl shadow-sm border border-slate-200 overflow-hidden">
        <div className="bg-gradient-to-r from-slate-50 to-slate-100 border-b border-slate-200 px-2 sm:px-4">
          <nav className="flex gap-0.5 sm:gap-1 overflow-x-auto scrollbar-hide -mb-px" role="tablist">
            {TABS.map((tab) => (
              <button
                key={tab.id}
                onClick={() => handleTabChange(tab.id)}
                role="tab"
                aria-selected={activeTab === tab.id}
                className={`flex items-center gap-1.5 sm:gap-2 py-3 sm:py-3.5 px-2.5 sm:px-4 text-xs sm:text-sm font-medium transition-all relative whitespace-nowrap shrink-0 ${
                  activeTab === tab.id
                    ? "text-indigo-600"
                    : "text-slate-500 hover:text-slate-700"
                }`}
              >
                <span className={`p-1 sm:p-1.5 rounded-lg transition-colors ${
                  activeTab === tab.id 
                    ? "bg-indigo-100 text-indigo-600" 
                    : "bg-transparent"
                }`}>
                  {tab.icon}
                </span>
                <span className="hidden sm:inline">{tab.label}</span>
                <span className="sm:hidden">{tab.label}</span>
                {activeTab === tab.id && (
                  <span className="absolute bottom-0 left-0 right-0 h-0.5 bg-indigo-600 rounded-t-full" />
                )}
              </button>
            ))}
          </nav>
        </div>

        {/* Tab Content */}
        <div className="p-3 sm:p-4 lg:p-6">
          {activeTab === "overview" && (
            <OverviewTab courseId={courseId} isOwner={isOwner} onTabChange={(tab) => handleTabChange(tab as TabType)} />
          )}
          {activeTab === "documents" && (
            <DocumentsTab courseId={courseId} isOwner={isOwner} />
          )}
          {activeTab === "processing" && (
            <ProcessingTab courseId={courseId} isOwner={isOwner} />
          )}
          {activeTab === "chat" && (
            <ChatTab courseId={courseId} />
          )}
          {activeTab === "settings" && (
            <SettingsTab courseId={courseId} isOwner={isOwner} courseName={course.name} />
          )}
        </div>
      </div>
    </div>
  );
}
