"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { useAuth } from "@/lib/auth-context";
import { api, Course } from "@/lib/api";
import { toast } from "sonner";
import { BookOpen, ArrowRight, Loader2, MessageSquare, Sparkles } from "lucide-react";

export default function StudentCoursesPage() {
  const router = useRouter();
  const { user } = useAuth();
  const [courses, setCourses] = useState<Course[]>([]);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    loadCourses();
  }, []);

  const loadCourses = async () => {
    try {
      const data = await api.getCourses();
      setCourses(data);
    } catch {
      toast.error("Dersler yüklenirken hata oluştu");
    } finally {
      setIsLoading(false);
    }
  };

  const handleCourseSelect = (courseId: number) => {
    router.push(`/student/chat/${courseId}`);
  };

  if (!user) return null;

  // Renk paleti
  const colors = [
    { bg: "bg-indigo-50", border: "border-indigo-200", icon: "bg-indigo-100", iconText: "text-indigo-600", hover: "hover:border-indigo-400" },
    { bg: "bg-emerald-50", border: "border-emerald-200", icon: "bg-emerald-100", iconText: "text-emerald-600", hover: "hover:border-emerald-400" },
    { bg: "bg-amber-50", border: "border-amber-200", icon: "bg-amber-100", iconText: "text-amber-600", hover: "hover:border-amber-400" },
    { bg: "bg-rose-50", border: "border-rose-200", icon: "bg-rose-100", iconText: "text-rose-600", hover: "hover:border-rose-400" },
    { bg: "bg-purple-50", border: "border-purple-200", icon: "bg-purple-100", iconText: "text-purple-600", hover: "hover:border-purple-400" },
    { bg: "bg-cyan-50", border: "border-cyan-200", icon: "bg-cyan-100", iconText: "text-cyan-600", hover: "hover:border-cyan-400" },
  ];

  return (
    <div>
      {/* Welcome Banner */}
      <div className="bg-gradient-to-r from-indigo-600 to-purple-600 rounded-2xl p-6 mb-8 text-white">
        <div className="flex items-center gap-4">
          <div className="w-14 h-14 rounded-xl bg-white/20 flex items-center justify-center">
            <Sparkles className="w-7 h-7" />
          </div>
          <div>
            <h1 className="text-2xl font-bold">Merhaba, {user.full_name.split(" ")[0]}!</h1>
            <p className="text-indigo-100 mt-1">Bugün hangi dersi çalışmak istersin?</p>
          </div>
        </div>
        <div className="mt-4 flex items-center gap-6 text-sm">
          <div className="flex items-center gap-2">
            <BookOpen className="w-4 h-4" />
            <span>{courses.length} Ders Mevcut</span>
          </div>
          <div className="flex items-center gap-2">
            <MessageSquare className="w-4 h-4" />
            <span>AI Asistan ile Öğren</span>
          </div>
        </div>
      </div>

      {/* Section Title */}
      <div className="flex items-center gap-3 mb-6">
        <div className="w-10 h-10 rounded-lg bg-indigo-100 flex items-center justify-center">
          <BookOpen className="w-5 h-5 text-indigo-600" />
        </div>
        <div>
          <h2 className="text-lg font-semibold text-slate-900">Derslerim</h2>
          <p className="text-sm text-slate-500">Bir ders seçerek AI asistanından yardım alabilirsin</p>
        </div>
      </div>

      {/* Content */}
      {isLoading && (
        <div className="flex justify-center py-12">
          <Loader2 className="w-6 h-6 text-slate-400 animate-spin" />
        </div>
      )}
      
      {!isLoading && courses.length === 0 && (
        <div className="bg-white rounded-xl border border-slate-200 p-12 text-center">
          <BookOpen className="w-12 h-12 text-slate-300 mx-auto mb-3" />
          <p className="text-slate-500 text-lg">Henüz ders yok</p>
          <p className="text-slate-400 text-sm mt-1">Öğretmenin ders eklediğinde burada görünecek</p>
        </div>
      )}
      
      {!isLoading && courses.length > 0 && (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-5">
          {courses.map((course, index) => {
            const color = colors[index % colors.length];
            return (
              <button
                key={course.id}
                type="button"
                onClick={() => handleCourseSelect(course.id)}
                className={`${color.bg} rounded-xl border-2 ${color.border} p-5 ${color.hover} hover:shadow-lg transition-all cursor-pointer group text-left w-full`}
              >
                <div className="flex items-start gap-4">
                  <div className={`w-12 h-12 rounded-xl ${color.icon} flex items-center justify-center shrink-0`}>
                    <BookOpen className={`w-6 h-6 ${color.iconText}`} />
                  </div>
                  <div className="flex-1 min-w-0">
                    <h3 className="font-semibold text-slate-900 text-lg">
                      {course.name}
                    </h3>
                    <p className="text-sm text-slate-600 mt-1 line-clamp-2">
                      {course.description || "Açıklama yok"}
                    </p>
                  </div>
                </div>
                <div className="mt-4 pt-4 border-t border-slate-200/50 flex items-center justify-between">
                  <span className="text-xs text-slate-500">
                    {new Date(course.created_at).toLocaleDateString("tr-TR")}
                  </span>
                  <span className={`text-sm ${color.iconText} font-semibold flex items-center gap-1 group-hover:gap-2 transition-all`}>
                    Çalışmaya Başla <ArrowRight className="w-4 h-4" />
                  </span>
                </div>
              </button>
            );
          })}
        </div>
      )}
    </div>
  );
}
