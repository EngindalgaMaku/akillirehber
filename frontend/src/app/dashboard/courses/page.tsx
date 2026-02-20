"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { useAuth } from "@/lib/auth-context";
import { api, Course } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { Switch } from "@/components/ui/switch";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import { toast } from "sonner";
import {
  Plus,
  BookOpen,
  ArrowRight,
  Loader2,
  Calendar,
  FileText,
  GraduationCap,
  Sparkles,
} from "lucide-react";
import { PageHeader } from "@/components/ui/page-header";

// Course card color schemes
const cardColors = [
  {
    bg: "from-violet-500 to-purple-600",
    icon: "bg-white/20",
    accent: "violet",
  },
  {
    bg: "from-blue-500 to-cyan-600",
    icon: "bg-white/20",
    accent: "blue",
  },
  {
    bg: "from-emerald-500 to-teal-600",
    icon: "bg-white/20",
    accent: "emerald",
  },
  {
    bg: "from-orange-500 to-amber-600",
    icon: "bg-white/20",
    accent: "orange",
  },
  {
    bg: "from-pink-500 to-rose-600",
    icon: "bg-white/20",
    accent: "pink",
  },
  {
    bg: "from-indigo-500 to-blue-600",
    icon: "bg-white/20",
    accent: "indigo",
  },
];

export default function CoursesPage() {
  const { user } = useAuth();
  const [courses, setCourses] = useState<Course[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [isCreateOpen, setIsCreateOpen] = useState(false);
  const [isCreating, setIsCreating] = useState(false);
  const [newCourse, setNewCourse] = useState({ name: "", description: "" });
  const [showInactive, setShowInactive] = useState(true); // Show inactive courses by default

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

  const handleCreateCourse = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsCreating(true);
    try {
      const course = await api.createCourse(newCourse);
      setCourses([...courses, course]);
      setNewCourse({ name: "", description: "" });
      setIsCreateOpen(false);
      toast.success("Ders başarıyla oluşturuldu");
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "Hata oluştu");
    } finally {
      setIsCreating(false);
    }
  };

  const getColorScheme = (index: number) => {
    return cardColors[index % cardColors.length];
  };

  const formatDate = (dateString: string) => {
    const date = new Date(dateString);
    return date.toLocaleDateString("tr-TR", {
      day: "numeric",
      month: "long",
      year: "numeric",
    });
  };

  // Filter courses based on showInactive toggle
  const filteredCourses = showInactive 
    ? courses 
    : courses.filter(c => c.is_active);

  const isTeacherOrAdmin = user?.role === "teacher" || user?.role === "admin";

  if (!user) return null;

  return (
    <div className="space-y-8">
      {/* Header */}
      <PageHeader
        icon={GraduationCap}
        title="Derslerim"
        description={
          user.role === "teacher"
            ? "Derslerinizi yönetin, içerik ekleyin ve öğrencilerinizle etkileşime geçin"
            : "Kayıtlı olduğunuz dersleri görüntüleyin ve içeriklere erişin"
        }
      >
        {(user.role === "teacher" || user.role === "admin") && (
          <Dialog open={isCreateOpen} onOpenChange={setIsCreateOpen}>
            <DialogTrigger asChild>
              <Button
                size="lg"
                className="bg-gradient-to-r from-indigo-600 to-violet-600
                  hover:from-indigo-700 hover:to-violet-700
                  shadow-lg shadow-indigo-500/25
                  transition-all duration-300 hover:shadow-xl hover:shadow-indigo-500/30
                  hover:-translate-y-0.5"
              >
                <Plus className="w-5 h-5 mr-2" />
                Yeni Ders Oluştur
              </Button>
            </DialogTrigger>
            <DialogContent className="sm:max-w-[500px]">
              <form onSubmit={handleCreateCourse}>
                <DialogHeader>
                  <DialogTitle className="flex items-center gap-2 text-xl">
                    <Sparkles className="w-5 h-5 text-indigo-600" />
                    Yeni Ders Oluştur
                  </DialogTitle>
                  <DialogDescription>
                    Yeni bir ders oluşturun ve öğrencilerinizle paylaşın.
                  </DialogDescription>
                </DialogHeader>
                <div className="space-y-5 py-6">
                  <div className="space-y-2">
                    <Label htmlFor="name" className="text-sm font-medium">
                      Ders Adı
                    </Label>
                    <Input
                      id="name"
                      placeholder="Örn: Yapay Zeka ve Makine Öğrenmesi"
                      value={newCourse.name}
                      onChange={(e) =>
                        setNewCourse({ ...newCourse, name: e.target.value })
                      }
                      className="h-11"
                      required
                    />
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="description" className="text-sm font-medium">
                      Açıklama
                    </Label>
                    <Textarea
                      id="description"
                      placeholder="Dersin içeriği ve hedefleri hakkında kısa bir açıklama yazın..."
                      value={newCourse.description}
                      onChange={(e) =>
                        setNewCourse({
                          ...newCourse,
                          description: e.target.value,
                        })
                      }
                      className="min-h-[100px] resize-none"
                    />
                  </div>
                </div>
                <DialogFooter className="gap-2">
                  <Button
                    type="button"
                    variant="outline"
                    onClick={() => setIsCreateOpen(false)}
                  >
                    İptal
                  </Button>
                  <Button
                    type="submit"
                    disabled={isCreating || !newCourse.name.trim()}
                    className="bg-gradient-to-r from-indigo-600 to-violet-600
                      hover:from-indigo-700 hover:to-violet-700 min-w-[120px]"
                  >
                    {isCreating ? (
                      <>
                        <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                        Oluşturuluyor
                      </>
                    ) : (
                      <>
                        <Plus className="w-4 h-4 mr-2" />
                        Oluştur
                      </>
                    )}
                  </Button>
                </DialogFooter>
              </form>
            </DialogContent>
          </Dialog>
        )}
      </PageHeader>

      {/* Stats Bar */}
      {!isLoading && courses.length > 0 && (
        <div className="space-y-4">
          {/* Filter Toggle for Teachers/Admins */}
          {isTeacherOrAdmin && (
            <div className="flex items-center justify-between bg-white rounded-xl border border-slate-200 p-3 sm:p-4 shadow-sm">
              <div className="flex items-center gap-3 min-w-0">
                <div className="w-10 h-10 rounded-lg bg-slate-100 flex items-center justify-center shrink-0">
                  <FileText className="w-5 h-5 text-slate-600" />
                </div>
                <div className="min-w-0">
                  <p className="font-medium text-slate-900 text-sm sm:text-base">Pasif Dersleri Göster</p>
                  <p className="text-xs sm:text-sm text-slate-500 hidden sm:block">Aktif olmayan dersleri listede göster</p>
                </div>
              </div>
              <Switch
                checked={showInactive}
                onCheckedChange={setShowInactive}
                className="data-[state=checked]:bg-indigo-600 shrink-0 ml-3"
              />
            </div>
          )}

          <div className="grid grid-cols-3 gap-3 sm:gap-4">
            <div className="bg-white rounded-xl border border-slate-200 p-3 sm:p-4 flex items-center gap-3 sm:gap-4 shadow-sm">
              <div className="w-10 h-10 sm:w-12 sm:h-12 rounded-xl bg-gradient-to-br from-indigo-500 to-violet-600 flex items-center justify-center shadow-lg shadow-indigo-500/20 shrink-0">
                <BookOpen className="w-5 h-5 sm:w-6 sm:h-6 text-white" />
              </div>
              <div className="min-w-0">
                <p className="text-xl sm:text-2xl font-bold text-slate-900">{filteredCourses.length}</p>
                <p className="text-xs sm:text-sm text-slate-500 truncate">{showInactive ? "Toplam Ders" : "Aktif Ders"}</p>
              </div>
            </div>
            <div className="bg-white rounded-xl border border-slate-200 p-3 sm:p-4 flex items-center gap-3 sm:gap-4 shadow-sm">
              <div className="w-10 h-10 sm:w-12 sm:h-12 rounded-xl bg-gradient-to-br from-emerald-500 to-teal-600 flex items-center justify-center shadow-lg shadow-emerald-500/20 shrink-0">
                <FileText className="w-5 h-5 sm:w-6 sm:h-6 text-white" />
              </div>
              <div className="min-w-0">
                <p className="text-xl sm:text-2xl font-bold text-slate-900">{courses.filter((c) => c.is_active).length}</p>
                <p className="text-xs sm:text-sm text-slate-500">Aktif</p>
              </div>
            </div>
            <div className="bg-white rounded-xl border border-slate-200 p-3 sm:p-4 flex items-center gap-3 sm:gap-4 shadow-sm">
              <div className="w-10 h-10 sm:w-12 sm:h-12 rounded-xl bg-gradient-to-br from-slate-500 to-slate-600 flex items-center justify-center shadow-lg shadow-slate-500/20 shrink-0">
                <FileText className="w-5 h-5 sm:w-6 sm:h-6 text-white" />
              </div>
              <div className="min-w-0">
                <p className="text-xl sm:text-2xl font-bold text-slate-900">{courses.filter((c) => !c.is_active).length}</p>
                <p className="text-xs sm:text-sm text-slate-500">Pasif</p>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Content */}
      {isLoading ? (
        <div className="flex flex-col items-center justify-center py-20">
          <div
            className="w-16 h-16 rounded-2xl bg-gradient-to-br from-indigo-500 to-violet-600
            flex items-center justify-center shadow-lg shadow-indigo-500/30 mb-4"
          >
            <Loader2 className="w-8 h-8 text-white animate-spin" />
          </div>
          <p className="text-slate-500">Dersler yükleniyor...</p>
        </div>
      ) : courses.length === 0 ? (
        <div
          className="bg-gradient-to-br from-slate-50 to-slate-100
          rounded-2xl border border-slate-200 p-16 text-center"
        >
          <div
            className="w-20 h-20 rounded-2xl bg-gradient-to-br from-slate-200 to-slate-300
            flex items-center justify-center mx-auto mb-6"
          >
            <BookOpen className="w-10 h-10 text-slate-400" />
          </div>
          <h3 className="text-xl font-semibold text-slate-700 mb-2">
            {(user.role === "teacher" || user.role === "admin")
              ? "Henüz ders oluşturmadınız"
              : "Henüz kayıtlı ders yok"}
          </h3>
          <p className="text-slate-500 mb-6 max-w-md mx-auto">
            {(user.role === "teacher" || user.role === "admin")
              ? "İlk dersinizi oluşturarak öğrencilerinizle içerik paylaşmaya başlayın."
              : "Öğretmeniniz ders oluşturduğunda burada görünecektir."}
          </p>
          {(user.role === "teacher" || user.role === "admin") && (
            <Button
              onClick={() => setIsCreateOpen(true)}
              size="lg"
              className="bg-gradient-to-r from-indigo-600 to-violet-600
                hover:from-indigo-700 hover:to-violet-700
                shadow-lg shadow-indigo-500/25"
            >
              <Plus className="w-5 h-5 mr-2" />
              İlk Dersinizi Oluşturun
            </Button>
          )}
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {filteredCourses.map((course, index) => {
            const colorScheme = getColorScheme(index);
            const isInactive = !course.is_active;
            
            return (
              <Link
                key={course.id}
                href={`/dashboard/courses/${course.id}`}
                className="group block"
              >
                <div
                  className={`bg-white rounded-2xl border overflow-hidden shadow-sm hover:shadow-xl
                  transition-all duration-300 hover:-translate-y-1 ${
                    isInactive ? 'border-slate-300 opacity-75' : 'border-slate-200'
                  }`}
                >
                  {/* Card Header with Gradient */}
                  <div
                    className={`h-32 bg-gradient-to-br ${colorScheme.bg} ${
                      isInactive ? 'opacity-60' : ''
                    } relative overflow-hidden`}
                  >
                    {/* Decorative Elements */}
                    <div
                      className="absolute top-0 right-0 w-32 h-32
                      bg-white/10 rounded-full -translate-y-1/2 translate-x-1/2"
                    />
                    <div
                      className="absolute bottom-0 left-0 w-24 h-24
                      bg-white/10 rounded-full translate-y-1/2 -translate-x-1/2"
                    />

                    {/* Icon */}
                    <div className="absolute bottom-4 left-5">
                      <div
                        className={`w-14 h-14 rounded-xl ${colorScheme.icon}
                        backdrop-blur-sm flex items-center justify-center
                        shadow-lg`}
                      >
                        <BookOpen className="w-7 h-7 text-white" />
                      </div>
                    </div>

                    {/* Arrow Icon */}
                    <div
                      className="absolute top-4 right-4 w-8 h-8 rounded-full
                      bg-white/20 backdrop-blur-sm flex items-center justify-center
                      opacity-0 group-hover:opacity-100 transition-opacity duration-300
                      group-hover:translate-x-1"
                    >
                      <ArrowRight className="w-4 h-4 text-white" />
                    </div>
                  </div>

                  {/* Card Content */}
                  <div className="p-5">
                    <div className="flex items-start justify-between gap-2 mb-2">
                      <h3
                        className={`font-semibold text-lg mb-0
                        group-hover:text-indigo-600 transition-colors line-clamp-1 ${
                          isInactive ? 'text-slate-600' : 'text-slate-900'
                        }`}
                      >
                        {course.name}
                      </h3>
                      {isInactive && (
                        <span className="px-2 py-0.5 bg-slate-200 text-slate-600 text-xs font-medium rounded-full shrink-0">
                          Pasif
                        </span>
                      )}
                    </div>
                    <p className={`text-sm mb-4 line-clamp-2 min-h-[40px] ${
                      isInactive ? 'text-slate-400' : 'text-slate-500'
                    }`}>
                      {course.description || "Açıklama eklenmemiş"}
                    </p>

                    {/* Footer */}
                    <div
                      className="flex items-center justify-between pt-4
                      border-t border-slate-100"
                    >
                      <div className="flex items-center gap-2 text-slate-400">
                        <Calendar className="w-4 h-4" />
                        <span className="text-xs">
                          {formatDate(course.created_at)}
                        </span>
                      </div>
                      {(user.role === "teacher" || user.role === "admin") ? (
                        <div
                          className="flex items-center gap-2"
                          onClick={(e) => e.preventDefault()}
                        >
                          <Switch
                            checked={course.is_active}
                            onCheckedChange={async (checked) => {
                              try {
                                await api.updateCourse(course.id, {
                                  is_active: checked,
                                });
                                // Reload courses to get fresh data
                                await loadCourses();
                                toast.success(
                                  checked
                                    ? "Ders aktif edildi"
                                    : "Ders pasif edildi"
                                );
                              } catch {
                                toast.error("Durum güncellenirken hata oluştu");
                              }
                            }}
                            className="data-[state=checked]:bg-emerald-500"
                          />
                          <span
                            className={`text-xs font-medium ${
                              course.is_active
                                ? "text-emerald-600"
                                : "text-slate-500"
                            }`}
                          >
                            {course.is_active ? "Aktif" : "Pasif"}
                          </span>
                        </div>
                      ) : (
                        <div
                          className={`px-2.5 py-1 rounded-full text-xs font-medium
                          ${
                            course.is_active
                              ? "bg-emerald-100 text-emerald-700"
                              : "bg-slate-100 text-slate-600"
                          }`}
                        >
                          {course.is_active ? "Aktif" : "Pasif"}
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              </Link>
            );
          })}

          {/* Add New Course Card (for teachers and admins) */}
          {(user.role === "teacher" || user.role === "admin") && (
            <button
              onClick={() => setIsCreateOpen(true)}
              className="group block w-full text-left"
            >
              <div
                className="bg-slate-50 rounded-2xl border-2 border-dashed
                border-slate-300 overflow-hidden h-full min-h-[200px] sm:min-h-[280px]
                hover:border-indigo-400 hover:bg-indigo-50/50
                transition-all duration-300 flex flex-col items-center
                justify-center gap-3 sm:gap-4 p-4"
              >
                <div
                  className="w-16 h-16 rounded-2xl bg-slate-200
                  group-hover:bg-indigo-100 flex items-center justify-center
                  transition-colors duration-300"
                >
                  <Plus
                    className="w-8 h-8 text-slate-400
                    group-hover:text-indigo-600 transition-colors duration-300"
                  />
                </div>
                <div className="text-center">
                  <p
                    className="font-medium text-slate-600
                    group-hover:text-indigo-600 transition-colors"
                  >
                    Yeni Ders Ekle
                  </p>
                  <p className="text-sm text-slate-400 mt-1">
                    Tıklayarak yeni ders oluşturun
                  </p>
                </div>
              </div>
            </button>
          )}
        </div>
      )}
    </div>
  );
}
