"use client";

import { useCallback, useEffect, useState } from "react";
import { useAuth } from "@/lib/auth-context";
import { api, Course, RagasGroupInfo } from "@/lib/api";
import { toast } from "sonner";
import Link from "next/link";
import { BookOpen, History, ArrowLeft, RefreshCw, SquarePen, Trash2 } from "lucide-react";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Button } from "@/components/ui/button";
import { Dialog, DialogContent, DialogFooter, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";

export default function RagasResultsPage() {
  const { user } = useAuth();
  const [courses, setCourses] = useState<Course[]>([]);
  const [selectedCourseId, setSelectedCourseId] = useState<number | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  const [groups, setGroups] = useState<RagasGroupInfo[]>([]);
  const [isGroupsLoading, setIsGroupsLoading] = useState(false);
  const [deletingGroupName, setDeletingGroupName] = useState<string | null>(null);

  const [isRenameOpen, setIsRenameOpen] = useState(false);
  const [renamingGroupName, setRenamingGroupName] = useState<string | null>(null);
  const [newGroupName, setNewGroupName] = useState("");
  const [isRenaming, setIsRenaming] = useState(false);

  const PAGE_SIZE = 10;
  const [page, setPage] = useState(1);

  const loadGroups = useCallback(async () => {
    if (!selectedCourseId) return;
    setIsGroupsLoading(true);
    try {
      const data = await api.getQuickTestResults(selectedCourseId, undefined, 0, 1);
      setGroups(data.groups || []);
    } catch (error) {
      console.error("Failed to load groups:", error);
      setGroups([]);
      toast.error("Gruplar yüklenemedi");
    } finally {
      setIsGroupsLoading(false);
    }
  }, [selectedCourseId]);

  const handleDeleteGroup = useCallback(
    async (groupName: string, testCount?: number | null) => {
      if (!selectedCourseId) return;
      if (!groupName) return;

      const msg = `"${groupName}" grubunu ve içindeki ${testCount ?? "tüm"} sonucu silmek istediğinizden emin misiniz?`;
      if (!confirm(msg)) return;

      setDeletingGroupName(groupName);
      try {
        const res = await api.deleteRagasGroup(selectedCourseId, groupName);
        toast.success(res?.message || "Grup silindi");
        await loadGroups();
      } catch (error) {
        console.error(error);
        toast.error(error instanceof Error ? error.message : "Grup silinemedi");
      } finally {
        setDeletingGroupName(null);
      }
    },
    [loadGroups, selectedCourseId]
  );

  const openRenameDialog = useCallback((groupName: string) => {
    setRenamingGroupName(groupName);
    setNewGroupName(groupName);
    setIsRenameOpen(true);
  }, []);

  const handleRenameGroup = useCallback(async () => {
    if (!selectedCourseId) return;
    if (!renamingGroupName) return;

    const nextName = newGroupName.trim();
    if (!nextName) {
      toast.error("Yeni grup adı boş olamaz");
      return;
    }
    if (nextName === renamingGroupName) {
      setIsRenameOpen(false);
      return;
    }
    if (groups.some((g) => g.name === nextName)) {
      toast.error("Bu isimde bir grup zaten var");
      return;
    }

    setIsRenaming(true);
    try {
      const res = await api.renameRagasGroup(selectedCourseId, renamingGroupName, nextName);
      toast.success(res?.message || "Grup adı güncellendi");
      setIsRenameOpen(false);
      setRenamingGroupName(null);
      await loadGroups();
    } catch (error) {
      console.error(error);
      toast.error(error instanceof Error ? error.message : "Grup adı güncellenemedi");
    } finally {
      setIsRenaming(false);
    }
  }, [groups, loadGroups, newGroupName, renamingGroupName, selectedCourseId]);

  useEffect(() => {
    loadCourses();
  }, []);

  useEffect(() => {
    if (selectedCourseId) {
      loadGroups();
      setPage(1);
    }
  }, [selectedCourseId, loadGroups]);

  const loadCourses = async () => {
    try {
      const data = await api.getCourses();
      setCourses(data);
      if (data.length > 0) {
        const savedCourseId = localStorage.getItem("ragas_selected_course_id");
        if (savedCourseId && data.find((c) => c.id === parseInt(savedCourseId))) {
          setSelectedCourseId(parseInt(savedCourseId));
        } else {
          setSelectedCourseId(data[0].id);
        }
      }
    } catch {
      toast.error("Dersler yüklenirken hata oluştu");
    } finally {
      setIsLoading(false);
    }
  };

  if (!user) return null;

  if (isLoading) {
    return (
      <div className="min-h-[60vh] flex items-center justify-center">
        <div className="text-center">
          <div className="w-16 h-16 border-4 border-slate-200 border-t-slate-700 rounded-full animate-spin mx-auto"></div>
          <p className="mt-4 text-slate-600 font-medium">Sonuçlar yükleniyor...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <Dialog open={isRenameOpen} onOpenChange={setIsRenameOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Grup adını değiştir</DialogTitle>
          </DialogHeader>

          <div className="space-y-2">
            <div className="text-sm text-slate-600">Eski ad: <span className="font-medium text-slate-900">{renamingGroupName || "-"}</span></div>
            <Input
              value={newGroupName}
              onChange={(e) => setNewGroupName(e.target.value)}
              placeholder="Yeni grup adı"
              autoFocus
            />
          </div>

          <DialogFooter>
            <Button variant="outline" onClick={() => setIsRenameOpen(false)} disabled={isRenaming}>
              İptal
            </Button>
            <Button onClick={handleRenameGroup} disabled={isRenaming}>
              Kaydet
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      <div className="bg-white rounded-2xl border border-slate-200 p-6 shadow-sm">
        <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between gap-4">
          <div className="flex items-center gap-3">
            <div className="p-3 bg-slate-100 rounded-xl">
              <History className="w-6 h-6 text-slate-700" />
            </div>
            <div>
              <h1 className="text-2xl font-bold text-slate-900">RAGAS Test Sonuçları</h1>
              <p className="text-sm text-slate-600">Önce test listesi, sonra detay</p>
            </div>
          </div>

          <div className="flex items-center gap-3">
            <Link href="/dashboard/ragas">
              <Button variant="outline">
                <ArrowLeft className="w-4 h-4 mr-2" />
                RAGAS&#39;a Dön
              </Button>
            </Link>

            <Button variant="outline" onClick={loadGroups} disabled={!selectedCourseId || isGroupsLoading}>
              <RefreshCw className={`w-4 h-4 mr-2 ${isGroupsLoading ? "animate-spin" : ""}`} />
              Yenile
            </Button>

            <Select
              value={selectedCourseId?.toString() || ""}
              onValueChange={(v) => {
                const courseId = Number(v);
                setSelectedCourseId(courseId);
                localStorage.setItem("ragas_selected_course_id", courseId.toString());
              }}
            >
              <SelectTrigger className="w-72">
                <BookOpen className="w-4 h-4 mr-2" />
                <SelectValue placeholder="Ders seçin" />
              </SelectTrigger>
              <SelectContent>
                {courses.map((course) => (
                  <SelectItem key={course.id} value={course.id.toString()}>
                    {course.name}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
        </div>
      </div>

      {!selectedCourseId ? (
        <div className="bg-white rounded-2xl border border-slate-200 p-12 text-center shadow-sm">
          <p className="text-slate-500">Sonuçları görmek için ders seçin.</p>
        </div>
      ) : isGroupsLoading ? (
        <div className="bg-white rounded-2xl border border-slate-200 p-12 text-center shadow-sm">
          <p className="text-slate-500">Gruplar yükleniyor...</p>
        </div>
      ) : groups.length === 0 ? (
        <div className="bg-white rounded-2xl border border-slate-200 p-12 text-center shadow-sm">
          <p className="text-slate-500">Bu ders için kayıtlı test grubu bulunamadı.</p>
        </div>
      ) : (
        <div className="bg-white rounded-2xl border border-slate-200 p-6 shadow-sm">
          {(() => {
            const cleaned = groups
              .filter((g) => g.name && g.name.trim().length > 0)
              .sort((a, b) => {
                if (!a.created_at && !b.created_at) return 0;
                if (!a.created_at) return 1;
                if (!b.created_at) return -1;
                return new Date(b.created_at).getTime() - new Date(a.created_at).getTime();
              });

            const totalPages = Math.max(1, Math.ceil(cleaned.length / PAGE_SIZE));
            const safePage = Math.min(Math.max(page, 1), totalPages);
            const start = (safePage - 1) * PAGE_SIZE;
            const end = Math.min(start + PAGE_SIZE, cleaned.length);
            const pageItems = cleaned.slice(start, end);

            return (
              <>
                <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-3 mb-3">
                  <div className="text-xs text-slate-500">
                    Gösterilen: <span className="font-medium text-slate-700">{cleaned.length === 0 ? 0 : start + 1}-{end}</span>
                    <span className="text-slate-400"> / </span>
                    <span className="font-medium text-slate-700">{cleaned.length}</span>
                  </div>

                  <div className="flex items-center gap-2">
                    <Button
                      type="button"
                      variant="outline"
                      size="sm"
                      className="h-8"
                      onClick={() => setPage((p) => Math.max(1, p - 1))}
                      disabled={safePage <= 1}
                    >
                      Önceki
                    </Button>
                    <div className="text-xs text-slate-600 min-w-[80px] text-center">
                      {safePage} / {totalPages}
                    </div>
                    <Button
                      type="button"
                      variant="outline"
                      size="sm"
                      className="h-8"
                      onClick={() => setPage((p) => Math.min(totalPages, p + 1))}
                      disabled={safePage >= totalPages}
                    >
                      Sonraki
                    </Button>
                  </div>
                </div>

                <div className="overflow-auto border border-slate-200 rounded-xl">
                  <table className="w-full text-xs">
                    <thead className="bg-slate-50 sticky top-0">
                      <tr>
                        <th className="px-3 py-2 text-left font-medium text-slate-600">Grup</th>
                        <th className="px-3 py-2 text-right font-medium text-slate-600">Test</th>
                        <th className="px-3 py-2 text-left font-medium text-slate-600">Ortalama</th>
                        <th className="px-3 py-2 text-left font-medium text-slate-600">LLM</th>
                        <th className="px-3 py-2 text-left font-medium text-slate-600">Eval</th>
                        <th className="px-3 py-2 text-left font-medium text-slate-600">Embedding</th>
                        <th className="px-3 py-2 text-right font-medium text-slate-600">TopK</th>
                        <th className="px-3 py-2 text-right font-medium text-slate-600">Alpha</th>
                        <th className="px-3 py-2 text-left font-medium text-slate-600">Reranker</th>
                        <th className="px-3 py-2 text-right font-medium text-slate-600"></th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-100">
                      {pageItems.map((g, idx) => {
                        const llm = g.llm_provider && g.llm_model
                          ? `${g.llm_provider}/${g.llm_model}`
                          : g.llm_model || g.llm_provider || "-";

                        const reranker = g.reranker_used
                          ? `${g.reranker_provider || "-"}/${g.reranker_model || "-"}`
                          : "Kapalı";

                        return (
                          <tr key={g.name} className={idx % 2 === 0 ? "bg-white hover:bg-slate-50" : "bg-slate-50/40 hover:bg-slate-50"}>
                            <td className="px-3 py-2">
                              <Link
                                href={`/dashboard/ragas/results/${encodeURIComponent(g.name)}`}
                                className="font-semibold text-slate-900 hover:underline"
                              >
                                {g.name}
                              </Link>
                              <div className="text-[11px] text-slate-500 mt-0.5 whitespace-nowrap">
                                {g.created_at ? new Date(g.created_at).toLocaleString("tr-TR") : "-"}
                              </div>
                            </td>
                            <td className="px-3 py-2 text-right">
                              <span className="inline-flex items-center rounded-md bg-slate-100 px-2 py-0.5 font-medium text-slate-700">
                                {g.test_count ?? "-"}
                              </span>
                            </td>
                            <td className="px-3 py-2">
                              <div className="text-xs flex gap-2 flex-wrap">
                                {typeof g.avg_faithfulness === "number" && Number.isFinite(g.avg_faithfulness) && (
                                  <span className="inline-flex items-center rounded bg-blue-50 px-1.5 py-0.5 text-blue-700 border border-blue-200">
                                    <span className="font-medium">F:</span>
                                    <span>{(g.avg_faithfulness * 100).toFixed(0)}%</span>
                                  </span>
                                )}
                                {typeof g.avg_answer_relevancy === "number" && Number.isFinite(g.avg_answer_relevancy) && (
                                  <span className="inline-flex items-center rounded bg-green-50 px-1.5 py-0.5 text-green-700 border border-green-200">
                                    <span className="font-medium">AR:</span>
                                    <span>{(g.avg_answer_relevancy * 100).toFixed(0)}%</span>
                                  </span>
                                )}
                                {typeof g.avg_context_precision === "number" && Number.isFinite(g.avg_context_precision) && (
                                  <span className="inline-flex items-center rounded bg-purple-50 px-1.5 py-0.5 text-purple-700 border border-purple-200">
                                    <span className="font-medium">CP:</span>
                                    <span>{(g.avg_context_precision * 100).toFixed(0)}%</span>
                                  </span>
                                )}
                                {typeof g.avg_context_recall === "number" && Number.isFinite(g.avg_context_recall) && (
                                  <span className="inline-flex items-center rounded bg-orange-50 px-1.5 py-0.5 text-orange-700 border border-orange-200">
                                    <span className="font-medium">CR:</span>
                                    <span>{(g.avg_context_recall * 100).toFixed(0)}%</span>
                                  </span>
                                )}
                                {typeof g.avg_answer_correctness === "number" && Number.isFinite(g.avg_answer_correctness) && (
                                  <span className="inline-flex items-center rounded bg-red-50 px-1.5 py-0.5 text-red-700 border border-red-200">
                                    <span className="font-medium">AC:</span>
                                    <span>{(g.avg_answer_correctness * 100).toFixed(0)}%</span>
                                  </span>
                                )}
                              </div>
                            </td>
                            <td className="px-3 py-2 whitespace-nowrap">
                              <span className="inline-flex items-center rounded-full bg-indigo-50 px-2 py-0.5 text-indigo-700 border border-indigo-200">
                                {llm}
                              </span>
                            </td>
                            <td className="px-3 py-2 whitespace-nowrap">
                              <span className="inline-flex items-center rounded-full bg-purple-50 px-2 py-0.5 text-purple-700 border border-purple-200">
                                {g.evaluation_model || "-"}
                              </span>
                            </td>
                            <td className="px-3 py-2 whitespace-nowrap">
                              <span className="inline-flex items-center rounded-full bg-slate-50 px-2 py-0.5 text-slate-700 border border-slate-200">
                                {g.embedding_model || "-"}
                              </span>
                            </td>
                            <td className="px-3 py-2 text-right text-slate-700">{g.search_top_k ?? "-"}</td>
                            <td className="px-3 py-2 text-right text-slate-700">{g.search_alpha ?? "-"}</td>
                            <td className="px-3 py-2 whitespace-nowrap">
                              <span
                                className={`inline-flex items-center rounded-full px-2 py-0.5 border ${
                                  g.reranker_used
                                    ? "bg-emerald-50 text-emerald-700 border-emerald-200"
                                    : "bg-slate-100 text-slate-600 border-slate-200"
                                }`}
                              >
                                {reranker}
                              </span>
                            </td>
                            <td className="px-3 py-2 text-right">
                              <div className="inline-flex items-center gap-1">
                                <Link href={`/dashboard/ragas/results/${encodeURIComponent(g.name)}`}>
                                  <Button
                                    variant="ghost"
                                    size="sm"
                                    className="h-8 w-8 p-0"
                                    type="button"
                                    title="Detay"
                                  >
                                    <SquarePen className="w-4 h-4" />
                                  </Button>
                                </Link>

                                <Button
                                  variant="ghost"
                                  size="sm"
                                  className="h-8 w-8 p-0"
                                  type="button"
                                  title="Grup adını değiştir"
                                  onClick={() => openRenameDialog(g.name)}
                                >
                                  <History className="w-4 h-4" />
                                </Button>

                                <Button
                                  variant="ghost"
                                  size="sm"
                                  className="h-8 w-8 p-0 text-red-600 hover:text-red-700"
                                  type="button"
                                  title="Sil"
                                  onClick={() => handleDeleteGroup(g.name, g.test_count)}
                                  disabled={deletingGroupName === g.name}
                                >
                                  <Trash2 className={`w-4 h-4 ${deletingGroupName === g.name ? "opacity-50" : ""}`} />
                                </Button>
                              </div>
                            </td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              </>
            );
          })()}
        </div>
      )}
    </div>
  );
}
