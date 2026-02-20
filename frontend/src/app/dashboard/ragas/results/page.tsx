"use client";

import { useCallback, useEffect, useState } from "react";
import { useAuth } from "@/lib/auth-context";
import { api, Course, RagasGroupInfo } from "@/lib/api";
import { toast } from "sonner";
import Link from "next/link";
import { BookOpen, History, ArrowLeft, RefreshCw, SquarePen, Trash2, BarChart3 } from "lucide-react";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Button } from "@/components/ui/button";
import { Dialog, DialogContent, DialogFooter, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Checkbox } from "@/components/ui/checkbox";

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
  
  // Multi-select for statistics
  const [selectedGroupNames, setSelectedGroupNames] = useState<Set<string>>(new Set());
  const [showStatisticsModal, setShowStatisticsModal] = useState(false);
  
  // Filters
  const [searchQuery, setSearchQuery] = useState("");
  const [filterLLM, setFilterLLM] = useState<string>("all");
  const [filterEvalModel, setFilterEvalModel] = useState<string>("all");
  const [filterReranker, setFilterReranker] = useState<string>("all");

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

  const toggleSelectAll = useCallback(() => {
    if (selectedGroupNames.size === groups.length) {
      setSelectedGroupNames(new Set());
    } else {
      setSelectedGroupNames(new Set(groups.map(g => g.name)));
    }
  }, [groups, selectedGroupNames.size]);

  const toggleSelectGroup = useCallback((groupName: string) => {
    const newSet = new Set(selectedGroupNames);
    if (newSet.has(groupName)) {
      newSet.delete(groupName);
    } else {
      newSet.add(groupName);
    }
    setSelectedGroupNames(newSet);
  }, [selectedGroupNames]);

  const calculateStatistics = useCallback(() => {
    const selectedGroups = groups.filter(g => selectedGroupNames.has(g.name));
    
    if (selectedGroups.length === 0) return null;

    const metrics = {
      faithfulness: selectedGroups.map(g => g.avg_faithfulness).filter((v): v is number => v != null && Number.isFinite(v)),
      answer_relevancy: selectedGroups.map(g => g.avg_answer_relevancy).filter((v): v is number => v != null && Number.isFinite(v)),
      context_precision: selectedGroups.map(g => g.avg_context_precision).filter((v): v is number => v != null && Number.isFinite(v)),
      context_recall: selectedGroups.map(g => g.avg_context_recall).filter((v): v is number => v != null && Number.isFinite(v)),
      answer_correctness: selectedGroups.map(g => g.avg_answer_correctness).filter((v): v is number => v != null && Number.isFinite(v)),
    };

    const calculateStats = (values: number[]) => {
      if (values.length === 0) return null;
      
      const mean = values.reduce((a, b) => a + b, 0) / values.length;
      const variance = values.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / values.length;
      const stdDev = Math.sqrt(variance);
      const sorted = [...values].sort((a, b) => a - b);
      const median = sorted.length % 2 === 0
        ? (sorted[sorted.length / 2 - 1] + sorted[sorted.length / 2]) / 2
        : sorted[Math.floor(sorted.length / 2)];
      const min = Math.min(...values);
      const max = Math.max(...values);

      return { mean, stdDev, median, min, max, count: values.length };
    };

    const totalTests = selectedGroups.reduce((sum, g) => sum + (g.test_count || 0), 0);

    return {
      faithfulness: calculateStats(metrics.faithfulness),
      answer_relevancy: calculateStats(metrics.answer_relevancy),
      context_precision: calculateStats(metrics.context_precision),
      context_recall: calculateStats(metrics.context_recall),
      answer_correctness: calculateStats(metrics.answer_correctness),
      totalSelected: selectedGroups.length,
      totalTests,
    };
  }, [groups, selectedGroupNames]);

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
          localStorage.setItem("ragas_selected_course_id", data[0].id.toString());
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

      <div className="bg-white rounded-2xl border border-slate-200 p-4 sm:p-6 shadow-sm">
        <div className="flex flex-col gap-4">
          <div className="flex items-center gap-3">
            <div className="p-2 sm:p-3 bg-slate-100 rounded-xl shrink-0">
              <History className="w-5 h-5 sm:w-6 sm:h-6 text-slate-700" />
            </div>
            <div className="min-w-0">
              <h1 className="text-lg sm:text-2xl font-bold text-slate-900">RAGAS Test Sonuçları</h1>
              <p className="text-xs sm:text-sm text-slate-600">Önce test listesi, sonra detay</p>
            </div>
          </div>

          <div className="flex flex-wrap items-center gap-2 sm:gap-3">
            {selectedGroupNames.size > 0 && (
              <Button 
                variant="outline"
                size="sm"
                onClick={() => setShowStatisticsModal(true)}
                className="bg-blue-50 hover:bg-blue-100 text-blue-700 border-blue-200"
              >
                <BarChart3 className="w-4 h-4 sm:mr-2" />
                <span className="hidden sm:inline">İstatistikler</span> ({selectedGroupNames.size})
              </Button>
            )}

            <Link href="/dashboard/ragas">
              <Button variant="outline" size="sm">
                <ArrowLeft className="w-4 h-4 sm:mr-2" />
                <span className="hidden sm:inline">RAGAS&#39;a Dön</span>
              </Button>
            </Link>

            <Button variant="outline" size="sm" onClick={loadGroups} disabled={!selectedCourseId || isGroupsLoading}>
              <RefreshCw className={`w-4 h-4 sm:mr-2 ${isGroupsLoading ? "animate-spin" : ""}`} />
              <span className="hidden sm:inline">Yenile</span>
            </Button>

            <Select
              value={selectedCourseId?.toString() || ""}
              onValueChange={(v) => {
                const courseId = Number(v);
                setSelectedCourseId(courseId);
                localStorage.setItem("ragas_selected_course_id", courseId.toString());
              }}
            >
              <SelectTrigger className="w-full sm:w-72">
                <BookOpen className="w-4 h-4 mr-2 shrink-0" />
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
        <div className="bg-white rounded-2xl border border-slate-200 p-3 sm:p-6 shadow-sm">
          {(() => {
            // Apply filters
            let filtered = groups
              .filter((g) => g.name && g.name.trim().length > 0)
              .filter((g) => {
                // Search filter
                if (searchQuery) {
                  const query = searchQuery.toLowerCase();
                  return g.name.toLowerCase().includes(query);
                }
                return true;
              })
              .filter((g) => {
                // LLM filter
                if (filterLLM !== "all") {
                  const llm = g.llm_provider && g.llm_model
                    ? `${g.llm_provider}/${g.llm_model}`
                    : g.llm_model || g.llm_provider || "";
                  return llm.toLowerCase().includes(filterLLM.toLowerCase());
                }
                return true;
              })
              .filter((g) => {
                // Eval model filter
                if (filterEvalModel !== "all") {
                  return (g.evaluation_model || "").toLowerCase().includes(filterEvalModel.toLowerCase());
                }
                return true;
              })
              .filter((g) => {
                // Reranker filter
                if (filterReranker === "enabled") {
                  return g.reranker_used === true;
                } else if (filterReranker === "disabled") {
                  return !g.reranker_used;
                }
                return true;
              });

            // Sort by date
            const cleaned = filtered.sort((a, b) => {
              if (!a.created_at && !b.created_at) return 0;
              if (!a.created_at) return 1;
              if (!b.created_at) return -1;
              return new Date(b.created_at).getTime() - new Date(a.created_at).getTime();
            });

            // Get unique values for filter dropdowns
            const uniqueLLMs = Array.from(new Set(
              groups.map(g => {
                if (g.llm_provider && g.llm_model) return `${g.llm_provider}/${g.llm_model}`;
                return g.llm_model || g.llm_provider || "";
              }).filter(Boolean)
            )).sort();

            const uniqueEvalModels = Array.from(new Set(
              groups.map(g => g.evaluation_model).filter(Boolean)
            )).sort();

            const totalPages = Math.max(1, Math.ceil(cleaned.length / PAGE_SIZE));
            const safePage = Math.min(Math.max(page, 1), totalPages);
            const start = (safePage - 1) * PAGE_SIZE;
            const end = Math.min(start + PAGE_SIZE, cleaned.length);
            const pageItems = cleaned.slice(start, end);

            return (
              <>
                {/* Filters */}
                <div className="mb-4 p-3 sm:p-4 bg-slate-50 rounded-xl border border-slate-200">
                  <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-3">
                    <div>
                      <label className="text-xs font-medium text-slate-700 mb-1 block">Ara</label>
                      <Input
                        placeholder="Grup adı..."
                        value={searchQuery}
                        onChange={(e) => {
                          setSearchQuery(e.target.value);
                          setPage(1);
                        }}
                        className="h-9 text-sm"
                      />
                    </div>

                    <div>
                      <label className="text-xs font-medium text-slate-700 mb-1 block">LLM Model</label>
                      <Select
                        value={filterLLM}
                        onValueChange={(v) => {
                          setFilterLLM(v);
                          setPage(1);
                        }}
                      >
                        <SelectTrigger className="h-9 text-sm">
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="all">Tümü</SelectItem>
                          {uniqueLLMs.map((llm) => (
                            <SelectItem key={llm} value={llm}>
                              {llm}
                            </SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                    </div>

                    <div>
                      <label className="text-xs font-medium text-slate-700 mb-1 block">Eval Model</label>
                      <Select
                        value={filterEvalModel}
                        onValueChange={(v) => {
                          setFilterEvalModel(v);
                          setPage(1);
                        }}
                      >
                        <SelectTrigger className="h-9 text-sm">
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="all">Tümü</SelectItem>
                          {uniqueEvalModels.map((model) => (
                            <SelectItem key={model} value={model}>
                              {model}
                            </SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                    </div>

                    <div>
                      <label className="text-xs font-medium text-slate-700 mb-1 block">Reranker</label>
                      <Select
                        value={filterReranker}
                        onValueChange={(v) => {
                          setFilterReranker(v);
                          setPage(1);
                        }}
                      >
                        <SelectTrigger className="h-9 text-sm">
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="all">Tümü</SelectItem>
                          <SelectItem value="enabled">Açık</SelectItem>
                          <SelectItem value="disabled">Kapalı</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>
                  </div>

                  {(searchQuery || filterLLM !== "all" || filterEvalModel !== "all" || filterReranker !== "all") && (
                    <div className="mt-3 flex items-center gap-2">
                      <span className="text-xs text-slate-600">Aktif filtreler:</span>
                      <Button
                        variant="outline"
                        size="sm"
                        className="h-7 text-xs"
                        onClick={() => {
                          setSearchQuery("");
                          setFilterLLM("all");
                          setFilterEvalModel("all");
                          setFilterReranker("all");
                          setPage(1);
                        }}
                      >
                        Tümünü Temizle
                      </Button>
                    </div>
                  )}
                </div>

                <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-3 mb-3">
                  <div className="text-xs text-slate-500">
                    Gösterilen: <span className="font-medium text-slate-700">{cleaned.length === 0 ? 0 : start + 1}-{end}</span>
                    <span className="text-slate-400"> / </span>
                    <span className="font-medium text-slate-700">{cleaned.length}</span>
                    {filtered.length !== groups.length && (
                      <span className="text-slate-400"> (toplam {groups.length} gruptan filtrelendi)</span>
                    )}
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

                <div className="overflow-auto border border-slate-200 rounded-xl -mx-3 sm:mx-0">
                  <table className="w-full text-xs min-w-[900px]">
                    <thead className="bg-slate-50 sticky top-0">
                      <tr>
                        <th className="px-3 py-2 text-left font-medium text-slate-600 w-[40px]">
                          <Checkbox
                            checked={selectedGroupNames.size === cleaned.length && cleaned.length > 0}
                            onCheckedChange={toggleSelectAll}
                          />
                        </th>
                        <th className="px-3 py-2 text-left font-medium text-slate-600">Grup</th>
                        <th className="px-3 py-2 text-right font-medium text-slate-600">Test</th>
                        <th className="px-3 py-2 text-left font-medium text-slate-600 min-w-[220px]">Ortalama</th>
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
                              <Checkbox
                                checked={selectedGroupNames.has(g.name)}
                                onCheckedChange={() => toggleSelectGroup(g.name)}
                              />
                            </td>
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
                              <div className="text-xs flex gap-1 flex-nowrap">
                                {typeof g.avg_faithfulness === "number" && Number.isFinite(g.avg_faithfulness) && (
                                  <span className="inline-flex items-center rounded bg-blue-50 px-1 py-0.5 text-blue-700 border border-blue-200" title="Faithfulness">
                                    <span className="font-medium">F:</span>
                                    <span>{(g.avg_faithfulness * 100).toFixed(0)}</span>
                                  </span>
                                )}
                                {typeof g.avg_answer_relevancy === "number" && Number.isFinite(g.avg_answer_relevancy) && (
                                  <span className="inline-flex items-center rounded bg-green-50 px-1 py-0.5 text-green-700 border border-green-200" title="Answer Relevancy">
                                    <span className="font-medium">AR:</span>
                                    <span>{(g.avg_answer_relevancy * 100).toFixed(0)}</span>
                                  </span>
                                )}
                                {typeof g.avg_context_precision === "number" && Number.isFinite(g.avg_context_precision) && (
                                  <span className="inline-flex items-center rounded bg-purple-50 px-1 py-0.5 text-purple-700 border border-purple-200" title="Context Precision">
                                    <span className="font-medium">CP:</span>
                                    <span>{(g.avg_context_precision * 100).toFixed(0)}</span>
                                  </span>
                                )}
                                {typeof g.avg_context_recall === "number" && Number.isFinite(g.avg_context_recall) && (
                                  <span className="inline-flex items-center rounded bg-orange-50 px-1 py-0.5 text-orange-700 border border-orange-200" title="Context Recall">
                                    <span className="font-medium">CR:</span>
                                    <span>{(g.avg_context_recall * 100).toFixed(0)}</span>
                                  </span>
                                )}
                                {typeof g.avg_answer_correctness === "number" && Number.isFinite(g.avg_answer_correctness) && (
                                  <span className="inline-flex items-center rounded bg-red-50 px-1 py-0.5 text-red-700 border border-red-200" title="Answer Correctness">
                                    <span className="font-medium">AC:</span>
                                    <span>{(g.avg_answer_correctness * 100).toFixed(0)}</span>
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

      {/* Statistics Modal */}
      <Dialog open={showStatisticsModal} onOpenChange={setShowStatisticsModal}>
        <DialogContent className="max-w-6xl max-h-[90vh] overflow-y-auto">
          <DialogHeader>
            <DialogTitle>Grup İstatistikleri - {selectedGroupNames.size} Grup Seçildi</DialogTitle>
          </DialogHeader>

          {(() => {
            const stats = calculateStatistics();
            if (!stats) return <p className="text-slate-500">İstatistik hesaplanamadı</p>;

            const StatCard = ({ title, data }: { title: string; data: ReturnType<typeof calculateStatistics>['faithfulness'] }) => {
              if (!data) return null;
              
              return (
                <div className="p-4 rounded-xl border border-slate-200 bg-gradient-to-br from-white to-slate-50">
                  <h3 className="text-sm font-semibold text-slate-900 mb-3">{title}</h3>
                  <div className="grid grid-cols-2 gap-3">
                    <div>
                      <div className="text-xs text-slate-500">Ortalama</div>
                      <div className="text-lg font-bold text-blue-600">
                        {(data.mean * 100).toFixed(1)}%
                      </div>
                    </div>
                    <div>
                      <div className="text-xs text-slate-500">Std. Sapma</div>
                      <div className="text-lg font-bold text-purple-600">
                        {(data.stdDev * 100).toFixed(1)}%
                      </div>
                    </div>
                    <div>
                      <div className="text-xs text-slate-500">Medyan</div>
                      <div className="text-base font-semibold text-slate-700">
                        {(data.median * 100).toFixed(1)}%
                      </div>
                    </div>
                    <div>
                      <div className="text-xs text-slate-500">Aralık</div>
                      <div className="text-base font-semibold text-slate-700">
                        {(data.min * 100).toFixed(1)}% - {(data.max * 100).toFixed(1)}%
                      </div>
                    </div>
                  </div>
                  
                  <div className="mt-3 pt-3 border-t border-slate-200">
                    <div className="flex items-center gap-2 text-xs">
                      <div className="flex-1">
                        <div className="h-2 bg-slate-200 rounded-full overflow-hidden">
                          <div 
                            className="h-full bg-gradient-to-r from-blue-500 to-purple-500 rounded-full transition-all"
                            style={{ width: `${data.mean * 100}%` }}
                          />
                        </div>
                      </div>
                      <span className="text-slate-600 font-medium">{data.count} grup</span>
                    </div>
                  </div>
                </div>
              );
            };

            return (
              <div className="space-y-6">
                {/* Summary */}
                <div className="p-4 rounded-xl bg-gradient-to-r from-blue-50 to-purple-50 border border-blue-200">
                  <div className="flex items-center justify-between">
                    <div>
                      <h3 className="text-lg font-bold text-slate-900">Genel Özet</h3>
                      <p className="text-sm text-slate-600 mt-1">
                        {stats.totalSelected} grup, toplam {stats.totalTests} test üzerinden hesaplanmıştır
                      </p>
                    </div>
                    <BarChart3 className="w-12 h-12 text-blue-500 opacity-50" />
                  </div>
                </div>

                {/* RAGAS Metrics */}
                <div>
                  <h3 className="text-base font-semibold text-slate-900 mb-3">RAGAS Metrikleri (Grup Ortalamaları)</h3>
                  <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                    <StatCard title="Faithfulness" data={stats.faithfulness} />
                    <StatCard title="Answer Relevancy" data={stats.answer_relevancy} />
                    <StatCard title="Context Precision" data={stats.context_precision} />
                    <StatCard title="Context Recall" data={stats.context_recall} />
                    <StatCard title="Answer Correctness" data={stats.answer_correctness} />
                  </div>
                </div>

                {/* Comparison Chart */}
                <div className="p-4 rounded-xl border border-slate-200 bg-white">
                  <h3 className="text-base font-semibold text-slate-900 mb-4">Metrik Karşılaştırması</h3>
                  <div className="space-y-3">
                    {stats.faithfulness && (
                      <div>
                        <div className="flex items-center justify-between text-xs mb-1">
                          <span className="font-medium text-slate-700">Faithfulness</span>
                          <span className="text-slate-600">{(stats.faithfulness.mean * 100).toFixed(1)}%</span>
                        </div>
                        <div className="h-3 bg-slate-100 rounded-full overflow-hidden">
                          <div 
                            className="h-full bg-blue-500 rounded-full transition-all"
                            style={{ width: `${stats.faithfulness.mean * 100}%` }}
                          />
                        </div>
                      </div>
                    )}
                    {stats.answer_relevancy && (
                      <div>
                        <div className="flex items-center justify-between text-xs mb-1">
                          <span className="font-medium text-slate-700">Answer Relevancy</span>
                          <span className="text-slate-600">{(stats.answer_relevancy.mean * 100).toFixed(1)}%</span>
                        </div>
                        <div className="h-3 bg-slate-100 rounded-full overflow-hidden">
                          <div 
                            className="h-full bg-purple-500 rounded-full transition-all"
                            style={{ width: `${stats.answer_relevancy.mean * 100}%` }}
                          />
                        </div>
                      </div>
                    )}
                    {stats.context_precision && (
                      <div>
                        <div className="flex items-center justify-between text-xs mb-1">
                          <span className="font-medium text-slate-700">Context Precision</span>
                          <span className="text-slate-600">{(stats.context_precision.mean * 100).toFixed(1)}%</span>
                        </div>
                        <div className="h-3 bg-slate-100 rounded-full overflow-hidden">
                          <div 
                            className="h-full bg-emerald-500 rounded-full transition-all"
                            style={{ width: `${stats.context_precision.mean * 100}%` }}
                          />
                        </div>
                      </div>
                    )}
                    {stats.context_recall && (
                      <div>
                        <div className="flex items-center justify-between text-xs mb-1">
                          <span className="font-medium text-slate-700">Context Recall</span>
                          <span className="text-slate-600">{(stats.context_recall.mean * 100).toFixed(1)}%</span>
                        </div>
                        <div className="h-3 bg-slate-100 rounded-full overflow-hidden">
                          <div 
                            className="h-full bg-amber-500 rounded-full transition-all"
                            style={{ width: `${stats.context_recall.mean * 100}%` }}
                          />
                        </div>
                      </div>
                    )}
                    {stats.answer_correctness && (
                      <div>
                        <div className="flex items-center justify-between text-xs mb-1">
                          <span className="font-medium text-slate-700">Answer Correctness</span>
                          <span className="text-slate-600">{(stats.answer_correctness.mean * 100).toFixed(1)}%</span>
                        </div>
                        <div className="h-3 bg-slate-100 rounded-full overflow-hidden">
                          <div 
                            className="h-full bg-rose-500 rounded-full transition-all"
                            style={{ width: `${stats.answer_correctness.mean * 100}%` }}
                          />
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              </div>
            );
          })()}
        </DialogContent>
      </Dialog>
    </div>
  );
}
