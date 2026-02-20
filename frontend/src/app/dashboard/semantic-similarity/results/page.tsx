"use client";

import { useCallback, useEffect, useState } from "react";
import { useAuth } from "@/lib/auth-context";
import { api, Course, SemanticSimilarityGroupInfo } from "@/lib/api";
import { toast } from "sonner";
import Link from "next/link";
import { BookOpen, History, ArrowLeft, RefreshCw, Trash2, BarChart3 } from "lucide-react";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Button } from "@/components/ui/button";
import { Dialog, DialogContent, DialogFooter, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Checkbox } from "@/components/ui/checkbox";

export default function SemanticSimilarityResultsPage() {
  const { user } = useAuth();
  const [courses, setCourses] = useState<Course[]>([]);
  const [selectedCourseId, setSelectedCourseId] = useState<number | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  const [groups, setGroups] = useState<SemanticSimilarityGroupInfo[]>([]);
  const [isGroupsLoading, setIsGroupsLoading] = useState(false);
  const [deletingGroupName, setDeletingGroupName] = useState<string | null>(null);

  const [isRenameOpen, setIsRenameOpen] = useState(false);
  const [renamingGroupName, setRenamingGroupName] = useState<string | null>(null);
  const [newGroupName, setNewGroupName] = useState("");
  const [isRenaming, setIsRenaming] = useState(false);

  const PAGE_SIZE = 10;
  const [page, setPage] = useState(1);
  
  // Multi-select for analysis
  const [selectedGroupNames, setSelectedGroupNames] = useState<Set<string>>(new Set());
  
  // Filters
  const [searchQuery, setSearchQuery] = useState("");
  const [filterLLM, setFilterLLM] = useState("all");
  const [filterEmbedding, setFilterEmbedding] = useState("all");
  const [filterReranker, setFilterReranker] = useState("all");

  const loadGroups = useCallback(async () => {
    if (!selectedCourseId) return;
    setIsGroupsLoading(true);
    try {
      const data = await api.getSemanticSimilarityResults(selectedCourseId, undefined, 0, 1);
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
        await api.deleteSemanticSimilarityGroup(selectedCourseId, groupName);
        toast.success("Grup silindi");
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
    if (groups.some((g) => g.group_name === nextName)) {
      toast.error("Bu isimde bir grup zaten var");
      return;
    }

    setIsRenaming(true);
    try {
      await api.renameSemanticSimilarityGroup(selectedCourseId, renamingGroupName, nextName);
      toast.success("Grup adı güncellendi");
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

  useEffect(() => {
    loadCourses();
  }, []);

  useEffect(() => {
    if (selectedCourseId) {
      loadGroups();
      setPage(1);
      setSelectedGroupNames(new Set()); // Clear selection when course changes
    }
  }, [selectedCourseId, loadGroups]);

  const loadCourses = async () => {
    try {
      const data = await api.getCourses();
      setCourses(data);
      if (data.length > 0) {
        const savedCourseId = localStorage.getItem("semantic_similarity_selected_course_id");
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
              <h1 className="text-2xl font-bold text-slate-900">Semantic Similarity Sonuçları</h1>
              <p className="text-sm text-slate-600">Test gruplarını listele ve analiz et</p>
            </div>
          </div>

          <div className="flex items-center gap-3">
            {selectedGroupNames.size > 1 && (
              <Link href={`/dashboard/semantic-similarity/analysis?groups=${Array.from(selectedGroupNames).join(",")}&course=${selectedCourseId}`}>
                <Button 
                  variant="outline"
                  className="bg-blue-50 hover:bg-blue-100 text-blue-700 border-blue-200"
                >
                  <BarChart3 className="w-4 h-4 mr-2" />
                  Karşılaştır ({selectedGroupNames.size})
                </Button>
              </Link>
            )}

            <Link href="/dashboard/semantic-similarity">
              <Button variant="outline">
                <ArrowLeft className="w-4 h-4 mr-2" />
                Geri Dön
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
                localStorage.setItem("semantic_similarity_selected_course_id", courseId.toString());
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
            // Get unique values for filters
            const uniqueLLMs = Array.from(new Set(groups.map(g => g.llm_model).filter(Boolean))) as string[];
            const uniqueEmbeddings = Array.from(new Set(groups.map(g => g.embedding_model).filter(Boolean))) as string[];

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
                  return g.llm_model === filterLLM;
                }
                return true;
              })
              .filter((g) => {
                // Embedding filter
                if (filterEmbedding !== "all") {
                  return g.embedding_model === filterEmbedding;
                }
                return true;
              })
              .filter((g) => {
                // Reranker filter
                if (filterReranker === "enabled") {
                  return g.reranker_used === true;
                } else if (filterReranker === "disabled") {
                  return g.reranker_used !== true;
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

            const totalPages = Math.max(1, Math.ceil(cleaned.length / PAGE_SIZE));
            const safePage = Math.min(Math.max(page, 1), totalPages);
            const start = (safePage - 1) * PAGE_SIZE;
            const end = Math.min(start + PAGE_SIZE, cleaned.length);
            const pageItems = cleaned.slice(start, end);

            return (
              <>
                {/* Filters */}
                <div className="mb-4 p-4 bg-slate-50 rounded-xl border border-slate-200">
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
                      <label className="text-xs font-medium text-slate-700 mb-1 block">Embedding Model</label>
                      <Select
                        value={filterEmbedding}
                        onValueChange={(v) => {
                          setFilterEmbedding(v);
                          setPage(1);
                        }}
                      >
                        <SelectTrigger className="h-9 text-sm">
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="all">Tümü</SelectItem>
                          {uniqueEmbeddings.map((emb) => (
                            <SelectItem key={emb} value={emb}>
                              {emb}
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

                  {(searchQuery || filterLLM !== "all" || filterEmbedding !== "all" || filterReranker !== "all") && (
                    <div className="mt-3 flex items-center gap-2">
                      <span className="text-xs text-slate-600">Aktif filtreler:</span>
                      <Button
                        variant="outline"
                        size="sm"
                        className="h-7 text-xs"
                        onClick={() => {
                          setSearchQuery("");
                          setFilterLLM("all");
                          setFilterEmbedding("all");
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

                <div className="overflow-auto border border-slate-200 rounded-xl">
                  <table className="w-full text-xs">
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
                        <th className="px-3 py-2 text-left font-medium text-slate-600">Ortalama Skorlar</th>
                        <th className="px-3 py-2 text-left font-medium text-slate-600">LLM Model</th>
                        <th className="px-3 py-2 text-left font-medium text-slate-600">Embedding</th>
                        <th className="px-3 py-2 text-right font-medium text-slate-600">Top-K</th>
                        <th className="px-3 py-2 text-right font-medium text-slate-600">Alpha</th>
                        <th className="px-3 py-2 text-left font-medium text-slate-600">Reranker</th>
                        <th className="px-3 py-2 text-right font-medium text-slate-600"></th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-100">
                      {pageItems.map((g, idx) => {
                        return (
                          <tr key={g.name} className={idx % 2 === 0 ? "bg-white hover:bg-slate-50" : "bg-slate-50/40 hover:bg-slate-50"}>
                            <td className="px-3 py-2">
                              <Checkbox
                                checked={selectedGroupNames.has(g.name)}
                                onCheckedChange={() => toggleSelectGroup(g.name)}
                              />
                            </td>
                            <td className="px-3 py-2">
                              <div className="font-semibold text-slate-900">
                                {g.name}
                              </div>
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
                              <div className="flex flex-nowrap gap-1.5 text-xs">
                                {(() => {
                                  return null;
                                })()}
                                {typeof g.avg_rouge1 === "number" && Number.isFinite(g.avg_rouge1) && (
                                  <span className="inline-flex items-center rounded bg-blue-50 px-1.5 py-0.5 text-blue-700 border border-blue-200">
                                    <span className="font-medium">R1:</span>
                                    <span>{(g.avg_rouge1 * 100).toFixed(0)}%</span>
                                  </span>
                                )}
                                {typeof g.avg_rouge2 === "number" && Number.isFinite(g.avg_rouge2) && (
                                  <span className="inline-flex items-center rounded bg-green-50 px-1.5 py-0.5 text-green-700 border border-green-200">
                                    <span className="font-medium">R2:</span>
                                    <span>{(g.avg_rouge2 * 100).toFixed(0)}%</span>
                                  </span>
                                )}
                                {typeof g.avg_rougel === "number" && Number.isFinite(g.avg_rougel) && (
                                  <span className="inline-flex items-center rounded bg-purple-50 px-1.5 py-0.5 text-purple-700 border border-purple-200">
                                    <span className="font-medium">RL:</span>
                                    <span>{(g.avg_rougel * 100).toFixed(0)}%</span>
                                  </span>
                                )}
                                {typeof g.avg_original_bertscore_f1 === "number" && Number.isFinite(g.avg_original_bertscore_f1) && (
                                  <span className="inline-flex items-center rounded bg-orange-50 px-1.5 py-0.5 text-orange-700 border border-orange-200">
                                    <span className="font-medium">BERT:</span>
                                    <span>{(g.avg_original_bertscore_f1 * 100).toFixed(0)}%</span>
                                  </span>
                                )}
                              </div>
                            </td>
                            <td className="px-3 py-2 max-w-[120px]">
                              <span className="inline-flex items-center rounded-full bg-indigo-50 px-2 py-0.5 text-indigo-700 border border-indigo-200 text-[11px] truncate max-w-full" title={g.llm_model || "-"}>
                                {g.llm_model ? g.llm_model.split("/").pop() : "-"}
                              </span>
                            </td>
                            <td className="px-3 py-2 max-w-[120px]">
                              <span className="inline-flex items-center rounded-full bg-slate-50 px-2 py-0.5 text-slate-700 border border-slate-200 text-[11px] truncate max-w-full" title={g.embedding_model || "-"}>
                                {g.embedding_model ? g.embedding_model.split("/").pop() : "-"}
                              </span>
                            </td>
                            <td className="px-3 py-2 text-right text-slate-700">{g.search_top_k ?? "-"}</td>
                            <td className="px-3 py-2 text-right text-slate-700">
                              {g.search_alpha !== null && g.search_alpha !== undefined ? g.search_alpha.toFixed(2) : "-"}
                            </td>
                            <td className="px-3 py-2 max-w-[110px]">
                              {(() => {
                                if (!g.reranker_used) {
                                  return (
                                    <span className="inline-flex items-center rounded-full bg-slate-100 px-2 py-0.5 text-slate-600 border border-slate-200 text-[11px]">
                                      Yok
                                    </span>
                                  );
                                }
                                const fullText = g.reranker_provider && g.reranker_model
                                  ? `${g.reranker_provider}/${g.reranker_model}`
                                  : "Var";
                                const shortText = g.reranker_model ? g.reranker_model.split("/").pop() : "Var";
                                return (
                                  <span className="inline-flex items-center rounded-full bg-emerald-50 px-2 py-0.5 text-emerald-700 border border-emerald-200 text-[11px] truncate max-w-full" title={fullText}>
                                    {shortText}
                                  </span>
                                );
                              })()}
                            </td>
                            <td className="px-3 py-2 text-right">
                              <div className="inline-flex items-center gap-1">
                                <Button
                                  variant="ghost"
                                  size="sm"
                                  className="h-8 w-8 p-0 text-purple-600 hover:text-purple-700 hover:bg-purple-50"
                                  onClick={async () => {
                                    if (!selectedCourseId) return;
                                    try {
                                      toast.loading("W&B'ye aktarılıyor...", { id: "wandb-export" });
                                      const res = await api.wandbExportSemanticSimilarityGroup({
                                        course_id: selectedCourseId,
                                        group_name: g.name,
                                      });
                                      if (res.run_url) {
                                        toast.success(`Aktarıldı: ${res.exported_count} kayıt`, { id: "wandb-export" });
                                        window.open(res.run_url, "_blank");
                                      } else {
                                        toast.success(`Aktarıldı: ${res.exported_count} kayıt`, { id: "wandb-export" });
                                      }
                                    } catch (e) {
                                      const msg = e instanceof Error ? e.message : "W&B aktarımı başarısız";
                                      toast.error(msg, { id: "wandb-export" });
                                    }
                                  }}
                                  title="W&B'ye gönder"
                                >
                                  <BarChart3 className="w-4 h-4" />
                                </Button>
                                <Button
                                  variant="ghost"
                                  size="sm"
                                  className="h-8 w-8 p-0"
                                  onClick={() => openRenameDialog(g.name)}
                                  title="Grup adını değiştir"
                                >
                                  <History className="w-4 h-4" />
                                </Button>
                                <Button
                                  variant="ghost"
                                  size="sm"
                                  className="h-8 w-8 p-0 text-red-600 hover:text-red-700 hover:bg-red-50"
                                  onClick={() => handleDeleteGroup(g.name, g.test_count)}
                                  disabled={deletingGroupName === g.name}
                                  title="Grubu sil"
                                >
                                  {deletingGroupName === g.name ? (
                                    <RefreshCw className="w-4 h-4 animate-spin" />
                                  ) : (
                                    <Trash2 className="w-4 h-4" />
                                  )}
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
