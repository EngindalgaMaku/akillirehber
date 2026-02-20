"use client";

import { useEffect, useState, useCallback, useRef } from "react";
import { useAuth } from "@/lib/auth-context";
import { api, Course } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Card } from "@/components/ui/card";
import { 
  Select, 
  SelectContent, 
  SelectItem, 
  SelectTrigger, 
  SelectValue 
} from "@/components/ui/select";
import { 
  Dialog, 
  DialogContent, 
  DialogDescription, 
  DialogHeader, 
  DialogTitle 
} from "@/components/ui/dialog";
import { toast } from "sonner";
import {
  Settings,
  RefreshCw,
  ExternalLink,
  Loader2,
  Database,
  Clock,
  CheckCircle,
  AlertTriangle,
  XCircle,
  Search,
  Filter,
  ArrowUpDown,
  Tag,
  Eye,
  Edit,
  Save,
  ChevronLeft,
  ChevronRight,
} from "lucide-react";

interface WandbRun {
  id: string;
  name: string;
  state: string;
  created_at: string | null;
  config: Record<string, unknown>;
  missing_fields: string[];
}

type SortField = "name" | "state" | "created_at";
type SortOrder = "asc" | "desc";

export default function WandbRunsPage() {
  const { user } = useAuth();
  const [courses, setCourses] = useState<Course[]>([]);
  const [selectedCourseId, setSelectedCourseId] = useState<number | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [isLoadingRuns, setIsLoadingRuns] = useState(false);
  
  // W&B Runs State
  const [wandbRuns, setWandbRuns] = useState<WandbRun[]>([]);
  const [paginationInfo, setPaginationInfo] = useState<{
    currentPage: number;
    totalPages: number;
    totalItems: number;
    itemsPerPage: number;
  } | null>(null);
  const [selectedRun, setSelectedRun] = useState<WandbRun | null>(null);
  const [updatingRunIds, setUpdatingRunIds] = useState<Set<string>>(new Set());
  
  // Filter and Sort State
  const [searchQuery, setSearchQuery] = useState("");
  const [stateFilter, setStateFilter] = useState<string>("all");
  const [tagFilter, setTagFilter] = useState<string>("");
  const [sortField, setSortField] = useState<SortField>("created_at");
  const [sortOrder, setSortOrder] = useState<SortOrder>("desc");
  
  // Pagination State
  const [currentPage, setCurrentPage] = useState(1);
  const [itemsPerPage, setItemsPerPage] = useState(25);
  
  // Edit Dialog State
  const [editingRun, setEditingRun] = useState<WandbRun | null>(null);
  const [editForm, setEditForm] = useState<Record<string, any>>({});
  const [isSaving, setIsSaving] = useState(false);

  // Debounce for search
  const debounceRef = useRef<NodeJS.Timeout | null>(null);

  const handleSearchChange = (value: string) => {
    setSearchQuery(value);
    if (debounceRef.current) {
      clearTimeout(debounceRef.current);
    }
    debounceRef.current = setTimeout(() => {
      setCurrentPage(1); // Reset to first page when searching
    }, 500); // 500ms debounce
  };

  const loadCourses = async () => {
    try {
      const data = await api.getCourses();
      setCourses(data);
      if (data.length > 0) {
        const savedCourseId = localStorage.getItem('wandb_runs_selected_course_id');
        if (savedCourseId && data.find(c => c.id === parseInt(savedCourseId))) {
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

  const loadWandbRuns = useCallback(async () => {
    if (!selectedCourseId) return;
    setIsLoadingRuns(true);
    try {
      const data = await api.getWandbRuns(
        selectedCourseId, 
        currentPage, 
        itemsPerPage, 
        searchQuery, 
        stateFilter, 
        tagFilter
      );
      setWandbRuns(data.runs);
      setPaginationInfo(data.pagination);
    } catch (e) {
      const msg = e instanceof Error ? e.message : "W&B run'ları alınamadı";
      toast.error(msg);
    } finally {
      setIsLoadingRuns(false);
    }
  }, [selectedCourseId, currentPage, itemsPerPage, searchQuery, stateFilter, tagFilter]);

  useEffect(() => {
    loadCourses();
  }, []);

  useEffect(() => {
    if (selectedCourseId) {
      loadWandbRuns();
    }
  }, [loadWandbRuns]);

  useEffect(() => {
    setCurrentPage(1);
  }, [searchQuery, stateFilter, tagFilter, sortField, sortOrder]);

  // Pagination calculations from backend
  const totalPages = paginationInfo?.totalPages || 1;
  const totalItems = paginationInfo?.totalItems || 0;
  const paginatedRuns = wandbRuns; // Backend already paginated
  
  const handlePageChange = (page: number) => {
    setCurrentPage(page);
  };
  
  const handleItemsPerPageChange = (value: string) => {
    const newItemsPerPage = parseInt(value);
    setItemsPerPage(newItemsPerPage);
    setCurrentPage(1);
  };
  
  const openEditDialog = (run: WandbRun) => {
    setEditingRun(run);
    setEditForm({ ...run.config });
  };
  
  const closeEditDialog = () => {
    setEditingRun(null);
    setEditForm({});
  };
  
  const handleEditFormChange = (field: string, value: any) => {
    setEditForm(prev => ({
      ...prev,
      [field]: value
    }));
  };
  
  const saveEditedRun = async () => {
    if (!editingRun || !selectedCourseId) {
      toast.error("Düzenleme için gerekli bilgiler eksik");
      return;
    }
    
    setIsSaving(true);
    try {
      const res = await api.updateWandbRun({
        run_id: editingRun.id,
        group_name: editForm.group_name || editingRun.config.group_name,
        course_id: selectedCourseId,
        tags: editForm.tags,
        llm_model_used: editForm.llm_model_used,
        embedding_model_used: editForm.embedding_model_used,
        llm_provider: editForm.llm_provider,
        total_tests: editForm.total_tests,
      });
      
      if (res.success) {
        toast.success(`Run başarıyla güncellendi: ${res.run_name}`);
        closeEditDialog();
        await loadWandbRuns();
      } else {
        toast.error(res.message || "Güncelleme başarısız");
      }
    } catch (e) {
      const msg = e instanceof Error ? e.message : "Güncelleme hatası";
      toast.error(msg);
    } finally {
      setIsSaving(false);
    }
  };

  const updateSelectedRun = async (run: WandbRun) => {
    if (!selectedCourseId) {
      toast.error("Lütfen bir ders seçin");
      return;
    }
    const groupName = run.config.group_name as string | undefined;
    if (!groupName) {
      toast.error("Run'ın group_name bilgisi bulunamadı");
      return;
    }
    setUpdatingRunIds((prev) => new Set(prev).add(run.id));
    try {
      const res = await api.updateWandbRun({
        run_id: run.id,
        group_name: groupName,
        course_id: selectedCourseId,
      });
      if (res.success) {
        toast.success(`Güncellendi: ${res.run_name} (${res.updated_fields?.join(", ")})`);
        // Refresh runs list
        await loadWandbRuns();
      } else {
        toast.error(res.message || "Güncelleme başarısız");
      }
    } catch (e) {
      const msg = e instanceof Error ? e.message : "Güncelleme hatası";
      toast.error(msg);
    } finally {
      setUpdatingRunIds((prev) => {
        const next = new Set(prev);
        next.delete(run.id);
        return next;
      });
    }
  };

  const getStateIcon = (state: string) => {
    switch (state.toLowerCase()) {
      case "finished":
      case "completed":
        return <CheckCircle className="w-4 h-4 text-emerald-600" />;
      case "running":
      case "pending":
        return <Loader2 className="w-4 h-4 text-blue-600 animate-spin" />;
      case "failed":
      case "crashed":
        return <XCircle className="w-4 h-4 text-red-600" />;
      default:
        return <AlertTriangle className="w-4 h-4 text-amber-600" />;
    }
  };

  const getStateColor = (state: string) => {
    switch (state.toLowerCase()) {
      case "finished":
      case "completed":
        return "bg-emerald-100 text-emerald-700 border-emerald-200";
      case "running":
      case "pending":
        return "bg-blue-100 text-blue-700 border-blue-200";
      case "failed":
      case "crashed":
        return "bg-red-100 text-red-700 border-red-200";
      default:
        return "bg-slate-100 text-slate-700 border-slate-200";
    }
  };

  const handleSort = (field: SortField) => {
    if (sortField === field) {
      setSortOrder(sortOrder === "asc" ? "desc" : "asc");
    } else {
      setSortField(field);
      setSortOrder("desc");
    }
  };

  if (!user) return null;

  if (isLoading) {
    return (
      <div className="min-h-[60vh] flex items-center justify-center">
        <div className="text-center">
          <div className="relative">
            <div className="w-16 h-16 border-4 border-indigo-200 border-t-indigo-600 rounded-full animate-spin mx-auto"></div>
            <Database className="w-6 h-6 text-indigo-600 absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2" />
          </div>
          <p className="mt-4 text-slate-600 font-medium">Yükleniyor...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Hero Header */}
      <div className="relative overflow-hidden bg-gradient-to-br from-indigo-600 via-purple-600 to-pink-600 rounded-2xl p-8 text-white shadow-xl">
        <div className="absolute inset-0 opacity-30" style={{backgroundImage: "url(\"data:image/svg+xml,%3Csvg width='60' height='60' viewBox='0 0 60 60' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' fill-rule='evenodd'%3E%3Cg fill='%23fff' fill-opacity='0.1'%3E%3Ccircle cx='30' cy='30' r='2'/%3E%3C/g%3E%3C/svg%3E\")"}}></div>
        <div className="absolute -top-24 -right-24 w-96 h-96 bg-white/10 rounded-full blur-3xl"></div>
        <div className="absolute -bottom-24 -left-24 w-96 h-96 bg-indigo-500/20 rounded-full blur-3xl"></div>
        
        <div className="relative z-10">
          <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between gap-6">
            <div className="flex items-center gap-4">
              <div className="p-3 bg-white/20 rounded-xl backdrop-blur-sm">
                <Database className="w-8 h-8" />
              </div>
              <div>
                <h1 className="text-3xl font-bold">W&B Runs</h1>
                <p className="text-indigo-200 mt-1">Weights & Biases run'larını yönetin</p>
              </div>
            </div>
            
            <div className="flex flex-wrap items-center gap-3">
              <Button
                variant="secondary"
                size="sm"
                className="bg-white/20 hover:bg-white/30 text-white border-0 backdrop-blur-sm h-10"
                onClick={loadWandbRuns}
                disabled={isLoadingRuns}
              >
                {isLoadingRuns ? (
                  <>
                    <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                    Yükleniyor...
                  </>
                ) : (
                  <>
                    <RefreshCw className="w-4 h-4 mr-2" />
                    Yenile
                  </>
                )}
              </Button>

              <Select 
                value={selectedCourseId?.toString() || ""} 
                onValueChange={(v) => {
                  const courseId = Number(v);
                  setSelectedCourseId(courseId);
                  localStorage.setItem('wandb_runs_selected_course_id', courseId.toString());
                }}
              >
                <SelectTrigger className="w-56 bg-white/20 border-0 text-white hover:bg-white/30 backdrop-blur-sm h-10">
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

          {selectedCourseId && (
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mt-8">
              <div className="bg-white/10 backdrop-blur-sm rounded-xl p-4 border border-white/20">
                <div className="flex items-center gap-3">
                  <div className="p-2 bg-white/20 rounded-lg">
                    <Database className="w-5 h-5" />
                  </div>
                  <div>
                    <p className="text-indigo-200 text-sm">Toplam Run</p>
                    <p className="text-2xl font-bold">{wandbRuns.length}</p>
                  </div>
                </div>
              </div>
              <div className="bg-white/10 backdrop-blur-sm rounded-xl p-4 border border-white/20">
                <div className="flex items-center gap-3">
                  <div className="p-2 bg-white/20 rounded-lg">
                    <CheckCircle className="w-5 h-5" />
                  </div>
                  <div>
                    <p className="text-indigo-200 text-sm">Tamamlanan</p>
                    <p className="text-2xl font-bold">{wandbRuns.filter(r => r.state === 'finished' || r.state === 'completed').length}</p>
                  </div>
                </div>
              </div>
              <div className="bg-white/10 backdrop-blur-sm rounded-xl p-4 border border-white/20">
                <div className="flex items-center gap-3">
                  <div className="p-2 bg-white/20 rounded-lg">
                    <Loader2 className="w-5 h-5" />
                  </div>
                  <div>
                    <p className="text-indigo-200 text-sm">Devam Eden</p>
                    <p className="text-2xl font-bold">{wandbRuns.filter(r => r.state === 'running' || r.state === 'pending').length}</p>
                  </div>
                </div>
              </div>
              <div className="bg-white/10 backdrop-blur-sm rounded-xl p-4 border border-white/20">
                <div className="flex items-center gap-3">
                  <div className="p-2 bg-white/20 rounded-lg">
                    <AlertTriangle className="w-5 h-5" />
                  </div>
                  <div>
                    <p className="text-indigo-200 text-sm">Güncelleme Gereken</p>
                    <p className="text-2xl font-bold">{wandbRuns.filter(r => r.missing_fields.length > 0).length}</p>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>

      {!selectedCourseId ? (
        <div className="bg-white rounded-2xl border border-slate-200 p-16 text-center shadow-sm">
          <div className="w-20 h-20 bg-indigo-100 rounded-full flex items-center justify-center mx-auto mb-6">
            <Database className="w-10 h-10 text-indigo-600" />
          </div>
          <h3 className="text-xl font-semibold text-slate-900 mb-2">Ders Seçin</h3>
          <p className="text-slate-500 max-w-md mx-auto">
            W&B run'larını görüntülemek için yukarıdan bir ders seçin.
          </p>
        </div>
      ) : (
        <div className="space-y-6">
          {/* Filters Card */}
          <Card className="p-6">
            <div className="flex flex-col lg:flex-row gap-4">
              <div className="flex-1">
                <Label className="text-sm font-medium text-slate-700 mb-2 block">Ara</Label>
                <div className="relative">
                  <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-slate-400" />
                  <Input
                    value={searchQuery}
                    onChange={(e) => handleSearchChange(e.target.value)}
                    placeholder="Run adı, ID veya grup adı ara..."
                    className="pl-10"
                  />
                </div>
              </div>
              
              <div className="w-full lg:w-48">
                <Label className="text-sm font-medium text-slate-700 mb-2 block">Durum</Label>
                <Select value={stateFilter} onValueChange={setStateFilter}>
                  <SelectTrigger>
                    <SelectValue placeholder="Tümü" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="all">Tümü</SelectItem>
                    <SelectItem value="finished">Tamamlandı</SelectItem>
                    <SelectItem value="running">Devam Ediyor</SelectItem>
                    <SelectItem value="failed">Başarısız</SelectItem>
                    <SelectItem value="pending">Beklemede</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              
              <div className="w-full lg:w-48">
                <Label className="text-sm font-medium text-slate-700 mb-2 block">Tag</Label>
                <div className="relative">
                  <Tag className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-slate-400" />
                  <Input
                    value={tagFilter}
                    onChange={(e) => setTagFilter(e.target.value)}
                    placeholder="Tag ile filtrele..."
                    className="pl-10"
                  />
                </div>
              </div>

              <div className="w-full lg:w-48">
                <Label className="text-sm font-medium text-slate-700 mb-2 block">Sırala</Label>
                <Select 
                  value={`${sortField}-${sortOrder}`} 
                  onValueChange={(v) => {
                    const [field, order] = v.split('-') as [SortField, SortOrder];
                    setSortField(field);
                    setSortOrder(order);
                  }}
                >
                  <SelectTrigger>
                    <SelectValue placeholder="Oluşturma Tarihi" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="created_at-desc">Oluşturma Tarihi (Yeni)</SelectItem>
                    <SelectItem value="created_at-asc">Oluşturma Tarihi (Eski)</SelectItem>
                    <SelectItem value="name-asc">Ad (A-Z)</SelectItem>
                    <SelectItem value="name-desc">Ad (Z-A)</SelectItem>
                    <SelectItem value="state-asc">Durum (A-Z)</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            </div>
          </Card>

          {/* Runs List */}
          {isLoadingRuns ? (
            <div className="bg-white rounded-2xl border border-slate-200 p-16 text-center">
              <Loader2 className="w-8 h-8 animate-spin mx-auto mb-4 text-indigo-600" />
              <p className="text-slate-600">Run'lar yükleniyor...</p>
            </div>
          ) : paginatedRuns.length === 0 ? (
            <div className="bg-white rounded-2xl border border-slate-200 p-16 text-center">
              <Database className="w-16 h-16 mx-auto mb-4 text-slate-300" />
              <h3 className="text-xl font-semibold text-slate-900 mb-2">Run Bulunamadı</h3>
              <p className="text-slate-500 max-w-md mx-auto">
                {searchQuery || stateFilter !== "all" 
                  ? "Arama kriterlerinize uygun run bulunamadı."
                  : "Bu ders için henüz W&B run'ı yok."}
              </p>
            </div>
          ) : (
            <div>
              <div className="grid gap-4">
              {paginatedRuns.map((run) => (
                <Card key={run.id} className="p-6 hover:shadow-lg transition-shadow">
                  <div className="flex flex-col lg:flex-row lg:items-start lg:justify-between gap-4">
                    <div className="flex-1 min-w-0">
                      <div className="flex items-start gap-3 mb-3">
                        <div className={`p-2 rounded-lg border ${getStateColor(run.state)}`}>
                          {getStateIcon(run.state)}
                        </div>
                        <div className="flex-1 min-w-0">
                          <h3 className="text-lg font-semibold text-slate-900 truncate mb-1">
                            {run.name}
                          </h3>
                          <div className="flex flex-wrap items-center gap-3 text-sm text-slate-600">
                            <span className="flex items-center gap-1">
                              <Clock className="w-3 h-3" />
                              {run.created_at 
                                ? new Date(run.created_at).toLocaleString("tr-TR")
                                : "Tarih bilinmiyor"}
                            </span>
                            <span className="flex items-center gap-1">
                              <Database className="w-3 h-3" />
                              ID: {run.id}
                            </span>
                          </div>
                        </div>
                      </div>

                      {/* Config Details */}
                      <div className="space-y-2 mb-3">
                        {run.config.group_name && (
                          <div className="flex items-center gap-2 text-sm">
                            <span className="text-slate-500 font-medium">Grup:</span>
                            <span className="text-slate-900 font-medium bg-slate-100 px-2 py-0.5 rounded">
                              {run.config.group_name as string}
                            </span>
                          </div>
                        )}
                        {run.config.llm_model_used && (
                          <div className="flex items-center gap-2 text-sm">
                            <span className="text-slate-500 font-medium">LLM Model:</span>
                            <span className="px-2 py-1 bg-gradient-to-r from-blue-500 to-indigo-600 text-white rounded-lg text-xs font-semibold shadow-sm">
                              {run.config.llm_model_used as string}
                            </span>
                          </div>
                        )}
                        {run.config.embedding_model_used && (
                          <div className="flex items-center gap-2 text-sm">
                            <span className="text-slate-500 font-medium">Embedding Model:</span>
                            <span className="px-2 py-1 bg-gradient-to-r from-emerald-500 to-teal-600 text-white rounded-lg text-xs font-semibold shadow-sm">
                              {run.config.embedding_model_used as string}
                            </span>
                          </div>
                        )}
                        <div className="flex items-center gap-2 text-sm">
                          <span className="text-slate-500 font-medium">Tags:</span>
                          <div className="flex flex-wrap gap-1">
                            {((run.config.tags as string[]) && (run.config.tags as string[]).length > 0) 
                              ? (run.config.tags as string[]).map((tag, index) => (
                                  <span
                                    key={index}
                                    className="px-2 py-1 bg-indigo-100 text-indigo-800 rounded-md text-xs font-medium"
                                  >
                                    {tag}
                                  </span>
                                ))
                              : (
                                <span className="text-slate-400 text-xs italic">Henüz tag eklenmemiş</span>
                              )
                            }
                          </div>
                        </div>
                      </div>

                      {/* Missing Fields Warning */}
                      {run.missing_fields.length > 0 && (
                        <div className="bg-amber-50 border border-amber-200 rounded-lg p-3">
                          <div className="flex items-start gap-2">
                            <AlertTriangle className="w-5 h-5 text-amber-600 flex-shrink-0 mt-0.5" />
                            <div>
                              <p className="text-sm font-semibold text-amber-900 mb-1">
                                Eksik Alanlar
                              </p>
                              <div className="flex flex-wrap gap-2">
                                {run.missing_fields.map((field) => (
                                  <span
                                    key={field}
                                    className="px-2 py-1 bg-amber-100 text-amber-800 rounded text-xs font-medium"
                                  >
                                    {field}
                                  </span>
                                ))}
                              </div>
                            </div>
                          </div>
                        </div>
                      )}
                    </div>

                    {/* Actions */}
                    <div className="flex flex-col gap-2 lg:w-auto">
                      {run.missing_fields.length > 0 ? (
                        <Button
                          variant="default"
                          size="sm"
                          onClick={() => updateSelectedRun(run)}
                          disabled={updatingRunIds.has(run.id)}
                          className="bg-gradient-to-r from-indigo-600 to-purple-600 hover:from-indigo-700 hover:to-purple-700"
                        >
                          {updatingRunIds.has(run.id) ? (
                            <>
                              <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                              Güncelleniyor...
                            </>
                          ) : (
                            <>
                              <RefreshCw className="w-4 h-4 mr-2" />
                              Güncelle
                            </>
                          )}
                        </Button>
                      ) : (
                        <div className="flex items-center justify-center h-9 px-4 bg-emerald-50 text-emerald-700 rounded-lg border border-emerald-200 text-sm font-medium">
                          <CheckCircle className="w-4 h-4 mr-2" />
                          Güncel
                        </div>
                      )}
                      
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() => setSelectedRun(run)}
                        className="border-indigo-200 text-indigo-600 hover:bg-indigo-50"
                      >
                        <Eye className="w-4 h-4 mr-2" />
                        Detaylar
                      </Button>
                      
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() => openEditDialog(run)}
                        className="border-emerald-200 text-emerald-600 hover:bg-emerald-50"
                      >
                        <Edit className="w-4 h-4 mr-2" />
                        Düzenle
                      </Button>
                    </div>
                  </div>
                </Card>
              ))}
            </div>
            
            {totalPages > 1 && (
              <div className="flex flex-col sm:flex-row items-center justify-between gap-4 mt-6 p-4 bg-white rounded-xl border border-slate-200">
                <div className="flex items-center gap-2 text-sm text-slate-600">
                  <span>Sayfa {currentPage} / {totalPages}</span>
                  <span className="text-slate-400">•</span>
                  <span>Toplam {totalItems} run</span>
                </div>
                
                <div className="flex items-center gap-2">
                  <Select value={itemsPerPage.toString()} onValueChange={handleItemsPerPageChange}>
                    <SelectTrigger className="w-20 h-8">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="5">5</SelectItem>
                      <SelectItem value="10">10</SelectItem>
                      <SelectItem value="20">20</SelectItem>
                      <SelectItem value="50">50</SelectItem>
                    </SelectContent>
                  </Select>
                  
                  <div className="flex items-center gap-1">
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => handlePageChange(currentPage - 1)}
                      disabled={currentPage === 1}
                      className="h-8 w-8 p-0"
                    >
                      <ChevronLeft className="w-4 h-4" />
                    </Button>
                    
                    <div className="flex items-center gap-1">
                      {Array.from({ length: Math.min(5, totalPages) }, (_, i) => {
                        const pageNum: number = (() => {
                          if (totalPages <= 5) {
                            return i + 1;
                          } else if (currentPage <= 3) {
                            return i + 1;
                          } else if (currentPage >= totalPages - 2) {
                            return totalPages - 4 + i;
                          } else {
                            return currentPage - 2 + i;
                          }
                        })();
                        
                        return (
                          <Button
                            key={pageNum}
                            variant={currentPage === pageNum ? "default" : "outline"}
                            size="sm"
                            onClick={() => handlePageChange(pageNum)}
                            className="h-8 w-8 p-0"
                          >
                            {pageNum}
                          </Button>
                        );
                      })}
                    </div>
                    
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => handlePageChange(currentPage + 1)}
                      disabled={currentPage === totalPages}
                      className="h-8 w-8 p-0"
                    >
                      <ChevronRight className="w-4 h-4" />
                    </Button>
                  </div>
                </div>
              </div>
            )}
            </div>
          )}
        </div>
      )}

      {/* View Details Dialog */}
      <Dialog open={!!selectedRun} onOpenChange={(open) => !open && setSelectedRun(null)}>
        <DialogContent className="max-w-3xl max-h-[90vh] overflow-y-auto">
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2">
              <Database className="w-5 h-5 text-indigo-600" />
              Run Detayları
            </DialogTitle>
            <DialogDescription>
              {selectedRun?.name}
            </DialogDescription>
          </DialogHeader>
          
          {selectedRun && (
            <div className="space-y-6 py-4">
              {/* Status Info */}
              <div className="flex items-center gap-4 p-4 bg-slate-50 rounded-xl">
                <div className={`p-3 rounded-lg border ${getStateColor(selectedRun.state)}`}>
                  {getStateIcon(selectedRun.state)}
                </div>
                <div>
                  <p className="text-sm text-slate-600">Durum</p>
                  <p className="text-lg font-bold text-slate-900">
                    {selectedRun.state}
                  </p>
                </div>
                <div className="ml-auto text-right">
                  <p className="text-sm text-slate-600">Oluşturulma Tarihi</p>
                  <p className="text-sm font-medium text-slate-900">
                    {selectedRun.created_at 
                      ? new Date(selectedRun.created_at).toLocaleString("tr-TR")
                      : "Bilinmiyor"}
                  </p>
                </div>
              </div>

              {/* Configuration */}
              <div>
                <h3 className="text-lg font-semibold text-slate-900 mb-4 flex items-center gap-2">
                  <Settings className="w-5 h-5 text-indigo-600" />
                  Yapılandırma
                </h3>
                <div className="space-y-3">
                  <div className="grid grid-cols-2 gap-4">
                    <div className="p-4 bg-white rounded-lg border border-slate-200">
                      <p className="text-xs text-slate-600 mb-1">Run ID</p>
                      <p className="text-sm font-mono font-semibold text-slate-900 break-all">
                        {selectedRun.id}
                      </p>
                    </div>
                    <div className="p-4 bg-white rounded-lg border border-slate-200">
                      <p className="text-xs text-slate-600 mb-1">Ders ID</p>
                      <p className="text-sm font-semibold text-slate-900">
                        {selectedRun.config.course_id as number || "-"}
                      </p>
                    </div>
                  </div>
                  
                  <div className="grid grid-cols-2 gap-4">
                    <div className="p-4 bg-gradient-to-br from-purple-50 to-pink-50 rounded-lg border border-purple-200">
                      <p className="text-xs text-purple-600 mb-1">Grup Adı</p>
                      <p className="text-sm font-semibold text-slate-900">
                        {selectedRun.config.group_name as string || "-"}
                      </p>
                    </div>
                    <div className="p-4 bg-gradient-to-br from-blue-50 to-cyan-50 rounded-lg border border-blue-200">
                      <p className="text-xs text-blue-600 mb-1">Test Sayısı</p>
                      <p className="text-sm font-semibold text-slate-900">
                        {selectedRun.config.total_tests as number || "-"}
                      </p>
                    </div>
                  </div>

                  <div className="grid grid-cols-2 gap-4">
                    <div className="p-4 bg-gradient-to-br from-indigo-50 to-purple-50 rounded-lg border border-indigo-200">
                      <p className="text-xs text-indigo-600 mb-1">LLM Model</p>
                      <p className="text-sm font-semibold text-slate-900 break-all">
                        {selectedRun.config.llm_model_used as string || "-"}
                      </p>
                    </div>
                    <div className="p-4 bg-gradient-to-br from-teal-50 to-cyan-50 rounded-lg border border-teal-200">
                      <p className="text-xs text-teal-600 mb-1">Embedding Model</p>
                      <p className="text-sm font-semibold text-slate-900 break-all">
                        {selectedRun.config.embedding_model_used as string || "-"}
                      </p>
                    </div>
                  </div>

                  {selectedRun.config.llm_provider && (
                    <div className="p-4 bg-white rounded-lg border border-slate-200">
                      <p className="text-xs text-slate-600 mb-1">LLM Provider</p>
                      <p className="text-sm font-semibold text-slate-900">
                        {selectedRun.config.llm_provider as string}
                      </p>
                    </div>
                  )}
                  
                  <div className="p-4 bg-white rounded-lg border border-slate-200">
                    <p className="text-xs text-slate-600 mb-1">Tags</p>
                    <div className="flex flex-wrap gap-1">
                      {((selectedRun.config.tags as string[]) && (selectedRun.config.tags as string[]).length > 0)
                        ? (selectedRun.config.tags as string[]).map((tag, index) => (
                            <span
                              key={index}
                              className="px-2 py-1 bg-indigo-100 text-indigo-800 rounded-md text-xs font-medium"
                            >
                              {tag}
                            </span>
                          ))
                        : (
                          <span className="text-slate-400 text-xs italic">Henüz tag eklenmemiş</span>
                        )
                      }
                    </div>
                  </div>
                </div>
              </div>

              {/* Missing Fields */}
              {selectedRun.missing_fields.length > 0 && (
                <div>
                  <h3 className="text-lg font-semibold text-slate-900 mb-4 flex items-center gap-2">
                    <AlertTriangle className="w-5 h-5 text-amber-600" />
                    Eksik Alanlar
                  </h3>
                  <div className="bg-amber-50 border border-amber-200 rounded-xl p-4">
                    <div className="flex flex-wrap gap-2">
                      {selectedRun.missing_fields.map((field) => (
                        <span
                          key={field}
                          className="px-3 py-1.5 bg-amber-100 text-amber-800 rounded-lg text-sm font-medium"
                        >
                          {field}
                        </span>
                      ))}
                    </div>
                    <p className="text-sm text-amber-700 mt-3">
                      Bu alanlar veritabanındaki bilgiler kullanılarak güncellenebilir.
                      Yukarıdaki "Güncelle" butonunu kullanın.
                    </p>
                  </div>
                </div>
              )}

              {/* Raw Config */}
              <div>
                <h3 className="text-lg font-semibold text-slate-900 mb-4 flex items-center gap-2">
                  <Database className="w-5 h-5 text-slate-600" />
                  Ham Yapılandırma (JSON)
                </h3>
                <div className="bg-slate-900 rounded-xl p-4 overflow-x-auto">
                  <pre className="text-xs text-green-400 font-mono">
                    {JSON.stringify(selectedRun.config, null, 2)}
                  </pre>
                </div>
              </div>
            </div>
          )}
          
          <div className="flex justify-end">
            <Button variant="outline" onClick={() => setSelectedRun(null)}>
              Kapat
            </Button>
          </div>
        </DialogContent>
      </Dialog>

      {/* Edit Dialog */}
      <Dialog open={!!editingRun} onOpenChange={(open) => !open && closeEditDialog()}>
        <DialogContent className="max-w-2xl max-h-[90vh] overflow-y-auto">
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2">
              <Edit className="w-5 h-5 text-emerald-600" />
              Run Düzenle
            </DialogTitle>
            <DialogDescription>
              {editingRun?.name}
            </DialogDescription>
          </DialogHeader>
          
          {editingRun && (
            <div className="space-y-6 py-4">
              {/* Basic Info */}
              <div className="p-4 bg-slate-50 rounded-xl">
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <p className="text-xs text-slate-600 mb-1">Run ID</p>
                    <p className="text-sm font-mono font-semibold text-slate-900 break-all">
                      {editingRun.id}
                    </p>
                  </div>
                  <div>
                    <p className="text-xs text-slate-600 mb-1">Durum</p>
                    <div className={`inline-flex items-center gap-1 px-2 py-1 rounded-lg border text-xs font-medium ${getStateColor(editingRun.state)}`}>
                      {getStateIcon(editingRun.state)}
                      {editingRun.state}
                    </div>
                  </div>
                </div>
              </div>

              {/* Edit Form */}
              <div className="space-y-4">
                <h3 className="text-lg font-semibold text-slate-900 flex items-center gap-2">
                  <Settings className="w-5 h-5 text-indigo-600" />
                  Yapılandırma Düzenle
                </h3>
                
                <div className="grid grid-cols-1 gap-4">
                  <div>
                    <Label className="text-sm font-medium text-slate-700 mb-2 block">
                      Grup Adı
                    </Label>
                    <Input
                      value={editForm.group_name || ""}
                      onChange={(e) => handleEditFormChange("group_name", e.target.value)}
                      placeholder="Grup adını girin"
                    />
                  </div>
                  
                  <div>
                    <Label className="text-sm font-medium text-slate-700 mb-2 block">
                      LLM Model
                    </Label>
                    <Input
                      value={editForm.llm_model_used || ""}
                      onChange={(e) => handleEditFormChange("llm_model_used", e.target.value)}
                      placeholder="LLM modelini girin"
                    />
                  </div>
                  
                  <div>
                    <Label className="text-sm font-medium text-slate-700 mb-2 block">
                      Embedding Model
                    </Label>
                    <Input
                      value={editForm.embedding_model_used || ""}
                      onChange={(e) => handleEditFormChange("embedding_model_used", e.target.value)}
                      placeholder="Embedding modelini girin"
                    />
                  </div>
                  
                  <div>
                    <Label className="text-sm font-medium text-slate-700 mb-2 block">
                      LLM Provider
                    </Label>
                    <Input
                      value={editForm.llm_provider || ""}
                      onChange={(e) => handleEditFormChange("llm_provider", e.target.value)}
                      placeholder="LLM provider'ı girin"
                    />
                  </div>
                  
                  <div>
                    <Label className="text-sm font-medium text-slate-700 mb-2 block">
                      Test Sayısı
                    </Label>
                    <Input
                      type="number"
                      value={editForm.total_tests || ""}
                      onChange={(e) => handleEditFormChange("total_tests", parseInt(e.target.value) || 0)}
                      placeholder="Test sayısını girin"
                    />
                  </div>
                  
                  <div>
                    <Label className="text-sm font-medium text-slate-700 mb-2 block">
                      Tags
                    </Label>
                    <Input
                      value={editForm.tags ? (Array.isArray(editForm.tags) ? editForm.tags.join(", ") : editForm.tags) : ""}
                      onChange={(e) => handleEditFormChange("tags", e.target.value.split(",").map(tag => tag.trim()).filter(tag => tag))}
                      placeholder="Virgülle ayrılmış tag'ler girin (örn: production, v1.0, test)"
                    />
                    <p className="text-xs text-slate-500 mt-1">
                      Virgülle ayrılmış birden fazla tag ekleyebilirsiniz
                    </p>
                  </div>
                </div>
              </div>

              {/* Current vs Preview */}
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <h4 className="text-sm font-semibold text-slate-700 mb-2">Mevcut Değerler</h4>
                  <div className="bg-slate-100 rounded-lg p-3 text-xs">
                    <pre className="text-slate-600">
                      {JSON.stringify(editingRun.config, null, 2)}
                    </pre>
                  </div>
                </div>
                <div>
                  <h4 className="text-sm font-semibold text-slate-700 mb-2">Yeni Değerler</h4>
                  <div className="bg-emerald-50 rounded-lg p-3 text-xs">
                    <pre className="text-emerald-700">
                      {JSON.stringify(editForm, null, 2)}
                    </pre>
                  </div>
                </div>
              </div>
            </div>
          )}
          
          <div className="flex justify-end gap-3">
            <Button variant="outline" onClick={closeEditDialog}>
              İptal
            </Button>
            <Button 
              onClick={saveEditedRun}
              disabled={isSaving}
              className="bg-emerald-600 hover:bg-emerald-700"
            >
              {isSaving ? (
                <>
                  <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                  Kaydediliyor...
                </>
              ) : (
                <>
                  <Save className="w-4 h-4 mr-2" />
                  Kaydet ve W&B'ye Gönder
                </>
              )}
            </Button>
          </div>
        </DialogContent>
      </Dialog>
    </div>
  );
}
