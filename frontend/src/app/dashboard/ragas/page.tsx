
"use client";

import { useEffect, useState, useCallback } from "react";
import { useAuth } from "@/lib/auth-context";
import { useModelProviders } from "@/hooks/useModelProviders";
import { api, Course, TestSet, EvaluationRun, RagasSettings, RagasProvider, QuickTestResponse, QuickTestResult } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { Card } from "@/components/ui/card";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog";
import { toast } from "sonner";
import { FlaskConical, Plus, Play, Trash2, FileText, Loader2, CheckCircle, XCircle, Clock, BarChart3, ArrowRight, Settings, Zap, Sparkles, X, Save, ChevronDown, History, Eye, Bookmark, Target, Layers, BookOpen, Activity } from "lucide-react";
import Link from "next/link";

export default function RagasPage() {
  const { user } = useAuth();
  const { getLLMProviders, getLLMModels } = useModelProviders();
  const [courses, setCourses] = useState<Course[]>([]);
  const [selectedCourseId, setSelectedCourseId] = useState<number | null>(null);
  const [testSets, setTestSets] = useState<TestSet[]>([]);
  const [runs, setRuns] = useState<EvaluationRun[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [isCreateOpen, setIsCreateOpen] = useState(false);
  const [isCreating, setIsCreating] = useState(false);
  const [newTestSet, setNewTestSet] = useState({ name: "", description: "" });
  const [ragasSettings, setRagasSettings] = useState<RagasSettings | null>(null);
  const [ragasProviders, setRagasProviders] = useState<RagasProvider[]>([]);
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);
  const [selectedProvider, setSelectedProvider] = useState<string>("");
  const [selectedModel, setSelectedModel] = useState<string>("");
  const [isSavingSettings, setIsSavingSettings] = useState(false);
  const [isQuickTestExpanded, setIsQuickTestExpanded] = useState(false);
  const [quickTestQuestion, setQuickTestQuestion] = useState("");
  const [quickTestGroundTruth, setQuickTestGroundTruth] = useState("");
  const [quickTestAlternatives, setQuickTestAlternatives] = useState<string[]>([]);
  const [quickTestSystemPrompt, setQuickTestSystemPrompt] = useState("");
  const [quickTestLlmModel, setQuickTestLlmModel] = useState("");
  const [isQuickTesting, setIsQuickTesting] = useState(false);
  const [quickTestResult, setQuickTestResult] = useState<QuickTestResponse | null>(null);
  const [savedResults, setSavedResults] = useState<QuickTestResult[]>([]);
  const [savedResultsGroups, setSavedResultsGroups] = useState<string[]>([]);
  const [savedResultsTotal, setSavedResultsTotal] = useState(0);
  const [isSavedResultsExpanded, setIsSavedResultsExpanded] = useState(false);
  const [selectedGroup, setSelectedGroup] = useState<string>("");
  const [isSaveDialogOpen, setIsSaveDialogOpen] = useState(false);
  const [saveGroupName, setSaveGroupName] = useState("");
  const [isSaving, setIsSaving] = useState(false);
  const [viewingResult, setViewingResult] = useState<QuickTestResult | null>(null);
  const [resultsPage, setResultsPage] = useState(0);
  const [isLoadingMore, setIsLoadingMore] = useState(false);
  const RESULTS_PER_PAGE = 10;

  const loadTestSets = useCallback(async () => {
    if (!selectedCourseId) return;
    try { const data = await api.getTestSets(selectedCourseId); setTestSets(data); } catch { toast.error("Test setleri yüklenirken hata oluştu"); }
  }, [selectedCourseId]);

  const loadRuns = useCallback(async () => {
    if (!selectedCourseId) return;
    try { const data = await api.getEvaluationRuns(selectedCourseId); setRuns(data); } catch { toast.error("Değerlendirmeler yüklenirken hata oluştu"); }
  }, [selectedCourseId]);

  useEffect(() => { loadCourses(); loadRagasSettings(); }, []);

  const loadSavedResults = useCallback(async (reset: boolean = true) => {
    if (!selectedCourseId) return;
    try {
      let groupFilter: string | undefined;
      if (selectedGroup === "__all__" || selectedGroup === "") { groupFilter = undefined; }
      else if (selectedGroup === "__no_group__") { groupFilter = ""; }
      else { groupFilter = selectedGroup; }
      const skip = reset ? 0 : resultsPage * RESULTS_PER_PAGE;
      const data = await api.getQuickTestResults(selectedCourseId, groupFilter, skip, RESULTS_PER_PAGE);
      if (reset) { setSavedResults(data.results); setResultsPage(1); }
      else { setSavedResults(prev => [...prev, ...data.results]); setResultsPage(prev => prev + 1); }
      setSavedResultsTotal(data.total);
      setSavedResultsGroups(data.groups);
    } catch { /* ignore */ }
  }, [selectedCourseId, selectedGroup, resultsPage]);

  const loadMoreResults = async () => { setIsLoadingMore(true); await loadSavedResults(false); setIsLoadingMore(false); };

  useEffect(() => { if (selectedCourseId) { loadTestSets(); loadRuns(); loadSavedResults(true); } }, [selectedCourseId, loadTestSets, loadRuns]);
  useEffect(() => { if (selectedCourseId) { setResultsPage(0); loadSavedResults(true); } }, [selectedGroup]);

  const loadCourses = async () => {
    try {
      const data = await api.getCourses();
      setCourses(data);
      if (data.length > 0) {
        // Check localStorage for previously selected course
        const savedCourseId = localStorage.getItem('ragas_selected_course_id');
        if (savedCourseId && data.find(c => c.id === parseInt(savedCourseId))) {
          setSelectedCourseId(parseInt(savedCourseId));
        } else {
          setSelectedCourseId(data[0].id);
        }
      }
    }
    catch { toast.error("Dersler yüklenirken hata oluştu"); }
    finally { setIsLoading(false); }
  };

  const loadRagasSettings = async () => {
    if (!selectedCourseId) return;
    try {
      const [settings, providersData] = await Promise.all([
        api.getRagasSettings(),
        api.getRagasProviders(),
      ]);
      setRagasSettings(settings);
      setRagasProviders(providersData.providers);
      setSelectedProvider(settings.current_provider || ""); setSelectedModel(settings.current_model || "");
    } catch { console.log("RAGAS settings not available"); }
  };

  const getProviderModels = (providerName: string) => {
    const ragasModels = ragasProviders.find((p) => p.name === providerName)?.models || [];
    const backendModels = getLLMModels(providerName);
    if (ragasModels.length === 0) return backendModels;
    if (backendModels.length === 0) return ragasModels;
    const seen = new Set<string>();
    const merged: string[] = [];
    for (const m of ragasModels) {
      if (seen.has(m)) continue;
      seen.add(m);
      merged.push(m);
    }
    for (const m of backendModels) {
      if (seen.has(m)) continue;
      seen.add(m);
      merged.push(m);
    }
    return merged;
  };

  const handleSaveSettings = async () => {
    setIsSavingSettings(true);
    try { const result = await api.updateRagasSettings({ provider: selectedProvider || "", model: selectedModel || "" }); setRagasSettings(result); toast.success("RAGAS ayarları güncellendi"); setIsSettingsOpen(false); }
    catch { toast.error("Ayarlar kaydedilirken hata oluştu"); }
    finally { setIsSavingSettings(false); }
  };

  const handleCreateTestSet = async (e: React.FormEvent) => {
    e.preventDefault(); if (!selectedCourseId) return;
    setIsCreating(true);
    try { await api.createTestSet({ course_id: selectedCourseId, name: newTestSet.name, description: newTestSet.description || undefined }); setNewTestSet({ name: "", description: "" }); setIsCreateOpen(false); loadTestSets(); toast.success("Test seti oluşturuldu"); }
    catch (error) { toast.error(error instanceof Error ? error.message : "Hata oluştu"); }
    finally { setIsCreating(false); }
  };

  const handleDeleteTestSet = async (id: number) => {
    if (!confirm("Bu test setini silmek istediğinizden emin misiniz?")) return;
    try { await api.deleteTestSet(id); loadTestSets(); toast.success("Test seti silindi"); }
    catch (error) { toast.error(error instanceof Error ? error.message : "Hata oluştu"); }
  };

  const handleDeleteRun = async (id: number) => {
    if (!confirm("Bu değerlendirmeyi silmek istediğinizden emin misiniz?")) return;
    try { await api.deleteEvaluationRun(id); loadRuns(); toast.success("Değerlendirme silindi"); }
    catch (error) { toast.error(error instanceof Error ? error.message : "Hata oluştu"); }
  };

  const handleQuickTest = async () => {
    if (!selectedCourseId || !quickTestQuestion || !quickTestGroundTruth) { toast.error("Lütfen ders, soru ve doğru cevap alanlarını doldurun"); return; }
    setIsQuickTesting(true); setQuickTestResult(null);
    try {
      const result = await api.quickTest({ course_id: selectedCourseId, question: quickTestQuestion, ground_truth: quickTestGroundTruth, alternative_ground_truths: quickTestAlternatives.filter(a => a.trim() !== ""), system_prompt: quickTestSystemPrompt || undefined, llm_provider: quickTestLlmModel ? "openrouter" : undefined, llm_model: quickTestLlmModel || undefined });
      setQuickTestResult(result); toast.success("Test tamamlandı");
    } catch (error) { toast.error(error instanceof Error ? error.message : "Test başarısız"); }
    finally { setIsQuickTesting(false); }
  };

  const handleSaveResult = async () => {
    if (!selectedCourseId || !quickTestResult) return;
    setIsSaving(true);
    try {
      await api.saveQuickTestResult({ course_id: selectedCourseId, group_name: saveGroupName || undefined, question: quickTestResult.question, ground_truth: quickTestResult.ground_truth, alternative_ground_truths: quickTestAlternatives.filter(a => a.trim() !== ""), system_prompt: quickTestResult.system_prompt_used, llm_provider: quickTestResult.llm_provider_used, llm_model: quickTestResult.llm_model_used, generated_answer: quickTestResult.generated_answer, retrieved_contexts: quickTestResult.retrieved_contexts, faithfulness: quickTestResult.faithfulness, answer_relevancy: quickTestResult.answer_relevancy, context_precision: quickTestResult.context_precision, context_recall: quickTestResult.context_recall, answer_correctness: quickTestResult.answer_correctness, latency_ms: quickTestResult.latency_ms });
      toast.success("Sonuç kaydedildi"); setIsSaveDialogOpen(false); setSaveGroupName(""); loadSavedResults();
    } catch (error) { toast.error(error instanceof Error ? error.message : "Kaydetme başarısız"); }
    finally { setIsSaving(false); }
  };

  const handleDeleteSavedResult = async (id: number) => {
    if (!confirm("Bu sonucu silmek istediğinizden emin misiniz?")) return;
    try { await api.deleteQuickTestResult(id); toast.success("Sonuç silindi"); loadSavedResults(); if (viewingResult?.id === id) { setViewingResult(null); } }
    catch (error) { toast.error(error instanceof Error ? error.message : "Silme başarısız"); }
  };

  const handleSaveSystemPrompt = async () => {
    if (!selectedCourseId || !quickTestSystemPrompt) { toast.error("Sistem promptu boş olamaz"); return; }
    try { await api.updateCourseSettings(selectedCourseId, { system_prompt: quickTestSystemPrompt }); toast.success("Sistem promptu kaydedildi"); }
    catch (error) { toast.error(error instanceof Error ? error.message : "Kaydetme başarısız"); }
  };

  const addAlternative = () => { setQuickTestAlternatives([...quickTestAlternatives, ""]); };
  const removeAlternative = (index: number) => { setQuickTestAlternatives(quickTestAlternatives.filter((_, i) => i !== index)); };
  const updateAlternative = (index: number, value: string) => { const newAlternatives = [...quickTestAlternatives]; newAlternatives[index] = value; setQuickTestAlternatives(newAlternatives); };

  const loadSystemPrompt = async () => {
    if (!selectedCourseId) return;
    try { const settings = await api.getCourseSettings(selectedCourseId); setQuickTestSystemPrompt(settings.system_prompt || ""); } catch { /* ignore */ }
  };

  useEffect(() => { if (selectedCourseId) { loadSystemPrompt(); } }, [selectedCourseId]);

  const getMetricColor = (value?: number) => { if (value === undefined || value === null) return "text-slate-400"; if (value >= 0.8) return "text-emerald-600"; if (value >= 0.6) return "text-amber-600"; return "text-red-600"; };
  const getMetricBgColor = (value?: number) => { if (value === undefined || value === null) return "bg-slate-50 border-slate-200"; if (value >= 0.8) return "bg-emerald-50 border-emerald-200"; if (value >= 0.6) return "bg-amber-50 border-amber-200"; return "bg-red-50 border-red-200"; };

  const getStatusBadge = (status: string) => {
    switch (status) {
      case "completed": return <span className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium rounded-full bg-emerald-100 text-emerald-700 border border-emerald-200"><CheckCircle className="w-3.5 h-3.5" /> Tamamlandı</span>;
      case "running": return <span className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium rounded-full bg-blue-100 text-blue-700 border border-blue-200"><Loader2 className="w-3.5 h-3.5 animate-spin" /> Çalışıyor</span>;
      case "failed": return <span className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium rounded-full bg-red-100 text-red-700 border border-red-200"><XCircle className="w-3.5 h-3.5" /> Başarısız</span>;
      default: return <span className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium rounded-full bg-slate-100 text-slate-600 border border-slate-200"><Clock className="w-3.5 h-3.5" /> Bekliyor</span>;
    }
  };

  const completedRuns = runs.filter(r => r.status === "completed");
  const totalQuestions = testSets.reduce((acc, ts) => acc + ts.question_count, 0);

  if (!user) return null;

  if (isLoading) {
    return (
      <div className="min-h-[60vh] flex items-center justify-center">
        <div className="text-center">
          <div className="relative"><div className="w-16 h-16 border-4 border-purple-200 border-t-purple-600 rounded-full animate-spin mx-auto"></div><FlaskConical className="w-6 h-6 text-purple-600 absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2" /></div>
          <p className="mt-4 text-slate-600 font-medium">RAGAS yükleniyor...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-8">
      {/* Hero Header */}
      <div className="relative overflow-hidden bg-gradient-to-br from-purple-600 via-indigo-600 to-blue-700 rounded-2xl p-8 text-white shadow-xl">
        <div className="absolute inset-0 opacity-30" style={{backgroundImage: "url(\"data:image/svg+xml,%3Csvg width='60' height='60' viewBox='0 0 60 60' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' fill-rule='evenodd'%3E%3Cg fill='%23fff' fill-opacity='0.1'%3E%3Ccircle cx='30' cy='30' r='2'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E\")"}}></div>
        <div className="absolute -top-24 -right-24 w-96 h-96 bg-white/10 rounded-full blur-3xl"></div>
        <div className="absolute -bottom-24 -left-24 w-96 h-96 bg-purple-500/20 rounded-full blur-3xl"></div>
        
        <div className="relative z-10">
          <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between gap-6">
            <div className="flex items-center gap-4">
              <div className="p-3 bg-white/20 rounded-xl backdrop-blur-sm"><FlaskConical className="w-8 h-8" /></div>
              <div>
                <h1 className="text-3xl font-bold">RAGAS Değerlendirme</h1>
                <p className="text-purple-200 mt-1">RAG sisteminizin kalitesini test edin ve ölçün</p>
              </div>
            </div>
            
            <div className="flex flex-wrap items-center gap-3">
              <Dialog open={isSettingsOpen} onOpenChange={setIsSettingsOpen}>
                <DialogTrigger asChild>
                  <Button variant="secondary" size="sm" className="bg-white/20 hover:bg-white/30 text-white border-0 backdrop-blur-sm h-10">
                    <Settings className="w-4 h-4 mr-2" />Ayarlar
                    {ragasSettings?.is_free && <span className="ml-2 flex items-center gap-1 px-2 py-0.5 bg-green-500/30 rounded-full text-xs"><Zap className="w-3 h-3" /> Ücretsiz</span>}
                  </Button>
                </DialogTrigger>
                <DialogContent className="sm:max-w-lg">
                  <DialogHeader>
                    <DialogTitle className="flex items-center gap-2"><Settings className="w-5 h-5 text-purple-600" />RAGAS Ayarları</DialogTitle>
                    <DialogDescription>Değerlendirme için kullanılacak LLM sağlayıcısını seçin.</DialogDescription>
                  </DialogHeader>
                  <div className="space-y-4 py-4">
                    <div className="space-y-2">
                      <Label className="text-sm font-medium">LLM Sağlayıcı</Label>
                      <Select value={selectedProvider || "auto"} onValueChange={(v) => { const newProvider = v === "auto" ? "" : v; setSelectedProvider(newProvider); if (newProvider) { const provider = ragasProviders.find(p => p.name === newProvider); setSelectedModel(provider?.default_model || ""); } else { setSelectedModel(""); } }}>
                        <SelectTrigger className="h-11"><SelectValue placeholder="Otomatik seç" /></SelectTrigger>
                        <SelectContent>
                          <SelectItem value="auto"><span className="flex items-center gap-2"><Zap className="w-4 h-4 text-amber-500" />Otomatik Seçim</span></SelectItem>
                          {ragasProviders.map((p) => (<SelectItem key={p.name} value={p.name} disabled={!p.available}><span className="flex items-center gap-2">{p.name}{!p.available && <span className="text-xs text-slate-400 bg-slate-100 px-2 py-0.5 rounded">(API key yok)</span>}</span></SelectItem>))}
                        </SelectContent>
                      </Select>
                    </div>
                    {selectedProvider && selectedProvider !== "auto" && (
                      <div className="space-y-2">
                        <Label className="text-sm font-medium">Model</Label>
                        <Select value={selectedModel || ""} onValueChange={setSelectedModel}>
                          <SelectTrigger className="h-11"><SelectValue placeholder="Model seçin" /></SelectTrigger>
                          <SelectContent>
                            {getProviderModels(selectedProvider).map((model) => (
                              <SelectItem key={model} value={model}>
                                {model}
                              </SelectItem>
                            ))}
                          </SelectContent>
                        </Select>
                      </div>
                    )}
                    {ragasSettings && (
                      <div className="bg-gradient-to-r from-slate-50 to-slate-100 rounded-xl p-4 border border-slate-200">
                        <p className="font-semibold text-slate-700 mb-2 flex items-center gap-2"><Activity className="w-4 h-4" />Mevcut Ayarlar</p>
                        <div className="grid grid-cols-2 gap-3 text-sm">
                          <div><p className="text-slate-500">Sağlayıcı</p><p className="font-medium text-slate-900">{ragasSettings.current_provider || "Otomatik"}</p></div>
                          <div><p className="text-slate-500">Model</p><p className="font-medium text-slate-900">{ragasSettings.current_model || "Varsayılan"}</p></div>
                        </div>
                      </div>
                    )}
                  </div>
                  <DialogFooter>
                    <Button variant="outline" onClick={() => setIsSettingsOpen(false)}>İptal</Button>
                    <Button onClick={handleSaveSettings} disabled={isSavingSettings} className="bg-purple-600 hover:bg-purple-700">{isSavingSettings ? <><Loader2 className="w-4 h-4 mr-2 animate-spin" />Kaydediliyor...</> : <><Save className="w-4 h-4 mr-2" />Kaydet</>}</Button>
                  </DialogFooter>
                </DialogContent>
              </Dialog>

              <Select value={selectedCourseId?.toString() || ""} onValueChange={(v) => {
                const courseId = Number(v);
                setSelectedCourseId(courseId);
                localStorage.setItem('ragas_selected_course_id', courseId.toString());
              }}>
                <SelectTrigger className="w-56 bg-white/20 border-0 text-white hover:bg-white/30 backdrop-blur-sm h-10"><BookOpen className="w-4 h-4 mr-2" /><SelectValue placeholder="Ders seçin" /></SelectTrigger>
                <SelectContent>{courses.map((course) => (<SelectItem key={course.id} value={course.id.toString()}>{course.name}</SelectItem>))}</SelectContent>
              </Select>
            </div>
          </div>

          {selectedCourseId && (
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mt-8">
              <div className="bg-white/10 backdrop-blur-sm rounded-xl p-4 border border-white/20"><div className="flex items-center gap-3"><div className="p-2 bg-white/20 rounded-lg"><Layers className="w-5 h-5" /></div><div><p className="text-purple-200 text-sm">Test Setleri</p><p className="text-2xl font-bold">{testSets.length}</p></div></div></div>
              <div className="bg-white/10 backdrop-blur-sm rounded-xl p-4 border border-white/20"><div className="flex items-center gap-3"><div className="p-2 bg-white/20 rounded-lg"><FileText className="w-5 h-5" /></div><div><p className="text-purple-200 text-sm">Toplam Soru</p><p className="text-2xl font-bold">{totalQuestions}</p></div></div></div>
              <div className="bg-white/10 backdrop-blur-sm rounded-xl p-4 border border-white/20"><div className="flex items-center gap-3"><div className="p-2 bg-white/20 rounded-lg"><BarChart3 className="w-5 h-5" /></div><div><p className="text-purple-200 text-sm">Değerlendirmeler</p><p className="text-2xl font-bold">{runs.length}</p></div></div></div>
              <div className="bg-white/10 backdrop-blur-sm rounded-xl p-4 border border-white/20"><div className="flex items-center gap-3"><div className="p-2 bg-white/20 rounded-lg"><Target className="w-5 h-5" /></div><div><p className="text-purple-200 text-sm">Tamamlanan</p><p className="text-2xl font-bold">{completedRuns.length}</p></div></div></div>
            </div>
          )}
        </div>
      </div>

      {!selectedCourseId ? (
        <div className="bg-white rounded-2xl border border-slate-200 p-16 text-center shadow-sm">
          <div className="w-20 h-20 bg-purple-100 rounded-full flex items-center justify-center mx-auto mb-6"><FlaskConical className="w-10 h-10 text-purple-600" /></div>
          <h3 className="text-xl font-semibold text-slate-900 mb-2">Ders Seçin</h3>
          <p className="text-slate-500 max-w-md mx-auto">RAGAS değerlendirmesi yapmak için yukarıdan bir ders seçin.</p>
        </div>
      ) : (
        <div className="space-y-6">
          {/* Quick Test */}
          <Card className="overflow-hidden border-0 shadow-lg bg-gradient-to-br from-purple-50 via-white to-indigo-50">
            <button onClick={() => setIsQuickTestExpanded(!isQuickTestExpanded)} className="w-full px-6 py-5 flex items-center justify-between hover:bg-purple-50/50 transition-all duration-200">
              <div className="flex items-center gap-4">
                <div className="p-3 bg-gradient-to-br from-purple-500 to-indigo-600 rounded-xl shadow-lg shadow-purple-200"><Sparkles className="w-6 h-6 text-white" /></div>
                <div className="text-left"><h2 className="text-xl font-bold text-slate-900">Hızlı Test</h2><p className="text-sm text-slate-600">Tek bir soru için anında RAGAS değerlendirmesi</p></div>
              </div>
              <div className={`p-2 rounded-full bg-purple-100 transition-transform duration-200 ${isQuickTestExpanded ? 'rotate-180' : ''}`}><ChevronDown className="w-5 h-5 text-purple-600" /></div>
            </button>
            {isQuickTestExpanded && (
              <div className="px-6 pb-6 pt-2 border-t border-purple-100">
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                  <div className="space-y-4">
                    <div><Label className="text-sm font-medium text-slate-700">Soru</Label><Textarea value={quickTestQuestion} onChange={(e) => setQuickTestQuestion(e.target.value)} placeholder="Test etmek istediğiniz soruyu girin..." rows={3} className="mt-1.5 border-slate-200 focus:border-purple-400 focus:ring-purple-400" /></div>
                    <div><Label className="text-sm font-medium text-slate-700">Doğru Cevap (Ground Truth)</Label><Textarea value={quickTestGroundTruth} onChange={(e) => setQuickTestGroundTruth(e.target.value)} placeholder="Beklenen doğru cevabı girin..." rows={3} className="mt-1.5 border-slate-200 focus:border-purple-400 focus:ring-purple-400" /></div>
                    <div>
                      <div className="flex items-center justify-between mb-2"><Label className="text-sm font-medium text-slate-700">Alternatif Doğru Cevaplar</Label><Button type="button" variant="outline" size="sm" onClick={addAlternative} className="h-7 text-xs border-purple-200 text-purple-600 hover:bg-purple-50"><Plus className="w-3 h-3 mr-1" /> Ekle</Button></div>
                      {quickTestAlternatives.map((alt, index) => (<div key={index} className="flex gap-2 mb-2"><Input value={alt} onChange={(e) => updateAlternative(index, e.target.value)} placeholder={`Alternatif ${index + 1}`} className="flex-1 border-slate-200" /><Button type="button" variant="ghost" size="sm" onClick={() => removeAlternative(index)} className="text-red-500 hover:text-red-700 hover:bg-red-50"><X className="w-4 h-4" /></Button></div>))}
                    </div>
                    <div>
                      <div className="flex items-center justify-between mb-2"><Label className="text-sm font-medium text-slate-700">Sistem Promptu</Label><Button type="button" variant="outline" size="sm" onClick={handleSaveSystemPrompt} disabled={!quickTestSystemPrompt} className="h-7 text-xs border-purple-200 text-purple-600 hover:bg-purple-50"><Save className="w-3 h-3 mr-1" /> Kaydet</Button></div>
                      <Textarea value={quickTestSystemPrompt} onChange={(e) => setQuickTestSystemPrompt(e.target.value)} placeholder="Sistem promptu (boş bırakılırsa ders ayarlarından alınır)" rows={3} className="mt-1 border-slate-200 focus:border-purple-400 focus:ring-purple-400" />
                    </div>
                    <div><Label className="text-sm font-medium text-slate-700">OpenRouter Model (Opsiyonel)</Label><Input value={quickTestLlmModel} onChange={(e) => setQuickTestLlmModel(e.target.value)} placeholder="örn: openai/gpt-4o-mini" className="mt-1.5 border-slate-200" /></div>
                    <Button onClick={handleQuickTest} disabled={isQuickTesting || !quickTestQuestion || !quickTestGroundTruth} className="w-full bg-gradient-to-r from-purple-600 to-indigo-600 hover:from-purple-700 hover:to-indigo-700 shadow-lg shadow-purple-200">{isQuickTesting ? <><Loader2 className="w-4 h-4 mr-2 animate-spin" />Test Ediliyor...</> : <><Play className="w-4 h-4 mr-2" />Test Et</>}</Button>
                  </div>
                  <div className="space-y-4">
                    {quickTestResult ? (
                      <>
                        <div><Label className="text-sm font-medium text-slate-700">Üretilen Cevap</Label><div className="mt-1.5 p-4 bg-white rounded-xl border border-slate-200 text-sm shadow-sm">{quickTestResult.generated_answer}</div></div>
                        <div>
                          <Label className="text-sm font-medium text-slate-700">Metrikler</Label>
                          <div className="grid grid-cols-2 gap-3 mt-2">
                            <div className={`p-4 rounded-xl border ${getMetricBgColor(quickTestResult.faithfulness)}`}><p className="text-xs text-slate-600 font-medium">Faithfulness</p><p className={`text-2xl font-bold ${getMetricColor(quickTestResult.faithfulness)}`}>{quickTestResult.faithfulness !== undefined && quickTestResult.faithfulness !== null ? `${(quickTestResult.faithfulness * 100).toFixed(1)}%` : "N/A"}</p></div>
                            <div className={`p-4 rounded-xl border ${getMetricBgColor(quickTestResult.answer_relevancy)}`}><p className="text-xs text-slate-600 font-medium">Answer Relevancy</p><p className={`text-2xl font-bold ${getMetricColor(quickTestResult.answer_relevancy)}`}>{quickTestResult.answer_relevancy !== undefined && quickTestResult.answer_relevancy !== null ? `${(quickTestResult.answer_relevancy * 100).toFixed(1)}%` : "N/A"}</p></div>
                            <div className={`p-4 rounded-xl border ${getMetricBgColor(quickTestResult.context_precision)}`}><p className="text-xs text-slate-600 font-medium">Context Precision</p><p className={`text-2xl font-bold ${getMetricColor(quickTestResult.context_precision)}`}>{quickTestResult.context_precision !== undefined && quickTestResult.context_precision !== null ? `${(quickTestResult.context_precision * 100).toFixed(1)}%` : "N/A"}</p></div>
                            <div className={`p-4 rounded-xl border ${getMetricBgColor(quickTestResult.context_recall)}`}><p className="text-xs text-slate-600 font-medium">Context Recall</p><p className={`text-2xl font-bold ${getMetricColor(quickTestResult.context_recall)}`}>{quickTestResult.context_recall !== undefined && quickTestResult.context_recall !== null ? `${(quickTestResult.context_recall * 100).toFixed(1)}%` : "N/A"}</p></div>
                            <div className={`p-4 rounded-xl border col-span-2 ${getMetricBgColor(quickTestResult.answer_correctness)}`}><p className="text-xs text-slate-600 font-medium">Answer Correctness</p><p className={`text-2xl font-bold ${getMetricColor(quickTestResult.answer_correctness)}`}>{quickTestResult.answer_correctness !== undefined && quickTestResult.answer_correctness !== null ? `${(quickTestResult.answer_correctness * 100).toFixed(1)}%` : "N/A"}</p></div>
                          </div>
                        </div>
                        <div className="grid grid-cols-2 gap-3">
                          <div className="p-4 bg-white rounded-xl border border-slate-200 shadow-sm"><p className="text-xs text-slate-500 font-medium">Gecikme</p><p className="text-lg font-bold text-slate-900">{quickTestResult.latency_ms}ms</p></div>
                          <div className="p-4 bg-white rounded-xl border border-slate-200 shadow-sm"><p className="text-xs text-slate-500 font-medium">Model</p><p className="text-sm font-medium text-slate-900 truncate">{quickTestResult.llm_model_used}</p></div>
                        </div>
                        {quickTestResult.reranker_used && (
                          <div className="p-4 bg-gradient-to-r from-amber-50 to-orange-50 rounded-xl border border-amber-200 shadow-sm">
                            <div className="flex items-center gap-2 mb-1">
                              <Sparkles className="w-4 h-4 text-amber-600" />
                              <p className="text-xs text-amber-700 font-semibold">Reranker Kullanıldı</p>
                            </div>
                            <p className="text-sm font-medium text-amber-900">
                              {quickTestResult.reranker_provider}/{quickTestResult.reranker_model || 'default'}
                            </p>
                          </div>
                        )}
                        <Dialog open={isSaveDialogOpen} onOpenChange={setIsSaveDialogOpen}>
                          <DialogTrigger asChild><Button variant="outline" className="w-full border-purple-200 text-purple-700 hover:bg-purple-50"><Bookmark className="w-4 h-4 mr-2" />Sonucu Kaydet</Button></DialogTrigger>
                          <DialogContent>
                            <DialogHeader><DialogTitle>Sonucu Kaydet</DialogTitle><DialogDescription>Bu test sonucunu daha sonra görüntülemek için kaydedin.</DialogDescription></DialogHeader>
                            <div className="space-y-4 py-4">
                              <div className="space-y-2"><Label>Grup Adı (Opsiyonel)</Label><Input value={saveGroupName} onChange={(e) => setSaveGroupName(e.target.value)} placeholder="örn: Deneme 1" /></div>
                              {savedResultsGroups.length > 0 && <div className="flex flex-wrap gap-1">{savedResultsGroups.map((g) => (<button key={g} type="button" onClick={() => setSaveGroupName(g)} className="px-2 py-1 text-xs bg-slate-100 hover:bg-slate-200 rounded">{g}</button>))}</div>}
                            </div>
                            <DialogFooter><Button variant="outline" onClick={() => setIsSaveDialogOpen(false)}>İptal</Button><Button onClick={handleSaveResult} disabled={isSaving}>{isSaving ? "Kaydediliyor..." : "Kaydet"}</Button></DialogFooter>
                          </DialogContent>
                        </Dialog>
                      </>
                    ) : (
                      <div className="flex items-center justify-center h-full min-h-[400px] text-slate-400">
                        <div className="text-center">
                          <div className="w-20 h-20 bg-purple-100 rounded-full flex items-center justify-center mx-auto mb-4"><Sparkles className="w-10 h-10 text-purple-400" /></div>
                          <p className="text-sm font-medium">Test sonuçları burada görünecek</p>
                          <p className="text-xs text-slate-400 mt-1">Soru ve doğru cevabı girerek test başlatın</p>
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              </div>
            )}
          </Card>

          {/* Saved Results */}
          <Card className="overflow-hidden border-0 shadow-lg bg-white">
            <button onClick={() => setIsSavedResultsExpanded(!isSavedResultsExpanded)} className="w-full px-6 py-5 flex items-center justify-between hover:bg-slate-50 transition-all duration-200">
              <div className="flex items-center gap-4">
                <div className="p-3 bg-gradient-to-br from-slate-600 to-slate-800 rounded-xl shadow-lg"><History className="w-6 h-6 text-white" /></div>
                <div className="text-left"><h2 className="text-xl font-bold text-slate-900">Kaydedilen Sonuçlar</h2><p className="text-sm text-slate-600">{savedResultsTotal} kayıtlı test sonucu</p></div>
              </div>
              <div className={`p-2 rounded-full bg-slate-100 transition-transform duration-200 ${isSavedResultsExpanded ? 'rotate-180' : ''}`}><ChevronDown className="w-5 h-5 text-slate-600" /></div>
            </button>
            {isSavedResultsExpanded && (
              <div className="px-6 pb-6 pt-2 border-t border-slate-100">
                {savedResultsGroups.length > 0 && (
                  <div className="mb-4">
                    <Select value={selectedGroup === "" ? "__all__" : selectedGroup} onValueChange={(v) => setSelectedGroup(v === "__all__" ? "" : v)}>
                      <SelectTrigger className="w-48"><SelectValue placeholder="Tüm gruplar" /></SelectTrigger>
                      <SelectContent>
                        <SelectItem value="__all__">Tüm gruplar</SelectItem>
                        {savedResultsGroups.filter((g) => g && g.trim() !== "").map((g) => (<SelectItem key={g} value={g}>{g}</SelectItem>))}
                        {savedResultsGroups.some((g) => !g || g.trim() === "") && <SelectItem value="__no_group__">Grupsuz</SelectItem>}
                      </SelectContent>
                    </Select>
                  </div>
                )}
                {savedResults.length === 0 ? (
                  <div className="text-center py-12 text-slate-400"><History className="w-12 h-12 mx-auto mb-3 opacity-50" /><p className="font-medium">Henüz kaydedilmiş sonuç yok</p></div>
                ) : (
                  <div className="space-y-3">
                    {savedResults.map((result) => (
                      <div key={result.id} className="p-4 bg-slate-50 rounded-xl border border-slate-200 hover:border-slate-300 transition-colors">
                        <div className="flex items-start justify-between">
                          <div className="flex-1 min-w-0">
                            <div className="flex items-center gap-2 mb-1">{result.group_name && <span className="px-2 py-0.5 text-xs bg-purple-100 text-purple-700 rounded-full font-medium">{result.group_name}</span>}<span className="text-xs text-slate-500">{new Date(result.created_at).toLocaleString("tr-TR")}</span></div>
                            <p className="text-sm font-medium text-slate-900 truncate">{result.question}</p>
                            <div className="flex flex-wrap items-center gap-3 mt-2 text-xs">
                              <span className={getMetricColor(result.faithfulness)}>Sadakat: {result.faithfulness !== undefined && result.faithfulness !== null ? `${(result.faithfulness * 100).toFixed(0)}%` : "N/A"}</span>
                              <span className={getMetricColor(result.answer_correctness)}>Doğruluk: {result.answer_correctness !== undefined && result.answer_correctness !== null ? `${(result.answer_correctness * 100).toFixed(0)}%` : "N/A"}</span>
                              <span className="text-slate-400">{result.latency_ms}ms</span>
                              {result.reranker_used && (
                                <span className="flex items-center gap-1 px-2 py-0.5 bg-amber-100 text-amber-700 rounded-full font-medium">
                                  <Sparkles className="w-3 h-3" />
                                  {result.reranker_provider}
                                </span>
                              )}
                            </div>
                          </div>
                          <div className="flex items-center gap-1 ml-2">
                            <Button variant="ghost" size="sm" onClick={() => setViewingResult(result)} className="text-slate-600 hover:text-purple-600"><Eye className="w-4 h-4" /></Button>
                            <Button variant="ghost" size="sm" onClick={() => handleDeleteSavedResult(result.id)} className="text-slate-400 hover:text-red-600"><Trash2 className="w-4 h-4" /></Button>
                          </div>
                        </div>
                      </div>
                    ))}
                    {savedResults.length < savedResultsTotal && <Button variant="outline" onClick={loadMoreResults} disabled={isLoadingMore} className="w-full">{isLoadingMore ? <><Loader2 className="w-4 h-4 mr-2 animate-spin" />Yükleniyor...</> : <>Daha Fazla ({savedResults.length}/{savedResultsTotal})</>}</Button>}
                  </div>
                )}
              </div>
            )}
          </Card>

          {/* View Result Dialog */}
          <Dialog open={!!viewingResult} onOpenChange={(open) => !open && setViewingResult(null)}>
            <DialogContent className="max-w-4xl max-h-[90vh] overflow-y-auto">
              <DialogHeader><DialogTitle>Sonuç Detayları</DialogTitle>{viewingResult?.group_name && <DialogDescription>Grup: {viewingResult.group_name}</DialogDescription>}</DialogHeader>
              {viewingResult && (
                <div className="space-y-4 py-4">
                  <div><Label className="text-sm font-medium">Soru</Label><p className="mt-1 p-3 bg-slate-50 rounded-lg text-sm">{viewingResult.question}</p></div>
                  <div><Label className="text-sm font-medium">Doğru Cevap</Label><p className="mt-1 p-3 bg-slate-50 rounded-lg text-sm">{viewingResult.ground_truth}</p></div>
                  <div><Label className="text-sm font-medium">Üretilen Cevap</Label><p className="mt-1 p-3 bg-slate-50 rounded-lg text-sm">{viewingResult.generated_answer}</p></div>
                  <div className="grid grid-cols-3 gap-3">
                    <div className={`p-3 rounded-lg border ${getMetricBgColor(viewingResult.faithfulness)}`}><p className="text-xs text-slate-600">Faithfulness</p><p className={`text-xl font-bold ${getMetricColor(viewingResult.faithfulness)}`}>{viewingResult.faithfulness !== undefined && viewingResult.faithfulness !== null ? `${(viewingResult.faithfulness * 100).toFixed(1)}%` : "N/A"}</p></div>
                    <div className={`p-3 rounded-lg border ${getMetricBgColor(viewingResult.answer_relevancy)}`}><p className="text-xs text-slate-600">Answer Relevancy</p><p className={`text-xl font-bold ${getMetricColor(viewingResult.answer_relevancy)}`}>{viewingResult.answer_relevancy !== undefined && viewingResult.answer_relevancy !== null ? `${(viewingResult.answer_relevancy * 100).toFixed(1)}%` : "N/A"}</p></div>
                    <div className={`p-3 rounded-lg border ${getMetricBgColor(viewingResult.context_precision)}`}><p className="text-xs text-slate-600">Context Precision</p><p className={`text-xl font-bold ${getMetricColor(viewingResult.context_precision)}`}>{viewingResult.context_precision !== undefined && viewingResult.context_precision !== null ? `${(viewingResult.context_precision * 100).toFixed(1)}%` : "N/A"}</p></div>
                    <div className={`p-3 rounded-lg border ${getMetricBgColor(viewingResult.context_recall)}`}><p className="text-xs text-slate-600">Context Recall</p><p className={`text-xl font-bold ${getMetricColor(viewingResult.context_recall)}`}>{viewingResult.context_recall !== undefined && viewingResult.context_recall !== null ? `${(viewingResult.context_recall * 100).toFixed(1)}%` : "N/A"}</p></div>
                    <div className={`p-3 rounded-lg border ${getMetricBgColor(viewingResult.answer_correctness)}`}><p className="text-xs text-slate-600">Answer Correctness</p><p className={`text-xl font-bold ${getMetricColor(viewingResult.answer_correctness)}`}>{viewingResult.answer_correctness !== undefined && viewingResult.answer_correctness !== null ? `${(viewingResult.answer_correctness * 100).toFixed(1)}%` : "N/A"}</p></div>
                    <div className="p-3 rounded-lg bg-slate-50 border border-slate-200"><p className="text-xs text-slate-600">Gecikme</p><p className="text-xl font-bold text-slate-900">{viewingResult.latency_ms}ms</p></div>
                  </div>
                  {viewingResult.reranker_used && (
                    <div className="p-4 bg-gradient-to-r from-amber-50 to-orange-50 rounded-xl border border-amber-200">
                      <div className="flex items-center gap-2 mb-1">
                        <Sparkles className="w-5 h-5 text-amber-600" />
                        <p className="text-sm text-amber-700 font-semibold">Reranker Kullanıldı</p>
                      </div>
                      <p className="text-base font-medium text-amber-900">
                        {viewingResult.reranker_provider}/{viewingResult.reranker_model || 'default'}
                      </p>
                    </div>
                  )}
                </div>
              )}
              <DialogFooter><Button variant="outline" onClick={() => setViewingResult(null)}>Kapat</Button></DialogFooter>
            </DialogContent>
          </Dialog>

          {/* Test Sets with Integrated Evaluations */}
          <Card className="overflow-hidden border-0 shadow-lg">
            <div className="flex items-center justify-between px-6 py-4 bg-gradient-to-r from-indigo-500 to-purple-600 text-white">
              <div className="flex items-center gap-3"><FileText className="w-5 h-5" /><h2 className="font-semibold">Test Setleri</h2></div>
              <div className="flex items-center gap-2">
                <Dialog open={isCreateOpen} onOpenChange={setIsCreateOpen}>
                  <DialogTrigger asChild><Button size="sm" variant="secondary" className="bg-white/20 hover:bg-white/30 text-white border-0"><Plus className="w-4 h-4 mr-1" /> Yeni Test Seti</Button></DialogTrigger>
                  <DialogContent>
                    <form onSubmit={handleCreateTestSet}>
                      <DialogHeader><DialogTitle>Yeni Test Seti</DialogTitle><DialogDescription>Soru-cevap çiftleri içeren bir test seti oluşturun.</DialogDescription></DialogHeader>
                      <div className="space-y-4 py-4">
                        <div className="space-y-2"><Label>İsim</Label><Input value={newTestSet.name} onChange={(e) => setNewTestSet({ ...newTestSet, name: e.target.value })} placeholder="Test seti adı" required /></div>
                        <div className="space-y-2"><Label>Açıklama</Label><Input value={newTestSet.description} onChange={(e) => setNewTestSet({ ...newTestSet, description: e.target.value })} placeholder="Opsiyonel açıklama" /></div>
                      </div>
                      <DialogFooter><Button type="button" variant="outline" onClick={() => setIsCreateOpen(false)}>İptal</Button><Button type="submit" disabled={isCreating}>{isCreating ? "Oluşturuluyor..." : "Oluştur"}</Button></DialogFooter>
                    </form>
                  </DialogContent>
                </Dialog>
              </div>
            </div>
            {testSets.length === 0 ? (
              <div className="p-8 text-center text-slate-400"><FileText className="w-10 h-10 mx-auto mb-2 opacity-50" /><p className="text-sm">Henüz test seti yok</p><p className="text-xs mt-1">Yeni bir test seti oluşturarak başlayın</p></div>
            ) : (
              <div className="divide-y divide-slate-100">
                {testSets.map((ts) => {
                  const testSetRun = runs.find(r => r.test_set_id === ts.id);
                  return (
                    <div key={ts.id} className="px-6 py-4 hover:bg-slate-50 transition-colors">
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-4 flex-1">
                          <div className="p-2 bg-indigo-100 rounded-lg">
                            <Layers className="w-5 h-5 text-indigo-600" />
                          </div>
                          <div className="flex-1">
                            <div className="flex items-center gap-3 mb-1">
                              <p className="font-semibold text-slate-900">{ts.name}</p>
                              {testSetRun && getStatusBadge(testSetRun.status)}
                            </div>
                            <div className="flex items-center gap-4 text-xs text-slate-500">
                              <span>{ts.question_count} soru</span>
                              <span>•</span>
                              <span>{new Date(ts.created_at).toLocaleDateString("tr-TR")}</span>
                              {testSetRun && (
                                <>
                                  <span>•</span>
                                  <span>{testSetRun.processed_questions}/{testSetRun.total_questions} değerlendirildi</span>
                                </>
                              )}
                            </div>
                            {testSetRun && testSetRun.status === "completed" && testSetRun.avg_faithfulness !== null && testSetRun.avg_faithfulness !== undefined && (
                              <div className="flex items-center gap-3 mt-3">
                                {(() => {
                                  const overallAvg = (
                                    (testSetRun.avg_faithfulness ?? 0) +
                                    (testSetRun.avg_answer_relevancy ?? 0) +
                                    (testSetRun.avg_context_precision ?? 0) +
                                    (testSetRun.avg_context_recall ?? 0) +
                                    (testSetRun.avg_answer_correctness ?? 0)
                                  ) / 5;
                                  
                                  return (
                                    <>
                                      <div className={`px-3 py-1.5 rounded-full border-2 ${getMetricBgColor(overallAvg)} flex items-center gap-2`}>
                                        <span className="text-xs font-medium text-slate-600">Genel Ortalama:</span>
                                        <span className={`text-lg font-bold ${getMetricColor(overallAvg)}`}>
                                          {(overallAvg * 100).toFixed(1)}%
                                        </span>
                                      </div>
                                      <div className="flex items-center gap-2 text-xs">
                                        <span className={`px-2 py-1 rounded ${getMetricBgColor(testSetRun.avg_faithfulness)}`}>
                                          <span className="text-slate-500">F:</span> <span className={`font-semibold ${getMetricColor(testSetRun.avg_faithfulness)}`}>{((testSetRun.avg_faithfulness ?? 0) * 100).toFixed(0)}%</span>
                                        </span>
                                        <span className={`px-2 py-1 rounded ${getMetricBgColor(testSetRun.avg_answer_relevancy)}`}>
                                          <span className="text-slate-500">AR:</span> <span className={`font-semibold ${getMetricColor(testSetRun.avg_answer_relevancy)}`}>{((testSetRun.avg_answer_relevancy ?? 0) * 100).toFixed(0)}%</span>
                                        </span>
                                        <span className={`px-2 py-1 rounded ${getMetricBgColor(testSetRun.avg_context_precision)}`}>
                                          <span className="text-slate-500">CP:</span> <span className={`font-semibold ${getMetricColor(testSetRun.avg_context_precision)}`}>{((testSetRun.avg_context_precision ?? 0) * 100).toFixed(0)}%</span>
                                        </span>
                                        <span className={`px-2 py-1 rounded ${getMetricBgColor(testSetRun.avg_context_recall)}`}>
                                          <span className="text-slate-500">CR:</span> <span className={`font-semibold ${getMetricColor(testSetRun.avg_context_recall)}`}>{((testSetRun.avg_context_recall ?? 0) * 100).toFixed(0)}%</span>
                                        </span>
                                        <span className={`px-2 py-1 rounded ${getMetricBgColor(testSetRun.avg_answer_correctness)}`}>
                                          <span className="text-slate-500">AC:</span> <span className={`font-semibold ${getMetricColor(testSetRun.avg_answer_correctness)}`}>{((testSetRun.avg_answer_correctness ?? 0) * 100).toFixed(0)}%</span>
                                        </span>
                                      </div>
                                    </>
                                  );
                                })()}
                              </div>
                            )}
                          </div>
                        </div>
                        <div className="flex items-center gap-2">
                          {testSetRun && (testSetRun.status === "completed" || testSetRun.status === "running") && (
                            <Link href={`/dashboard/ragas/runs/${testSetRun.id}`}>
                              <Button variant="outline" size="sm" className={testSetRun.status === "completed" ? "border-emerald-200 text-emerald-600 hover:bg-emerald-50" : "border-blue-200 text-blue-600 hover:bg-blue-50"}>
                                <BarChart3 className="w-4 h-4 mr-1" />
                                {testSetRun.status === "running" ? "Görüntüle" : "Sonuçlar"}
                              </Button>
                            </Link>
                          )}
                          <Link href={`/dashboard/ragas/test-sets/${ts.id}`}>
                            <Button variant="ghost" size="sm" className="text-indigo-600 hover:text-indigo-700 hover:bg-indigo-50">
                              Düzenle <ArrowRight className="w-4 h-4 ml-1" />
                            </Button>
                          </Link>
                          <Button variant="ghost" size="sm" onClick={() => handleDeleteTestSet(ts.id)} className="text-slate-400 hover:text-red-600 hover:bg-red-50">
                            <Trash2 className="w-4 h-4" />
                          </Button>
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>
            )}
          </Card>
        </div>
      )}
    </div>
  );
}
