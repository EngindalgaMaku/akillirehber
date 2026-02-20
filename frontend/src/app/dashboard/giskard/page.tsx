"use client";

import { useEffect, useState, useCallback } from "react";
import { useAuth } from "@/lib/auth-context";
import { api, Course, GiskardTestSet, GiskardEvaluationRun, GiskardQuickTestSavedResult, GiskardQuickTestResultListResponse, GiskardQuickTestResponse } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { Card } from "@/components/ui/card";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog";
import { toast } from "sonner";
import { Shield, Plus, Play, Trash2, FileText, Loader2, CheckCircle, XCircle, Clock, BarChart3, ArrowRight, Save, ChevronDown, History, Eye, Bookmark, Target, Layers, BookOpen, Sparkles, HelpCircle } from "lucide-react";
import Link from "next/link";

export default function GiskardPage() {
  const { user } = useAuth();
  const [courses, setCourses] = useState<Course[]>([]);
  const [selectedCourseId, setSelectedCourseId] = useState<number | null>(null);
  const [testSets, setTestSets] = useState<GiskardTestSet[]>([]);
  const [runs, setRuns] = useState<GiskardEvaluationRun[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [isCreateOpen, setIsCreateOpen] = useState(false);
  const [isCreating, setIsCreating] = useState(false);
  const [newTestSet, setNewTestSet] = useState({ name: "", description: "" });
  const [isQuickTestExpanded, setIsQuickTestExpanded] = useState(false);
  const [quickTestQuestion, setQuickTestQuestion] = useState("");
  const [quickTestQuestionType, setQuickTestQuestionType] = useState<"relevant" | "irrelevant">("relevant");
  const [quickTestExpectedAnswer, setQuickTestExpectedAnswer] = useState("");
  const [quickTestSystemPrompt, setQuickTestSystemPrompt] = useState("");
  const [quickTestLlmModel, setQuickTestLlmModel] = useState("");
  const [isQuickTesting, setIsQuickTesting] = useState(false);
  const [quickTestResult, setQuickTestResult] = useState<GiskardQuickTestResponse | null>(null);
  const [savedResults, setSavedResults] = useState<GiskardQuickTestSavedResult[]>([]);
  const [savedResultsGroups, setSavedResultsGroups] = useState<string[]>([]);
  const [savedResultsTotal, setSavedResultsTotal] = useState(0);
  const [isSavedResultsExpanded, setIsSavedResultsExpanded] = useState(false);
  const [selectedGroup, setSelectedGroup] = useState<string>("");
  const [isSaveDialogOpen, setIsSaveDialogOpen] = useState(false);
  const [saveGroupName, setSaveGroupName] = useState("");
  const [isSaving, setIsSaving] = useState(false);
  const [viewingResult, setViewingResult] = useState<GiskardQuickTestSavedResult | null>(null);
  const [resultsPage, setResultsPage] = useState(0);
  const [isLoadingMore, setIsLoadingMore] = useState(false);
  const [isMetricExplanationOpen, setIsMetricExplanationOpen] = useState(false);
  const [selectedMetric, setSelectedMetric] = useState<string | null>(null);
  const RESULTS_PER_PAGE = 10;

  const metricExplanations = {
    score: {
      title: "Genel Skor (Overall Score)",
      description: "RAG sisteminin performansını tek bir sayı ile özetleyen kümülatif bir performans metriğidir.",
      details: [
        "Alakalı Sorular Skoru: %50 ağırlık",
        "Alakasız Sorular Skoru: %40 ağırlık",
        "Dil Tutarlılığı Skoru: %10 ağırlık",
        "",
        "Matematiksel Formül:",
        "Genel Skor = (Alakalı_Skor × 0.5) + (Alakasız_Skor × 0.4) + (Dil_Skor × 0.1)",
        "",
        "İdeal Değerler:",
        "• Mükemmel: > 90%",
        "• İyi: 80% - 90%",
        "• Orta: 60% - 80%",
        "• Zayıf: < 60%"
      ]
    },
    hallucination: {
      title: "Halüsinasyon (Hallucination)",
      description: "Bir yapay zeka modelinin eğitim verilerinde veya bilgi tabanında bulunmayan, gerçek olmayan veya doğrulanamayan bilgiler üretmesidir.",
      details: [
        "Tespit Yöntemi:",
        "• Alakasız sorular sorulur",
        "• Model 'Bilmiyorum' demeli",
        "• Uydurma cevap verirse halüsinasyon yapmış kabul edilir",
        "",
        "Matematiksel Formül:",
        "Halüsinasyon Oranı = (Halüsinasyon Yapılan Soru Sayısı) / (Toplam Alakasız Soru Sayısı)",
        "",
        "Bilimsel Önemi:",
        "• Güvenilirlik: Sistemin güvenilirliğini ciddi şekilde etkiler",
        "• Yanlış Bilgi Yayılımı: Yanlış bilgilerin yayılmasına neden olabilir",
        "• Etik Standartlar: AI etik standartları halüsinasyon riskini minimize etmeyi zorunlu kılar"
      ]
    },
    correct_refusal: {
      title: "Doğru Reddetme (Correct Rejection)",
      description: "Bir modelin bilgi tabanında bulunmayan veya yanıt veremeyeceği soruları doğru bir şekilde reddetme yeteneğidir.",
      details: [
        "Tespit Yöntemi:",
        "• Reddetme İfadeleri: 'bilmiyorum', 'bilgim yok', 'bilgim bulunmuyor', 'bu konuda bilgi veremem', 'notlarda bu konu yok'",
        "• Bu ifadelerden biri varsa → Doğru Reddetme",
        "• Yoksa → Yanlış Reddetme (Halüsinasyon)",
        "",
        "Matematiksel Formül:",
        "Doğru Reddetme Oranı = 1 - Halüsinasyon Oranı",
        "",
        "İdeal Değerler:",
        "• Mükemmel: > 95%",
        "• İyi: 85% - 95%",
        "• Orta: 70% - 85%",
        "• Zayıf: < 70%"
      ]
    },
    quality_score: {
      title: "Kalite Skoru (Quality Score)",
      description: "Tek bir cevabın genel kalitesini değerlendiren bileşik bir metriktir. Bu skor, iki ana bileşenin ağırlıklı toplamı olarak hesaplanır.",
      details: [
        "Bileşenler:",
        "• Doğruluk Skoru (Correctness): %70 ağırlık",
        "• Dil Skoru (Language): %30 ağırlık",
        "",
        "Matematiksel Formül:",
        "Kalite Skoru = (Doğruluk_Skor × 0.7) + (Dil_Skor × 0.3)",
        "",
        "Doğruluk Skoru:",
        "• Alakalı sorular için: Cevap sağlanmış ve yeterli uzunlukta (>20 karakter) → 1.0",
        "• Alakasız sorular için: 'Bilmiyorum' veya benzeri reddetme → 1.0",
        "• Aksi halde → 0.0",
        "",
        "Dil Skoru:",
        "• Cevap tamamen Türkçe → 0.3",
        "• Cevap Türkçe değil → 0.0"
      ]
    },
    language: {
      title: "Dil (Language)",
      description: "Modelin ürettiği cevapların dilini belirler. Bu metrik, dil tutarlılığı ve dil tespiti tekniklerini kullanır.",
      details: [
        "Tespit Yöntemi:",
        "• Türkçe Karakter Kontrolü: ç, ğ, ı, ö, ş, ü (küçük harfler) ve Ç, Ğ, İ, Ö, Ş, Ü (büyük harfler)",
        "• İngilizce Kelime Kontrolü: the, and, is, of, to, in, that",
        "",
        "Karar Mekanizması:",
        "• Türkçe karakter var VE İngilizce kelime yok → Türkçe",
        "• İngilizce kelime var VE Türkçe karakter yok → İngilizce",
        "• Her ikisi de var → Karışık",
        "",
        "Bilimsel Önemi:",
        "• Kullanıcı Beklentisi: Türkçe bir sistemden Türkçe cevap beklenir",
        "• Kültürel Uygunluk: Dil, kültürel bağlamın önemli bir parçasıdır",
        "• Eğitim Kalitesi: Öğrencilerin anadillerinde eğitim almaları öğrenme verimliliğini artırır"
      ]
    }
  };

  const loadTestSets = useCallback(async () => {
    if (!selectedCourseId) return;
    try {
      const data = await api.getGiskardTestSets(selectedCourseId);
      setTestSets(data);
    } catch {
      toast.error("Test setleri yüklenirken hata oluştu");
    }
  }, [selectedCourseId]);

  const loadRuns = useCallback(async () => {
    if (!selectedCourseId) return;
    try {
      const data = await api.getGiskardEvaluationRuns(selectedCourseId);
      setRuns(data);
    } catch {
      toast.error("Değerlendirmeler yüklenirken hata oluştu");
    }
  }, [selectedCourseId]);

  useEffect(() => {
    loadCourses();
  }, []);

  const loadSavedResults = useCallback(async (reset: boolean = true) => {
    if (!selectedCourseId) return;
    try {
      let groupFilter: string | undefined;
      if (selectedGroup === "__all__" || selectedGroup === "") {
        groupFilter = undefined;
      } else if (selectedGroup === "__no_group__") {
        groupFilter = "";
      } else {
        groupFilter = selectedGroup;
      }
      const skip = reset ? 0 : resultsPage * RESULTS_PER_PAGE;
      const data = await api.getGiskardQuickTestResults(
        selectedCourseId,
        groupFilter,
        skip,
        RESULTS_PER_PAGE
      );
      if (reset) {
        setSavedResults(data.results);
        setResultsPage(1);
      } else {
        setSavedResults(prev => [...prev, ...data.results]);
        setResultsPage(prev => prev + 1);
      }
      setSavedResultsTotal(data.total);
      setSavedResultsGroups(data.groups);
    } catch {
    }
  }, [selectedCourseId, selectedGroup, resultsPage]);

  const loadMoreResults = async () => {
    setIsLoadingMore(true);
    await loadSavedResults(false);
    setIsLoadingMore(false);
  };

  useEffect(() => {
    if (selectedCourseId) {
      loadTestSets();
      loadRuns();
      loadSavedResults(true);
    }
  }, [selectedCourseId, loadTestSets, loadRuns]);
  useEffect(() => {
    if (selectedCourseId) {
      setResultsPage(0);
      loadSavedResults(true);
    }
  }, [selectedGroup]);

  const loadCourses = async () => {
    try {
      const data = await api.getCourses();
      setCourses(data);
      if (data.length > 0) {
        const savedCourseId = localStorage.getItem('giskard_selected_course_id');
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

  const handleCreateTestSet = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!selectedCourseId) return;
    setIsCreating(true);
    try {
      await api.createGiskardTestSet({
        course_id: selectedCourseId,
        name: newTestSet.name,
        description: newTestSet.description || undefined
      });
      setNewTestSet({ name: "", description: "" });
      setIsCreateOpen(false);
      loadTestSets();
      toast.success("Test seti oluşturuldu");
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "Hata oluştu");
    } finally {
      setIsCreating(false);
    }
  };

  const handleDeleteTestSet = async (id: number) => {
    if (!confirm("Bu test setini silmek istediğinizden emin misiniz?")) return;
    try {
      await api.deleteGiskardTestSet(id);
      loadTestSets();
      toast.success("Test seti silindi");
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "Hata oluştu");
    }
  };

  const handleDeleteRun = async (id: number) => {
    if (!confirm("Bu değerlendirmeyi silmek istediğinizden emin misiniz?")) return;
    try {
      await api.deleteGiskardEvaluationRun(id);
      loadRuns();
      toast.success("Değerlendirme silindi");
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "Hata oluştu");
    }
  };

  const handleQuickTest = async () => {
    if (!selectedCourseId || !quickTestQuestion || !quickTestExpectedAnswer) {
      toast.error("Lütfen ders, soru ve beklenen cevap alanlarını doldurun");
      return;
    }
    setIsQuickTesting(true);
    setQuickTestResult(null);
    try {
      const result = await api.giskardQuickTest({
        course_id: selectedCourseId,
        question: quickTestQuestion,
        question_type: quickTestQuestionType,
        expected_answer: quickTestExpectedAnswer,
        system_prompt: quickTestSystemPrompt || undefined,
        llm_provider: quickTestLlmModel ? "openrouter" : undefined,
        llm_model: quickTestLlmModel || undefined
      });
      setQuickTestResult(result);
      toast.success("Test tamamlandı");
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "Test başarısız");
    } finally {
      setIsQuickTesting(false);
    }
  };

  const handleSaveResult = async () => {
    if (!selectedCourseId || !quickTestResult) return;
    setIsSaving(true);
    try {
      await api.saveGiskardQuickTestResult({
        course_id: selectedCourseId,
        group_name: saveGroupName || undefined,
        question: quickTestResult.question,
        question_type: quickTestResult.question_type || "relevant",
        expected_answer: quickTestResult.expected_answer,
        generated_answer: quickTestResult.generated_answer,
        score: quickTestResult.score,
        correct_refusal: quickTestResult.correct_refusal,
        hallucinated: quickTestResult.hallucinated,
        provided_answer: quickTestResult.provided_answer,
        language: quickTestResult.language,
        quality_score: quickTestResult.quality_score,
        system_prompt: quickTestResult.system_prompt_used,
        llm_provider: quickTestResult.llm_provider_used,
        llm_model: quickTestResult.llm_model_used,
        embedding_model: quickTestResult.embedding_model_used,
        latency_ms: quickTestResult.latency_ms,
        error_message: quickTestResult.error_message,
      });
      toast.success("Sonuç kaydedildi");
      setIsSaveDialogOpen(false);
      setSaveGroupName("");
      loadSavedResults();
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "Kaydetme başarısız");
    } finally {
      setIsSaving(false);
    }
  };

  const handleDeleteSavedResult = async (id: number) => {
    if (!confirm("Bu sonucu silmek istediğinizden emin misiniz?")) return;
    try {
      await api.deleteGiskardQuickTestResult(id);
      toast.success("Sonuç silindi");
      loadSavedResults();
      if (viewingResult?.id === id) {
        setViewingResult(null);
      }
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "Silme başarısız");
    }
  };

  const getMetricColor = (value?: number) => {
    if (value === undefined || value === null) return "text-slate-400";
    if (value >= 0.8) return "text-emerald-600";
    if (value >= 0.6) return "text-amber-600";
    return "text-red-600";
  };

  const getMetricBgColor = (value?: number) => {
    if (value === undefined || value === null) return "bg-slate-50 border-slate-200";
    if (value >= 0.8) return "bg-emerald-50 border-emerald-200";
    if (value >= 0.6) return "bg-amber-50 border-amber-200";
    return "bg-red-50 border-red-200";
  };

  const getStatusBadge = (status: string) => {
    switch (status) {
      case "completed":
        return (
          <span className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium rounded-full bg-emerald-100 text-emerald-700 border border-emerald-200">
            <CheckCircle className="w-3.5 h-3.5" />
            Tamamlandı
          </span>
        );
      case "running":
        return (
          <span className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium rounded-full bg-blue-100 text-blue-700 border border-blue-200">
            <Loader2 className="w-3.5 h-3.5 animate-spin" />
            Çalışıyor
          </span>
        );
      case "failed":
        return (
          <span className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium rounded-full bg-red-100 text-red-700 border border-red-200">
            <XCircle className="w-3.5 h-3.5" />
            Başarısız
          </span>
        );
      default:
        return (
          <span className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium rounded-full bg-slate-100 text-slate-600 border border-slate-200">
            <Clock className="w-3.5 h-3.5" />
            Bekliyor
          </span>
        );
    }
  };

  const completedRuns = runs.filter(r => r.status === "completed");
  const totalQuestions = testSets.reduce((acc, ts) => acc + ts.question_count, 0);

  if (!user) return null;

  if (isLoading) {
    return (
      <div className="min-h-[60vh] flex items-center justify-center">
        <div className="text-center">
          <div className="relative">
            <div className="w-16 h-16 border-4 border-emerald-200 border-t-emerald-600 rounded-full animate-spin mx-auto"></div>
            <Shield className="w-6 h-6 text-emerald-600 absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2" />
          </div>
          <p className="mt-4 text-slate-600 font-medium">Giskard yükleniyor...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-8">
      <div className="relative overflow-hidden bg-gradient-to-br from-emerald-600 via-teal-600 to-cyan-700 rounded-2xl p-8 text-white shadow-xl">
        <div className="absolute inset-0 opacity-30" style={{backgroundImage: "url(\"data:image/svg+xml,%3Csvg width='60' height='60' viewBox='0 0 60 60' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' fill-rule='evenodd'%3E%3Cg fill='%23fff' fill-opacity='0.1'%3E%3Ccircle cx='30' cy='30' r='2'/%3E%3C/g%3E%3C/svg%3E\")"}}></div>
        <div className="absolute -top-24 -right-24 w-96 h-96 bg-white/10 rounded-full blur-3xl"></div>
        <div className="absolute -bottom-24 -left-24 w-96 h-96 bg-emerald-500/20 rounded-full blur-3xl"></div>
        <div className="relative z-10">
          <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between gap-6">
            <div className="flex items-center gap-4">
              <div className="p-3 bg-white/20 rounded-xl backdrop-blur-sm">
                <Shield className="w-8 h-8" />
              </div>
              <div>
                <h1 className="text-3xl font-bold">Giskard Halüsinasyon Testi</h1>
                <p className="text-emerald-200 mt-1">RAG sisteminizin güvenliğini test edin</p>
              </div>
            </div>
            <div className="flex flex-wrap items-center gap-3">
              <Select 
                value={selectedCourseId?.toString() || ""} 
                onValueChange={(v) => {
                  const courseId = Number(v);
                  setSelectedCourseId(courseId);
                  localStorage.setItem('giskard_selected_course_id', courseId.toString());
                }}
              >
                <SelectTrigger className="w-56 bg-white/20 border-0 text-white hover:bg-white/30 backdrop-blur-sm h-10">
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
          {selectedCourseId && (
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mt-8">
              <div className="bg-white/10 backdrop-blur-sm rounded-xl p-4 border border-white/20">
                <div className="flex items-center gap-3">
                  <div className="p-2 bg-emerald-100 rounded-lg">
                    <Layers className="w-5 h-5 text-emerald-600" />
                  </div>
                  <div>
                    <p className="text-emerald-200 text-sm">Test Setleri</p>
                    <p className="text-2xl font-bold">{testSets.length}</p>
                  </div>
                </div>
              </div>
              <div className="bg-white/10 backdrop-blur-sm rounded-xl p-4 border border-white/20">
                <div className="flex items-center gap-3">
                  <div className="p-2 bg-teal-100 rounded-lg">
                    <FileText className="w-5 h-5 text-teal-600" />
                  </div>
                  <div>
                    <p className="text-teal-200 text-sm">Toplam Soru</p>
                    <p className="text-2xl font-bold">{totalQuestions}</p>
                  </div>
                </div>
              </div>
              <div className="bg-white/10 backdrop-blur-sm rounded-xl p-4 border border-white/20">
                <div className="flex items-center gap-3">
                  <div className="p-2 bg-cyan-100 rounded-lg">
                    <BarChart3 className="w-5 h-5 text-cyan-600" />
                  </div>
                  <div>
                    <p className="text-cyan-200 text-sm">Değerlendirmeler</p>
                    <p className="text-2xl font-bold">{runs.length}</p>
                  </div>
                </div>
              </div>
              <div className="bg-white/10 backdrop-blur-sm rounded-xl p-4 border border-white/20">
                <div className="flex items-center gap-3">
                  <div className="p-2 bg-emerald-100 rounded-lg">
                    <Target className="w-5 h-5 text-emerald-600" />
                  </div>
                  <div>
                    <p className="text-emerald-200 text-sm">Tamamlanan</p>
                    <p className="text-2xl font-bold">{completedRuns.length}</p>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
      
      {!selectedCourseId ? (
        <div className="bg-white rounded-2xl border border-slate-200 p-16 text-center shadow-sm">
          <div className="w-20 h-20 bg-emerald-100 rounded-full flex items-center justify-center mx-auto mb-6">
            <Shield className="w-10 h-10 text-emerald-600" />
          </div>
          <h3 className="text-xl font-semibold text-slate-900 mb-2">Ders Seçin</h3>
          <p className="text-slate-500 max-w-md mx-auto">Giskard halüsinasyon testi yapmak için yukarıdan bir ders seçin.</p>
        </div>
      ) : (
        <div className="space-y-6">
          <Card className="overflow-hidden border-0 shadow-lg bg-gradient-to-br from-emerald-50 via-white to-teal-50">
            <button 
              onClick={() => setIsQuickTestExpanded(!isQuickTestExpanded)} 
              className="w-full px-6 py-5 flex items-center justify-between hover:bg-emerald-50/50 transition-all duration-200"
            >
              <div className="flex items-center gap-4">
                <div className="p-3 bg-gradient-to-br from-emerald-500 to-teal-600 rounded-xl shadow-lg shadow-emerald-200">
                  <Sparkles className="w-6 h-6 text-white" />
                </div>
                <div className="text-left">
                  <h2 className="text-xl font-bold text-slate-900">Hızlı Test</h2>
                  <p className="text-sm text-slate-600">Tek bir soru için anında halüsinasyon testi</p>
                </div>
              </div>
              <div className={`p-2 rounded-full bg-emerald-100 transition-transform duration-200 ${isQuickTestExpanded ? 'rotate-180' : ''}`}>
                <ChevronDown className="w-5 h-5 text-emerald-600" />
              </div>
            </button>
            
            {isQuickTestExpanded && (
              <div className="px-6 pb-6 pt-2 border-t border-emerald-100">
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                  <div className="space-y-4">
                    <div>
                      <Label className="text-sm font-medium text-slate-700">Soru</Label>
                      <Textarea 
                        value={quickTestQuestion} 
                        onChange={(e) => setQuickTestQuestion(e.target.value)} 
                        placeholder="Test etmek istediğiniz soruyu girin..." 
                        rows={3} 
                        className="mt-1.5 border-slate-200 focus:border-emerald-400 focus:ring-emerald-400" 
                      />
                    </div>
                    <div>
                      <Label className="text-sm font-medium text-slate-700">Soru Tipi</Label>
                      <Select 
                        value={quickTestQuestionType} 
                        onValueChange={(v) => setQuickTestQuestionType(v as "relevant" | "irrelevant")}
                      >
                        <SelectTrigger className="h-11">
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="relevant">İlgili (Notlarda var)</SelectItem>
                          <SelectItem value="irrelevant">İlginç (Notlarda yok)</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>
                    <div>
                      <Label className="text-sm font-medium text-slate-700">Beklenen Cevap</Label>
                      <Textarea 
                        value={quickTestExpectedAnswer} 
                        onChange={(e) => setQuickTestExpectedAnswer(e.target.value)} 
                        placeholder="Beklenen doğru cevabı girin..." 
                        rows={3} 
                        className="mt-1.5 border-slate-200 focus:border-emerald-400 focus:ring-emerald-400" 
                      />
                    </div>
                    <div>
                      <div className="flex items-center justify-between mb-2">
                        <Label className="text-sm font-medium text-slate-700">Sistem Promptu</Label>
                        <Button 
                          type="button" 
                          variant="outline" 
                          size="sm" 
                          onClick={() => {}} 
                          disabled={!quickTestSystemPrompt} 
                          className="h-7 text-xs border-emerald-200 text-emerald-600 hover:bg-emerald-50"
                        >
                          <Save className="w-3 h-3 mr-1" />
                          Kaydet
                        </Button>
                      </div>
                      <Textarea 
                        value={quickTestSystemPrompt} 
                        onChange={(e) => setQuickTestSystemPrompt(e.target.value)} 
                        placeholder="Sistem promptu (boş bırakılırsa ders ayarlarından alınır)" 
                        rows={3} 
                        className="mt-1 border-slate-200 focus:border-emerald-400 focus:ring-emerald-400" 
                      />
                    </div>
                    <div>
                      <Label className="text-sm font-medium text-slate-700">OpenRouter Model (Opsiyonel)</Label>
                      <Input 
                        value={quickTestLlmModel} 
                        onChange={(e) => setQuickTestLlmModel(e.target.value)} 
                        placeholder="örn: openai/gpt-4o-mini" 
                        className="mt-1.5 border-slate-200" 
                      />
                    </div>
                    <Button 
                      onClick={handleQuickTest} 
                      disabled={isQuickTesting || !quickTestQuestion || !quickTestExpectedAnswer} 
                      className="w-full bg-gradient-to-r from-emerald-600 to-teal-600 hover:from-emerald-700 hover:to-teal-700 shadow-lg shadow-emerald-200"
                    >
                      {isQuickTesting ? (
                        <>
                          <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                          Test Ediliyor...
                        </>
                      ) : (
                        <>
                          <Play className="w-4 h-4 mr-2" />
                          Test Et
                        </>
                      )}
                    </Button>
                  </div>
                  <div className="space-y-4">
                    {quickTestResult ? (
                      <div>
                        <div>
                          <Label className="text-sm font-medium text-slate-700">Üretilen Cevap</Label>
                          <div className="mt-1.5 p-4 bg-white rounded-xl border border-slate-200 text-sm shadow-sm">
                            {quickTestResult.generated_answer}
                          </div>
                        </div>
                        <div>
                          <Label className="text-sm font-medium text-slate-700">Metrikler</Label>
                          <div className="grid grid-cols-2 gap-3 mt-2">
                            <div className={`p-4 rounded-xl border ${getMetricBgColor(quickTestResult.score)}`}>
                              <div className="flex items-center justify-between">
                                <p className="text-xs text-slate-600 font-medium">Skor</p>
                                <button
                                  type="button"
                                  onClick={() => { setSelectedMetric('score'); setIsMetricExplanationOpen(true); }}
                                  className="text-slate-400 hover:text-emerald-600 transition-colors"
                                >
                                  <HelpCircle className="w-4 h-4" />
                                </button>
                              </div>
                              <p className={`text-2xl font-bold ${getMetricColor(quickTestResult.score)}`}>
                                {quickTestResult.score !== undefined && quickTestResult.score !== null ? `${(quickTestResult.score * 100).toFixed(1)}%` : "N/A"}
                              </p>
                            </div>
                            <div className={`p-4 rounded-xl border ${quickTestResult.hallucinated ? "bg-red-50 border-red-200" : "bg-emerald-50 border-emerald-200"}`}>
                              <div className="flex items-center justify-between">
                                <p className="text-xs text-slate-600 font-medium">Halüsinasyon</p>
                                <button
                                  type="button"
                                  onClick={() => { setSelectedMetric('hallucination'); setIsMetricExplanationOpen(true); }}
                                  className="text-slate-400 hover:text-emerald-600 transition-colors"
                                >
                                  <HelpCircle className="w-4 h-4" />
                                </button>
                              </div>
                              <p className={`text-2xl font-bold ${quickTestResult.hallucinated !== undefined && quickTestResult.hallucinated !== null ? (quickTestResult.hallucinated ? "text-red-600" : "text-emerald-600") : "text-slate-400"}`}>
                                {quickTestResult.hallucinated !== undefined && quickTestResult.hallucinated !== null ? (quickTestResult.hallucinated ? "Evet" : "Hayır") : "N/A"}
                              </p>
                            </div>
                            <div className={`p-4 rounded-xl border ${quickTestResult.correct_refusal ? "bg-emerald-50 border-emerald-200" : "bg-red-50 border-red-200"}`}>
                              <div className="flex items-center justify-between">
                                <p className="text-xs text-slate-600 font-medium">Doğru Reddetme</p>
                                <button
                                  type="button"
                                  onClick={() => { setSelectedMetric('correct_refusal'); setIsMetricExplanationOpen(true); }}
                                  className="text-slate-400 hover:text-emerald-600 transition-colors"
                                >
                                  <HelpCircle className="w-4 h-4" />
                                </button>
                              </div>
                              <p className={`text-2xl font-bold ${quickTestResult.correct_refusal !== undefined && quickTestResult.correct_refusal !== null ? (quickTestResult.correct_refusal ? "text-emerald-600" : "text-red-600") : "text-slate-400"}`}>
                                {quickTestResult.correct_refusal !== undefined && quickTestResult.correct_refusal !== null ? (quickTestResult.correct_refusal ? "Evet" : "Hayır") : "N/A"}
                              </p>
                            </div>
                            <div className={`p-4 rounded-xl border ${getMetricBgColor(quickTestResult.quality_score)}`}>
                              <div className="flex items-center justify-between">
                                <p className="text-xs text-slate-600 font-medium">Kalite Skoru</p>
                                <button
                                  type="button"
                                  onClick={() => { setSelectedMetric('quality_score'); setIsMetricExplanationOpen(true); }}
                                  className="text-slate-400 hover:text-emerald-600 transition-colors"
                                >
                                  <HelpCircle className="w-4 h-4" />
                                </button>
                              </div>
                              <p className={`text-2xl font-bold ${getMetricColor(quickTestResult.quality_score)}`}>
                                {quickTestResult.quality_score !== undefined && quickTestResult.quality_score !== null ? `${(quickTestResult.quality_score * 100).toFixed(1)}%` : "N/A"}
                              </p>
                            </div>
                            <div className={`p-4 rounded-xl border ${quickTestResult.language === "Türkçe" ? "bg-emerald-50 border-emerald-200" : quickTestResult.language === "İngilizce" ? "bg-red-50 border-red-200" : "bg-amber-50 border-amber-200"}`}>
                              <div className="flex items-center justify-between">
                                <p className="text-xs text-slate-600 font-medium">Dil</p>
                                <button
                                  type="button"
                                  onClick={() => { setSelectedMetric('language'); setIsMetricExplanationOpen(true); }}
                                  className="text-slate-400 hover:text-emerald-600 transition-colors"
                                >
                                  <HelpCircle className="w-4 h-4" />
                                </button>
                              </div>
                              <p className={`text-2xl font-bold ${quickTestResult.language || "N/A"}`}>
                                {quickTestResult.language || "N/A"}
                              </p>
                            </div>
                          </div>
                          <div className="grid grid-cols-2 gap-3 mt-2">
                            <div className="p-4 bg-white rounded-xl border border-slate-200 shadow-sm">
                              <p className="text-xs text-slate-500 font-medium">Gecikme</p>
                              <p className="text-lg font-bold text-slate-900">{quickTestResult.latency_ms}ms</p>
                            </div>
                            <div className="p-4 bg-white rounded-xl border border-slate-200 shadow-sm">
                              <p className="text-xs text-slate-500 font-medium">Model</p>
                              <p className="text-sm font-medium text-slate-900 truncate">{quickTestResult.llm_model_used || "N/A"}</p>
                            </div>
                          </div>
                        </div>
                        <div className="flex items-center gap-2 mt-4">
                          <Dialog open={isSaveDialogOpen} onOpenChange={setIsSaveDialogOpen}>
                            <DialogTrigger asChild>
                              <Button 
                                variant="outline" 
                                className="w-full border-emerald-200 text-emerald-700 hover:bg-emerald-50"
                              >
                                <Bookmark className="w-4 h-4 mr-2" />
                                Sonucu Kaydet
                              </Button>
                            </DialogTrigger>
                            <DialogContent>
                              <DialogHeader>
                                <DialogTitle>Sonuçu Kaydet</DialogTitle>
                                <DialogDescription>Bu test sonucunu daha sonra görüntülemek için kaydedin.</DialogDescription>
                              </DialogHeader>
                              <div className="space-y-4 py-4">
                                <div className="space-y-2">
                                  <Label>Grup Adı (Opsiyonel)</Label>
                                  <Input 
                                    value={saveGroupName} 
                                    onChange={(e) => setSaveGroupName(e.target.value)} 
                                    placeholder="örn: Deneme 1" 
                                  />
                                  {savedResultsGroups.length > 0 && (
                                    <div className="flex flex-wrap gap-1">
                                      {savedResultsGroups.filter((g) => g && g.trim() !== "").map((g) => (
                                        <button 
                                          key={g} 
                                          type="button" 
                                          onClick={() => setSaveGroupName(g)} 
                                          className="px-2 py-1 text-xs bg-slate-100 hover:bg-slate-200 rounded"
                                        >
                                          {g}
                                        </button>
                                      ))}
                                    </div>
                                  )}
                                </div>
                              </div>
                              <DialogFooter>
                                <Button variant="outline" onClick={() => setIsSaveDialogOpen(false)}>İptal</Button>
                                <Button onClick={handleSaveResult} disabled={isSaving}>
                                  {isSaving ? "Kaydediliyor..." : "Kaydet"}
                                </Button>
                              </DialogFooter>
                            </DialogContent>
                          </Dialog>
                        </div>
                      </div>
                    ) : (
                      <div className="flex items-center justify-center h-full min-h-[400px] text-slate-400">
                        <div className="text-center">
                          <div className="w-20 h-20 bg-emerald-100 rounded-full flex items-center justify-center mx-auto mb-4">
                            <Sparkles className="w-10 h-10 text-emerald-400" />
                          </div>
                          <p className="text-sm font-medium">Test sonuçları burada görünecek</p>
                          <p className="text-xs text-slate-400 mt-1">Soru ve beklenen cevabı girerek test başlatın</p>
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              </div>
            )}
          </Card>
          
          <Card className="overflow-hidden border-0 shadow-lg bg-white">
            <button 
              onClick={() => setIsSavedResultsExpanded(!isSavedResultsExpanded)} 
              className="w-full px-6 py-5 flex items-center justify-between hover:bg-slate-50 transition-all duration-200"
            >
              <div className="flex items-center gap-4">
                <div className="p-3 bg-gradient-to-br from-slate-600 to-slate-800 rounded-xl shadow-lg">
                  <History className="w-6 h-6 text-white" />
                </div>
                <div className="text-left">
                  <h2 className="text-xl font-bold text-slate-900">Kaydedilen Sonuçlar</h2>
                  <p className="text-sm text-slate-600">{savedResultsTotal} kayıtlı test sonucu</p>
                </div>
              </div>
              <div className={`p-2 rounded-full bg-slate-100 transition-transform duration-200 ${isSavedResultsExpanded ? 'rotate-180' : ''}`}>
                <ChevronDown className="w-5 h-5 text-slate-600" />
              </div>
            </button>
            
            {isSavedResultsExpanded && (
              <div className="px-6 pb-6 pt-2 border-t border-slate-100">
                {savedResultsGroups.length > 0 && (
                  <div className="mb-4">
                    <Select 
                      value={selectedGroup === "" ? "__all__" : selectedGroup} 
                      onValueChange={(v) => setSelectedGroup(v === "__all__" ? "" : v)}
                    >
                      <SelectTrigger className="w-48">
                        <SelectValue placeholder="Tüm gruplar" />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="__all__">Tüm gruplar</SelectItem>
                        {savedResultsGroups.filter((g) => g && g.trim() !== "").map((g) => (
                          <SelectItem key={g} value={g}>
                            {g}
                          </SelectItem>
                        ))}
                        {savedResultsGroups.some((g) => !g || g.trim() === "") && (
                          <SelectItem value="__no_group__">Grupsuz</SelectItem>
                        )}
                      </SelectContent>
                    </Select>
                  </div>
                )}
                
                {savedResults.length === 0 ? (
                  <div className="text-center py-12 text-slate-400">
                    <History className="w-12 h-12 mx-auto mb-3 opacity-50" />
                    <p className="font-medium">Henüz kaydedilmiş sonuç yok</p>
                  </div>
                ) : (
                  <div className="space-y-3">
                    {savedResults.map((result) => (
                      <div key={result.id} className="p-4 bg-slate-50 rounded-xl border border-slate-200 hover:border-slate-300 transition-colors">
                        <div className="flex items-start justify-between">
                          <div className="flex-1 min-w-0">
                            <div className="flex items-center gap-2 mb-1">
                              {result.group_name && (
                                <span className="px-2 py-0.5 text-xs bg-emerald-100 text-emerald-700 rounded-full font-medium">
                                  {result.group_name}
                                </span>
                              )}
                              <span className="text-xs text-slate-500">
                                {new Date(result.created_at).toLocaleString("tr-TR")}
                              </span>
                            </div>
                            <p className="text-sm font-medium text-slate-900 truncate">{result.question}</p>
                          </div>
                          <div className="flex items-center gap-2">
                            <Button 
                              variant="ghost" 
                              size="sm" 
                              onClick={() => setViewingResult(result)} 
                              className="text-slate-600 hover:text-emerald-600 hover:bg-emerald-50"
                            >
                              <Eye className="w-4 h-4" />
                            </Button>
                            <Button 
                              variant="ghost" 
                              size="sm" 
                              onClick={() => handleDeleteSavedResult(result.id)} 
                              className="text-slate-400 hover:text-red-600 hover:bg-red-50"
                            >
                              <Trash2 className="w-4 h-4" />
                            </Button>
                          </div>
                        </div>
                      </div>
                    ))}
                    {savedResults.length < savedResultsTotal && (
                      <Button 
                        variant="outline" 
                        onClick={loadMoreResults} 
                        disabled={isLoadingMore} 
                        className="w-full"
                      >
                        {isLoadingMore ? (
                          <>
                            <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                            Yükleniyor...
                          </>
                        ) : (
                          <>
                            Daha Fazla ({savedResults.length}/{savedResultsTotal})
                          </>
                        )}
                      </Button>
                    )}
                  </div>
                )}
              </div>
            )}
          </Card>
          
          <Card className="overflow-hidden border-0 shadow-lg">
            <div className="flex items-center justify-between px-6 py-4 bg-gradient-to-r from-emerald-500 to-teal-600 text-white">
              <div className="flex items-center gap-3">
                <FileText className="w-5 h-5" />
                <h2 className="font-semibold">Test Setleri</h2>
              </div>
              <div className="flex items-center gap-2">
                <Dialog open={isCreateOpen} onOpenChange={setIsCreateOpen}>
                  <DialogTrigger asChild>
                    <Button 
                      size="sm" 
                      variant="secondary" 
                      className="bg-white/20 hover:bg-white/30 text-white border-0"
                    >
                      <Plus className="w-4 h-4 mr-1" />
                      Yeni Test Seti
                    </Button>
                  </DialogTrigger>
                  <DialogContent>
                    <form onSubmit={handleCreateTestSet}>
                      <DialogHeader>
                        <DialogTitle>Yeni Test Seti</DialogTitle>
                        <DialogDescription>Soru-cevap çiftleri içeren bir test seti oluşturun.</DialogDescription>
                      </DialogHeader>
                      <div className="space-y-4 py-4">
                        <div className="space-y-2">
                          <Label>İsim</Label>
                          <Input 
                            value={newTestSet.name} 
                            onChange={(e) => setNewTestSet({ ...newTestSet, name: e.target.value })} 
                            placeholder="Test seti adı" 
                            required 
                          />
                        </div>
                        <div className="space-y-2">
                          <Label>Açıklama</Label>
                          <Input 
                            value={newTestSet.description} 
                            onChange={(e) => setNewTestSet({ ...newTestSet, description: e.target.value })} 
                            placeholder="Opsiyonel açıklama" 
                          />
                        </div>
                      </div>
                      <DialogFooter>
                        <Button type="button" variant="outline" onClick={() => setIsCreateOpen(false)}>İptal</Button>
                        <Button type="submit" disabled={isCreating}>
                          {isCreating ? "Oluşturuluyor..." : "Oluştur"}
                        </Button>
                      </DialogFooter>
                    </form>
                  </DialogContent>
                </Dialog>
              </div>
            </div>
            
            {testSets.length === 0 ? (
              <div className="p-8 text-center text-slate-400">
                <FileText className="w-10 h-10 mx-auto mb-3 opacity-50" />
                <p className="text-sm">Henüz test seti yok</p>
                <p className="text-xs mt-1">Yeni bir test seti oluşturarak başlayın</p>
              </div>
            ) : (
              <div className="divide-y divide-slate-100">
                {testSets.map((ts) => {
                  const testSetRun = runs.find(r => r.test_set_id === ts.id);
                  return (
                    <div key={ts.id} className="px-6 py-4 hover:bg-slate-50 transition-colors">
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-4 flex-1">
                          <div className="p-2 bg-emerald-100 rounded-lg">
                            <Layers className="w-5 h-5 text-emerald-600" />
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
                          </div>
                          <div className="flex items-center gap-2">
                            {testSetRun && (testSetRun.status === "completed" || testSetRun.status === "running") && (
                              <Link href={`/dashboard/giskard/runs/${testSetRun.id}`}>
                                <Button 
                                  variant="outline" 
                                  size="sm" 
                                  className="border-blue-200 text-blue-600 hover:bg-blue-50"
                                >
                                  <BarChart3 className="w-4 h-4 mr-1" />
                                  Sonuçlar
                                </Button>
                              </Link>
                            )}
                            <Link href={`/dashboard/giskard/test-sets/${ts.id}`}>
                              <Button 
                                variant="ghost" 
                                size="sm" 
                                className="text-emerald-600 hover:text-emerald-700 hover:bg-emerald-50"
                              >
                                Düzenle
                                <ArrowRight className="w-4 h-4 ml-1" />
                              </Button>
                            </Link>
                            <Button 
                              variant="ghost" 
                              size="sm" 
                              onClick={() => handleDeleteTestSet(ts.id)} 
                              className="text-slate-400 hover:text-red-600 hover:bg-red-50"
                            >
                              <Trash2 className="w-4 h-4" />
                            </Button>
                          </div>
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
      
      <Dialog open={!!viewingResult} onOpenChange={(open) => !open && setViewingResult(null)}>
        <DialogContent className="max-w-4xl max-h-[90vh] overflow-y-auto">
          <DialogHeader>
            <DialogTitle>Sonuç Detayları</DialogTitle>
            {viewingResult?.group_name && (
              <DialogDescription>Grup: {viewingResult.group_name}</DialogDescription>
            )}
          </DialogHeader>
          {viewingResult && (
            <div className="space-y-4 py-4">
              <div>
                <Label className="text-sm font-medium">Soru</Label>
                <p className="mt-1 p-3 bg-slate-50 rounded-lg text-sm">{viewingResult.question}</p>
              </div>
              <div>
                <Label className="text-sm font-medium">Beklenen Cevap</Label>
                <p className="mt-1 p-3 bg-slate-50 rounded-lg text-sm">{viewingResult.expected_answer}</p>
              </div>
              <div>
                <Label className="text-sm font-medium">Üretilen Cevap</Label>
                <p className="mt-1 p-3 bg-slate-50 rounded-lg text-sm">{viewingResult.generated_answer}</p>
              </div>
              <div className="grid grid-cols-2 gap-3 mt-2">
                <div className={`p-4 rounded-lg border ${getMetricBgColor(viewingResult.faithfulness)}`}>
                  <p className="text-xs text-slate-600 font-medium">İnanılırlık</p>
                  <p className={`text-2xl font-bold ${getMetricColor(viewingResult.faithfulness)}`}>
                    {viewingResult.faithfulness !== undefined && viewingResult.faithfulness !== null ? `${(viewingResult.faithfulness * 100).toFixed(1)}%` : "N/A"}
                  </p>
                </div>
                <div className={`p-4 rounded-lg border ${getMetricBgColor(viewingResult.answer_relevancy)}`}>
                  <p className="text-xs text-slate-600 font-medium">Cevap İlgililiği</p>
                  <p className={`text-2xl font-bold ${getMetricColor(viewingResult.answer_relevancy)}`}>
                    {viewingResult.answer_relevancy !== undefined && viewingResult.answer_relevancy !== null ? `${(viewingResult.answer_relevancy * 100).toFixed(1)}%` : "N/A"}
                  </p>
                </div>
                <div className={`p-4 rounded-lg border ${getMetricBgColor(viewingResult.context_precision)}`}>
                  <p className="text-xs text-slate-600 font-medium">Bağlam Hassasiyeti</p>
                  <p className={`text-2xl font-bold ${getMetricColor(viewingResult.context_precision)}`}>
                    {viewingResult.context_precision !== undefined && viewingResult.context_precision !== null ? `${(viewingResult.context_precision * 100).toFixed(1)}%` : "N/A"}
                  </p>
                </div>
                <div className={`p-4 rounded-lg border ${getMetricBgColor(viewingResult.context_recall)}`}>
                  <p className="text-xs text-slate-600 font-medium">Bağlam Geri Çağrısı</p>
                  <p className={`text-2xl font-bold ${getMetricColor(viewingResult.context_recall)}`}>
                    {viewingResult.context_recall !== undefined && viewingResult.context_recall !== null ? `${(viewingResult.context_recall * 100).toFixed(1)}%` : "N/A"}
                  </p>
                </div>
                <div className={`p-4 rounded-lg border ${getMetricBgColor(viewingResult.answer_correctness)}`}>
                  <p className="text-xs text-slate-600 font-medium">Cevap Doğruluğu</p>
                  <p className={`text-2xl font-bold ${getMetricColor(viewingResult.answer_correctness)}`}>
                    {viewingResult.answer_correctness !== undefined && viewingResult.answer_correctness !== null ? `${(viewingResult.answer_correctness * 100).toFixed(1)}%` : "N/A"}
                  </p>
                </div>
              </div>
              <div className="grid grid-cols-2 gap-3 mt-2">
                <div className="p-4 bg-white rounded-xl border border-slate-200 shadow-sm">
                  <p className="text-xs text-slate-500 font-medium">Gecikme</p>
                  <p className="text-lg font-bold text-slate-900">{viewingResult.latency_ms}ms</p>
                </div>
                <div className="p-4 bg-white rounded-xl border border-slate-200 shadow-sm">
                  <p className="text-xs text-slate-500 font-medium">Model</p>
                  <p className="text-sm font-medium text-slate-900 truncate">{viewingResult.llm_model || "N/A"}</p>
                </div>
              </div>
            </div>
          )}
          <DialogFooter>
            <Button variant="outline" onClick={() => setViewingResult(null)}>Kapat</Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      <Dialog open={isMetricExplanationOpen} onOpenChange={setIsMetricExplanationOpen}>
        <DialogContent className="max-w-2xl max-h-[90vh] overflow-y-auto">
          <DialogHeader>
            <DialogTitle className="text-xl font-bold text-slate-900">
              {selectedMetric && metricExplanations[selectedMetric as keyof typeof metricExplanations]?.title}
            </DialogTitle>
          </DialogHeader>
          {selectedMetric && metricExplanations[selectedMetric as keyof typeof metricExplanations] && (
            <div className="space-y-4 py-4">
              <div className="bg-gradient-to-r from-emerald-50 to-teal-50 p-4 rounded-xl border border-emerald-200">
                <p className="text-sm text-slate-700 leading-relaxed">
                  {metricExplanations[selectedMetric as keyof typeof metricExplanations].description}
                </p>
              </div>
              <div className="bg-white p-4 rounded-xl border border-slate-200">
                <h4 className="text-sm font-semibold text-slate-900 mb-3">Detaylar</h4>
                <div className="space-y-2">
                  {metricExplanations[selectedMetric as keyof typeof metricExplanations].details.map((detail, index) => (
                    <p key={index} className={`text-sm ${detail === '' ? 'mt-3' : ''} ${detail.startsWith('•') || detail.startsWith('-') ? 'text-slate-600' : 'text-slate-700'} leading-relaxed`}>
                      {detail === '' ? '\u00A0' : detail}
                    </p>
                  ))}
                </div>
              </div>
            </div>
          )}
          <DialogFooter>
            <Button variant="outline" onClick={() => setIsMetricExplanationOpen(false)}>Kapat</Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}
