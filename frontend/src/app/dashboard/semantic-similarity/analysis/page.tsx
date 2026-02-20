"use client";

import { useEffect, useState, useCallback, useRef } from "react";
import { useAuth } from "@/lib/auth-context";
import {
  api,
  Course,
  SemanticSimilarityResult,
  SemanticSimilarityResultListResponse
} from "@/lib/api";
import { Button } from "@/components/ui/button";
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
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Label } from "@/components/ui/label";
import { toast } from "sonner";
import {
  Target,
  Loader2,
  ChevronDown,
  History,
  BookOpen,
  TrendingUp,
  TrendingDown,
  BarChart3,
  FileText,
  Download,
  ArrowRight,
  CheckCircle2,
  AlertTriangle,
  Info,
  ArrowLeft,
  Settings,
  Trash2
} from "lucide-react";
import Link from "next/link";
import { useRouter, useSearchParams } from "next/navigation";
import { generateAnalysisPDF } from "./exportAnalysisToPDF";

export interface GroupComparison {
  groupName: string;
  results: SemanticSimilarityResult[];
  aggregate: {
    avg_rouge1?: number;
    avg_rouge2?: number;
    avg_rougel?: number;
    avg_bertscore_precision?: number;
    avg_bertscore_recall?: number;
    avg_bertscore_f1?: number;
    avg_latency_ms?: number;
    test_count: number;
  };
}

// NOTE: In the analysis page, "bertscore_f1" fields in GroupComparison.aggregate
// and QuestionComparison are populated from original_bertscore_f1 (bert-score library),
// NOT from the embedding-based bertscore_f1.

export interface QuestionComparison {
  question: string;
  groupResults: {
    groupName: string;
    rouge1?: number;
    rouge2?: number;
    rougel?: number;
    bertscore_f1?: number;
    generated_answer: string;
    ground_truth: string;
  }[];
  variance: {
    rouge1_variance: number;
    rouge1_std_dev: number;
    rouge2_variance: number;
    rouge2_std_dev: number;
    rougel_variance: number;
    rougel_std_dev: number;
    bertscore_f1_variance: number;
    bertscore_f1_std_dev: number;
  };
}

export default function SemanticSimilarityAnalysisPage() {
  const { user } = useAuth();
  const router = useRouter();
  const searchParams = useSearchParams();
  const [courses, setCourses] = useState<Course[]>([]);
  const [selectedCourseId, setSelectedCourseId] = useState<number | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  
  // Analysis State
  const [selectedGroups, setSelectedGroups] = useState<string[]>([]);
  const [availableGroups, setAvailableGroups] = useState<
    { name: string; created_at: string | null; question_count: number }[]
  >([]);
  const [groupComparisons, setGroupComparisons] = useState<GroupComparison[]>([]);
  const [questionComparisons, setQuestionComparisons] = useState<QuestionComparison[]>([]);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [courseSettings, setCourseSettings] = useState<{ default_embedding_model?: string; llm_provider?: string; llm_model?: string } | null>(null);

  // Group rename state
  const [isRenameModalOpen, setIsRenameModalOpen] = useState(false);
  const [groupToRename, setGroupToRename] = useState<{name: string; created_at: string | null} | null>(null);
  const [newGroupName, setNewGroupName] = useState("");
  const [isRenaming, setIsRenaming] = useState(false);

  // Pagination state
  const [groupsPage, setGroupsPage] = useState(1);
  const groupsPerPage = 9;

  // Delete group state
  const [isDeleteModalOpen, setIsDeleteModalOpen] = useState(false);
  const [groupToDelete, setGroupToDelete] = useState<{name: string; created_at: string | null} | null>(null);
  const [isDeleting, setIsDeleting] = useState(false);

  // PDF Export Modal State
  const [isPDFModalOpen, setIsPDFModalOpen] = useState(false);
  const [experimentInfo, setExperimentInfo] = useState({
    title: "",
    experimentName: "",
    date: new Date().toISOString().split('T')[0],
    experimenter: "",
    summary: "",
    objective: "",
    methodology: "",
    evaluation: ""
  });

  // UI State
  const [isOverviewExpanded, setIsOverviewExpanded] = useState(true);
  const [isGroupComparisonExpanded, setIsGroupComparisonExpanded] = useState(true);
  const [isQuestionComparisonExpanded, setIsQuestionComparisonExpanded] = useState(true);
  const [isReliabilityExpanded, setIsReliabilityExpanded] = useState(true);

  // Track if auto-analysis has been triggered
  const autoAnalyzedRef = useRef(false);

  useEffect(() => {
    loadCourses();
  }, []);

  useEffect(() => {
    if (selectedCourseId) {
      loadAvailableGroups();
      loadCourseSettings();
      setGroupsPage(1); // Reset to first page when course changes
    }
  }, [selectedCourseId]);

  // Handle URL parameters for auto-analysis
  useEffect(() => {
    if (autoAnalyzedRef.current) return; // Prevent multiple triggers
    
    const groupsParam = searchParams.get('groups');
    const courseParam = searchParams.get('course');
    
    console.log('URL params:', { groupsParam, courseParam });
    
    if (groupsParam && courseParam) {
      const courseId = parseInt(courseParam);
      const groupNames = groupsParam.split(',').filter(g => g.trim());
      
      console.log('Parsed params:', { courseId, groupNames, isArray: Array.isArray(groupNames) });
      
      if (groupNames.length > 0 && !isNaN(courseId)) {
        autoAnalyzedRef.current = true; // Mark as triggered
        
        // Set course if different
        if (selectedCourseId !== courseId) {
          setSelectedCourseId(courseId);
          localStorage.setItem('semantic_similarity_selected_course_id', courseId.toString());
        }
        
        // Set selected groups and trigger analysis
        setSelectedGroups(groupNames);
        
        // Wait a bit for course to load, then analyze
        setTimeout(() => {
          console.log('Calling runAnalysis with:', { groupNames, courseId });
          runAnalysis(groupNames, courseId);
        }, 500);
      }
    }
  }, [searchParams]);

  const loadCourses = async () => {
    try {
      const data = await api.getCourses();
      setCourses(data);
      if (data.length > 0) {
        // Check localStorage for previously selected course
        const savedCourseId = localStorage.getItem('semantic_similarity_selected_course_id');
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

  const loadAvailableGroups = async () => {
    if (!selectedCourseId) return;
    try {
      const data = await api.getSemanticSimilarityResults(
        selectedCourseId,
        undefined,
        0,
        10000  // Get all groups for pagination
      );
      const counts: Record<string, number> = {};
      for (const r of data.results) {
        const name = (r.group_name || "").trim();
        if (!name) continue;
        counts[name] = (counts[name] || 0) + 1;
      }

      const groups = data.groups
        .filter((g) => g && g.name && g.name.trim() !== "")
        .map((g) => ({
          ...g,
          question_count: counts[g.name] || 0,
        }));
      setAvailableGroups(groups);
    } catch {
      console.log("Failed to load groups");
    }
  };

  const handleRenameGroup = async () => {
    if (!groupToRename || !selectedCourseId) return;

    setIsRenaming(true);
    try {
      await api.renameSemanticSimilarityGroup(
        selectedCourseId,
        groupToRename.name,
        newGroupName.trim()
      );
      toast.success(`Grup adı değiştirildi: ${newGroupName}`);
      
      // Reload all groups and update selected groups
      await loadAvailableGroups();
      setSelectedGroups(selectedGroups.map(g => 
        g === groupToRename.name ? newGroupName.trim() : g
      ));
      
      // Reset to first page after rename
      setGroupsPage(1);
      
      setIsRenameModalOpen(false);
      setGroupToRename(null);
      setNewGroupName("");
    } catch {
      toast.error("Grup adı değiştirilemedi");
    } finally {
      setIsRenaming(false);
    }
  };

  const handleDeleteGroup = async () => {
    if (!groupToDelete || !selectedCourseId) return;
    // Just open the confirmation modal, don't delete yet
    setIsDeleteModalOpen(true);
  };

  const confirmDeleteGroup = async () => {
    if (!groupToDelete || !selectedCourseId) return;

    setIsDeleting(true);
    try {
      const result = await api.deleteSemanticSimilarityGroup(
        selectedCourseId,
        groupToDelete.name
      );
      toast.success(`Grup silindi: ${groupToDelete.name} - ${result.deleted_count} test silindi`);
      
      // Reload all groups and update selected groups
      await loadAvailableGroups();
      setSelectedGroups(selectedGroups.filter(g => g !== groupToDelete.name));
      
      // Reset to first page after delete
      setGroupsPage(1);
      
      setIsDeleteModalOpen(false);
      setGroupToDelete(null);
    } catch {
      toast.error("Grup silinemedi");
    } finally {
      setIsDeleting(false);
    }
  };

  const loadCourseSettings = async () => {
    if (!selectedCourseId) return;
    try {
      const settings = await api.getCourseSettings(selectedCourseId);
      setCourseSettings(settings);
    } catch {
      console.log("Failed to load course settings");
    }
  };

  const runAnalysis = async (groupsToAnalyze?: string[], courseId?: number) => {
    const groups = groupsToAnalyze || selectedGroups;
    const course = courseId || selectedCourseId;
    
    console.log('runAnalysis called with:', { groupsToAnalyze, courseId, selectedGroups, groups, isArray: Array.isArray(groups) });
    
    // Ensure groups is an array
    if (!Array.isArray(groups) || groups.length < 2) {
      toast.error("Lütfen en az 2 grup seçin");
      return;
    }

    if (!course) {
      toast.error("Lütfen bir ders seçin");
      return;
    }

    setIsAnalyzing(true);
    try {
      // Her grup için sonuçları yükle
      const comparisons: GroupComparison[] = [];
      
      for (const groupName of groups) {
        const data = await api.getSemanticSimilarityResults(
          course,
          groupName,
          0,
          10000
        );
        
        // Aggregate hesapla
        const rouge1Results = data.results.filter(r => r.rouge1 != null);
        const rouge2Results = data.results.filter(r => r.rouge2 != null);
        const rougelResults = data.results.filter(r => r.rougel != null);
        const bertPrecisionResults = data.results.filter(r => r.original_bertscore_precision != null);
        const bertRecallResults = data.results.filter(r => r.original_bertscore_recall != null);
        const bertF1Results = data.results.filter(r => r.original_bertscore_f1 != null);
        const latencyResults = data.results.filter(r => r.latency_ms != null);

        comparisons.push({
          groupName,
          results: data.results,
          aggregate: {
            avg_rouge1: rouge1Results.length > 0 
              ? rouge1Results.reduce((sum, r) => sum + r.rouge1!, 0) / rouge1Results.length 
              : undefined,
            avg_rouge2: rouge2Results.length > 0 
              ? rouge2Results.reduce((sum, r) => sum + r.rouge2!, 0) / rouge2Results.length 
              : undefined,
            avg_rougel: rougelResults.length > 0 
              ? rougelResults.reduce((sum, r) => sum + r.rougel!, 0) / rougelResults.length 
              : undefined,
            avg_bertscore_precision: bertPrecisionResults.length > 0 
              ? bertPrecisionResults.reduce((sum, r) => sum + r.original_bertscore_precision!, 0) / bertPrecisionResults.length 
              : undefined,
            avg_bertscore_recall: bertRecallResults.length > 0 
              ? bertRecallResults.reduce((sum, r) => sum + r.original_bertscore_recall!, 0) / bertRecallResults.length 
              : undefined,
            avg_bertscore_f1: bertF1Results.length > 0 
              ? bertF1Results.reduce((sum, r) => sum + r.original_bertscore_f1!, 0) / bertF1Results.length 
              : undefined,
            avg_latency_ms: latencyResults.length > 0 
              ? latencyResults.reduce((sum, r) => sum + r.latency_ms!, 0) / latencyResults.length 
              : undefined,
            test_count: data.results.length
          }
        });
      }

      setGroupComparisons(comparisons);

      // Soru bazlı karşılaştırma
      const questionMap = new Map<string, QuestionComparison["groupResults"]>();
      
      for (const comparison of comparisons) {
        for (const result of comparison.results) {
          if (!questionMap.has(result.question)) {
            questionMap.set(result.question, []);
          }
          questionMap.get(result.question)!.push({
            groupName: comparison.groupName,
            rouge1: result.rouge1,
            rouge2: result.rouge2,
            rougel: result.rougel,
            bertscore_f1: result.original_bertscore_f1,
            generated_answer: result.generated_answer,
            ground_truth: result.ground_truth
          });
        }
      }

      // Soru bazlı varyasyon hesapla
      const questionComps: QuestionComparison[] = [];
      for (const [question, groupResults] of questionMap.entries()) {
        if (groupResults.length >= 2) {
          const rouge1Values = groupResults.filter(g => g.rouge1 != null).map(g => g.rouge1!);
          const rouge2Values = groupResults.filter(g => g.rouge2 != null).map(g => g.rouge2!);
          const rougelValues = groupResults.filter(g => g.rougel != null).map(g => g.rougel!);
          const bertF1Values = groupResults.filter(g => g.bertscore_f1 != null).map(g => g.bertscore_f1!);
          
          const rouge1Variance = rouge1Values.length > 1
            ? rouge1Values.reduce((sum, v) => sum + Math.pow(v - rouge1Values.reduce((a, b) => a + b, 0) / rouge1Values.length, 2), 0) / rouge1Values.length
            : 0;
          const rouge1StdDev = Math.sqrt(rouge1Variance);
          
          const rouge2Variance = rouge2Values.length > 1
            ? rouge2Values.reduce((sum, v) => sum + Math.pow(v - rouge2Values.reduce((a, b) => a + b, 0) / rouge2Values.length, 2), 0) / rouge2Values.length
            : 0;
          const rouge2StdDev = Math.sqrt(rouge2Variance);
          
          const rougelVariance = rougelValues.length > 1
            ? rougelValues.reduce((sum, v) => sum + Math.pow(v - rougelValues.reduce((a, b) => a + b, 0) / rougelValues.length, 2), 0) / rougelValues.length
            : 0;
          const rougelStdDev = Math.sqrt(rougelVariance);
          
          const bertF1Variance = bertF1Values.length > 1
            ? bertF1Values.reduce((sum, v) => sum + Math.pow(v - bertF1Values.reduce((a, b) => a + b, 0) / bertF1Values.length, 2), 0) / bertF1Values.length
            : 0;
          const bertF1StdDev = Math.sqrt(bertF1Variance);

          questionComps.push({
            question,
            groupResults,
            variance: {
              rouge1_variance: rouge1Variance,
              rouge1_std_dev: rouge1StdDev,
              rouge2_variance: rouge2Variance,
              rouge2_std_dev: rouge2StdDev,
              rougel_variance: rougelVariance,
              rougel_std_dev: rougelStdDev,
              bertscore_f1_variance: bertF1Variance,
              bertscore_f1_std_dev: bertF1StdDev
            }
          });
        }
      }

      setQuestionComparisons(questionComps);
      toast.success("Analiz tamamlandı");
    } catch (error) {
      toast.error("Analiz başarısız");
      console.error(error);
    } finally {
      setIsAnalyzing(false);
    }
  };

  const getMetricColor = (value?: number | null) => {
    if (value === undefined || value === null) return "text-slate-400";
    if (value >= 0.8) return "text-emerald-600";
    if (value >= 0.6) return "text-amber-600";
    return "text-red-600";
  };

  const getVarianceColor = (stdDev: number) => {
    if (stdDev < 0.05) return "text-emerald-600 bg-emerald-50 border-emerald-200";
    if (stdDev < 0.1) return "text-amber-600 bg-amber-50 border-amber-200";
    return "text-red-600 bg-red-50 border-red-200";
  };

  const getVarianceLabel = (stdDev: number) => {
    if (stdDev < 0.05) return "Çok Düşük";
    if (stdDev < 0.1) return "Düşük";
    if (stdDev < 0.15) return "Orta";
    if (stdDev < 0.2) return "Yüksek";
    return "Çok Yüksek";
  };

  const exportAnalysisToExcel = () => {
    if (groupComparisons.length === 0) return;
    
    // Grup karşılaştırma
    const groupHeaders = ["Grup", "Test Sayısı", "Ort. ROUGE-1", "Ort. ROUGE-2", "Ort. ROUGE-L", "Ort. BERTScore F1", "Ort. Yanıt Süresi (ms)"];
    const groupRows = groupComparisons.map(g => [
      g.groupName,
      g.aggregate.test_count,
      g.aggregate.avg_rouge1 != null ? (g.aggregate.avg_rouge1 * 100).toFixed(2) + "%" : "-",
      g.aggregate.avg_rouge2 != null ? (g.aggregate.avg_rouge2 * 100).toFixed(2) + "%" : "-",
      g.aggregate.avg_rougel != null ? (g.aggregate.avg_rougel * 100).toFixed(2) + "%" : "-",
      g.aggregate.avg_bertscore_f1 != null ? (g.aggregate.avg_bertscore_f1 * 100).toFixed(2) + "%" : "-",
      g.aggregate.avg_latency_ms != null ? g.aggregate.avg_latency_ms.toFixed(0) : "-"
    ]);

    // Soru bazlı varyasyon
    const questionHeaders = ["Soru", "Grup Sayısı", "ROUGE-1 Std Sapma", "ROUGE-2 Std Sapma", "ROUGE-L Std Sapma", "BERTScore F1 Std Sapma", "Varyasyon Seviyesi"];
    const questionRows = questionComparisons.map(q => [
      `"${q.question.replace(/"/g, '""')}"`,
      q.groupResults.length,
      (q.variance.rouge1_std_dev * 100).toFixed(2) + "%",
      (q.variance.rouge2_std_dev * 100).toFixed(2) + "%",
      (q.variance.rougel_std_dev * 100).toFixed(2) + "%",
      (q.variance.bertscore_f1_std_dev * 100).toFixed(2) + "%",
      getVarianceLabel(q.variance.bertscore_f1_std_dev)
    ]);

    // Güvenilirlik analizi
    const reliabilityHeaders = ["Genel Varyasyon", "Ortalama ROUGE-1 Std Sapma", "Ortalama BERTScore F1 Std Sapma", "Düşük Varyasyonlu Sorular", "Yüksek Varyasyonlu Sorular"];
    const lowVarianceQuestions = questionComparisons.filter(q => q.variance.bertscore_f1_std_dev < 0.1);
    const highVarianceQuestions = questionComparisons.filter(q => q.variance.bertscore_f1_std_dev >= 0.15);
    
    const reliabilityRows: string[][] = [
      [
        questionComparisons.length > 0
          ? (questionComparisons.reduce((sum, q) => sum + q.variance.rouge1_std_dev, 0) / questionComparisons.length * 100).toFixed(2) + "%"
          : "-",
        questionComparisons.length > 0
          ? (questionComparisons.reduce((sum, q) => sum + q.variance.bertscore_f1_std_dev, 0) / questionComparisons.length * 100).toFixed(2) + "%"
          : "-",
        `${lowVarianceQuestions.length} / ${questionComparisons.length}`,
        questionComparisons.length > 0
          ? (lowVarianceQuestions.length / questionComparisons.length * 100).toFixed(0) + "%"
          : "0%",
        `${highVarianceQuestions.length} / ${questionComparisons.length}`,
        questionComparisons.length > 0
          ? (highVarianceQuestions.length / questionComparisons.length * 100).toFixed(0) + "%"
          : "0%"
      ]
    ];

    // Özet bilgileri
    const summaryHeaders = ["Analiz Tarihi", "Ders", "Gruplar", "Embedding Model", "LLM Model"];
    const summaryRows = [[
      new Date().toLocaleString('tr-TR'),
      courses.find(c => c.id === selectedCourseId)?.name || "-",
      selectedGroups.join(", "),
      courseSettings?.default_embedding_model || "-",
      courseSettings?.llm_provider && courseSettings?.llm_model
        ? `${courseSettings.llm_provider}/${courseSettings.llm_model}`
        : "-"
    ]];

    const tsv = [
      "GRUP KARŞILAŞTIRMASI",
      groupHeaders.join("\t"),
      ...groupRows.map(r => r.join("\t")),
      "",
      "",
      "SORU BAZLI VARYASYON ANALİZİ",
      questionHeaders.join("\t"),
      ...questionRows.map(r => r.join("\t")),
      "",
      "",
      "GÜVENİLİRLİK ANALİZİ",
      reliabilityHeaders.join("\t"),
      ...reliabilityRows.map(r => r.join("\t")),
      "",
      "",
      "ÖZET BİLGİLER",
      summaryHeaders.join("\t"),
      ...summaryRows.map(r => r.join("\t"))
    ].join("\n");

    const blob = new Blob(["\ufeff" + tsv], { type: "text/tab-separated-values;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `semantic-similarity-analysis-${Date.now()}.tsv`;
    a.click();
    URL.revokeObjectURL(url);
    toast.success("Analiz raporu indirildi");
  };

  const exportAnalysisToPDF = () => {
    if (groupComparisons.length === 0) return;

    // Validate form
    if (!experimentInfo.title.trim() || !experimentInfo.experimentName.trim() || 
        !experimentInfo.experimenter.trim() || !experimentInfo.summary.trim()) {
      toast.error("Lütfen zorunlu alanları doldurun");
      return;
    }

    const course = courses.find(c => c.id === selectedCourseId);
    const htmlContent = generateAnalysisPDF({
      experimentInfo,
      groupComparisons,
      questionComparisons,
      courseName: course?.name || "-",
      courseSettings
    });

    // Create a new window and print
    const printWindow = window.open('', '_blank');
    if (printWindow) {
      printWindow.document.write(htmlContent);
      printWindow.document.close();
      printWindow.onload = () => {
        printWindow.print();
      };
      toast.success("PDF raporu oluşturuluyor...");
      setIsPDFModalOpen(false);
    } else {
      toast.error("Pop-up engellendi. Lütfen pop-up'lara izin verin.");
    }
  };

  const openPDFModal = () => {
    setIsPDFModalOpen(true);
  };

  if (!user) return null;

  if (isLoading) {
    return (
      <div className="min-h-[60vh] flex items-center justify-center">
        <div className="text-center">
          <div className="relative">
            <div className="w-16 h-16 border-4 border-teal-200 border-t-teal-600 rounded-full animate-spin mx-auto"></div>
            <Target className="w-6 h-6 text-teal-600 absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2" />
          </div>
          <p className="mt-4 text-slate-600 font-medium">Yükleniyor...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="relative overflow-hidden bg-gradient-to-br from-indigo-600 via-purple-600 to-pink-600 rounded-2xl p-8 text-white shadow-xl">
        <div className="absolute top-4 left-4 z-20">
          <Link href="/dashboard/semantic-similarity/results">
            <Button
              variant="secondary"
              size="sm"
              className="bg-white/20 hover:bg-white/30 text-white border-0 backdrop-blur-sm"
            >
              <ArrowLeft className="w-4 h-4 mr-2" />
              Geri Dön
            </Button>
          </Link>
        </div>
        <div className="absolute inset-0 z-0 opacity-30" style={{backgroundImage: "url(\"data:image/svg+xml,%3Csvg width='60' height='60' viewBox='0 0 60 60' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' fill-rule='evenodd'%3E%3Cg fill='%23fff' fill-opacity='0.1'%3E%3Ccircle cx='30' cy='30' r='2'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E\")"}}></div>
        <div className="absolute -top-24 -right-24 z-0 w-96 h-96 bg-white/10 rounded-full blur-3xl"></div>
        
        <div className="relative z-10">
          <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between gap-6">
            <div className="flex items-center gap-4">
              <div className="p-3 bg-white/20 rounded-xl backdrop-blur-sm">
                <BarChart3 className="w-8 h-8" />
              </div>
              <div>
                <h1 className="text-3xl font-bold">Semantic Similarity Analizi</h1>
                <p className="text-indigo-200 mt-1">Test gruplarının istatistiksel karşılaştırması</p>
              </div>
            </div>
            
            <div className="flex items-center gap-3">
              <Select value={selectedCourseId?.toString() || ""} onValueChange={(v) => {
                const courseId = Number(v);
                setSelectedCourseId(courseId);
                localStorage.setItem('semantic_similarity_selected_course_id', courseId.toString());
              }}>
                <SelectTrigger className="w-56 bg-white/20 border-0 text-white hover:bg-white/30 backdrop-blur-sm h-10">
                  <BookOpen className="w-4 h-4 mr-2" />
                  <SelectValue placeholder="Ders seçin" />
                </SelectTrigger>
                <SelectContent>
                  {courses.map((course) => (
                    <SelectItem key={course.id} value={course.id.toString()}>{course.name}</SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          </div>
        </div>
      </div>

      {!selectedCourseId ? (
        <div className="bg-white rounded-2xl border border-slate-200 p-16 text-center shadow-sm">
          <div className="w-20 h-20 bg-indigo-100 rounded-full flex items-center justify-center mx-auto mb-6">
            <BarChart3 className="w-10 h-10 text-indigo-600" />
          </div>
          <h3 className="text-xl font-semibold text-slate-900 mb-2">Ders Seçin</h3>
          <p className="text-slate-500 max-w-md mx-auto">
            Analiz yapmak için yukarıdan bir ders seçin.
          </p>
        </div>
      ) : (
        <div className="space-y-6">
          {/* Group Selection Card */}
          <Card className="overflow-hidden border-0 shadow-lg bg-white">
            <button
              onClick={() => setIsOverviewExpanded(!isOverviewExpanded)}
              className="w-full px-6 py-5 flex items-center justify-between hover:bg-slate-50 transition-all duration-200"
            >
              <div className="flex items-center gap-4">
                <div className="p-3 bg-gradient-to-br from-indigo-500 to-purple-600 rounded-xl shadow-lg">
                  <Target className="w-6 h-6 text-white" />
                </div>
                <div className="text-left">
                  <h2 className="text-xl font-bold text-slate-900">Grup Seçimi</h2>
                  <p className="text-sm text-slate-600">
                    Karşılaştırmak istediğiniz test gruplarını seçin
                  </p>
                </div>
              </div>
              <div
                className={`p-2 rounded-full bg-indigo-100 transition-transform duration-200 ${
                  isOverviewExpanded ? "rotate-180" : ""
                }`}
              >
                <ChevronDown className="w-5 h-5 text-indigo-600" />
              </div>
            </button>

            {isOverviewExpanded && (
              <div className="px-6 pb-6 pt-2 border-t border-slate-100">
                <div className="space-y-4">
                  <div>
                    <div className="flex items-center justify-between mb-2">
                      <label className="text-sm font-medium text-slate-700">
                        Test Grupları (En az 2 grup seçin)
                      </label>
                      <div className="flex items-center gap-2 text-sm text-slate-500">
                        <span>
                          Toplam {availableGroups.length} grup
                        </span>
                        {availableGroups.length > groupsPerPage && (
                          <span>
                            - Sayfa {groupsPage} / {Math.ceil(availableGroups.length / groupsPerPage)}
                          </span>
                        )}
                      </div>
                    </div>
                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
                      {availableGroups
                        .slice((groupsPage - 1) * groupsPerPage, groupsPage * groupsPerPage)
                        .map((group) => (
                        <div
                          key={group.name}
                          className={`p-3 rounded-xl border-2 transition-all ${
                            selectedGroups.includes(group.name)
                              ? "bg-indigo-50 border-indigo-500"
                              : "bg-white border-slate-200"
                          }`}
                        >
                          <div className="flex items-center justify-between mb-2">
                            <div className="flex items-center gap-2 flex-1">
                              <button
                                type="button"
                                onClick={() => {
                                  if (selectedGroups.includes(group.name)) {
                                    setSelectedGroups(selectedGroups.filter(g => g !== group.name));
                                  } else {
                                    setSelectedGroups([...selectedGroups, group.name]);
                                  }
                                }}
                                className={`flex-1 text-left p-2 rounded-lg border-2 transition-all ${
                                  selectedGroups.includes(group.name)
                                    ? "bg-indigo-100 border-indigo-500 text-indigo-700"
                                    : "bg-white border-slate-200 text-slate-600 hover:border-indigo-300"
                                }`}
                              >
                                <span className="text-sm font-medium truncate">{group.name}</span>
                              </button>
                            </div>
                            {selectedGroups.includes(group.name) && (
                              <CheckCircle2 className="w-4 h-4 text-indigo-600 flex-shrink-0" />
                            )}
                          </div>
                          <div className="flex items-center justify-between">
                            <span className="text-xs text-slate-500">
                              {group.created_at 
                                ? new Date(group.created_at).toLocaleDateString('tr-TR', {
                                    year: 'numeric',
                                    month: 'long',
                                    day: 'numeric'
                                  })
                                : 'Tarih yok'}
                            </span>
                            <span className="text-xs font-medium text-slate-600 bg-slate-100 px-2 py-1 rounded-full">
                              {group.question_count} soru
                            </span>
                            <div className="flex items-center gap-2">
                              <button
                                type="button"
                                onClick={() => {
                                  setGroupToRename(group);
                                  setNewGroupName(group.name);
                                  setIsRenameModalOpen(true);
                                }}
                                className="text-xs text-indigo-600 hover:text-indigo-800 flex items-center gap-1"
                              >
                                <Settings className="w-3 h-3" />
                                <span>Yeniden Adlandır</span>
                              </button>
                              <button
                                type="button"
                                onClick={() => {
                                  setGroupToDelete(group);
                                  setIsDeleteModalOpen(true);
                                }}
                                className="text-xs text-red-600 hover:text-red-800 flex items-center gap-1"
                              >
                                <Trash2 className="w-3 h-3" />
                                <span>Sil</span>
                              </button>
                            </div>
                          </div>
                        </div>
                      ))}
                    </div>
                    {availableGroups.length > groupsPerPage && (
                      <div className="flex items-center justify-center gap-4 mt-4">
                        <Button
                          variant="outline"
                          onClick={() => setGroupsPage(Math.max(1, groupsPage - 1))}
                          disabled={groupsPage === 1}
                          className="border-slate-200 text-slate-600 hover:bg-slate-50"
                        >
                          Önceki
                        </Button>
                        <div className="flex items-center gap-2">
                          {Array.from({ length: Math.ceil(availableGroups.length / groupsPerPage) }, (_, i) => i + 1).map(pageNum => (
                            <button
                              key={pageNum}
                              onClick={() => setGroupsPage(pageNum)}
                              className={`w-8 h-8 rounded-full text-sm font-medium transition-all ${
                                pageNum === groupsPage
                                  ? "bg-indigo-600 text-white"
                                  : "bg-white border-slate-200 text-slate-600 hover:bg-slate-50"
                              }`}
                            >
                              {pageNum}
                            </button>
                          ))}
                        </div>
                        <Button
                          variant="outline"
                          onClick={() => setGroupsPage(Math.min(Math.ceil(availableGroups.length / groupsPerPage), groupsPage + 1))}
                          disabled={groupsPage >= Math.ceil(availableGroups.length / groupsPerPage)}
                          className="border-slate-200 text-slate-600 hover:bg-slate-50"
                        >
                          Sonraki
                        </Button>
                      </div>
                    )}
                  </div>

                  <div className="flex items-center justify-between">
                    <p className="text-sm text-slate-600">
                      {selectedGroups.length >= 2 
                        ? `${selectedGroups.length} grup seçildi` 
                        : "Lütfen en az 2 grup seçin"}
                    </p>
                    <Button
                      onClick={() => runAnalysis()}
                      disabled={isAnalyzing || selectedGroups.length < 2}
                      className="bg-gradient-to-r from-indigo-600 to-purple-600 hover:from-indigo-700 hover:to-purple-700"
                    >
                      {isAnalyzing ? (
                        <>
                          <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                          Analiz Ediliyor...
                        </>
                      ) : (
                        <>
                          <BarChart3 className="w-4 h-4 mr-2" />
                          Analizi Başlat
                        </>
                      )}
                    </Button>
                  </div>
                </div>
              </div>
            )}
          </Card>

          {groupComparisons.length > 0 && (
            <>
              {/* Group Comparison Card */}
              <Card className="overflow-hidden border-0 shadow-lg bg-white">
                <button
                  onClick={() => setIsGroupComparisonExpanded(!isGroupComparisonExpanded)}
                  className="w-full px-6 py-5 flex items-center justify-between hover:bg-slate-50 transition-all duration-200"
                >
                  <div className="flex items-center gap-4">
                    <div className="p-3 bg-gradient-to-br from-teal-500 to-cyan-600 rounded-xl shadow-lg">
                      <TrendingUp className="w-6 h-6 text-white" />
                    </div>
                    <div className="text-left">
                      <h2 className="text-xl font-bold text-slate-900">Grup Karşılaştırması</h2>
                      <p className="text-sm text-slate-600">
                        Seçilen grupların ortalama metrikleri
                      </p>
                    </div>
                  </div>
                  <div
                    className={`p-2 rounded-full bg-teal-100 transition-transform duration-200 ${
                      isGroupComparisonExpanded ? "rotate-180" : ""
                    }`}
                  >
                    <ChevronDown className="w-5 h-5 text-teal-600" />
                  </div>
                </button>

                {isGroupComparisonExpanded && (
                  <div className="px-6 pb-6 pt-2 border-t border-slate-100">
                    <div className="overflow-x-auto">
                      <table className="w-full text-sm">
                        <thead className="bg-slate-50">
                          <tr>
                            <th className="px-4 py-3 text-left font-medium text-slate-700">Grup</th>
                            <th className="px-4 py-3 text-center font-medium text-slate-700">Test Sayısı</th>
                            <th className="px-4 py-3 text-center font-medium text-slate-700">ROUGE-1</th>
                            <th className="px-4 py-3 text-center font-medium text-slate-700">ROUGE-2</th>
                            <th className="px-4 py-3 text-center font-medium text-slate-700">ROUGE-L</th>
                            <th className="px-4 py-3 text-center font-medium text-slate-700">BERTScore F1</th>
                            <th className="px-4 py-3 text-center font-medium text-slate-700">Ort. Yanıt Süresi</th>
                          </tr>
                        </thead>
                        <tbody className="divide-y divide-slate-100">
                          {groupComparisons.map((comp, idx) => (
                            <tr key={idx} className="hover:bg-slate-50">
                              <td className="px-4 py-3 font-medium text-slate-900">
                                <span className="px-2 py-1 bg-indigo-100 text-indigo-700 rounded-full text-xs">
                                  {comp.groupName}
                                </span>
                              </td>
                              <td className="px-4 py-3 text-center">
                                <span className="font-bold text-slate-700">{comp.aggregate.test_count}</span>
                              </td>
                              <td className="px-4 py-3 text-center">
                                {comp.aggregate.avg_rouge1 != null ? (
                                  <span className={`font-bold ${getMetricColor(comp.aggregate.avg_rouge1)}`}>
                                    {(comp.aggregate.avg_rouge1 * 100).toFixed(2)}%
                                  </span>
                                ) : (
                                  <span className="text-slate-400">-</span>
                                )}
                              </td>
                              <td className="px-4 py-3 text-center">
                                {comp.aggregate.avg_rouge2 != null ? (
                                  <span className={`font-bold ${getMetricColor(comp.aggregate.avg_rouge2)}`}>
                                    {(comp.aggregate.avg_rouge2 * 100).toFixed(2)}%
                                  </span>
                                ) : (
                                  <span className="text-slate-400">-</span>
                                )}
                              </td>
                              <td className="px-4 py-3 text-center">
                                {comp.aggregate.avg_rougel != null ? (
                                  <span className={`font-bold ${getMetricColor(comp.aggregate.avg_rougel)}`}>
                                    {(comp.aggregate.avg_rougel * 100).toFixed(2)}%
                                  </span>
                                ) : (
                                  <span className="text-slate-400">-</span>
                                )}
                              </td>
                              <td className="px-4 py-3 text-center">
                                {comp.aggregate.avg_bertscore_f1 != null ? (
                                  <span className={`font-bold ${getMetricColor(comp.aggregate.avg_bertscore_f1)}`}>
                                    {(comp.aggregate.avg_bertscore_f1 * 100).toFixed(2)}%
                                  </span>
                                ) : (
                                  <span className="text-slate-400">-</span>
                                )}
                              </td>
                              <td className="px-4 py-3 text-center">
                                {comp.aggregate.avg_latency_ms != null ? (
                                  <span className="font-bold text-slate-700">
                                    {comp.aggregate.avg_latency_ms.toFixed(0)} ms
                                  </span>
                                ) : (
                                  <span className="text-slate-400">-</span>
                                )}
                              </td>
                            </tr>
                          ))}
                          
                          {/* Overall Average Row */}
                          {(() => {
                            const rouge1Values = groupComparisons.filter(g => g.aggregate.avg_rouge1 != null).map(g => g.aggregate.avg_rouge1!);
                            const rouge2Values = groupComparisons.filter(g => g.aggregate.avg_rouge2 != null).map(g => g.aggregate.avg_rouge2!);
                            const rougelValues = groupComparisons.filter(g => g.aggregate.avg_rougel != null).map(g => g.aggregate.avg_rougel!);
                            const bertF1Values = groupComparisons.filter(g => g.aggregate.avg_bertscore_f1 != null).map(g => g.aggregate.avg_bertscore_f1!);
                            const latencyValues = groupComparisons.filter(g => g.aggregate.avg_latency_ms != null).map(g => g.aggregate.avg_latency_ms!);
                            const totalTests = groupComparisons.reduce((sum, g) => sum + g.aggregate.test_count, 0);
                            
                            const avgRouge1 = rouge1Values.length > 0 ? rouge1Values.reduce((a, b) => a + b, 0) / rouge1Values.length : null;
                            const avgRouge2 = rouge2Values.length > 0 ? rouge2Values.reduce((a, b) => a + b, 0) / rouge2Values.length : null;
                            const avgRougel = rougelValues.length > 0 ? rougelValues.reduce((a, b) => a + b, 0) / rougelValues.length : null;
                            const avgBertF1 = bertF1Values.length > 0 ? bertF1Values.reduce((a, b) => a + b, 0) / bertF1Values.length : null;
                            const avgLatency = latencyValues.length > 0 ? latencyValues.reduce((a, b) => a + b, 0) / latencyValues.length : null;
                            
                            return (
                              <tr className="bg-blue-50 border-t-2 border-blue-200 font-semibold">
                                <td className="px-4 py-3 text-slate-900">
                                  <span className="px-2 py-1 bg-blue-600 text-white rounded-full text-xs">
                                    Genel Ortalama
                                  </span>
                                </td>
                                <td className="px-4 py-3 text-center text-slate-900">{totalTests}</td>
                                <td className="px-4 py-3 text-center">
                                  {avgRouge1 != null ? (
                                    <span className={`font-bold ${getMetricColor(avgRouge1)}`}>
                                      {(avgRouge1 * 100).toFixed(2)}%
                                    </span>
                                  ) : (
                                    <span className="text-slate-400">-</span>
                                  )}
                                </td>
                                <td className="px-4 py-3 text-center">
                                  {avgRouge2 != null ? (
                                    <span className={`font-bold ${getMetricColor(avgRouge2)}`}>
                                      {(avgRouge2 * 100).toFixed(2)}%
                                    </span>
                                  ) : (
                                    <span className="text-slate-400">-</span>
                                  )}
                                </td>
                                <td className="px-4 py-3 text-center">
                                  {avgRougel != null ? (
                                    <span className={`font-bold ${getMetricColor(avgRougel)}`}>
                                      {(avgRougel * 100).toFixed(2)}%
                                    </span>
                                  ) : (
                                    <span className="text-slate-400">-</span>
                                  )}
                                </td>
                                <td className="px-4 py-3 text-center">
                                  {avgBertF1 != null ? (
                                    <span className={`font-bold ${getMetricColor(avgBertF1)}`}>
                                      {(avgBertF1 * 100).toFixed(2)}%
                                    </span>
                                  ) : (
                                    <span className="text-slate-400">-</span>
                                  )}
                                </td>
                                <td className="px-4 py-3 text-center">
                                  {avgLatency != null ? (
                                    <span className="font-bold text-slate-900">
                                      {avgLatency.toFixed(0)} ms
                                    </span>
                                  ) : (
                                    <span className="text-slate-400">-</span>
                                  )}
                                </td>
                              </tr>
                            );
                          })()}
                          
                          {/* Standard Deviation Row */}
                          {(() => {
                            const rouge1Values = groupComparisons.filter(g => g.aggregate.avg_rouge1 != null).map(g => g.aggregate.avg_rouge1!);
                            const rouge2Values = groupComparisons.filter(g => g.aggregate.avg_rouge2 != null).map(g => g.aggregate.avg_rouge2!);
                            const rougelValues = groupComparisons.filter(g => g.aggregate.avg_rougel != null).map(g => g.aggregate.avg_rougel!);
                            const bertF1Values = groupComparisons.filter(g => g.aggregate.avg_bertscore_f1 != null).map(g => g.aggregate.avg_bertscore_f1!);
                            const latencyValues = groupComparisons.filter(g => g.aggregate.avg_latency_ms != null).map(g => g.aggregate.avg_latency_ms!);
                            
                            const calcStdDev = (values: number[]) => {
                              if (values.length < 2) return null;
                              const mean = values.reduce((a, b) => a + b, 0) / values.length;
                              const variance = values.reduce((sum, v) => sum + Math.pow(v - mean, 2), 0) / values.length;
                              return Math.sqrt(variance);
                            };
                            
                            const stdRouge1 = calcStdDev(rouge1Values);
                            const stdRouge2 = calcStdDev(rouge2Values);
                            const stdRougel = calcStdDev(rougelValues);
                            const stdBertF1 = calcStdDev(bertF1Values);
                            const stdLatency = calcStdDev(latencyValues);
                            
                            return (
                              <tr className="bg-purple-50 border-t border-purple-200 font-semibold">
                                <td className="px-4 py-3 text-slate-900">
                                  <span className="px-2 py-1 bg-purple-600 text-white rounded-full text-xs">
                                    Standart Sapma (σ)
                                  </span>
                                </td>
                                <td className="px-4 py-3 text-center text-slate-400">-</td>
                                <td className="px-4 py-3 text-center">
                                  {stdRouge1 != null ? (
                                    <span className={`font-bold ${getVarianceColor(stdRouge1)} px-2 py-1 rounded border`}>
                                      ±{(stdRouge1 * 100).toFixed(2)}%
                                    </span>
                                  ) : (
                                    <span className="text-slate-400">-</span>
                                  )}
                                </td>
                                <td className="px-4 py-3 text-center">
                                  {stdRouge2 != null ? (
                                    <span className={`font-bold ${getVarianceColor(stdRouge2)} px-2 py-1 rounded border`}>
                                      ±{(stdRouge2 * 100).toFixed(2)}%
                                    </span>
                                  ) : (
                                    <span className="text-slate-400">-</span>
                                  )}
                                </td>
                                <td className="px-4 py-3 text-center">
                                  {stdRougel != null ? (
                                    <span className={`font-bold ${getVarianceColor(stdRougel)} px-2 py-1 rounded border`}>
                                      ±{(stdRougel * 100).toFixed(2)}%
                                    </span>
                                  ) : (
                                    <span className="text-slate-400">-</span>
                                  )}
                                </td>
                                <td className="px-4 py-3 text-center">
                                  {stdBertF1 != null ? (
                                    <span className={`font-bold ${getVarianceColor(stdBertF1)} px-2 py-1 rounded border`}>
                                      ±{(stdBertF1 * 100).toFixed(2)}%
                                    </span>
                                  ) : (
                                    <span className="text-slate-400">-</span>
                                  )}
                                </td>
                                <td className="px-4 py-3 text-center">
                                  {stdLatency != null ? (
                                    <span className="font-bold text-purple-700 bg-purple-100 px-2 py-1 rounded border border-purple-200">
                                      ±{stdLatency.toFixed(0)} ms
                                    </span>
                                  ) : (
                                    <span className="text-slate-400">-</span>
                                  )}
                                </td>
                              </tr>
                            );
                          })()}
                        </tbody>
                      </table>
                    </div>

                    {/* Gruplar arası farklar */}
                    {groupComparisons.length === 2 && (
                      <div className="mt-6 p-4 bg-gradient-to-r from-amber-50 to-orange-50 rounded-xl border border-amber-200">
                        <h3 className="text-sm font-semibold text-amber-900 mb-3 flex items-center gap-2">
                          <ArrowRight className="w-4 h-4" />
                          Gruplar Arası Farklar
                        </h3>
                        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                          {groupComparisons[0].aggregate.avg_rouge1 != null && groupComparisons[1].aggregate.avg_rouge1 != null && (
                            <div className="p-3 bg-white rounded-lg border border-amber-100">
                              <p className="text-xs text-amber-600 mb-1">ROUGE-1 Farkı</p>
                              <p className={`text-lg font-bold ${
                                groupComparisons[0].aggregate.avg_rouge1! > groupComparisons[1].aggregate.avg_rouge1!
                                  ? "text-emerald-600"
                                  : "text-red-600"
                              }`}>
                                {Math.abs(groupComparisons[0].aggregate.avg_rouge1! - groupComparisons[1].aggregate.avg_rouge1!) * 100 > 0 ? "+" : ""}
                                {((groupComparisons[0].aggregate.avg_rouge1! - groupComparisons[1].aggregate.avg_rouge1!) * 100).toFixed(2)}%
                              </p>
                            </div>
                          )}
                          {groupComparisons[0].aggregate.avg_rougel != null && groupComparisons[1].aggregate.avg_rougel != null && (
                            <div className="p-3 bg-white rounded-lg border border-amber-100">
                              <p className="text-xs text-amber-600 mb-1">ROUGE-L Farkı</p>
                              <p className={`text-lg font-bold ${
                                groupComparisons[0].aggregate.avg_rougel! > groupComparisons[1].aggregate.avg_rougel!
                                  ? "text-emerald-600"
                                  : "text-red-600"
                              }`}>
                                {Math.abs(groupComparisons[0].aggregate.avg_rougel! - groupComparisons[1].aggregate.avg_rougel!) * 100 > 0 ? "+" : ""}
                                {((groupComparisons[0].aggregate.avg_rougel! - groupComparisons[1].aggregate.avg_rougel!) * 100).toFixed(2)}%
                              </p>
                            </div>
                          )}
                          {groupComparisons[0].aggregate.avg_bertscore_f1 != null && groupComparisons[1].aggregate.avg_bertscore_f1 != null && (
                            <div className="p-3 bg-white rounded-lg border border-amber-100">
                              <p className="text-xs text-amber-600 mb-1">BERTScore F1 Farkı</p>
                              <p className={`text-lg font-bold ${
                                groupComparisons[0].aggregate.avg_bertscore_f1! > groupComparisons[1].aggregate.avg_bertscore_f1!
                                  ? "text-emerald-600"
                                  : "text-red-600"
                              }`}>
                                {Math.abs(groupComparisons[0].aggregate.avg_bertscore_f1! - groupComparisons[1].aggregate.avg_bertscore_f1!) * 100 > 0 ? "+" : ""}
                                {((groupComparisons[0].aggregate.avg_bertscore_f1! - groupComparisons[1].aggregate.avg_bertscore_f1!) * 100).toFixed(2)}%
                              </p>
                            </div>
                          )}
                        </div>
                      </div>
                    )}
                  </div>
                )}
              </Card>

              {/* Question Comparison Card */}
              <Card className="overflow-hidden border-0 shadow-lg bg-white">
                <button
                  onClick={() => setIsQuestionComparisonExpanded(!isQuestionComparisonExpanded)}
                  className="w-full px-6 py-5 flex items-center justify-between hover:bg-slate-50 transition-all duration-200"
                >
                  <div className="flex items-center gap-4">
                    <div className="p-3 bg-gradient-to-br from-purple-500 to-pink-600 rounded-xl shadow-lg">
                      <FileText className="w-6 h-6 text-white" />
                    </div>
                    <div className="text-left">
                      <h2 className="text-xl font-bold text-slate-900">Soru Bazlı Varyasyon Analizi</h2>
                      <p className="text-sm text-slate-600">
                        Her sorunun gruplar arasındaki varyasyonu
                      </p>
                    </div>
                  </div>
                  <div
                    className={`p-2 rounded-full bg-purple-100 transition-transform duration-200 ${
                      isQuestionComparisonExpanded ? "rotate-180" : ""
                    }`}
                  >
                    <ChevronDown className="w-5 h-5 text-purple-600" />
                  </div>
                </button>

                {isQuestionComparisonExpanded && (
                  <div className="px-6 pb-6 pt-2 border-t border-slate-100">
                    <div className="max-h-[500px] overflow-y-auto">
                      <table className="w-full text-sm">
                        <thead className="bg-slate-50 sticky top-0">
                          <tr>
                            <th className="px-4 py-3 text-left font-medium text-slate-700 w-64">Soru</th>
                            <th className="px-4 py-3 text-center font-medium text-slate-700">Grup Sayısı</th>
                            <th className="px-4 py-3 text-center font-medium text-slate-700">ROUGE-1 Std Sapma</th>
                            <th className="px-4 py-3 text-center font-medium text-slate-700">ROUGE-2 Std Sapma</th>
                            <th className="px-4 py-3 text-center font-medium text-slate-700">ROUGE-L Std Sapma</th>
                            <th className="px-4 py-3 text-center font-medium text-slate-700">BERTScore F1 Std Sapma</th>
                            <th className="px-4 py-3 text-center font-medium text-slate-700">Varyasyon Seviyesi</th>
                          </tr>
                        </thead>
                        <tbody className="divide-y divide-slate-100">
                          {questionComparisons.map((comp, idx) => (
                            <tr key={idx} className="hover:bg-slate-50">
                              <td className="px-4 py-3 text-slate-900">
                                <div className="max-h-16 overflow-y-auto whitespace-pre-wrap">
                                  {comp.question}
                                </div>
                              </td>
                              <td className="px-4 py-3 text-center">
                                <span className="px-2 py-1 bg-purple-100 text-purple-700 rounded-full text-xs font-bold">
                                  {comp.groupResults.length}
                                </span>
                              </td>
                              <td className="px-4 py-3 text-center">
                                <span className={`font-bold ${getMetricColor(comp.variance.rouge1_std_dev)}`}>
                                  {(comp.variance.rouge1_std_dev * 100).toFixed(2)}%
                                </span>
                              </td>
                              <td className="px-4 py-3 text-center">
                                <span className={`font-bold ${getMetricColor(comp.variance.rouge2_std_dev)}`}>
                                  {(comp.variance.rouge2_std_dev * 100).toFixed(2)}%
                                </span>
                              </td>
                              <td className="px-4 py-3 text-center">
                                <span className={`font-bold ${getMetricColor(comp.variance.rougel_std_dev)}`}>
                                  {(comp.variance.rougel_std_dev * 100).toFixed(2)}%
                                </span>
                              </td>
                              <td className="px-4 py-3 text-center">
                                <span className={`font-bold ${getMetricColor(comp.variance.bertscore_f1_std_dev)}`}>
                                  {(comp.variance.bertscore_f1_std_dev * 100).toFixed(2)}%
                                </span>
                              </td>
                              <td className="px-4 py-3 text-center">
                                <span className={`px-2 py-1 rounded-full text-xs font-medium border ${getVarianceColor(comp.variance.bertscore_f1_std_dev)}`}>
                                  {getVarianceLabel(comp.variance.bertscore_f1_std_dev)}
                                </span>
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                )}
              </Card>

              {/* Reliability Analysis Card */}
              <Card className="overflow-hidden border-0 shadow-lg bg-white">
                <button
                  onClick={() => setIsReliabilityExpanded(!isReliabilityExpanded)}
                  className="w-full px-6 py-5 flex items-center justify-between hover:bg-slate-50 transition-all duration-200"
                >
                  <div className="flex items-center gap-4">
                    <div className="p-3 bg-gradient-to-br from-slate-600 to-slate-800 rounded-xl shadow-lg">
                      <Info className="w-6 h-6 text-white" />
                    </div>
                    <div className="text-left">
                      <h2 className="text-xl font-bold text-slate-900">Güvenilirlik Analizi</h2>
                      <p className="text-sm text-slate-600">
                        Testin güvenilirlik değerlendirmesi
                      </p>
                    </div>
                  </div>
                  <div
                    className={`p-2 rounded-full bg-slate-100 transition-transform duration-200 ${
                      isReliabilityExpanded ? "rotate-180" : ""
                    }`}
                  >
                    <ChevronDown className="w-5 h-5 text-slate-600" />
                  </div>
                </button>

                {isReliabilityExpanded && (
                  <div className="px-6 pb-6 pt-2 border-t border-slate-100">
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      {/* Genel Varyasyon */}
                      <div className="p-4 bg-gradient-to-br from-emerald-50 to-teal-50 rounded-xl border border-emerald-200">
                        <h3 className="text-sm font-semibold text-emerald-900 mb-3 flex items-center gap-2">
                          <CheckCircle2 className="w-4 h-4" />
                          Genel Varyasyon
                        </h3>
                        <div className="space-y-3">
                          <div>
                            <p className="text-xs text-emerald-600 mb-1">Ortalama ROUGE-1 Std Sapma</p>
                            <p className="text-2xl font-bold text-emerald-700">
                              {questionComparisons.length > 0 
                                ? (questionComparisons.reduce((sum, q) => sum + q.variance.rouge1_std_dev, 0) / questionComparisons.length * 100).toFixed(2) + "%"
                                : "-"}
                            </p>
                          </div>
                          <div>
                            <p className="text-xs text-emerald-600 mb-1">Ortalama BERTScore F1 Std Sapma</p>
                            <p className="text-2xl font-bold text-emerald-700">
                              {questionComparisons.length > 0 
                                ? (questionComparisons.reduce((sum, q) => sum + q.variance.bertscore_f1_std_dev, 0) / questionComparisons.length * 100).toFixed(2) + "%"
                                : "-"}
                            </p>
                          </div>
                        </div>
                      </div>

                      {/* Güvenilirlik Puanı */}
                      <div className="p-4 bg-gradient-to-br from-blue-50 to-indigo-50 rounded-xl border border-blue-200">
                        <h3 className="text-sm font-semibold text-blue-900 mb-3 flex items-center gap-2">
                          <Target className="w-4 h-4" />
                          Güvenilirlik Puanı
                        </h3>
                        <div className="space-y-3">
                          <div>
                            <p className="text-xs text-blue-600 mb-1">Düşük Varyasyonlu Sorular</p>
                            <p className="text-2xl font-bold text-blue-700">
                              {questionComparisons.filter(q => q.variance.bertscore_f1_std_dev < 0.1).length} / {questionComparisons.length}
                            </p>
                            <p className="text-xs text-slate-500 mt-1">
                              % {(questionComparisons.length > 0 
                                ? (questionComparisons.filter(q => q.variance.bertscore_f1_std_dev < 0.1).length / questionComparisons.length * 100).toFixed(0)
                                : 0)}
                            </p>
                          </div>
                          <div>
                            <p className="text-xs text-blue-600 mb-1">Yüksek Varyasyonlu Sorular</p>
                            <p className="text-2xl font-bold text-red-600">
                              {questionComparisons.filter(q => q.variance.bertscore_f1_std_dev >= 0.15).length} / {questionComparisons.length}
                            </p>
                            <p className="text-xs text-slate-500 mt-1">
                              % {(questionComparisons.length > 0 
                                ? (questionComparisons.filter(q => q.variance.bertscore_f1_std_dev >= 0.15).length / questionComparisons.length * 100).toFixed(0)
                                : 0)}
                            </p>
                          </div>
                        </div>
                      </div>

                      {/* Uyarılar */}
                      <div className="p-4 bg-gradient-to-br from-amber-50 to-orange-50 rounded-xl border border-amber-200 md:col-span-2">
                        <h3 className="text-sm font-semibold text-amber-900 mb-3 flex items-center gap-2">
                          <AlertTriangle className="w-4 h-4" />
                          Dikkat Edilmesi Gereken Sorular
                        </h3>
                        <div className="space-y-2 max-h-48 overflow-y-auto">
                          {questionComparisons
                            .filter(q => q.variance.bertscore_f1_std_dev >= 0.15)
                            .slice(0, 5)
                            .map((q, idx) => (
                              <div key={idx} className="p-3 bg-white rounded-lg border border-amber-100">
                                <p className="text-xs text-slate-900 font-medium truncate">{q.question}</p>
                                <p className="text-xs text-red-600 mt-1">
                                  Std Sapma: {(q.variance.bertscore_f1_std_dev * 100).toFixed(2)}% - Yüksek varyasyon
                                </p>
                              </div>
                            ))}
                          {questionComparisons.filter(q => q.variance.bertscore_f1_std_dev >= 0.15).length === 0 && (
                            <p className="text-sm text-slate-600 text-center py-4">
                              Yüksek varyasyonlu soru bulunamadı
                            </p>
                          )}
                        </div>
                      </div>
                    </div>
                  </div>
                )}
              </Card>

              {/* Export Button */}
              <div className="flex justify-end gap-3">
                <Button
                  onClick={openPDFModal}
                  className="bg-gradient-to-r from-indigo-600 to-purple-600 hover:from-indigo-700 hover:to-purple-700"
                >
                  <Download className="w-4 h-4 mr-2" />
                  PDF Raporu Oluştur
                </Button>
                <Button
                  onClick={exportAnalysisToExcel}
                  variant="outline"
                  className="border-indigo-200 text-indigo-600 hover:bg-indigo-50"
                >
                  <Download className="w-4 h-4 mr-2" />
                  Excel İndir
                </Button>
                <Button
                  variant="outline"
                  onClick={() => {
                    const course = courses.find(c => c.id === selectedCourseId);
                    toast.info(
                      <div className="text-left">
                        <p className="font-semibold mb-2">Analiz Özeti</p>
                        <p className="text-sm">Ders: {course?.name || "-"}</p>
                        <p className="text-sm">Gruplar: {selectedGroups.join(", ")}</p>
                        <p className="text-sm">Embedding Model: {courseSettings?.default_embedding_model || "-"}</p>
                        <p className="text-sm">LLM Model: {courseSettings?.llm_provider && courseSettings?.llm_model ? `${courseSettings.llm_provider}/${courseSettings.llm_model}` : "-"}</p>
                        <p className="text-sm mt-2">Analiz Tarihi: {new Date().toLocaleString('tr-TR')}</p>
                      </div>
                    );
                  }}
                  className="border-indigo-200 text-indigo-600 hover:bg-indigo-50"
                >
                  <Settings className="w-4 h-4 mr-2" />
                  Özet
                </Button>
              </div>
            </>
          )}
        </div>
      )}

      {/* PDF Export Modal */}
      <Dialog open={isPDFModalOpen} onOpenChange={setIsPDFModalOpen}>
        <DialogContent className="max-w-2xl max-h-[90vh] overflow-y-auto">
          <DialogHeader>
            <DialogTitle className="text-2xl font-bold bg-gradient-to-r from-indigo-600 to-purple-600 bg-clip-text text-transparent">
              🔬 Deney Bilgileri
            </DialogTitle>
            <DialogDescription>
              PDF raporu oluşturmak için deney bilgilerini girin
            </DialogDescription>
          </DialogHeader>
          
          <div className="space-y-4 py-4">
            <div className="grid grid-cols-2 gap-4">
              <div className="space-y-2">
                <Label htmlFor="title" className="text-sm font-medium">
                  Başlık <span className="text-red-500">*</span>
                </Label>
                <Input
                  id="title"
                  placeholder="Örn: Semantic Similarity Analiz Raporu"
                  value={experimentInfo.title}
                  onChange={(e) => setExperimentInfo({...experimentInfo, title: e.target.value})}
                  className="w-full"
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="experimentName" className="text-sm font-medium">
                  Deney Adı <span className="text-red-500">*</span>
                </Label>
                <Input
                  id="experimentName"
                  placeholder="Örn: RAG Model Performans Testi"
                  value={experimentInfo.experimentName}
                  onChange={(e) => setExperimentInfo({...experimentInfo, experimentName: e.target.value})}
                  className="w-full"
                />
              </div>
            </div>

            <div className="grid grid-cols-2 gap-4">
              <div className="space-y-2">
                <Label htmlFor="date" className="text-sm font-medium">
                  Tarih <span className="text-red-500">*</span>
                </Label>
                <Input
                  id="date"
                  type="date"
                  value={experimentInfo.date}
                  onChange={(e) => setExperimentInfo({...experimentInfo, date: e.target.value})}
                  className="w-full"
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="experimenter" className="text-sm font-medium">
                  Deneyi Yapan <span className="text-red-500">*</span>
                </Label>
                <Input
                  id="experimenter"
                  placeholder="Örn: Dr. Ahmet Yılmaz"
                  value={experimentInfo.experimenter}
                  onChange={(e) => setExperimentInfo({...experimentInfo, experimenter: e.target.value})}
                  className="w-full"
                />
              </div>
            </div>

            <div className="space-y-2">
              <Label htmlFor="summary" className="text-sm font-medium">
                Özet <span className="text-red-500">*</span>
              </Label>
              <Textarea
                id="summary"
                placeholder="Deneyin kısa bir özetini girin..."
                value={experimentInfo.summary}
                onChange={(e) => setExperimentInfo({...experimentInfo, summary: e.target.value})}
                className="w-full min-h-[100px]"
                rows={4}
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="objective" className="text-sm font-medium">
                Amaç
              </Label>
              <Textarea
                id="objective"
                placeholder="Deneyin amacını açıklayın..."
                value={experimentInfo.objective}
                onChange={(e) => setExperimentInfo({...experimentInfo, objective: e.target.value})}
                className="w-full min-h-[80px]"
                rows={3}
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="methodology" className="text-sm font-medium">
                Metodoloji
              </Label>
              <Textarea
                id="methodology"
                placeholder="Kullanılan metodolojiyi açıklayın..."
                value={experimentInfo.methodology}
                onChange={(e) => setExperimentInfo({...experimentInfo, methodology: e.target.value})}
                className="w-full min-h-[80px]"
                rows={3}
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="evaluation" className="text-sm font-medium">
                Değerlendirme
              </Label>
              <Textarea
                id="evaluation"
                placeholder="Deney sonuçlarınızı değerlendirin ve yorumlarınızı yazın..."
                value={experimentInfo.evaluation}
                onChange={(e) => setExperimentInfo({...experimentInfo, evaluation: e.target.value})}
                className="w-full min-h-[120px]"
                rows={5}
              />
              <p className="text-xs text-slate-500 mt-1">
                Bu alan raporun suni olmaması için kendi değerlendirmenizi girmenizi sağlar.
              </p>
            </div>
          </div>

          <DialogFooter>
            <Button
              variant="outline"
              onClick={() => setIsPDFModalOpen(false)}
              className="border-slate-200 text-slate-600 hover:bg-slate-50"
            >
              İptal
            </Button>
            <Button
              onClick={exportAnalysisToPDF}
              className="bg-gradient-to-r from-indigo-600 to-purple-600 hover:from-indigo-700 hover:to-purple-700"
            >
              <Download className="w-4 h-4 mr-2" />
              PDF Raporu Oluştur
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Rename Group Modal */}
      <Dialog open={isRenameModalOpen} onOpenChange={setIsRenameModalOpen}>
        <DialogContent className="max-w-md">
          <DialogHeader>
            <DialogTitle className="text-xl font-bold bg-gradient-to-r from-indigo-600 to-purple-600 bg-clip-text text-transparent">
              Grup Adını Değiştir
            </DialogTitle>
            <DialogDescription>
              Grup adını değiştirmek için yeni ad girin
            </DialogDescription>
          </DialogHeader>
          
          <div className="space-y-4 py-4">
            <div className="space-y-2">
              <Label htmlFor="newGroupName" className="text-sm font-medium">
                Mevcut Grup Adı
              </Label>
              <Input
                id="oldGroupName"
                value={groupToRename?.name || ""}
                disabled
                className="w-full bg-slate-100"
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="newGroupName" className="text-sm font-medium">
                Yeni Grup Adı <span className="text-red-500">*</span>
              </Label>
              <Input
                id="newGroupName"
                placeholder="Yeni grup adı girin..."
                value={newGroupName}
                onChange={(e) => setNewGroupName(e.target.value)}
                className="w-full"
                autoFocus
              />
            </div>
          </div>

          <DialogFooter>
            <Button
              variant="outline"
              onClick={() => setIsRenameModalOpen(false)}
              className="border-slate-200 text-slate-600 hover:bg-slate-50"
              disabled={isRenaming}
            >
              İptal
            </Button>
            <Button
              onClick={handleRenameGroup}
              disabled={!newGroupName.trim() || isRenaming}
              className="bg-gradient-to-r from-indigo-600 to-purple-600 hover:from-indigo-700 hover:to-purple-700"
            >
              {isRenaming ? (
                <>
                  <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                  Değiştiriliyor...
                </>
              ) : (
                <>
                  <Settings className="w-4 h-4 mr-2" />
                  Grubu Değiştir
                </>
              )}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Delete Group Confirmation Modal */}
      <Dialog open={isDeleteModalOpen} onOpenChange={setIsDeleteModalOpen}>
        <DialogContent className="max-w-md">
          <DialogHeader>
            <DialogTitle className="text-xl font-bold text-red-600">
              <div className="flex items-center gap-2">
                <Trash2 className="w-5 h-5" />
                Grubu Sil
              </div>
            </DialogTitle>
            <DialogDescription>
              Bu işlem geri alınamaz
            </DialogDescription>
          </DialogHeader>
          
          <div className="space-y-4 py-4">
            <div className="p-4 bg-red-50 border border-red-200 rounded-lg">
              <p className="text-sm font-medium text-red-900 mb-2">
                &quot;{groupToDelete?.name}&quot; grubunu silmek istediğinize emin misiniz?
              </p>
              <p className="text-xs text-red-700">
                Bu işlem, grup altındaki tüm testleri kalıcı olarak silecektir.
              </p>
            </div>
            
            <div className="flex items-center gap-2 text-sm text-slate-600">
              <AlertTriangle className="w-4 h-4 text-amber-600" />
              <span>Grup silinirse altındaki tüm testler de silinecektir</span>
            </div>
          </div>

          <DialogFooter>
            <Button
              variant="outline"
              onClick={() => setIsDeleteModalOpen(false)}
              className="border-slate-200 text-slate-600 hover:bg-slate-50"
              disabled={isDeleting}
            >
              İptal
            </Button>
            <Button
              onClick={confirmDeleteGroup}
              disabled={isDeleting}
              className="bg-red-600 hover:bg-red-700 text-white"
            >
              {isDeleting ? (
                <>
                  <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                  Siliniyor...
                </>
              ) : (
                <>
                  <Trash2 className="w-4 h-4 mr-2" />
                  Grubu Sil
                </>
              )}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}
