"use client";

import { useEffect, useState, useCallback } from "react";
import { useAuth } from "@/lib/auth-context";
import { 
  api, 
  Course, 
  SemanticSimilarityQuickTestResponse,
  SemanticSimilarityBatchTestResponse,
  SemanticSimilarityResult
} from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
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
  DialogTrigger 
} from "@/components/ui/dialog";
import { toast } from "sonner";
import { 
  Target, 
  Plus, 
  Loader2, 
  X, 
  Save, 
  ChevronDown, 
  History, 
  Eye, 
  Trash2, 
  BookOpen, 
  Sparkles,
  FileJson,
  Download,
  Settings,
  FileText
} from "lucide-react";
import { generateSemanticSimilarityPDF } from "./exportToPDF";

export default function SemanticSimilarityPage() {
  const { user } = useAuth();
  const [courses, setCourses] = useState<Course[]>([]);
  const [selectedCourseId, setSelectedCourseId] = useState<number | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  // Settings State
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);
  const [selectedEmbeddingModel, setSelectedEmbeddingModel] = useState<string>("");
  const [selectedLlmProvider, setSelectedLlmProvider] = useState<string>("");
  const [selectedLlmModel, setSelectedLlmModel] = useState<string>("");
  const [isSavingSettings, setIsSavingSettings] = useState(false);
  const [llmProviders, setLlmProviders] = useState<Record<string, string[]>>({});
  const [availableLlmModels, setAvailableLlmModels] = useState<string[]>([]);

  // Quick Test State
  const [isQuickTestExpanded, setIsQuickTestExpanded] = useState(false);
  const [quickTestQuestion, setQuickTestQuestion] = useState("");
  const [quickTestGroundTruth, setQuickTestGroundTruth] = useState("");
  const [quickTestAlternatives, setQuickTestAlternatives] = useState<string[]>([]);
  const [quickTestGeneratedAnswer, setQuickTestGeneratedAnswer] = useState("");
  const [isQuickTesting, setIsQuickTesting] = useState(false);
  const [quickTestResult, setQuickTestResult] = useState<SemanticSimilarityQuickTestResponse | null>(null);

  // Batch Test State
  const [isBatchTestExpanded, setIsBatchTestExpanded] = useState(false);
  const [batchTestJson, setBatchTestJson] = useState("");
  const [isBatchTesting, setIsBatchTesting] = useState(false);
  const [batchTestResult, setBatchTestResult] = useState<SemanticSimilarityBatchTestResponse | null>(null);

  // Saved Results State
  const [isSavedResultsExpanded, setIsSavedResultsExpanded] = useState(false);
  const [savedResults, setSavedResults] = useState<SemanticSimilarityResult[]>([]);
  const [savedResultsGroups, setSavedResultsGroups] = useState<string[]>([]);
  const [savedResultsTotal, setSavedResultsTotal] = useState(0);
  const [savedResultsAggregate, setSavedResultsAggregate] = useState<any>(null);
  const [selectedGroup, setSelectedGroup] = useState<string>("");
  const [resultsPage, setResultsPage] = useState(0);
  const [isLoadingMore, setIsLoadingMore] = useState(false);
  const RESULTS_PER_PAGE = 10;

  // Save Dialog State
  const [isSaveDialogOpen, setIsSaveDialogOpen] = useState(false);
  const [saveGroupName, setSaveGroupName] = useState("");
  const [isSaving, setIsSaving] = useState(false);
  const [saveDialogMode, setSaveDialogMode] = useState<"quick" | "batch">("quick");

  // View Result Dialog State
  const [viewingResult, setViewingResult] = useState<SemanticSimilarityResult | null>(null);

  // PDF Export State
  const [isPdfGenerating, setIsPdfGenerating] = useState(false);

  useEffect(() => {
    loadCourses();
    loadLlmProviders();
  }, []);

  useEffect(() => {
    if (selectedCourseId) {
      loadSavedResults(true);
      loadCourseSettings();
    }
  }, [selectedCourseId]);

  useEffect(() => {
    if (selectedCourseId) {
      setResultsPage(0);
      loadSavedResults(true);
    }
  }, [selectedGroup]);

  useEffect(() => {
    // Load models when provider changes
    if (selectedLlmProvider && llmProviders[selectedLlmProvider]) {
      setAvailableLlmModels(llmProviders[selectedLlmProvider]);
    } else {
      setAvailableLlmModels([]);
    }
  }, [selectedLlmProvider, llmProviders]);

  const loadLlmProviders = async () => {
    try {
      const providers = await api.getLLMProviders();
      setLlmProviders(providers);
    } catch {
      console.log("LLM providers not available");
    }
  };

  const loadCourses = async () => {
    try {
      const data = await api.getCourses();
      setCourses(data);
      if (data.length > 0) {
        setSelectedCourseId(data[0].id);
      }
    } catch {
      toast.error("Dersler yüklenirken hata oluştu");
    } finally {
      setIsLoading(false);
    }
  };

  const loadCourseSettings = async () => {
    if (!selectedCourseId) return;
    try {
      const settings = await api.getCourseSettings(selectedCourseId);
      setSelectedEmbeddingModel(settings.default_embedding_model || "");
      // LLM ayarlarını da yükle (varsayılan olarak ders ayarlarını kullan)
      // Kullanıcı isterse ayarlardan değiştirebilir
      if (!selectedLlmProvider) {
        setSelectedLlmProvider(settings.llm_provider || "");
        setSelectedLlmModel(settings.llm_model || "");
      }
    } catch {
      console.log("Course settings not available");
    }
  };

  const handleSaveSettings = async () => {
    if (!selectedCourseId) return;
    setIsSavingSettings(true);
    try {
      await api.updateCourseSettings(selectedCourseId, {
        default_embedding_model: selectedEmbeddingModel
      });
      toast.success("Ayarlar güncellendi");
      setIsSettingsOpen(false);
    } catch {
      toast.error("Ayarlar kaydedilirken hata oluştu");
    } finally {
      setIsSavingSettings(false);
    }
  };

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
      const data = await api.getSemanticSimilarityResults(
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
      setSavedResultsAggregate(data.aggregate);
    } catch {
      /* ignore */
    }
  }, [selectedCourseId, selectedGroup, resultsPage]);

  const loadMoreResults = async () => {
    setIsLoadingMore(true);
    await loadSavedResults(false);
    setIsLoadingMore(false);
  };

  const handleQuickTest = async () => {
    if (!selectedCourseId || !quickTestQuestion || !quickTestGroundTruth) {
      toast.error("Lütfen ders, soru ve doğru cevap alanlarını doldurun");
      return;
    }

    setIsQuickTesting(true);
    setQuickTestResult(null);

    try {
      const result = await api.semanticSimilarityQuickTest({
        course_id: selectedCourseId,
        question: quickTestQuestion,
        ground_truth: quickTestGroundTruth,
        alternative_ground_truths: quickTestAlternatives.filter(a => a.trim() !== ""),
        generated_answer: quickTestGeneratedAnswer || undefined,
        llm_provider: selectedLlmProvider || undefined,
        llm_model: selectedLlmModel || undefined
      });
      setQuickTestResult(result);
      toast.success("Test tamamlandı");
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "Test başarısız");
    } finally {
      setIsQuickTesting(false);
    }
  };

  const handleBatchTest = async () => {
    if (!selectedCourseId || !batchTestJson) {
      toast.error("Lütfen ders ve JSON verisi girin");
      return;
    }

    setIsBatchTesting(true);
    setBatchTestResult(null);

    try {
      let parsedData = JSON.parse(batchTestJson);
      let testCases = parsedData;
      
      // Eğer RAGAS test set formatındaysa (questions array içinde), onu çıkar
      if (parsedData.questions && Array.isArray(parsedData.questions)) {
        testCases = parsedData.questions;
        // RAGAS'tan gelen name'i otomatik group name olarak ayarla
        if (parsedData.name && !saveGroupName) {
          setSaveGroupName(parsedData.name);
        }
      }
      
      console.log("Starting streaming batch test with", testCases.length, "test cases");
      
      // Get token from localStorage with correct key
      const token = localStorage.getItem('akilli_rehber_token');
      if (!token) {
        throw new Error('Oturum süresi dolmuş, lütfen tekrar giriş yapın');
      }
      
      // Streaming endpoint kullan
      const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'}/api/semantic-similarity/batch-test-stream`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`
        },
        body: JSON.stringify({
          course_id: selectedCourseId,
          test_cases: testCases,
          llm_provider: selectedLlmProvider || undefined,
          llm_model: selectedLlmModel || undefined
        })
      });

      if (!response.ok) {
        throw new Error('Batch test başarısız');
      }

      const reader = response.body?.getReader();
      const decoder = new TextDecoder();
      const results: any[] = [];
      
      while (reader) {
        const { done, value } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value);
        const lines = chunk.split('\n');

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const data = JSON.parse(line.slice(6));
            
            if (data.event === 'progress') {
              results.push(data.result);
              
              // Calculate aggregates
              const validResults = results.filter(r => r.similarity_score != null);
              const rouge1Results = results.filter(r => r.rouge1 != null);
              const rouge2Results = results.filter(r => r.rouge2 != null);
              const rougelResults = results.filter(r => r.rougel != null);
              const bertPrecisionResults = results.filter(r => r.bertscore_precision != null);
              const bertRecallResults = results.filter(r => r.bertscore_recall != null);
              const bertF1Results = results.filter(r => r.bertscore_f1 != null);
              
              // Geçici sonuçları göster
              setBatchTestResult({
                results: results,
                aggregate: {
                  avg_similarity: validResults.reduce((sum, r) => sum + r.similarity_score, 0) / validResults.length,
                  min_similarity: Math.min(...validResults.map(r => r.similarity_score)),
                  max_similarity: Math.max(...validResults.map(r => r.similarity_score)),
                  total_latency_ms: 0,
                  test_count: testCases.length,
                  successful_count: results.length,
                  failed_count: 0,
                  avg_rouge1: rouge1Results.length > 0 ? rouge1Results.reduce((sum, r) => sum + r.rouge1!, 0) / rouge1Results.length : undefined,
                  avg_rouge2: rouge2Results.length > 0 ? rouge2Results.reduce((sum, r) => sum + r.rouge2!, 0) / rouge2Results.length : undefined,
                  avg_rougel: rougelResults.length > 0 ? rougelResults.reduce((sum, r) => sum + r.rougel!, 0) / rougelResults.length : undefined,
                  avg_bertscore_precision: bertPrecisionResults.length > 0 ? bertPrecisionResults.reduce((sum, r) => sum + r.bertscore_precision!, 0) / bertPrecisionResults.length : undefined,
                  avg_bertscore_recall: bertRecallResults.length > 0 ? bertRecallResults.reduce((sum, r) => sum + r.bertscore_recall!, 0) / bertRecallResults.length : undefined,
                  avg_bertscore_f1: bertF1Results.length > 0 ? bertF1Results.reduce((sum, r) => sum + r.bertscore_f1!, 0) / bertF1Results.length : undefined,
                },
                embedding_model_used: selectedEmbeddingModel || '',
                llm_model_used: selectedLlmModel ? `${selectedLlmProvider}/${selectedLlmModel}` : undefined
              });
            } else if (data.event === 'complete') {
              // Final update with correct metadata from backend
              const validResults = results.filter(r => r.similarity_score != null);
              const rouge1Results = results.filter(r => r.rouge1 != null);
              const rouge2Results = results.filter(r => r.rouge2 != null);
              const rougelResults = results.filter(r => r.rougel != null);
              const bertPrecisionResults = results.filter(r => r.bertscore_precision != null);
              const bertRecallResults = results.filter(r => r.bertscore_recall != null);
              const bertF1Results = results.filter(r => r.bertscore_f1 != null);
              
              setBatchTestResult({
                results: results,
                aggregate: {
                  avg_similarity: validResults.reduce((sum, r) => sum + r.similarity_score, 0) / validResults.length,
                  min_similarity: Math.min(...validResults.map(r => r.similarity_score)),
                  max_similarity: Math.max(...validResults.map(r => r.similarity_score)),
                  total_latency_ms: 0,
                  test_count: testCases.length,
                  successful_count: results.length,
                  failed_count: 0,
                  avg_rouge1: rouge1Results.length > 0 ? rouge1Results.reduce((sum, r) => sum + r.rouge1!, 0) / rouge1Results.length : undefined,
                  avg_rouge2: rouge2Results.length > 0 ? rouge2Results.reduce((sum, r) => sum + r.rouge2!, 0) / rouge2Results.length : undefined,
                  avg_rougel: rougelResults.length > 0 ? rougelResults.reduce((sum, r) => sum + r.rougel!, 0) / rougelResults.length : undefined,
                  avg_bertscore_precision: bertPrecisionResults.length > 0 ? bertPrecisionResults.reduce((sum, r) => sum + r.bertscore_precision!, 0) / bertPrecisionResults.length : undefined,
                  avg_bertscore_recall: bertRecallResults.length > 0 ? bertRecallResults.reduce((sum, r) => sum + r.bertscore_recall!, 0) / bertRecallResults.length : undefined,
                  avg_bertscore_f1: bertF1Results.length > 0 ? bertF1Results.reduce((sum, r) => sum + r.bertscore_f1!, 0) / bertF1Results.length : undefined,
                },
                embedding_model_used: data.embedding_model_used || selectedEmbeddingModel || '',
                llm_model_used: data.llm_model_used || (selectedLlmModel ? `${selectedLlmProvider}/${selectedLlmModel}` : undefined)
              });
              toast.success(`Batch test tamamlandı: ${data.completed}/${data.total}`);
            } else if (data.event === 'error') {
              console.error('Test error:', data.error);
            }
          }
        }
      }
      
    } catch (error) {
      console.error("Batch test error:", error);
      if (error instanceof SyntaxError) {
        toast.error("Geçersiz JSON formatı");
      } else {
        toast.error(error instanceof Error ? error.message : "Test başarısız");
      }
    } finally {
      setIsBatchTesting(false);
    }
  };

  const handleSaveResult = async () => {
    if (!selectedCourseId || !quickTestResult) return;

    setIsSaving(true);
    try {
      await api.saveSemanticSimilarityResult({
        course_id: selectedCourseId,
        group_name: saveGroupName || undefined,
        question: quickTestResult.question,
        ground_truth: quickTestResult.ground_truth,
        alternative_ground_truths: quickTestAlternatives.filter(a => a.trim() !== ""),
        generated_answer: quickTestResult.generated_answer,
        similarity_score: quickTestResult.similarity_score,
        best_match_ground_truth: quickTestResult.best_match_ground_truth,
        all_scores: quickTestResult.all_scores,
        rouge1: quickTestResult.rouge1,
        rouge2: quickTestResult.rouge2,
        rougel: quickTestResult.rougel,
        bertscore_precision: quickTestResult.bertscore_precision,
        bertscore_recall: quickTestResult.bertscore_recall,
        bertscore_f1: quickTestResult.bertscore_f1,
        latency_ms: quickTestResult.latency_ms,
        embedding_model_used: quickTestResult.embedding_model_used,
        llm_model_used: quickTestResult.llm_model_used,
        retrieved_contexts: quickTestResult.retrieved_contexts,
        system_prompt_used: quickTestResult.system_prompt_used
      });
      toast.success("Sonuç kaydedildi");
      setIsSaveDialogOpen(false);
      setSaveGroupName("");
      loadSavedResults(true);
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "Kaydetme başarısız");
    } finally {
      setIsSaving(false);
    }
  };

  const handleSaveBatchResults = async () => {
    if (!selectedCourseId || !batchTestResult) return;

    setIsSaving(true);
    try {
      let successCount = 0;
      let failCount = 0;

      // Her test sonucunu ayrı ayrı kaydet
      for (const result of batchTestResult.results) {
        try {
          await api.saveSemanticSimilarityResult({
            course_id: selectedCourseId,
            group_name: saveGroupName || undefined,
            question: result.question,
            ground_truth: result.ground_truth,
            alternative_ground_truths: undefined,
            generated_answer: result.generated_answer,
            similarity_score: result.similarity_score,
            best_match_ground_truth: result.best_match_ground_truth,
            all_scores: undefined,
            rouge1: result.rouge1,
            rouge2: result.rouge2,
            rougel: result.rougel,
            bertscore_precision: result.bertscore_precision,
            bertscore_recall: result.bertscore_recall,
            bertscore_f1: result.bertscore_f1,
            latency_ms: result.latency_ms || 0,
            embedding_model_used: batchTestResult.embedding_model_used || selectedEmbeddingModel || "unknown",
            llm_model_used: batchTestResult.llm_model_used,
            retrieved_contexts: result.retrieved_contexts,
            system_prompt_used: result.system_prompt_used
          });
          successCount++;
        } catch (error) {
          console.error("Failed to save result:", error);
          failCount++;
        }
      }

      if (successCount > 0) {
        toast.success(`${successCount} sonuç kaydedildi${failCount > 0 ? `, ${failCount} başarısız` : ""}`);
      } else {
        toast.error("Hiçbir sonuç kaydedilemedi");
      }

      setIsSaveDialogOpen(false);
      setSaveGroupName("");
      loadSavedResults(true);
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "Kaydetme başarısız");
    } finally {
      setIsSaving(false);
    }
  };

  const handleDeleteSavedResult = async (id: number) => {
    if (!confirm("Bu sonucu silmek istediğinizden emin misiniz?")) return;

    try {
      await api.deleteSemanticSimilarityResult(id);
      toast.success("Sonuç silindi");
      loadSavedResults(true);
      if (viewingResult?.id === id) {
        setViewingResult(null);
      }
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "Silme başarısız");
    }
  };

  const exportBatchResults = () => {
    if (!batchTestResult) return;

    const csv = [
      ["Question", "Ground Truth", "Generated Answer", "Similarity Score", "Latency (ms)"].join(","),
      ...batchTestResult.results.map(r => [
        `"${r.question.replace(/"/g, '""')}"`,
        `"${r.ground_truth.replace(/"/g, '""')}"`,
        `"${r.generated_answer.replace(/"/g, '""')}"`,
        (r.similarity_score * 100).toFixed(2),
        r.latency_ms
      ].join(","))
    ].join("\n");

    const blob = new Blob([csv], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `semantic-similarity-batch-${Date.now()}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const exportAllResultsToCSV = async () => {
    if (!selectedCourseId) return;
    
    try {
      // Tüm sonuçları al (sayfalama olmadan)
      let groupFilter: string | undefined;
      if (selectedGroup === "__all__" || selectedGroup === "") {
        groupFilter = undefined;
      } else if (selectedGroup === "__no_group__") {
        groupFilter = "";
      } else {
        groupFilter = selectedGroup;
      }
      
      const data = await api.getSemanticSimilarityResults(
        selectedCourseId,
        groupFilter,
        0,
        10000 // Çok büyük bir limit - tüm sonuçları al
      );

      if (data.results.length === 0) {
        toast.error("İndirilecek sonuç yok");
        return;
      }

      // CSV başlıkları - sadece skorlar
      const headers = [
        "Soru No",
        "ROUGE-1 (%)",
        "ROUGE-2 (%)",
        "ROUGE-L (%)",
        "BERTScore P (%)",
        "BERTScore R (%)",
        "BERTScore F1 (%)",
        "Gecikme (ms)"
      ];

      const rows = data.results.map((r, idx) => [
        idx + 1,
        r.rouge1 != null ? (r.rouge1 * 100).toFixed(2) : "-",
        r.rouge2 != null ? (r.rouge2 * 100).toFixed(2) : "-",
        r.rougel != null ? (r.rougel * 100).toFixed(2) : "-",
        r.bertscore_precision != null ? (r.bertscore_precision * 100).toFixed(2) : "-",
        r.bertscore_recall != null ? (r.bertscore_recall * 100).toFixed(2) : "-",
        r.bertscore_f1 != null ? (r.bertscore_f1 * 100).toFixed(2) : "-",
        r.latency_ms
      ]);

      const csv = [headers.join(","), ...rows.map(row => row.join(","))].join("\n");

      // UTF-8 BOM ekle (Excel için Türkçe karakter desteği)
      const blob = new Blob(["\ufeff" + csv], { type: "text/csv;charset=utf-8;" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      const groupName = selectedGroup && selectedGroup !== "__all__" 
        ? (selectedGroup === "__no_group__" ? "grupsuz" : selectedGroup)
        : "tum-sonuclar";
      a.download = `semantic-similarity-scores-${groupName}-${Date.now()}.csv`;
      a.click();
      URL.revokeObjectURL(url);

      toast.success(`${data.results.length} sonuç CSV olarak indirildi`);
    } catch (error) {
      toast.error("CSV indirme başarısız");
      console.error(error);
    }
  };

  const exportToPDF = async () => {
    if (!selectedCourseId || !savedResultsAggregate) {
      toast.error("Lütfen bir ders seçin ve sonuçların yüklenmesini bekleyin");
      return;
    }

    try {
      setIsPdfGenerating(true);
      toast.loading("PDF raporu oluşturuluyor...", { id: "pdf-generation" });

      let groupFilter: string | undefined;
      if (selectedGroup === "__all__" || selectedGroup === "") {
        groupFilter = undefined;
      } else if (selectedGroup === "__no_group__") {
        groupFilter = "";
      } else {
        groupFilter = selectedGroup;
      }
      
      const data = await api.getSemanticSimilarityResults(
        selectedCourseId,
        groupFilter,
        0,
        10000
      );

      if (data.results.length === 0) {
        toast.error("İndirilecek sonuç yok", { id: "pdf-generation" });
        return;
      }

      const courseName = courses.find(c => c.id === selectedCourseId)?.name || "Bilinmeyen Ders";
      const groupName = selectedGroup && selectedGroup !== "__all__" 
        ? (selectedGroup === "__no_group__" ? "Grupsuz" : selectedGroup)
        : "Tüm Sonuçlar";

      const htmlContent = generateSemanticSimilarityPDF({
        results: data.results,
        aggregate: savedResultsAggregate,
        courseName,
        groupName
      });

      const printWindow = window.open('', '_blank');
      if (!printWindow) {
        toast.error("Popup engelleyicinizi kontrol edin", { id: "pdf-generation" });
        return;
      }

      printWindow.document.write(htmlContent);
      printWindow.document.close();
      printWindow.onload = () => {
        printWindow.print();
      };

      toast.success("PDF raporu oluşturuldu!", { id: "pdf-generation" });
    } catch (error) {
      console.error("PDF oluşturma hatası:", error);
      toast.error("PDF oluşturma başarısız", { id: "pdf-generation" });
    } finally {
      setIsPdfGenerating(false);
    }
  };

  const addAlternative = () => {
    setQuickTestAlternatives([...quickTestAlternatives, ""]);
  };

  const removeAlternative = (index: number) => {
    setQuickTestAlternatives(quickTestAlternatives.filter((_, i) => i !== index));
  };

  const updateAlternative = (index: number, value: string) => {
    const newAlternatives = [...quickTestAlternatives];
    newAlternatives[index] = value;
    setQuickTestAlternatives(newAlternatives);
  };

  const getMetricColor = (value: number) => {
    if (value >= 0.8) return "text-emerald-600";
    if (value >= 0.6) return "text-amber-600";
    return "text-red-600";
  };

  const getMetricBgColor = (value: number) => {
    if (value >= 0.8) return "bg-emerald-50 border-emerald-200";
    if (value >= 0.6) return "bg-amber-50 border-amber-200";
    return "bg-red-50 border-red-200";
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
    <div className="space-y-8">
      {/* Hero Header */}
      <div className="relative overflow-hidden bg-gradient-to-br from-teal-600 via-cyan-600 to-blue-700 rounded-2xl p-8 text-white shadow-xl">
        <div className="absolute inset-0 opacity-30" style={{backgroundImage: "url(\"data:image/svg+xml,%3Csvg width='60' height='60' viewBox='0 0 60 60' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' fill-rule='evenodd'%3E%3Cg fill='%23fff' fill-opacity='0.1'%3E%3Ccircle cx='30' cy='30' r='2'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E\")"}}></div>
        <div className="absolute -top-24 -right-24 w-96 h-96 bg-white/10 rounded-full blur-3xl"></div>
        <div className="absolute -bottom-24 -left-24 w-96 h-96 bg-teal-500/20 rounded-full blur-3xl"></div>
        
        <div className="relative z-10">
          <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between gap-6">
            <div className="flex items-center gap-4">
              <div className="p-3 bg-white/20 rounded-xl backdrop-blur-sm"><Target className="w-8 h-8" /></div>
              <div>
                <h1 className="text-3xl font-bold">Semantic Similarity Test</h1>
                <p className="text-teal-200 mt-1">Üretilen cevapların anlamsal benzerliğini ölçün</p>
              </div>
            </div>
            
            <div className="flex flex-wrap items-center gap-3">
              <Dialog open={isSettingsOpen} onOpenChange={setIsSettingsOpen}>
                <DialogTrigger asChild>
                  <Button variant="secondary" size="sm" className="bg-white/20 hover:bg-white/30 text-white border-0 backdrop-blur-sm h-10">
                    <Settings className="w-4 h-4 mr-2" />Ayarlar
                  </Button>
                </DialogTrigger>
                <DialogContent className="sm:max-w-lg">
                  <DialogHeader>
                    <DialogTitle className="flex items-center gap-2"><Settings className="w-5 h-5 text-teal-600" />Test Ayarları</DialogTitle>
                    <DialogDescription>Semantic similarity testi için kullanılacak modelleri seçin.</DialogDescription>
                  </DialogHeader>
                  <div className="space-y-4 py-4">
                    <div className="space-y-2">
                      <Label className="text-sm font-medium">Embedding Model</Label>
                      <Select value={selectedEmbeddingModel || ""} onValueChange={setSelectedEmbeddingModel}>
                        <SelectTrigger className="h-11"><SelectValue placeholder="Model seçin" /></SelectTrigger>
                        <SelectContent>
                          <SelectItem value="openai/text-embedding-3-small">OpenAI text-embedding-3-small (1536 dim)</SelectItem>
                          <SelectItem value="openai/text-embedding-3-large">OpenAI text-embedding-3-large (3072 dim)</SelectItem>
                          <SelectItem value="openai/text-embedding-ada-002">OpenAI text-embedding-ada-002 (1536 dim)</SelectItem>
                          <SelectItem value="alibaba/text-embedding-v4">Alibaba text-embedding-v4 (1024 dim)</SelectItem>
                          <SelectItem value="cohere/embed-multilingual-v3.0">Cohere embed-multilingual-v3.0 (1024 dim)</SelectItem>
                          <SelectItem value="cohere/embed-multilingual-light-v3.0">Cohere embed-multilingual-light-v3.0 (384 dim)</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>
                    <div className="space-y-2">
                      <Label className="text-sm font-medium">LLM Provider</Label>
                      <Select value={selectedLlmProvider || ""} onValueChange={(v) => { setSelectedLlmProvider(v); setSelectedLlmModel(""); }}>
                        <SelectTrigger className="h-11"><SelectValue placeholder="Provider seçin" /></SelectTrigger>
                        <SelectContent>
                          {Object.keys(llmProviders).map((provider) => (
                            <SelectItem key={provider} value={provider}>{provider}</SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                    </div>
                    <div className="space-y-2">
                      <Label className="text-sm font-medium">LLM Model</Label>
                      <Select value={selectedLlmModel || ""} onValueChange={setSelectedLlmModel} disabled={!selectedLlmProvider}>
                        <SelectTrigger className="h-11"><SelectValue placeholder={selectedLlmProvider ? "Model seçin" : "Önce provider seçin"} /></SelectTrigger>
                        <SelectContent>
                          {availableLlmModels.map((model) => (
                            <SelectItem key={model} value={model}>{model}</SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                    </div>
                    {(selectedEmbeddingModel || selectedLlmModel) && (
                      <div className="bg-gradient-to-r from-slate-50 to-slate-100 rounded-xl p-4 border border-slate-200">
                        <p className="font-semibold text-slate-700 mb-2 flex items-center gap-2"><Target className="w-4 h-4" />Seçili Modeller</p>
                        {selectedEmbeddingModel && <p className="text-sm text-slate-600">Embedding: <span className="font-medium text-slate-900">{selectedEmbeddingModel}</span></p>}
                        {selectedLlmModel && <p className="text-sm text-slate-600 mt-1">LLM: <span className="font-medium text-slate-900">{selectedLlmProvider}/{selectedLlmModel}</span></p>}
                      </div>
                    )}
                  </div>
                  <DialogFooter>
                    <Button variant="outline" onClick={() => setIsSettingsOpen(false)}>İptal</Button>
                    <Button onClick={handleSaveSettings} disabled={isSavingSettings} className="bg-teal-600 hover:bg-teal-700">{isSavingSettings ? <><Loader2 className="w-4 h-4 mr-2 animate-spin" />Kaydediliyor...</> : <><Save className="w-4 h-4 mr-2" />Kaydet</>}</Button>
                  </DialogFooter>
                </DialogContent>
              </Dialog>

              <Select value={selectedCourseId?.toString() || ""} onValueChange={(v) => setSelectedCourseId(Number(v))}>
                <SelectTrigger className="w-56 bg-white/20 border-0 text-white hover:bg-white/30 backdrop-blur-sm h-10"><BookOpen className="w-4 h-4 mr-2" /><SelectValue placeholder="Ders seçin" /></SelectTrigger>
                <SelectContent>{courses.map((course) => (<SelectItem key={course.id} value={course.id.toString()}>{course.name}</SelectItem>))}</SelectContent>
              </Select>
            </div>
          </div>

          {selectedCourseId && (
            <div className="grid grid-cols-2 md:grid-cols-3 gap-4 mt-8">
              <div className="bg-white/10 backdrop-blur-sm rounded-xl p-4 border border-white/20"><div className="flex items-center gap-3"><div className="p-2 bg-white/20 rounded-lg"><History className="w-5 h-5" /></div><div><p className="text-teal-200 text-sm">Kayıtlı Sonuçlar</p><p className="text-2xl font-bold">{savedResultsTotal}</p></div></div></div>
              <div className="bg-white/10 backdrop-blur-sm rounded-xl p-4 border border-white/20"><div className="flex items-center gap-3"><div className="p-2 bg-white/20 rounded-lg"><Target className="w-5 h-5" /></div><div><p className="text-teal-200 text-sm">Gruplar</p><p className="text-2xl font-bold">{savedResultsGroups.length}</p></div></div></div>
              <div className="bg-white/10 backdrop-blur-sm rounded-xl p-4 border border-white/20"><div className="flex items-center gap-3"><div className="p-2 bg-white/20 rounded-lg"><BookOpen className="w-5 h-5" /></div><div><p className="text-teal-200 text-sm">Seçili Ders</p><p className="text-lg font-bold truncate">{courses.find(c => c.id === selectedCourseId)?.name || "-"}</p></div></div></div>
            </div>
          )}
        </div>
      </div>

      {!selectedCourseId ? (
        <div className="bg-white rounded-2xl border border-slate-200 p-16 text-center shadow-sm">
          <div className="w-20 h-20 bg-teal-100 rounded-full flex items-center justify-center mx-auto mb-6">
            <Target className="w-10 h-10 text-teal-600" />
          </div>
          <h3 className="text-xl font-semibold text-slate-900 mb-2">Ders Seçin</h3>
          <p className="text-slate-500 max-w-md mx-auto">
            Semantic similarity testi yapmak için yukarıdan bir ders seçin.
          </p>
        </div>
      ) : (
        <div className="space-y-6">
          {/* Quick Test Card */}
          <Card className="overflow-hidden border-0 shadow-lg bg-gradient-to-br from-teal-50 via-white to-cyan-50">
            <button
              onClick={() => setIsQuickTestExpanded(!isQuickTestExpanded)}
              className="w-full px-6 py-5 flex items-center justify-between hover:bg-teal-50/50 transition-all duration-200"
            >
              <div className="flex items-center gap-4">
                <div className="p-3 bg-gradient-to-br from-teal-500 to-cyan-600 rounded-xl shadow-lg shadow-teal-200">
                  <Sparkles className="w-6 h-6 text-white" />
                </div>
                <div className="text-left">
                  <h2 className="text-xl font-bold text-slate-900">Hızlı Test</h2>
                  <p className="text-sm text-slate-600">
                    Tek bir soru için anında benzerlik testi
                  </p>
                </div>
              </div>
              <div
                className={`p-2 rounded-full bg-teal-100 transition-transform duration-200 ${
                  isQuickTestExpanded ? "rotate-180" : ""
                }`}
              >
                <ChevronDown className="w-5 h-5 text-teal-600" />
              </div>
            </button>

            {isQuickTestExpanded && (
              <div className="px-6 pb-6 pt-2 border-t border-teal-100">
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                  {/* Input Section */}
                  <div className="space-y-4">
                    <div>
                      <Label className="text-sm font-medium text-slate-700">Soru</Label>
                      <Textarea
                        value={quickTestQuestion}
                        onChange={(e) => setQuickTestQuestion(e.target.value)}
                        placeholder="Test etmek istediğiniz soruyu girin..."
                        rows={3}
                        className="mt-1.5 border-slate-200 focus:border-teal-400 focus:ring-teal-400"
                      />
                    </div>

                    <div>
                      <Label className="text-sm font-medium text-slate-700">
                        Doğru Cevap (Ground Truth)
                      </Label>
                      <Textarea
                        value={quickTestGroundTruth}
                        onChange={(e) => setQuickTestGroundTruth(e.target.value)}
                        placeholder="Beklenen doğru cevabı girin..."
                        rows={3}
                        className="mt-1.5 border-slate-200 focus:border-teal-400 focus:ring-teal-400"
                      />
                    </div>

                    <div>
                      <div className="flex items-center justify-between mb-2">
                        <Label className="text-sm font-medium text-slate-700">
                          Alternatif Doğru Cevaplar
                        </Label>
                        <Button
                          type="button"
                          variant="outline"
                          size="sm"
                          onClick={addAlternative}
                          className="h-7 text-xs border-teal-200 text-teal-600 hover:bg-teal-50"
                        >
                          <Plus className="w-3 h-3 mr-1" /> Ekle
                        </Button>
                      </div>
                      {quickTestAlternatives.map((alt, index) => (
                        <div key={index} className="flex gap-2 mb-2">
                          <Input
                            value={alt}
                            onChange={(e) => updateAlternative(index, e.target.value)}
                            placeholder={`Alternatif ${index + 1}`}
                            className="flex-1 border-slate-200"
                          />
                          <Button
                            type="button"
                            variant="ghost"
                            size="sm"
                            onClick={() => removeAlternative(index)}
                            className="text-red-500 hover:text-red-700 hover:bg-red-50"
                          >
                            <X className="w-4 h-4" />
                          </Button>
                        </div>
                      ))}
                    </div>

                    <div>
                      <Label className="text-sm font-medium text-slate-700">
                        Üretilen Cevap (Opsiyonel)
                      </Label>
                      <Textarea
                        value={quickTestGeneratedAnswer}
                        onChange={(e) => setQuickTestGeneratedAnswer(e.target.value)}
                        placeholder="Boş bırakılırsa otomatik üretilir..."
                        rows={3}
                        className="mt-1.5 border-slate-200 focus:border-teal-400 focus:ring-teal-400"
                      />
                      <p className="text-xs text-slate-500 mt-1">
                        Boş bırakırsanız, RAG sistemi kullanılarak cevap üretilir
                      </p>
                    </div>

                    <Button
                      onClick={handleQuickTest}
                      disabled={isQuickTesting || !quickTestQuestion || !quickTestGroundTruth}
                      className="w-full bg-gradient-to-r from-teal-600 to-cyan-600 hover:from-teal-700 hover:to-cyan-700 shadow-lg shadow-teal-200"
                    >
                      {isQuickTesting ? (
                        <>
                          <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                          Test Ediliyor...
                        </>
                      ) : (
                        <>
                          <Target className="w-4 h-4 mr-2" />
                          Test Et
                        </>
                      )}
                    </Button>
                  </div>

                  {/* Results Section */}
                  <div className="space-y-4">
                    {quickTestResult ? (
                      <>
                        <div>
                          <Label className="text-sm font-medium text-slate-700">
                            Üretilen Cevap
                          </Label>
                          <div className="mt-1.5 p-4 bg-white rounded-xl border border-slate-200 text-sm shadow-sm">
                            {quickTestResult.generated_answer}
                          </div>
                        </div>

                        {quickTestResult.all_scores && quickTestResult.all_scores.length > 1 && (
                          <div>
                            <Label className="text-sm font-medium text-slate-700">
                              Tüm Ground Truth Skorları
                            </Label>
                            <div className="mt-2 space-y-2">
                              {quickTestResult.all_scores.map((score, idx) => (
                                <div
                                  key={idx}
                                  className="p-3 bg-white rounded-lg border border-slate-200 flex justify-between items-center"
                                >
                                  <span className="text-xs text-slate-600 flex-1 truncate">
                                    {score.ground_truth.substring(0, 40)}...
                                  </span>
                                  <span
                                    className={`text-sm font-bold ${getMetricColor(score.score)}`}
                                  >
                                    {(score.score * 100).toFixed(1)}%
                                  </span>
                                </div>
                              ))}
                            </div>
                          </div>
                        )}

                        <div className="grid grid-cols-2 gap-3">
                          <div className="p-4 bg-white rounded-xl border border-slate-200 shadow-sm">
                            <p className="text-xs text-slate-500 font-medium">Gecikme</p>
                            <p className="text-lg font-bold text-slate-900">
                              {quickTestResult.latency_ms}ms
                            </p>
                          </div>
                          <div className="p-4 bg-white rounded-xl border border-slate-200 shadow-sm">
                            <p className="text-xs text-slate-500 font-medium">Embedding Model</p>
                            <p className="text-sm font-medium text-slate-900 truncate">
                              {quickTestResult.embedding_model_used}
                            </p>
                          </div>
                        </div>

                        {quickTestResult.llm_model_used && (
                          <div className="p-4 bg-white rounded-xl border border-slate-200 shadow-sm">
                            <p className="text-xs text-slate-500 font-medium">LLM Model</p>
                            <p className="text-sm font-medium text-slate-900 truncate">
                              {quickTestResult.llm_model_used}
                            </p>
                          </div>
                        )}

                        {/* ROUGE and BERTScore Metrics */}
                        {(quickTestResult.rouge1 !== null || quickTestResult.rouge2 !== null || quickTestResult.rougel !== null || 
                          quickTestResult.bertscore_f1 !== null) && (
                          <div>
                            <Label className="text-sm font-medium text-slate-700">
                              Ek Metrikler
                            </Label>
                            <div className="mt-2 grid grid-cols-2 gap-3">
                              {quickTestResult.rouge1 !== null && (
                                <div className="p-4 bg-gradient-to-br from-purple-50 to-purple-100 rounded-xl border border-purple-200 shadow-sm">
                                  <p className="text-xs text-purple-600 font-medium">ROUGE-1</p>
                                  <p className={`text-2xl font-bold ${getMetricColor(quickTestResult.rouge1)}`}>
                                    {(quickTestResult.rouge1 * 100).toFixed(1)}%
                                  </p>
                                </div>
                              )}
                              {quickTestResult.rouge2 !== null && (
                                <div className="p-4 bg-gradient-to-br from-purple-50 to-purple-100 rounded-xl border border-purple-200 shadow-sm">
                                  <p className="text-xs text-purple-600 font-medium">ROUGE-2</p>
                                  <p className={`text-2xl font-bold ${getMetricColor(quickTestResult.rouge2)}`}>
                                    {(quickTestResult.rouge2 * 100).toFixed(1)}%
                                  </p>
                                </div>
                              )}
                              {quickTestResult.rougel !== null && (
                                <div className="p-4 bg-gradient-to-br from-purple-50 to-purple-100 rounded-xl border border-purple-200 shadow-sm">
                                  <p className="text-xs text-purple-600 font-medium">ROUGE-L</p>
                                  <p className={`text-2xl font-bold ${getMetricColor(quickTestResult.rougel)}`}>
                                    {(quickTestResult.rougel * 100).toFixed(1)}%
                                  </p>
                                </div>
                              )}
                              {quickTestResult.bertscore_f1 !== null && (
                                <div className="p-4 bg-gradient-to-br from-indigo-50 to-indigo-100 rounded-xl border border-indigo-200 shadow-sm">
                                  <p className="text-xs text-indigo-600 font-medium">BERTScore F1</p>
                                  <p className={`text-2xl font-bold ${getMetricColor(quickTestResult.bertscore_f1)}`}>
                                    {(quickTestResult.bertscore_f1 * 100).toFixed(1)}%
                                  </p>
                                </div>
                              )}
                            </div>
                          </div>
                        )}

                        {/* Retrieved Contexts / Sources */}
                        {quickTestResult.retrieved_contexts && quickTestResult.retrieved_contexts.length > 0 && (
                          <div>
                            <Label className="text-sm font-medium text-slate-700 flex items-center gap-2">
                              <FileText className="w-4 h-4" />
                              Kullanılan Kaynaklar ({quickTestResult.retrieved_contexts.length})
                            </Label>
                            <div className="mt-2 space-y-2 max-h-64 overflow-y-auto">
                              {quickTestResult.retrieved_contexts.map((context, idx) => (
                                <div
                                  key={idx}
                                  className="p-3 bg-white rounded-lg border border-slate-200 text-xs text-slate-600"
                                >
                                  <div className="flex items-center gap-2 mb-1">
                                    <span className="bg-teal-100 text-teal-700 px-2 py-0.5 rounded text-xs font-medium">
                                      Kaynak {idx + 1}
                                    </span>
                                  </div>
                                  <p className="line-clamp-3">{context}</p>
                                </div>
                              ))}
                            </div>
                          </div>
                        )}

                        <Button
                          variant="outline"
                          className="w-full border-teal-200 text-teal-600 hover:bg-teal-50"
                          onClick={() => {
                            setSaveDialogMode("quick");
                            setIsSaveDialogOpen(true);
                          }}
                        >
                          <Save className="w-4 h-4 mr-2" />
                          Sonucu Kaydet
                        </Button>
                      </>
                    ) : (
                      <div className="flex items-center justify-center h-full min-h-[400px] text-slate-400">
                        <div className="text-center">
                          <div className="w-20 h-20 bg-teal-100 rounded-full flex items-center justify-center mx-auto mb-4">
                            <Sparkles className="w-10 h-10 text-teal-400" />
                          </div>
                          <p className="text-sm font-medium">Test sonuçları burada görünecek</p>
                          <p className="text-xs text-slate-400 mt-1">
                            Soru ve doğru cevabı girerek test başlatın
                          </p>
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              </div>
            )}
          </Card>

          {/* Batch Test Card */}
          <Card className="overflow-hidden border-0 shadow-lg bg-gradient-to-br from-cyan-50 via-white to-blue-50">
            <button
              onClick={() => setIsBatchTestExpanded(!isBatchTestExpanded)}
              className="w-full px-6 py-5 flex items-center justify-between hover:bg-cyan-50/50 transition-all duration-200"
            >
              <div className="flex items-center gap-4">
                <div className="p-3 bg-gradient-to-br from-cyan-500 to-blue-600 rounded-xl shadow-lg shadow-cyan-200">
                  <FileJson className="w-6 h-6 text-white" />
                </div>
                <div className="text-left">
                  <h2 className="text-xl font-bold text-slate-900">Batch Test</h2>
                  <p className="text-sm text-slate-600">
                    Birden fazla test için JSON formatında toplu test
                  </p>
                </div>
              </div>
              <div
                className={`p-2 rounded-full bg-cyan-100 transition-transform duration-200 ${
                  isBatchTestExpanded ? "rotate-180" : ""
                }`}
              >
                <ChevronDown className="w-5 h-5 text-cyan-600" />
              </div>
            </button>

            {isBatchTestExpanded && (
              <div className="px-6 pb-6 pt-2 border-t border-cyan-100">
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                  {/* Input Section */}
                  <div className="space-y-4">
                    <div>
                      <Label className="text-sm font-medium text-slate-700">Test Verileri (JSON)</Label>
                      <Textarea
                        value={batchTestJson}
                        onChange={(e) => setBatchTestJson(e.target.value)}
                        placeholder={`[\n  {\n    "question": "Soru 1",\n    "ground_truth": "Doğru cevap 1",\n    "alternative_ground_truths": ["Alt 1"],\n    "generated_answer": "Üretilen cevap 1"\n  }\n]`}
                        rows={6}
                        className="mt-1.5 border-slate-200 focus:border-cyan-400 focus:ring-cyan-400 font-mono text-xs max-h-48 overflow-y-auto resize-none"
                      />
                      <p className="text-xs text-slate-500 mt-2">
                        JSON array formatında test verilerini girin. generated_answer opsiyoneldir.
                      </p>
                    </div>

                    <Button
                      onClick={handleBatchTest}
                      disabled={isBatchTesting || !batchTestJson}
                      className="w-full bg-gradient-to-r from-cyan-600 to-blue-600 hover:from-cyan-700 hover:to-blue-700 shadow-lg shadow-cyan-200"
                    >
                      {isBatchTesting ? (
                        <>
                          <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                          Test Ediliyor...
                        </>
                      ) : (
                        <>
                          <FileJson className="w-4 h-4 mr-2" />
                          Batch Test Başlat
                        </>
                      )}
                    </Button>
                  </div>

                  {/* Results Section */}
                  <div className="space-y-4">
                    {batchTestResult ? (
                      <>
                        <div>
                          <Label className="text-sm font-medium text-slate-700">
                            Özet İstatistikler
                          </Label>
                          <div className="mt-2 grid grid-cols-3 gap-3">
                            <div className="p-4 bg-white rounded-xl border border-slate-200 shadow-sm">
                              <p className="text-xs text-slate-500 font-medium">Test Sayısı</p>
                              <p className="text-2xl font-bold text-slate-900">
                                {batchTestResult.aggregate.test_count}
                              </p>
                            </div>
                            {batchTestResult.aggregate.avg_rouge1 != null && (
                              <div className="p-4 bg-gradient-to-br from-purple-50 to-pink-50 rounded-xl border border-purple-200 shadow-sm">
                                <p className="text-xs text-purple-600 font-medium">Ort. ROUGE-1</p>
                                <p className="text-xl font-bold text-purple-700">
                                  {(batchTestResult.aggregate.avg_rouge1 * 100).toFixed(1)}%
                                </p>
                              </div>
                            )}
                            {batchTestResult.aggregate.avg_rouge2 != null && (
                              <div className="p-4 bg-gradient-to-br from-purple-50 to-pink-50 rounded-xl border border-purple-200 shadow-sm">
                                <p className="text-xs text-purple-600 font-medium">Ort. ROUGE-2</p>
                                <p className="text-xl font-bold text-purple-700">
                                  {(batchTestResult.aggregate.avg_rouge2 * 100).toFixed(1)}%
                                </p>
                              </div>
                            )}
                            {batchTestResult.aggregate.avg_rougel != null && (
                              <div className="p-4 bg-gradient-to-br from-purple-50 to-pink-50 rounded-xl border border-purple-200 shadow-sm">
                                <p className="text-xs text-purple-600 font-medium">Ort. ROUGE-L</p>
                                <p className="text-xl font-bold text-purple-700">
                                  {(batchTestResult.aggregate.avg_rougel * 100).toFixed(1)}%
                                </p>
                              </div>
                            )}
                            {batchTestResult.aggregate.avg_bertscore_f1 != null && (
                              <div className="p-4 bg-gradient-to-br from-blue-50 to-cyan-50 rounded-xl border border-blue-200 shadow-sm">
                                <p className="text-xs text-blue-600 font-medium">Ort. BERTScore F1</p>
                                <p className="text-xl font-bold text-blue-700">
                                  {(batchTestResult.aggregate.avg_bertscore_f1 * 100).toFixed(1)}%
                                </p>
                              </div>
                            )}
                          </div>
                        </div>

                        <div>
                          <div className="flex items-center justify-between mb-2">
                            <Label className="text-sm font-medium text-slate-700">
                              Test Sonuçları
                            </Label>
                            <div className="flex gap-2">
                              <Button
                                variant="outline"
                                size="sm"
                                onClick={() => {
                                  setSaveDialogMode("batch");
                                  setIsSaveDialogOpen(true);
                                }}
                                className="h-7 text-xs border-teal-200 text-teal-600 hover:bg-teal-50"
                              >
                                <Save className="w-3 h-3 mr-1" /> Tümünü Kaydet
                              </Button>
                              <Button
                                variant="outline"
                                size="sm"
                                onClick={exportBatchResults}
                                className="h-7 text-xs border-cyan-200 text-cyan-600 hover:bg-cyan-50"
                              >
                                <Download className="w-3 h-3 mr-1" /> CSV İndir
                              </Button>
                            </div>
                          </div>
                          <div className="max-h-[400px] overflow-y-auto border border-slate-200 rounded-xl">
                            <table className="w-full text-xs">
                              <thead className="bg-slate-50 sticky top-0">
                                <tr>
                                  <th className="px-3 py-2 text-left font-medium text-slate-600">
                                    Soru
                                  </th>
                                  <th className="px-3 py-2 text-center font-medium text-slate-600">
                                    ROUGE-1
                                  </th>
                                  <th className="px-3 py-2 text-center font-medium text-slate-600">
                                    ROUGE-L
                                  </th>
                                  <th className="px-3 py-2 text-center font-medium text-slate-600">
                                    BERTScore
                                  </th>
                                  <th className="px-3 py-2 text-center font-medium text-slate-600">
                                    Süre
                                  </th>
                                </tr>
                              </thead>
                              <tbody className="divide-y divide-slate-100">
                                {batchTestResult.results.map((result, idx) => (
                                  <tr key={idx} className="hover:bg-slate-50">
                                    <td className="px-3 py-2 text-slate-700 truncate max-w-[200px]">
                                      {result.question}
                                    </td>
                                    <td className="px-3 py-2 text-center">
                                      {result.rouge1 !== null && result.rouge1 !== undefined ? (
                                        <span className={`font-medium ${getMetricColor(result.rouge1)}`}>
                                          {(result.rouge1 * 100).toFixed(0)}%
                                        </span>
                                      ) : (
                                        <span className="text-slate-400">-</span>
                                      )}
                                    </td>
                                    <td className="px-3 py-2 text-center">
                                      {result.rougel !== null && result.rougel !== undefined ? (
                                        <span className={`font-medium ${getMetricColor(result.rougel)}`}>
                                          {(result.rougel * 100).toFixed(0)}%
                                        </span>
                                      ) : (
                                        <span className="text-slate-400">-</span>
                                      )}
                                    </td>
                                    <td className="px-3 py-2 text-center">
                                      {result.bertscore_f1 !== null && result.bertscore_f1 !== undefined ? (
                                        <span className={`font-medium ${getMetricColor(result.bertscore_f1)}`}>
                                          {(result.bertscore_f1 * 100).toFixed(0)}%
                                        </span>
                                      ) : (
                                        <span className="text-slate-400">-</span>
                                      )}
                                    </td>
                                    <td className="px-3 py-2 text-center text-slate-600">
                                      {result.latency_ms}ms
                                    </td>
                                  </tr>
                                ))}
                              </tbody>
                            </table>
                          </div>
                        </div>
                      </>
                    ) : (
                      <div className="flex items-center justify-center h-full min-h-[400px] text-slate-400">
                        <div className="text-center">
                          <div className="w-20 h-20 bg-cyan-100 rounded-full flex items-center justify-center mx-auto mb-4">
                            <FileJson className="w-10 h-10 text-cyan-400" />
                          </div>
                          <p className="text-sm font-medium">Batch test sonuçları burada görünecek</p>
                          <p className="text-xs text-slate-400 mt-1">
                            JSON formatında test verilerini girerek başlatın
                          </p>
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              </div>
            )}
          </Card>

          {/* Saved Results Card */}
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
                  <p className="text-sm text-slate-600">
                    {savedResultsTotal} kayıtlı test sonucu
                  </p>
                </div>
              </div>
              <div
                className={`p-2 rounded-full bg-slate-100 transition-transform duration-200 ${
                  isSavedResultsExpanded ? "rotate-180" : ""
                }`}
              >
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
                      <SelectTrigger className="w-full max-w-md">
                        <SelectValue placeholder="Tüm gruplar" />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="__all__">Tüm gruplar</SelectItem>
                        {savedResultsGroups
                          .filter((g) => g && g.trim() !== "")
                          .map((g) => (
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

                {/* Grup Özet İstatistikleri */}
                {savedResultsAggregate && (
                  <div className="mb-6 p-4 bg-gradient-to-br from-teal-50 to-cyan-50 rounded-xl border border-teal-200">
                    <div className="flex items-center justify-between mb-3">
                      <h3 className="text-sm font-semibold text-teal-900">
                        {selectedGroup && selectedGroup !== "__all__" 
                          ? `"${selectedGroup === "__no_group__" ? "Grupsuz" : selectedGroup}" Grup İstatistikleri (Tüm ${savedResultsAggregate.test_count} Test)`
                          : `Genel İstatistikler (Tüm ${savedResultsAggregate.test_count} Test)`}
                      </h3>
                      <div className="flex gap-2">
                        <Button
                          variant="outline"
                          size="sm"
                          onClick={exportAllResultsToCSV}
                          className="h-7 text-xs border-teal-300 text-teal-700 hover:bg-teal-100"
                        >
                          <Download className="w-3 h-3 mr-1" /> CSV İndir
                        </Button>
                        <Button
                          variant="outline"
                          size="sm"
                          onClick={exportToPDF}
                          disabled={isPdfGenerating}
                          className="h-7 text-xs border-cyan-300 text-cyan-700 hover:bg-cyan-100"
                        >
                          {isPdfGenerating ? (
                            <>
                              <Loader2 className="w-3 h-3 mr-1 animate-spin" /> Oluşturuluyor...
                            </>
                          ) : (
                            <>
                              <FileText className="w-3 h-3 mr-1" /> PDF Rapor
                            </>
                          )}
                        </Button>
                      </div>
                    </div>
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                      {savedResultsAggregate.avg_rouge1 != null && (
                        <div className="p-3 bg-white rounded-lg border border-purple-100">
                          <p className="text-xs text-purple-600 mb-1">Ort. ROUGE-1</p>
                          <p className="text-xl font-bold text-purple-700">
                            {(savedResultsAggregate.avg_rouge1 * 100).toFixed(1)}%
                          </p>
                        </div>
                      )}
                      {savedResultsAggregate.avg_rouge2 != null && (
                        <div className="p-3 bg-white rounded-lg border border-purple-100">
                          <p className="text-xs text-purple-600 mb-1">Ort. ROUGE-2</p>
                          <p className="text-xl font-bold text-purple-700">
                            {(savedResultsAggregate.avg_rouge2 * 100).toFixed(1)}%
                          </p>
                        </div>
                      )}
                      {savedResultsAggregate.avg_rougel != null && (
                        <div className="p-3 bg-white rounded-lg border border-purple-100">
                          <p className="text-xs text-purple-600 mb-1">Ort. ROUGE-L</p>
                          <p className="text-xl font-bold text-purple-700">
                            {(savedResultsAggregate.avg_rougel * 100).toFixed(1)}%
                          </p>
                        </div>
                      )}
                      {savedResultsAggregate.avg_bertscore_f1 != null && (
                        <div className="p-3 bg-white rounded-lg border border-blue-100">
                          <p className="text-xs text-blue-600 mb-1">Ort. BERTScore</p>
                          <p className="text-xl font-bold text-blue-700">
                            {(savedResultsAggregate.avg_bertscore_f1 * 100).toFixed(1)}%
                          </p>
                        </div>
                      )}
                      <div className="p-3 bg-white rounded-lg border border-slate-100">
                        <p className="text-xs text-slate-600 mb-1">Test Sayısı</p>
                        <p className="text-xl font-bold text-slate-900">
                          {savedResultsAggregate.test_count}
                        </p>
                      </div>
                    </div>
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
                      <div
                        key={result.id}
                        className="p-4 bg-slate-50 rounded-xl border border-slate-200 hover:border-slate-300 transition-colors"
                      >
                        <div className="flex items-start justify-between">
                          <div className="flex-1 min-w-0">
                            <div className="flex items-center gap-2 mb-1">
                              {result.group_name && (
                                <span className="px-2 py-0.5 text-xs bg-teal-100 text-teal-700 rounded-full font-medium">
                                  {result.group_name}
                                </span>
                              )}
                              <span className="text-xs text-slate-500">
                                {new Date(result.created_at).toLocaleString("tr-TR")}
                              </span>
                            </div>
                            <p className="text-sm font-medium text-slate-900 truncate">
                              {result.question}
                            </p>
                            <div className="flex flex-wrap items-center gap-3 mt-2 text-xs">
                              {result.bertscore_f1 != null && (
                                <span className={getMetricColor(result.bertscore_f1)}>
                                  BERTScore: {(result.bertscore_f1 * 100).toFixed(0)}%
                                </span>
                              )}
                              {result.rouge1 != null && (
                                <span className="text-purple-600">
                                  ROUGE-1: {(result.rouge1 * 100).toFixed(0)}%
                                </span>
                              )}
                              <span className="text-slate-400">{result.latency_ms}ms</span>
                            </div>
                          </div>
                          <div className="flex items-center gap-1 ml-2">
                            <Button
                              variant="ghost"
                              size="sm"
                              onClick={() => setViewingResult(result)}
                              className="text-slate-600 hover:text-teal-600"
                            >
                              <Eye className="w-4 h-4" />
                            </Button>
                            <Button
                              variant="ghost"
                              size="sm"
                              onClick={() => handleDeleteSavedResult(result.id)}
                              className="text-slate-400 hover:text-red-600"
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

          {/* Save Dialog */}
          <Dialog open={isSaveDialogOpen} onOpenChange={setIsSaveDialogOpen}>
            <DialogContent>
              <DialogHeader>
                <DialogTitle>
                  {saveDialogMode === "quick" ? "Sonucu Kaydet" : "Batch Sonuçlarını Kaydet"}
                </DialogTitle>
                <DialogDescription>
                  {saveDialogMode === "quick" 
                    ? "Bu test sonucunu daha sonra görüntülemek için kaydedin."
                    : `${batchTestResult?.results.length || 0} test sonucunu kaydetmek üzeresiniz.`
                  }
                </DialogDescription>
              </DialogHeader>
              <div className="space-y-4 py-4">
                <div className="space-y-2">
                  <Label>Grup Adı (Opsiyonel)</Label>
                  <Input
                    value={saveGroupName}
                    onChange={(e) => setSaveGroupName(e.target.value)}
                    placeholder="örn: Deneme 1"
                  />
                </div>
                {savedResultsGroups.length > 0 && (
                  <div className="flex flex-wrap gap-1">
                    {savedResultsGroups.map((g) => (
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
              <DialogFooter>
                <Button
                  variant="outline"
                  onClick={() => setIsSaveDialogOpen(false)}
                >
                  İptal
                </Button>
                <Button 
                  onClick={saveDialogMode === "quick" ? handleSaveResult : handleSaveBatchResults} 
                  disabled={isSaving}
                >
                  {isSaving ? (
                    <>
                      <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                      Kaydediliyor...
                    </>
                  ) : (
                    "Kaydet"
                  )}
                </Button>
              </DialogFooter>
            </DialogContent>
          </Dialog>

          {/* View Result Dialog */}
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
                    <p className="mt-1 p-3 bg-slate-50 rounded-lg text-sm">
                      {viewingResult.question}
                    </p>
                  </div>

                  <div>
                    <Label className="text-sm font-medium">Doğru Cevap</Label>
                    <p className="mt-1 p-3 bg-slate-50 rounded-lg text-sm">
                      {viewingResult.ground_truth}
                    </p>
                  </div>

                  {viewingResult.alternative_ground_truths &&
                    viewingResult.alternative_ground_truths.length > 0 && (
                      <div>
                        <Label className="text-sm font-medium">Alternatif Doğru Cevaplar</Label>
                        <div className="mt-1 space-y-1">
                          {viewingResult.alternative_ground_truths.map((alt, idx) => (
                            <p key={idx} className="p-2 bg-slate-50 rounded text-xs">
                              {idx + 1}. {alt}
                            </p>
                          ))}
                        </div>
                      </div>
                    )}

                  <div>
                    <Label className="text-sm font-medium">Üretilen Cevap</Label>
                    <p className="mt-1 p-3 bg-slate-50 rounded-lg text-sm">
                      {viewingResult.generated_answer}
                    </p>
                  </div>

                  {viewingResult.all_scores && viewingResult.all_scores.length > 1 && (
                    <div>
                      <Label className="text-sm font-medium">Tüm Ground Truth Skorları</Label>
                      <div className="mt-2 space-y-2">
                        {viewingResult.all_scores.map((score, idx) => (
                          <div
                            key={idx}
                            className="p-3 bg-slate-50 rounded-lg border border-slate-200 flex justify-between items-center"
                          >
                            <span className="text-xs text-slate-600 flex-1">
                              {score.ground_truth}
                            </span>
                            <span
                              className={`text-sm font-bold ml-3 ${getMetricColor(score.score)}`}
                            >
                              {(score.score * 100).toFixed(1)}%
                            </span>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* ROUGE ve BERTScore Metrikleri */}
                  {(viewingResult.rouge1 != null || viewingResult.rouge2 != null || 
                    viewingResult.rougel != null || viewingResult.bertscore_f1 != null) && (
                    <div>
                      <Label className="text-sm font-medium">Metrikler</Label>
                      <div className="mt-2 grid grid-cols-2 gap-3">
                        {viewingResult.rouge1 != null && (
                          <div className="p-4 bg-gradient-to-br from-purple-50 to-pink-50 rounded-xl border border-purple-200">
                            <p className="text-xs text-purple-600 font-medium mb-1">ROUGE-1</p>
                            <p className="text-2xl font-bold text-purple-700">
                              {(viewingResult.rouge1 * 100).toFixed(1)}%
                            </p>
                          </div>
                        )}
                        {viewingResult.rouge2 != null && (
                          <div className="p-4 bg-gradient-to-br from-purple-50 to-pink-50 rounded-xl border border-purple-200">
                            <p className="text-xs text-purple-600 font-medium mb-1">ROUGE-2</p>
                            <p className="text-2xl font-bold text-purple-700">
                              {(viewingResult.rouge2 * 100).toFixed(1)}%
                            </p>
                          </div>
                        )}
                        {viewingResult.rougel != null && (
                          <div className="p-4 bg-gradient-to-br from-purple-50 to-pink-50 rounded-xl border border-purple-200">
                            <p className="text-xs text-purple-600 font-medium mb-1">ROUGE-L</p>
                            <p className="text-2xl font-bold text-purple-700">
                              {(viewingResult.rougel * 100).toFixed(1)}%
                            </p>
                          </div>
                        )}
                        {viewingResult.bertscore_precision != null && (
                          <div className="p-4 bg-gradient-to-br from-blue-50 to-cyan-50 rounded-xl border border-blue-200">
                            <p className="text-xs text-blue-600 font-medium mb-1">BERTScore Precision</p>
                            <p className="text-2xl font-bold text-blue-700">
                              {(viewingResult.bertscore_precision * 100).toFixed(1)}%
                            </p>
                          </div>
                        )}
                        {viewingResult.bertscore_recall != null && (
                          <div className="p-4 bg-gradient-to-br from-blue-50 to-cyan-50 rounded-xl border border-blue-200">
                            <p className="text-xs text-blue-600 font-medium mb-1">BERTScore Recall</p>
                            <p className="text-2xl font-bold text-blue-700">
                              {(viewingResult.bertscore_recall * 100).toFixed(1)}%
                            </p>
                          </div>
                        )}
                        {viewingResult.bertscore_f1 != null && (
                          <div className="p-4 bg-gradient-to-br from-blue-50 to-cyan-50 rounded-xl border border-blue-200">
                            <p className="text-xs text-blue-600 font-medium mb-1">BERTScore F1</p>
                            <p className="text-2xl font-bold text-blue-700">
                              {(viewingResult.bertscore_f1 * 100).toFixed(1)}%
                            </p>
                          </div>
                        )}
                      </div>
                    </div>
                  )}

                  {viewingResult.all_scores && viewingResult.all_scores.length > 1 && (
                    <div>
                      <Label className="text-sm font-medium">Tüm Skorlar</Label>
                      <div className="mt-2 space-y-2">
                        {viewingResult.all_scores.map((score, idx) => (
                          <div
                            key={idx}
                            className="p-3 bg-slate-50 rounded-lg border border-slate-200 flex justify-between items-center"
                          >
                            <span className="text-xs text-slate-600 flex-1">
                              {score.ground_truth}
                            </span>
                            <span
                              className={`text-sm font-bold ml-3 ${getMetricColor(score.score)}`}
                            >
                              {(score.score * 100).toFixed(1)}%
                            </span>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  <div className="grid grid-cols-2 gap-3">
                    <div className="p-4 bg-slate-50 rounded-lg border border-slate-200">
                      <p className="text-xs text-slate-600">Gecikme</p>
                      <p className="text-xl font-bold text-slate-900">
                        {viewingResult.latency_ms}ms
                      </p>
                    </div>
                    <div className="p-4 bg-slate-50 rounded-lg border border-slate-200">
                      <p className="text-xs text-slate-600">Embedding Model</p>
                      <p className="text-sm font-medium text-slate-900 truncate">
                        {viewingResult.embedding_model_used}
                      </p>
                    </div>
                    {viewingResult.llm_model_used && (
                      <div className="p-4 bg-slate-50 rounded-lg border border-slate-200 col-span-2">
                        <p className="text-xs text-slate-600">LLM Model</p>
                        <p className="text-sm font-medium text-slate-900">
                          {viewingResult.llm_model_used}
                        </p>
                      </div>
                    )}
                  </div>

                  {viewingResult.system_prompt_used && (
                    <div>
                      <Label className="text-sm font-medium text-amber-700">⚠️ Kullanılan Sistem Promptu</Label>
                      <div className="mt-2 p-4 bg-amber-50 rounded-lg border border-amber-200">
                        <pre className="text-xs text-slate-700 whitespace-pre-wrap font-mono">
                          {viewingResult.system_prompt_used}
                        </pre>
                      </div>
                    </div>
                  )}

                  {viewingResult.retrieved_contexts && viewingResult.retrieved_contexts.length > 0 && (
                    <div>
                      <Label className="text-sm font-medium">Alınan Bağlamlar ({viewingResult.retrieved_contexts.length})</Label>
                      <div className="mt-2 space-y-2 max-h-60 overflow-y-auto">
                        {viewingResult.retrieved_contexts.map((context, idx) => (
                          <div key={idx} className="p-3 bg-blue-50 rounded-lg border border-blue-200">
                            <p className="text-xs text-blue-600 font-medium mb-1">Bağlam {idx + 1}</p>
                            <p className="text-xs text-slate-700 whitespace-pre-wrap">
                              {context.length > 300 ? context.substring(0, 300) + '...' : context}
                            </p>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              )}
              <DialogFooter>
                <Button variant="outline" onClick={() => setViewingResult(null)}>
                  Kapat
                </Button>
              </DialogFooter>
            </DialogContent>
          </Dialog>
        </div>
      )}
    </div>
  );
}
