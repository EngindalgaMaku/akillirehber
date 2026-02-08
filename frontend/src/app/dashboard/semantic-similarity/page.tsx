"use client";

import { useEffect, useState, useCallback, useRef } from "react";
import { useAuth } from "@/lib/auth-context";
import { useModelProviders } from "@/hooks/useModelProviders";
import { 
  api, 
  Course, 
  SemanticSimilarityQuickTestResponse,
  SemanticSimilarityBatchTestResponse,
  SemanticSimilarityResult,
  BatchTestSession
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
  FileText,
  History,
  Download,
  Trash2,
  Save,
  Play,
  Pause,
  Square,
  Loader2,
  Settings,
  RefreshCw,
  BarChart3,
  BookOpen,
  Sparkles,
  ChevronDown,
  Plus,
  X,
  FileJson,
  Eye,
} from "lucide-react";
import { generateSemanticSimilarityPDF } from "./exportToPDF";

export default function SemanticSimilarityPage() {
  const { user } = useAuth();
  const { getEmbeddingModels, getLLMProviders, getLLMModels, isLoading: providersLoading } = useModelProviders();
  const [courses, setCourses] = useState<Course[]>([]);
  const [selectedCourseId, setSelectedCourseId] = useState<number | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  // Settings State
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);
  const [selectedEmbeddingProvider, setSelectedEmbeddingProvider] = useState<string>(() => {
    if (typeof window !== 'undefined') {
      return localStorage.getItem('semantic_similarity_embedding_provider') || "";
    }
    return "";
  });
  const [selectedEmbeddingModel, setSelectedEmbeddingModel] = useState<string>(() => {
    if (typeof window !== 'undefined') {
      return localStorage.getItem('semantic_similarity_embedding_model') || "";
    }
    return "";
  });
  const [selectedLlmProvider, setSelectedLlmProvider] = useState<string>(() => {
    if (typeof window !== 'undefined') {
      return localStorage.getItem('semantic_similarity_llm_provider') || "";
    }
    return "";
  });
  const [selectedLlmModel, setSelectedLlmModel] = useState<string>(() => {
    if (typeof window !== 'undefined') {
      return localStorage.getItem('semantic_similarity_llm_model') || "";
    }
    return "";
  });
  // Reranker state
  const [selectedRerankerProvider, setSelectedRerankerProvider] = useState<string>(() => {
    if (typeof window !== 'undefined') {
      return localStorage.getItem('semantic_similarity_reranker_provider') || "";
    }
    return "";
  });
  const [selectedRerankerModel, setSelectedRerankerModel] = useState<string>(() => {
    if (typeof window !== 'undefined') {
      return localStorage.getItem('semantic_similarity_reranker_model') || "";
    }
    return "";
  });
  const [isRerankerEnabled, setIsRerankerEnabled] = useState<boolean>(() => {
    if (typeof window !== 'undefined') {
      return localStorage.getItem('semantic_similarity_reranker_enabled') === 'true';
    }
    return false;
  });
  const [isDirectLlmEnabled, setIsDirectLlmEnabled] = useState<boolean>(() => {
    if (typeof window !== 'undefined') {
      return localStorage.getItem('semantic_similarity_direct_llm_enabled') === 'true';
    }
    return false;
  });
  const [isSavingSettings, setIsSavingSettings] = useState(false);

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
  const batchTestStartTimeRef = useRef<number | null>(null);
  const [batchTestElapsedTime, setBatchTestElapsedTime] = useState<string>("00:00:00");
  const [repeatCount, setRepeatCount] = useState<number>(3);
  const [currentRunNumber, setCurrentRunNumber] = useState<number>(1);
  const [totalRuns, setTotalRuns] = useState<number>(3);
  const [allRunResults, setAllRunResults] = useState<SemanticSimilarityBatchTestResponse[]>([]);

  // Batch Test Sessions State
  const [batchTestSessions, setBatchTestSessions] = useState<BatchTestSession[]>([]);
  const [isSessionsLoading, setIsSessionsLoading] = useState(false);
  const [isResumingSession, setIsResumingSession] = useState(false);
  
  // Cancellation State
  const [currentTestId, setCurrentTestId] = useState<string | null>(null);
  const [isCancelling, setIsCancelling] = useState(false);

  // Test Dataset State
  const [testDatasets, setTestDatasets] = useState<any[]>([]);
  const [selectedDataset, setSelectedDataset] = useState<string>("");
  const [isDatasetsLoading, setIsDatasetsLoading] = useState(false);
  const [showSaveDatasetDialog, setShowSaveDatasetDialog] = useState(false);
  const [datasetName, setDatasetName] = useState("");
  const [datasetDescription, setDatasetDescription] = useState("");

  // Save Dialog State
  const [isSaveDialogOpen, setIsSaveDialogOpen] = useState(false);
  const [saveGroupName, setSaveGroupName] = useState("");
  const [isSaving, setIsSaving] = useState(false);
  const [saveDialogMode, setSaveDialogMode] = useState<"quick" | "batch">("quick");

  // View Result Dialog State
  const [viewingResult, setViewingResult] = useState<SemanticSimilarityResult | null>(null);

  // Temporary placeholders for removed features (to prevent errors)
  const selectedGroup = "";
  const savedResultsTotal = 0;
  const savedResultsGroups: any[] = [];
  const savedResultsAggregate: any = null;
  const statisticsModalResults: any[] = [];
  const isWandbRunsModalOpen = false;
  const setIsWandbRunsModalOpen = () => {};
  const isLoadingWandbRuns = false;
  const wandbRuns: any[] = [];
  const updatingRunIds = new Set<string>();
  const isStatisticsModalOpen = false;
  const setIsStatisticsModalOpen = () => {};
  const isLoadingStatisticsResults = false;
  const isPdfGenerating = false;
  const isWandbExporting = false;

  useEffect(() => {
    loadCourses();
  }, []);

  useEffect(() => {
    if (!isSettingsOpen) return;
  }, [isSettingsOpen]);

  // Save settings to localStorage when they change
  useEffect(() => {
    if (selectedEmbeddingModel) {
      localStorage.setItem('semantic_similarity_embedding_model', selectedEmbeddingModel);
    }
  }, [selectedEmbeddingModel]);

  useEffect(() => {
    if (selectedLlmProvider) {
      localStorage.setItem('semantic_similarity_llm_provider', selectedLlmProvider);
    }
  }, [selectedLlmProvider]);

  useEffect(() => {
    if (selectedLlmModel) {
      localStorage.setItem('semantic_similarity_llm_model', selectedLlmModel);
    }
  }, [selectedLlmModel]);

  useEffect(() => {
    if (selectedRerankerProvider) {
      localStorage.setItem('semantic_similarity_reranker_provider', selectedRerankerProvider);
    }
  }, [selectedRerankerProvider]);

  useEffect(() => {
    if (selectedRerankerModel) {
      localStorage.setItem('semantic_similarity_reranker_model', selectedRerankerModel);
    }
  }, [selectedRerankerModel]);

  useEffect(() => {
    localStorage.setItem('semantic_similarity_reranker_enabled', isRerankerEnabled.toString());
  }, [isRerankerEnabled]);

  useEffect(() => {
    localStorage.setItem('semantic_similarity_direct_llm_enabled', isDirectLlmEnabled.toString());
  }, [isDirectLlmEnabled]);

  useEffect(() => {
    if (selectedCourseId) {
      loadCourseSettings();
      loadBatchTestSessions();
      loadTestDatasets();
    }
  }, [selectedCourseId]);

  const exportSelectedGroupToWandb = async () => {
    if (!selectedCourseId) {
      toast.error("Lütfen bir ders seçin");
      return;
    }

    if (!selectedGroup || selectedGroup === "__all__" || selectedGroup === "__no_group__") {
      toast.error("W&B'ye göndermek için bir grup seçin");
      return;
    }

    setIsWandbExporting(true);
    try {
      toast.loading("W&B'ye aktarılıyor...", { id: "wandb-export" });
      const res = await api.wandbExportSemanticSimilarityGroup({
        course_id: selectedCourseId,
        group_name: selectedGroup,
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
    } finally {
      setIsWandbExporting(false);
    }
  };

  const loadWandbRuns = async () => {
    if (!selectedCourseId) {
      toast.error("Lütfen bir ders seçin");
      return;
    }
    setIsLoadingWandbRuns(true);
    try {
      const data = await api.getWandbRuns(selectedCourseId);
      setWandbRuns(data.runs);
    } catch (e) {
      const msg = e instanceof Error ? e.message : "W&B run’ları alınamadı";
      toast.error(msg);
    } finally {
      setIsLoadingWandbRuns(false);
    }
  };

  const updateSelectedWandbRun = async (run: {
    id: string;
    name: string;
    config: Record<string, unknown>;
    missing_fields: string[];
  }) => {
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

  const loadCourseSettings = async () => {
    if (!selectedCourseId) return;
    try {
      const settings = await api.getCourseSettings(selectedCourseId);
      console.log("Course settings loaded:", settings); // Debug log
      setSelectedEmbeddingModel(settings.default_embedding_model || "");
      // LLM ayarlarını da yükle (varsayılan olarak ders ayarlarını kullan)
      // Kullanıcı isterse ayarlardan değiştirebilir
      if (!selectedLlmProvider) {
        setSelectedLlmProvider(settings.llm_provider || "");
        setSelectedLlmModel(settings.llm_model || "");
      }
      // Reranker ayarlarını yükle
      setIsRerankerEnabled(settings.enable_reranker || false);
      setSelectedRerankerProvider(settings.reranker_provider || "");
      setSelectedRerankerModel(settings.reranker_model || "");
      // Direct LLM ayarını yükle
      setIsDirectLlmEnabled(settings.enable_direct_llm || false);
      console.log("Reranker settings:", {
        enabled: settings.enable_reranker,
        provider: settings.reranker_provider,
        model: settings.reranker_model
      }); // Debug log
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

  const loadBatchTestSessions = async () => {
    if (!selectedCourseId) return;
    setIsSessionsLoading(true);
    try {
      const data = await api.getBatchTestSessions(selectedCourseId);
      // Sadece in_progress ve failed durumundaki oturumları göster
      const filteredSessions = data.sessions.filter(session => 
        session.status === 'in_progress' || session.status === 'failed'
      );
      setBatchTestSessions(filteredSessions);
    } catch (error) {
      console.log("Failed to load batch test sessions:", error);
      toast.error("Batch test oturumları yüklenirken hata oluştu");
    } finally {
      setIsSessionsLoading(false);
    }
  };

  const loadTestDatasets = async () => {
    if (!selectedCourseId) return;
    setIsDatasetsLoading(true);
    try {
      const testSets = await api.getTestSets(selectedCourseId);
      const datasets = testSets.map(ts => ({
        id: ts.id,
        name: ts.name,
        description: ts.description,
        course_id: ts.course_id,
        total_test_cases: ts.question_count,
        question_count: ts.question_count,
        created_at: ts.created_at,
      }));
      setTestDatasets(datasets);
    } catch (error) {
      console.log("Failed to load test sets:", error);
      toast.error("Test setleri yüklenirken hata oluştu");
    } finally {
      setIsDatasetsLoading(false);
    }
  };

  const handleResumeSession = async (sessionId: number) => {
    if (!selectedCourseId) return;
    setIsResumingSession(true);
    batchTestStartTimeRef.current = Date.now();
    setBatchTestElapsedTime("00:00:00");
    try {
      // Get token from localStorage with correct key
      const token = localStorage.getItem('akilli_rehber_token');
      if (!token) {
        throw new Error('Oturum süresi dolmuş, lütfen tekrar giriş yapın');
      }
      
      // Use the resume endpoint to stream results
      const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'}/api/semantic-similarity/batch-test-sessions/${sessionId}/resume`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`
        }
      });

      if (!response.ok) {
        throw new Error('Oturum devam ettirilemedi');
      }

      const reader = response.body?.getReader();
      if (!reader) {
        throw new Error('Response body reader is not available');
      }
      
      const decoder = new TextDecoder();
      type StreamingResult = {
        question: string;
        ground_truth: string;
        generated_answer: string;
        similarity_score: number;
        best_match_ground_truth: string;
        latency_ms: number;
        retrieved_contexts?: string[];
        rouge1?: number | null;
        rouge2?: number | null;
        rougel?: number | null;
        bertscore_precision?: number | null;
        bertscore_recall?: number | null;
        bertscore_f1?: number | null;
        original_bertscore_precision?: number | null;
        original_bertscore_recall?: number | null;
        original_bertscore_f1?: number | null;
        hit_at_1?: number | null;
        mrr?: number | null;
        system_prompt_used?: string;
        error_message?: string;
      };

      const results: StreamingResult[] = [];
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        if (!value) continue;

        const chunk = decoder.decode(value, { stream: true });
        buffer += chunk;

        const lines = buffer.split("\n");
        buffer = lines.pop() ?? "";

        for (const line of lines) {
          if (!line.startsWith("data: ")) continue;
          const payload = line.slice(6);

          try {
            const data = JSON.parse(payload);

            if (data.event === 'progress') {
              results.push(data.result as StreamingResult);

              if (data.result?.error_message) {
                toast.error(
                  `Soru ${data.index + 1}/${data.total} başarısız: ${data.result.error_message}`
                );
              }

              // Calculate elapsed time
              if (batchTestStartTimeRef.current) {
                const elapsedMs = Date.now() - batchTestStartTimeRef.current;
                const elapsedSeconds = Math.floor(elapsedMs / 1000);
                const hours = Math.floor(elapsedSeconds / 3600);
                const minutes = Math.floor((elapsedSeconds % 3600) / 60);
                const seconds = elapsedSeconds % 60;
                setBatchTestElapsedTime(
                  `${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`
                );
              }

              const validResults = results.filter((r): r is StreamingResult => r.similarity_score != null);
              const rouge1Results = results.filter((r): r is StreamingResult => r.rouge1 != null);
              const rouge2Results = results.filter((r): r is StreamingResult => r.rouge2 != null);
              const rougelResults = results.filter((r): r is StreamingResult => r.rougel != null);
              const origBertPrecisionResults = results.filter((r): r is StreamingResult => r.original_bertscore_precision != null);
              const origBertRecallResults = results.filter((r): r is StreamingResult => r.original_bertscore_recall != null);
              const origBertF1Results = results.filter((r): r is StreamingResult => r.original_bertscore_f1 != null);

              setBatchTestResult({
                results: results as SemanticSimilarityBatchTestResponse['results'],
                aggregate: {
                  avg_similarity: validResults.reduce((sum, r) => sum + r.similarity_score!, 0) / validResults.length,
                  min_similarity: Math.min(...validResults.map(r => r.similarity_score!)),
                  max_similarity: Math.max(...validResults.map(r => r.similarity_score!)),
                  total_latency_ms: 0,
                  test_count: results.length,
                  avg_rouge1: rouge1Results.length > 0 ? rouge1Results.reduce((sum, r) => sum + r.rouge1!, 0) / rouge1Results.length : undefined,
                  avg_rouge2: rouge2Results.length > 0 ? rouge2Results.reduce((sum, r) => sum + r.rouge2!, 0) / rouge2Results.length : undefined,
                  avg_rougel: rougelResults.length > 0 ? rougelResults.reduce((sum, r) => sum + r.rougel!, 0) / rougelResults.length : undefined,
                  avg_original_bertscore_precision: origBertPrecisionResults.length > 0 ? origBertPrecisionResults.reduce((sum, r) => sum + r.original_bertscore_precision!, 0) / origBertPrecisionResults.length : undefined,
                  avg_original_bertscore_recall: origBertRecallResults.length > 0 ? origBertRecallResults.reduce((sum, r) => sum + r.original_bertscore_recall!, 0) / origBertRecallResults.length : undefined,
                  avg_original_bertscore_f1: origBertF1Results.length > 0 ? origBertF1Results.reduce((sum, r) => sum + r.original_bertscore_f1!, 0) / origBertF1Results.length : undefined,
                },
                embedding_model_used: data.embedding_model_used || '',
                llm_model_used: data.llm_model_used || undefined
              });
            } else if (data.event === 'complete') {
              const validResults = results.filter((r): r is StreamingResult => r.similarity_score != null);
              const rouge1Results = results.filter((r): r is StreamingResult => r.rouge1 != null);
              const rouge2Results = results.filter((r): r is StreamingResult => r.rouge2 != null);
              const rougelResults = results.filter((r): r is StreamingResult => r.rougel != null);
              const bertPrecisionResults = results.filter((r): r is StreamingResult => r.bertscore_precision != null);
              const bertRecallResults = results.filter((r): r is StreamingResult => r.bertscore_recall != null);
              const bertF1Results = results.filter((r): r is StreamingResult => r.bertscore_f1 != null);
              const origBertPrecisionResults = results.filter((r): r is StreamingResult => r.original_bertscore_precision != null);
              const origBertRecallResults = results.filter((r): r is StreamingResult => r.original_bertscore_recall != null);
              const origBertF1Results = results.filter((r): r is StreamingResult => r.original_bertscore_f1 != null);

              setBatchTestResult({
                results: results as SemanticSimilarityBatchTestResponse['results'],
                aggregate: {
                  avg_similarity: validResults.reduce((sum, r) => sum + r.similarity_score!, 0) / validResults.length,
                  min_similarity: Math.min(...validResults.map(r => r.similarity_score!)),
                  max_similarity: Math.max(...validResults.map(r => r.similarity_score!)),
                  total_latency_ms: 0,
                  test_count: results.length,
                  avg_rouge1: rouge1Results.length > 0 ? rouge1Results.reduce((sum, r) => sum + r.rouge1!, 0) / rouge1Results.length : undefined,
                  avg_rouge2: rouge2Results.length > 0 ? rouge2Results.reduce((sum, r) => sum + r.rouge2!, 0) / rouge2Results.length : undefined,
                  avg_rougel: rougelResults.length > 0 ? rougelResults.reduce((sum, r) => sum + r.rougel!, 0) / rougelResults.length : undefined,
                  avg_bertscore_precision: bertPrecisionResults.length > 0 ? bertPrecisionResults.reduce((sum, r) => sum + r.bertscore_precision!, 0) / bertPrecisionResults.length : undefined,
                  avg_bertscore_recall: bertRecallResults.length > 0 ? bertRecallResults.reduce((sum, r) => sum + r.bertscore_recall!, 0) / bertRecallResults.length : undefined,
                  avg_bertscore_f1: bertF1Results.length > 0 ? bertF1Results.reduce((sum, r) => sum + r.bertscore_f1!, 0) / bertF1Results.length : undefined,
                  avg_original_bertscore_precision: origBertPrecisionResults.length > 0 ? origBertPrecisionResults.reduce((sum, r) => sum + r.original_bertscore_precision!, 0) / origBertPrecisionResults.length : undefined,
                  avg_original_bertscore_recall: origBertRecallResults.length > 0 ? origBertRecallResults.reduce((sum, r) => sum + r.original_bertscore_recall!, 0) / origBertRecallResults.length : undefined,
                  avg_original_bertscore_f1: origBertF1Results.length > 0 ? origBertF1Results.reduce((sum, r) => sum + r.original_bertscore_f1!, 0) / origBertF1Results.length : undefined,
                },
                embedding_model_used: data.embedding_model_used || '',
                llm_model_used: data.llm_model_used || undefined
              });
              toast.success(`Oturum tamamlandı: ${data.completed}/${data.total}`);
              loadBatchTestSessions();
            } else if (data.event === 'error') {
              console.error('Test error:', data.error);
              loadBatchTestSessions();
            }
          } catch (jsonError) {
            // Log parse errors but don't crash - incomplete chunks will be completed in next iteration
            console.warn('SSE parse warning (will retry with next chunk):', {
              line: line.substring(0, 100) + (line.length > 100 ? '...' : ''),
              error: jsonError instanceof Error ? jsonError.message : String(jsonError)
            });
          }
        }
      }
    } catch (error) {
      console.error("Resume session error:", error);
      toast.error(error instanceof Error ? error.message : "Oturum devam ettirilemedi");
      loadBatchTestSessions();
    } finally {
      setIsResumingSession(false);
    }
  };

  const handleCancelSession = async (sessionId: number) => {
    if (!confirm("Bu oturumu iptal etmek istediğinizden emin misiniz?")) return;

    try {
      await api.cancelBatchTestSession(sessionId);
      toast.success("Oturum iptal edildi");
      loadBatchTestSessions();
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "İptal başarısız");
    }
  };

  const handleCancelCurrentTest = async () => {
    if (!currentTestId) {
      toast.error("Test henüz başlamadı veya test ID bulunamadı");
      return;
    }

    setIsCancelling(true);
    try {
      const token = localStorage.getItem('akilli_rehber_token');
      const response = await fetch(
        `${process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'}/api/semantic-similarity/cancel-test/${currentTestId}`,
        {
          method: 'POST',
          headers: {
            'Authorization': `Bearer ${token}`,
            'Content-Type': 'application/json'
          }
        }
      );

      if (!response.ok) {
        throw new Error('Test iptal edilemedi');
      }

      const data = await response.json();
      toast.success("Test iptal ediliyor...");
      setCurrentTestId(null);
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "İptal başarısız");
    } finally {
      setIsCancelling(false);
    }
  };

  const handleDeleteSession = async (sessionId: number) => {
    if (!confirm("Bu oturumu kalıcı olarak silmek istediğinizden emin misiniz? Bu işlem geri alınamaz.")) return;

    try {
      await api.deleteBatchTestSession(sessionId);
      toast.success("Oturum silindi");
      loadBatchTestSessions();
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "Silme başarısız");
    }
  };

  const getSessionStatusColor = (status: string) => {
    switch (status) {
      case 'completed':
        return 'bg-emerald-100 text-emerald-700';
      case 'in_progress':
        return 'bg-blue-100 text-blue-700';
      case 'failed':
        return 'bg-red-100 text-red-700';
      case 'cancelled':
        return 'bg-slate-100 text-slate-700';
      default:
        return 'bg-slate-100 text-slate-700';
    }
  };

  const getSessionStatusText = (status: string) => {
    switch (status) {
      case 'completed':
        return 'Tamamlandı';
      case 'in_progress':
        return 'Devam Ediyor';
      case 'failed':
        return 'Başarısız';
      case 'cancelled':
        return 'İptal Edildi';
      default:
        return status;
    }
  };

  const loadAllResultsForStatistics = async () => {
    if (!selectedCourseId) return;
    setIsLoadingStatisticsResults(true);
    try {
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
        10000 // Large limit to get all results
      );
      // Sort results alphabetically by question (Turkish locale)
      const sortedResults = [...data.results].sort((a, b) => 
        a.question.localeCompare(b.question, 'tr')
      );
      setStatisticsModalResults(sortedResults);
    } catch {
      console.log("Failed to load all results for statistics");
    } finally {
      setIsLoadingStatisticsResults(false);
    }
  };

  const copyTableToExcel = () => {
    const headers = ["#", "Soru", "Cosine Sim. (%)", "ROUGE-1 (%)", "ROUGE-2 (%)", "ROUGE-L (%)", "BERTScore P (%)", "BERTScore R (%)", "BERTScore F1 (%)", "Gecikme (ms)"];
    const rows = statisticsModalResults.map((r, idx) => [
      idx + 1,
        `"${r.question.replace(/"/g, '""')}"`,
        r.similarity_score != null ? (r.similarity_score * 100).toFixed(2) : "-",
        r.rouge1 != null ? (r.rouge1 * 100).toFixed(2) : "-",
        r.rouge2 != null ? (r.rouge2 * 100).toFixed(2) : "-",
        r.rougel != null ? (r.rougel * 100).toFixed(2) : "-",
        r.original_bertscore_precision != null ? (r.original_bertscore_precision * 100).toFixed(2) : "-",
        r.original_bertscore_recall != null ? (r.original_bertscore_recall * 100).toFixed(2) : "-",
        r.original_bertscore_f1 != null ? (r.original_bertscore_f1 * 100).toFixed(2) : "-",
        r.latency_ms
      ]);
    
    const csv = [headers.join("\t"), ...rows.map(row => row.join("\t"))].join("\n");
    navigator.clipboard.writeText(csv).then(() => {
      toast.success("Tablo kopyalandı! Excel'e yapıştırabilirsiniz.");
    }).catch(() => {
      toast.error("Kopyalama başarısız");
    });
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
        embedding_provider: selectedEmbeddingProvider || undefined,
        embedding_model: selectedEmbeddingModel || undefined,
        llm_provider: selectedLlmProvider || undefined,
        llm_model: selectedLlmModel || undefined,
        use_direct_llm: isDirectLlmEnabled ? true : undefined
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

    // Check if there's already an active session
    const activeSessions = batchTestSessions.filter(s => 
      s.status === 'in_progress' || s.status === 'failed'
    );
    
    if (activeSessions.length > 0) {
      const activeSession = activeSessions[0];
      if (confirm(
        `Bu ders için zaten aktif bir oturum var. Mevcut oturumu devam ettirmek ister misiniz?\n\nOturum: ${activeSession.group_name}\nDurum: ${activeSession.status}\nİlerleme: ${activeSession.completed_tests}/${activeSession.total_tests}`
      )) {
        handleResumeSession(activeSession.id);
        return;
      }
    }

    setIsBatchTesting(true);
    setBatchTestResult(null);
    setAllRunResults([]);
    setCurrentRunNumber(1);
    batchTestStartTimeRef.current = Date.now();
    setBatchTestElapsedTime("00:00:00");

    try {
      const parsedData = JSON.parse(batchTestJson);
      let testCases = parsedData;

      // Eğer RAGAS test set formatındaysa (questions array içinde), onu çıkar
      if (parsedData.questions && Array.isArray(parsedData.questions)) {
        testCases = parsedData.questions.map((q: any) => ({
          question: q.question,
          ground_truth: q.ground_truth,
          alternative_ground_truths: q.alternative_ground_truths,
          bloom_level: q.question_metadata?.bloom_level || null,
        }));
        // RAGAS'tan gelen name'i otomatik group name olarak ayarla
        if (parsedData.name && !saveGroupName) {
          setSaveGroupName(parsedData.name);
        }
      }

      // Normalize test cases: convert Turkish field names to English and fix data types
      testCases = testCases.map((tc: any) => ({
        question: tc.question || tc["Soru"],
        ground_truth: tc.ground_truth || tc["İdeal Cevap (GT)"],
        alternative_ground_truths: tc.alternative_ground_truths,
        generated_answer: tc.generated_answer,
        bloom_level: tc.bloom_level || tc["Bloom Seviyesi"] || null,
        question_metadata: tc.question_metadata,
        // Convert string context to array, handle Turkish field name
        expected_contexts: (() => {
          const ctx = tc.expected_contexts || tc["Bağlam (Context)"];
          if (!ctx) return null;
          if (Array.isArray(ctx)) return ctx;
          return [ctx]; // Wrap string in array
        })(),
        topic: tc.topic || tc["Konu"] || null,
        // Convert chunk_id to string, handle Turkish field name
        chunk_id: (() => {
          const cid = tc.chunk_id || tc["Chunk_ID"];
          if (cid === null || cid === undefined) return null;
          return String(cid);
        })(),
      }));

      console.log(`Starting ${totalRuns} repeated batch tests with ${testCases.length} test cases each`);
      
      const token = localStorage.getItem('akilli_rehber_token');
      if (!token) {
        throw new Error('Oturum süresi dolmuş, lütfen tekrar giriş yapın');
      }

      // Run the batch test multiple times using the NEW cancellable endpoint
      const allRunResults: SemanticSimilarityBatchTestResponse[] = [];
      
      for (let run = 1; run <= totalRuns; run++) {
        setCurrentRunNumber(run);
        
        toast.info(`Test başlatılıyor (${run}/${totalRuns})...`);
        
        // Run this single batch test using cancellable endpoint
        const runResult = await runCancellableBatchTest(testCases);
        allRunResults.push(runResult);
        
        // Small delay between runs
        if (run < totalRuns) {
          await new Promise(resolve => setTimeout(resolve, 1000));
        }
      }
      
      // Calculate combined statistics from all runs
      const combinedResults = allRunResults.flatMap(r => r.results);
      const combinedAggregate = calculateCombinedAggregate(allRunResults);
      
      setAllRunResults(allRunResults);
      setBatchTestResult({
        results: combinedResults,
        aggregate: combinedAggregate,
        embedding_model_used: selectedEmbeddingModel || '',
        llm_model_used: selectedLlmModel ? `${selectedLlmProvider}/${selectedLlmModel}` : undefined
      });
      
      toast.success(`Tüm ${totalRuns} test başarıyla tamamlandı!`);
      
    } catch (error) {
      console.error("Batch test error:", error);
      toast.error(error instanceof Error ? error.message : "Test başarısız");
      // Refresh the sessions list to update status
      loadBatchTestSessions();
    } finally {
      setIsBatchTesting(false);
      setCurrentTestId(null); // Clean up test ID
    }
  };

  // Test Dataset Handlers
  const handleSaveDataset = async () => {
    if (!selectedCourseId || !batchTestJson || !datasetName) {
      toast.error("Lütfen ders, JSON verisi ve veri seti adı girin");
      return;
    }

    try {
      const parsedData = JSON.parse(batchTestJson);
      let testCases = parsedData;

      // Eğer RAGAS test set formatındaysa (questions array içinde), onu çıkar
      if (parsedData.questions && Array.isArray(parsedData.questions)) {
        testCases = parsedData.questions.map((q: any) => ({
          question: q.question || q,
          ground_truth: q.ground_truth || q.answer || "",
          alternative_ground_truths: q.alternative_ground_truths || [],
        }));
      }

      // 1. Create test set via RAGAS API
      const testSet = await api.createTestSet({
        course_id: selectedCourseId,
        name: datasetName,
        description: datasetDescription,
      });

      // 2. Import questions into the test set
      const questions = testCases.map((tc: any) => ({
        question: tc.question,
        ground_truth: tc.ground_truth,
        alternative_ground_truths: tc.alternative_ground_truths || [],
        expected_contexts: tc.expected_contexts || [],
      }));

      await api.importQuestions(testSet.id, questions);

      toast.success("Veri seti başarıyla kaydedildi");
      setShowSaveDatasetDialog(false);
      setDatasetName("");
      setDatasetDescription("");
      loadTestDatasets();
    } catch (error) {
      console.error("Save dataset error:", error);
      toast.error(error instanceof Error ? error.message : "Veri seti kaydedilemedi");
    }
  };

  const handleLoadDataset = async (datasetId: string) => {
    if (!selectedCourseId) return;

    try {
      const testSet = await api.getTestSet(parseInt(datasetId));
      
      // Convert test set questions to test cases format and sort alphabetically
      const testCases = testSet.questions
        .sort((a, b) => a.question.localeCompare(b.question, 'tr-TR'))
        .map(q => ({
          question: q.question,
          ground_truth: q.ground_truth,
          alternative_ground_truths: q.alternative_ground_truths || [],
          expected_contexts: q.expected_contexts || [],
        }));
      
      setBatchTestJson(JSON.stringify(testCases, null, 2));
      setSelectedDataset(datasetId);
      toast.success(`"${testSet.name}" test seti yüklendi (${testCases.length} soru)`);
    } catch (error) {
      console.error("Load test set error:", error);
      toast.error(error instanceof Error ? error.message : "Test seti yüklenemedi");
    }
  };

  const runCancellableBatchTest = async (testCases: any[]): Promise<SemanticSimilarityBatchTestResponse> => {
    batchTestStartTimeRef.current = Date.now();
    setBatchTestElapsedTime("00:00:00");
    
    // Use the NEW cancellable endpoint
    const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'}/api/semantic-similarity/batch-test-stream-cancellable`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${localStorage.getItem('akilli_rehber_token')}`
      },
      body: JSON.stringify({
        course_id: selectedCourseId,
        test_cases: testCases,
        embedding_provider: selectedEmbeddingProvider || undefined,
        embedding_model: selectedEmbeddingModel || undefined,
        llm_provider: selectedLlmProvider || undefined,
        llm_model: selectedLlmModel || undefined,
        search_top_k: undefined,
        search_alpha: undefined,
        reranker_used: isRerankerEnabled,
        reranker_provider: isRerankerEnabled ? (selectedRerankerProvider || undefined) : undefined,
        reranker_model: isRerankerEnabled ? (selectedRerankerModel || undefined) : undefined,
        use_direct_llm: isDirectLlmEnabled ? true : undefined
      })
    });

    if (!response.ok) {
      throw new Error('Batch test başarısız');
    }

    const reader = response.body?.getReader();
    if (!reader) {
      throw new Error('Response body is not readable');
    }

    const decoder = new TextDecoder();
    type StreamingResult = {
      question: string;
      ground_truth: string;
      generated_answer: string;
      similarity_score: number;
      best_match_ground_truth: string;
      latency_ms: number;
      retrieved_contexts?: string[];
      rouge1?: number | null;
      rouge2?: number | null;
      rougel?: number | null;
      bertscore_precision?: number | null;
      bertscore_recall?: number | null;
      bertscore_f1?: number | null;
      original_bertscore_precision?: number | null;
      original_bertscore_recall?: number | null;
      original_bertscore_f1?: number | null;
      hit_at_1?: number | null;
      mrr?: number | null;
      system_prompt_used?: string;
      error_message?: string;
    };

    const results: StreamingResult[] = [];
    let buffer = "";
    let testId: string | null = null;
    
    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        if (!value) continue;

        const chunk = decoder.decode(value, { stream: true });
        buffer += chunk;

        const lines = buffer.split("\n");
        buffer = lines.pop() ?? "";

        for (const line of lines) {
          if (!line.startsWith("data: ")) continue;
          const payload = line.slice(6);

          try {
            const data = JSON.parse(payload);

            if (data.event === 'init') {
              // Store test_id for cancellation
              testId = data.test_id;
              setCurrentTestId(testId);
              console.log('Test started with ID:', testId);
            } else if (data.event === 'progress') {
              results.push(data.result as StreamingResult);

              // Debug: Log BERTScore values
              if (data.result) {
                console.log('Result BERTScore values:', {
                  question: data.result.question?.substring(0, 50),
                  bertscore_precision: data.result.bertscore_precision,
                  bertscore_recall: data.result.bertscore_recall,
                  bertscore_f1: data.result.bertscore_f1,
                  original_bertscore_precision: data.result.original_bertscore_precision,
                  original_bertscore_recall: data.result.original_bertscore_recall,
                  original_bertscore_f1: data.result.original_bertscore_f1,
                });
              }

              if (data.result?.error_message) {
                toast.error(
                  `Soru ${data.index + 1} başarısız: ${data.result.error_message}`
                );
              }

              // Calculate elapsed time
              if (batchTestStartTimeRef.current) {
                const elapsedMs = Date.now() - batchTestStartTimeRef.current;
                const elapsedSeconds = Math.floor(elapsedMs / 1000);
                const hours = Math.floor(elapsedSeconds / 3600);
                const minutes = Math.floor((elapsedSeconds % 3600) / 60);
                const seconds = elapsedSeconds % 60;
                setBatchTestElapsedTime(
                  `${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`
                );
              }

              const validResults = results.filter((r): r is StreamingResult => r.similarity_score != null);
              const rouge1Results = results.filter((r): r is StreamingResult => r.rouge1 != null);
              const rouge2Results = results.filter((r): r is StreamingResult => r.rouge2 != null);
              const rougelResults = results.filter((r): r is StreamingResult => r.rougel != null);
              const origBertPrecisionResults = results.filter((r): r is StreamingResult => r.original_bertscore_precision != null);
              const origBertRecallResults = results.filter((r): r is StreamingResult => r.original_bertscore_recall != null);
              const origBertF1Results = results.filter((r): r is StreamingResult => r.original_bertscore_f1 != null);

              setBatchTestResult({
                results: results as SemanticSimilarityBatchTestResponse['results'],
                aggregate: {
                  avg_similarity: validResults.reduce((sum, r) => sum + r.similarity_score!, 0) / validResults.length,
                  min_similarity: Math.min(...validResults.map(r => r.similarity_score!)),
                  max_similarity: Math.max(...validResults.map(r => r.similarity_score!)),
                  total_latency_ms: 0,
                  test_count: results.length,
                  avg_rouge1: rouge1Results.length > 0 ? rouge1Results.reduce((sum, r) => sum + r.rouge1!, 0) / rouge1Results.length : undefined,
                  avg_rouge2: rouge2Results.length > 0 ? rouge2Results.reduce((sum, r) => sum + r.rouge2!, 0) / rouge2Results.length : undefined,
                  avg_rougel: rougelResults.length > 0 ? rougelResults.reduce((sum, r) => sum + r.rougel!, 0) / rougelResults.length : undefined,
                  avg_original_bertscore_precision: origBertPrecisionResults.length > 0 ? origBertPrecisionResults.reduce((sum, r) => sum + r.original_bertscore_precision!, 0) / origBertPrecisionResults.length : undefined,
                  avg_original_bertscore_recall: origBertRecallResults.length > 0 ? origBertRecallResults.reduce((sum, r) => sum + r.original_bertscore_recall!, 0) / origBertRecallResults.length : undefined,
                  avg_original_bertscore_f1: origBertF1Results.length > 0 ? origBertF1Results.reduce((sum, r) => sum + r.original_bertscore_f1!, 0) / origBertF1Results.length : undefined,
                },
                embedding_model_used: selectedEmbeddingModel || '',
                llm_model_used: selectedLlmModel ? `${selectedLlmProvider}/${selectedLlmModel}` : undefined
              });
            } else if (data.event === 'complete') {
              const validResults = results.filter((r): r is StreamingResult => r.similarity_score != null);
              const rouge1Results = results.filter((r): r is StreamingResult => r.rouge1 != null);
              const rouge2Results = results.filter((r): r is StreamingResult => r.rouge2 != null);
              const rougelResults = results.filter((r): r is StreamingResult => r.rougel != null);
              const origBertPrecisionResults = results.filter((r): r is StreamingResult => r.original_bertscore_precision != null);
              const origBertRecallResults = results.filter((r): r is StreamingResult => r.original_bertscore_recall != null);
              const origBertF1Results = results.filter((r): r is StreamingResult => r.original_bertscore_f1 != null);

              const finalResult: SemanticSimilarityBatchTestResponse = {
                results: results as SemanticSimilarityBatchTestResponse['results'],
                aggregate: {
                  avg_similarity: validResults.reduce((sum, r) => sum + r.similarity_score!, 0) / validResults.length,
                  min_similarity: Math.min(...validResults.map(r => r.similarity_score!)),
                  max_similarity: Math.max(...validResults.map(r => r.similarity_score!)),
                  total_latency_ms: 0,
                  test_count: testCases.length,
                  avg_rouge1: rouge1Results.length > 0 ? rouge1Results.reduce((sum, r) => sum + r.rouge1!, 0) / rouge1Results.length : undefined,
                  avg_rouge2: rouge2Results.length > 0 ? rouge2Results.reduce((sum, r) => sum + r.rouge2!, 0) / rouge2Results.length : undefined,
                  avg_rougel: rougelResults.length > 0 ? rougelResults.reduce((sum, r) => sum + r.rougel!, 0) / rougelResults.length : undefined,
                  avg_original_bertscore_precision: origBertPrecisionResults.length > 0 ? origBertPrecisionResults.reduce((sum, r) => sum + r.original_bertscore_precision!, 0) / origBertPrecisionResults.length : undefined,
                  avg_original_bertscore_recall: origBertRecallResults.length > 0 ? origBertRecallResults.reduce((sum, r) => sum + r.original_bertscore_recall!, 0) / origBertRecallResults.length : undefined,
                  avg_original_bertscore_f1: origBertF1Results.length > 0 ? origBertF1Results.reduce((sum, r) => sum + r.original_bertscore_f1!, 0) / origBertF1Results.length : undefined,
                },
                embedding_model_used: data.embedding_model_used || selectedEmbeddingModel || '',
                llm_model_used: data.llm_model_used || (selectedLlmModel ? `${selectedLlmProvider}/${selectedLlmModel}` : undefined)
              };
              
              toast.success(`Test tamamlandı: ${data.completed}/${data.total}`);
              setCurrentTestId(null);
              
              // Auto-save results with batch endpoint (single request)
              try {
                const autoGroupName = `batch_${new Date().toISOString().replace(/[:.]/g, '-').slice(0, -5)}`;
                const settings = await api.getCourseSettings(selectedCourseId!);
                const isDirectLlm = isDirectLlmEnabled || settings.enable_direct_llm;
                
                // Filter only successful results (with valid similarity_score)
                const validResults = results.filter(r => r.similarity_score != null && r.similarity_score !== undefined);
                
                if (validResults.length > 0) {
                  const batchResponse = await api.saveSemanticSimilarityResultsBatch({
                    course_id: selectedCourseId!,
                    group_name: autoGroupName,
                    results: validResults.map(result => ({
                      question: result.question,
                      ground_truth: result.ground_truth,
                      generated_answer: result.generated_answer,
                      similarity_score: result.similarity_score,
                      best_match_ground_truth: result.best_match_ground_truth || result.ground_truth,
                      rouge1: result.rouge1 ?? undefined,
                      rouge2: result.rouge2 ?? undefined,
                      rougel: result.rougel ?? undefined,
                      bertscore_precision: result.bertscore_precision ?? undefined,
                      bertscore_recall: result.bertscore_recall ?? undefined,
                      bertscore_f1: result.bertscore_f1 ?? undefined,
                      original_bertscore_precision: result.original_bertscore_precision ?? undefined,
                      original_bertscore_recall: result.original_bertscore_recall ?? undefined,
                      original_bertscore_f1: result.original_bertscore_f1 ?? undefined,
                      latency_ms: result.latency_ms || 0,
                      embedding_model_used: isDirectLlm ? "N/A (Direct LLM)" : finalResult.embedding_model_used,
                      llm_model_used: finalResult.llm_model_used,
                      retrieved_contexts: result.retrieved_contexts,
                      system_prompt_used: isDirectLlm ? "Direct LLM - No system prompt" : result.system_prompt_used,
                      search_top_k: isDirectLlm ? undefined : settings.search_top_k,
                      search_alpha: isDirectLlm ? undefined : settings.search_alpha,
                      reranker_used: isDirectLlm ? false : settings.enable_reranker,
                      reranker_provider: isDirectLlm ? undefined : settings.reranker_provider,
                      reranker_model: isDirectLlm ? undefined : settings.reranker_model,
                    })),
                  });
                  
                  toast.success(`${batchResponse.saved_count} sonuç otomatik kaydedildi (Grup: ${autoGroupName})`);
                  if (batchResponse.failed_count > 0) {
                    toast.warning(`${batchResponse.failed_count} sonuç kaydedilemedi`);
                  }
                } else {
                  toast.warning("Kaydedilecek geçerli sonuç bulunamadı");
                }
              } catch (autoSaveError) {
                console.error("Auto-save failed:", autoSaveError);
                toast.warning("Sonuçlar otomatik kaydedilemedi. Manuel kaydetmeyi deneyin.");
              }
              
              return finalResult;
            } else if (data.event === 'cancelled') {
              toast.warning(`Test iptal edildi: ${data.completed}/${data.total} tamamlandı`);
              setCurrentTestId(null);
              throw new Error('Test cancelled by user');
            } else if (data.event === 'error') {
              console.error('Test error:', data.error);
              setCurrentTestId(null);
              throw new Error(data.error);
            }
          } catch (jsonError) {
            console.warn('SSE parse warning:', jsonError);
          }
        }
      }
      
      // If we get here without a complete event, throw error
      throw new Error('Test completed without final event');
    } catch (error) {
      console.error("Cancellable batch test error:", error);
      setCurrentTestId(null);
      throw error;
    }
  };

  const runSingleBatchTest = async (sessionId: number, runNumber: number, testCasesCount: number): Promise<SemanticSimilarityBatchTestResponse> => {
    batchTestStartTimeRef.current = Date.now();
    setBatchTestElapsedTime("00:00:00");
    
    // Use the resume endpoint to stream results
    const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'}/api/semantic-similarity/batch-test-sessions/${sessionId}/resume`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${localStorage.getItem('akilli_rehber_token')}`
      }
    });

    if (!response.ok) {
      throw new Error('Batch test başarısız');
    }

    const reader = response.body?.getReader();
    if (!reader) {
      throw new Error('Response body is not readable');
    }

    const decoder = new TextDecoder();
    type StreamingResult = {
      question: string;
      ground_truth: string;
      generated_answer: string;
      similarity_score: number;
      best_match_ground_truth: string;
      latency_ms: number;
      retrieved_contexts?: string[];
      rouge1?: number | null;
      rouge2?: number | null;
      rougel?: number | null;
      bertscore_precision?: number | null;
      bertscore_recall?: number | null;
      bertscore_f1?: number | null;
      hit_at_1?: number | null;
      mrr?: number | null;
      system_prompt_used?: string;
      error_message?: string;
    };

    const results: StreamingResult[] = [];
    let buffer = "";
    
    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        if (!value) continue;

        const chunk = decoder.decode(value, { stream: true });
        buffer += chunk;

        const lines = buffer.split("\n");
        buffer = lines.pop() ?? "";

        for (const line of lines) {
          if (!line.startsWith("data: ")) continue;
          const payload = line.slice(6);

          try {
            const data = JSON.parse(payload);

            if (data.event === 'progress') {
              results.push(data.result as StreamingResult);

              if (data.result?.error_message) {
                toast.error(
                  `Soru ${data.index + 1}/${data.total} başarısız: ${data.result.error_message}`
                );
              }

              // Calculate elapsed time
              if (batchTestStartTimeRef.current) {
                const elapsedMs = Date.now() - batchTestStartTimeRef.current;
                const elapsedSeconds = Math.floor(elapsedMs / 1000);
                const hours = Math.floor(elapsedSeconds / 3600);
                const minutes = Math.floor((elapsedSeconds % 3600) / 60);
                const seconds = elapsedSeconds % 60;
                setBatchTestElapsedTime(
                  `${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`
                );
              }

              const validResults = results.filter((r): r is StreamingResult => r.similarity_score != null);
              const rouge1Results = results.filter((r): r is StreamingResult => r.rouge1 != null);
              const rouge2Results = results.filter((r): r is StreamingResult => r.rouge2 != null);
              const rougelResults = results.filter((r): r is StreamingResult => r.rougel != null);
              const origBertPrecisionResults = results.filter((r): r is StreamingResult => r.original_bertscore_precision != null);
              const origBertRecallResults = results.filter((r): r is StreamingResult => r.original_bertscore_recall != null);
              const origBertF1Results = results.filter((r): r is StreamingResult => r.original_bertscore_f1 != null);

              setBatchTestResult({
                results: results as SemanticSimilarityBatchTestResponse['results'],
                aggregate: {
                  avg_similarity: validResults.reduce((sum, r) => sum + r.similarity_score!, 0) / validResults.length,
                  min_similarity: Math.min(...validResults.map(r => r.similarity_score!)),
                  max_similarity: Math.max(...validResults.map(r => r.similarity_score!)),
                  total_latency_ms: 0,
                  test_count: results.length,
                  avg_rouge1: rouge1Results.length > 0 ? rouge1Results.reduce((sum, r) => sum + r.rouge1!, 0) / rouge1Results.length : undefined,
                  avg_rouge2: rouge2Results.length > 0 ? rouge2Results.reduce((sum, r) => sum + r.rouge2!, 0) / rouge2Results.length : undefined,
                  avg_rougel: rougelResults.length > 0 ? rougelResults.reduce((sum, r) => sum + r.rougel!, 0) / rougelResults.length : undefined,
                  avg_original_bertscore_precision: origBertPrecisionResults.length > 0 ? origBertPrecisionResults.reduce((sum, r) => sum + r.original_bertscore_precision!, 0) / origBertPrecisionResults.length : undefined,
                  avg_original_bertscore_recall: origBertRecallResults.length > 0 ? origBertRecallResults.reduce((sum, r) => sum + r.original_bertscore_recall!, 0) / origBertRecallResults.length : undefined,
                  avg_original_bertscore_f1: origBertF1Results.length > 0 ? origBertF1Results.reduce((sum, r) => sum + r.original_bertscore_f1!, 0) / origBertF1Results.length : undefined,
                },
                embedding_model_used: data.embedding_model_used || selectedEmbeddingModel || '',
                llm_model_used: data.llm_model_used || (selectedLlmModel ? `${selectedLlmProvider}/${selectedLlmModel}` : undefined)
              });
            } else if (data.event === 'complete') {
              const validResults = results.filter((r): r is StreamingResult => r.similarity_score != null);
              const rouge1Results = results.filter((r): r is StreamingResult => r.rouge1 != null);
              const rouge2Results = results.filter((r): r is StreamingResult => r.rouge2 != null);
              const rougelResults = results.filter((r): r is StreamingResult => r.rougel != null);
              const origBertPrecisionResults = results.filter((r): r is StreamingResult => r.original_bertscore_precision != null);
              const origBertRecallResults = results.filter((r): r is StreamingResult => r.original_bertscore_recall != null);
              const origBertF1Results = results.filter((r): r is StreamingResult => r.original_bertscore_f1 != null);

              const finalResult: SemanticSimilarityBatchTestResponse = {
                results: results as SemanticSimilarityBatchTestResponse['results'],
                aggregate: {
                  avg_similarity: validResults.reduce((sum, r) => sum + r.similarity_score!, 0) / validResults.length,
                  min_similarity: Math.min(...validResults.map(r => r.similarity_score!)),
                  max_similarity: Math.max(...validResults.map(r => r.similarity_score!)),
                  total_latency_ms: 0,
                  test_count: testCasesCount,
                  avg_rouge1: rouge1Results.length > 0 ? rouge1Results.reduce((sum, r) => sum + r.rouge1!, 0) / rouge1Results.length : undefined,
                  avg_rouge2: rouge2Results.length > 0 ? rouge2Results.reduce((sum, r) => sum + r.rouge2!, 0) / rouge2Results.length : undefined,
                  avg_rougel: rougelResults.length > 0 ? rougelResults.reduce((sum, r) => sum + r.rougel!, 0) / rougelResults.length : undefined,
                  avg_original_bertscore_precision: origBertPrecisionResults.length > 0 ? origBertPrecisionResults.reduce((sum, r) => sum + r.original_bertscore_precision!, 0) / origBertPrecisionResults.length : undefined,
                  avg_original_bertscore_recall: origBertRecallResults.length > 0 ? origBertRecallResults.reduce((sum, r) => sum + r.original_bertscore_recall!, 0) / origBertRecallResults.length : undefined,
                  avg_original_bertscore_f1: origBertF1Results.length > 0 ? origBertF1Results.reduce((sum, r) => sum + r.original_bertscore_f1!, 0) / origBertF1Results.length : undefined,
                },
                embedding_model_used: data.embedding_model_used || selectedEmbeddingModel || '',
                llm_model_used: data.llm_model_used || (selectedLlmModel ? `${selectedLlmProvider}/${selectedLlmModel}` : undefined)
              };
              
              toast.success(`Run ${runNumber} tamamlandı: ${data.completed}/${data.total}`);
              loadBatchTestSessions();
              return finalResult;
            } else if (data.event === 'error') {
              console.error('Test error:', data.error);
              loadBatchTestSessions();
              throw new Error('Test başarısız');
            }
          } catch (jsonError) {
            // Log parse errors but don't crash - incomplete chunks will be completed in next iteration
            console.warn('SSE parse warning (will retry with next chunk):', {
              line: line.substring(0, 100) + (line.length > 100 ? '...' : ''),
              error: jsonError instanceof Error ? jsonError.message : String(jsonError)
            });
          }
        }
      }
    } catch (error) {
      console.error("Batch test error:", error);
      throw error;
    }
  };

  const calculateCombinedAggregate = (runs: SemanticSimilarityBatchTestResponse[]): SemanticSimilarityBatchTestResponse['aggregate'] => {
    // Combine all results from all runs
    const allResults = runs.flatMap(r => r.results);
    
    const validResults = allResults.filter((r): r is SemanticSimilarityBatchTestResponse['results'][number] => r.similarity_score != null);
    const rouge1Results = allResults.filter((r): r is SemanticSimilarityBatchTestResponse['results'][number] => r.rouge1 != null);
    const rouge2Results = allResults.filter((r): r is SemanticSimilarityBatchTestResponse['results'][number] => r.rouge2 != null);
    const rougelResults = allResults.filter((r): r is SemanticSimilarityBatchTestResponse['results'][number] => r.rougel != null);
    const bertPrecisionResults = allResults.filter((r): r is SemanticSimilarityBatchTestResponse['results'][number] => r.bertscore_precision != null);
    const bertRecallResults = allResults.filter((r): r is SemanticSimilarityBatchTestResponse['results'][number] => r.bertscore_recall != null);
    const bertF1Results = allResults.filter((r): r is SemanticSimilarityBatchTestResponse['results'][number] => r.bertscore_f1 != null);
    const origBertPrecisionResults = allResults.filter((r): r is SemanticSimilarityBatchTestResponse['results'][number] => r.original_bertscore_precision != null);
    const origBertRecallResults = allResults.filter((r): r is SemanticSimilarityBatchTestResponse['results'][number] => r.original_bertscore_recall != null);
    const origBertF1Results = allResults.filter((r): r is SemanticSimilarityBatchTestResponse['results'][number] => r.original_bertscore_f1 != null);

    return {
      avg_similarity: validResults.reduce((sum, r) => sum + r.similarity_score!, 0) / validResults.length,
      min_similarity: Math.min(...validResults.map(r => r.similarity_score!)),
      max_similarity: Math.max(...validResults.map(r => r.similarity_score!)),
      total_latency_ms: 0,
      test_count: allResults.length,
      avg_rouge1: rouge1Results.length > 0 ? rouge1Results.reduce((sum, r) => sum + r.rouge1!, 0) / rouge1Results.length : undefined,
      avg_rouge2: rouge2Results.length > 0 ? rouge2Results.reduce((sum, r) => sum + r.rouge2!, 0) / rouge2Results.length : undefined,
      avg_rougel: rougelResults.length > 0 ? rougelResults.reduce((sum, r) => sum + r.rougel!, 0) / rougelResults.length : undefined,
      avg_bertscore_precision: bertPrecisionResults.length > 0 ? bertPrecisionResults.reduce((sum, r) => sum + r.bertscore_precision!, 0) / bertPrecisionResults.length : undefined,
      avg_bertscore_recall: bertRecallResults.length > 0 ? bertRecallResults.reduce((sum, r) => sum + r.bertscore_recall!, 0) / bertRecallResults.length : undefined,
      avg_bertscore_f1: bertF1Results.length > 0 ? bertF1Results.reduce((sum, r) => sum + r.bertscore_f1!, 0) / bertF1Results.length : undefined,
      avg_original_bertscore_precision: origBertPrecisionResults.length > 0 ? origBertPrecisionResults.reduce((sum, r) => sum + r.original_bertscore_precision!, 0) / origBertPrecisionResults.length : undefined,
      avg_original_bertscore_recall: origBertRecallResults.length > 0 ? origBertRecallResults.reduce((sum, r) => sum + r.original_bertscore_recall!, 0) / origBertRecallResults.length : undefined,
      avg_original_bertscore_f1: origBertF1Results.length > 0 ? origBertF1Results.reduce((sum, r) => sum + r.original_bertscore_f1!, 0) / origBertF1Results.length : undefined,
    };
  };

  const handleSaveResult = async () => {
    if (!selectedCourseId || !quickTestResult) return;

    setIsSaving(true);
    try {
      // Get course settings for search and reranker config
      const settings = await api.getCourseSettings(selectedCourseId);
      const isDirectLlm = isDirectLlmEnabled || settings.enable_direct_llm;
      
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
        original_bertscore_precision: quickTestResult.original_bertscore_precision,
        original_bertscore_recall: quickTestResult.original_bertscore_recall,
        original_bertscore_f1: quickTestResult.original_bertscore_f1,
        latency_ms: quickTestResult.latency_ms,
        embedding_model_used: isDirectLlm ? "N/A (Direct LLM)" : quickTestResult.embedding_model_used,
        llm_model_used: quickTestResult.llm_model_used,
        retrieved_contexts: quickTestResult.retrieved_contexts,
        system_prompt_used: isDirectLlm ? "Direct LLM - No system prompt" : quickTestResult.system_prompt_used,
        search_top_k: isDirectLlm ? undefined : settings.search_top_k,
        search_alpha: isDirectLlm ? undefined : settings.search_alpha,
        reranker_used: isDirectLlm ? false : settings.enable_reranker,
        reranker_provider: isDirectLlm ? undefined : settings.reranker_provider,
        reranker_model: isDirectLlm ? undefined : settings.reranker_model
      });
      toast.success("Sonuç kaydedildi");
      setIsSaveDialogOpen(false);
      setSaveGroupName("");
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
      // Get course settings for search and reranker config
      const settings = await api.getCourseSettings(selectedCourseId);
      const isDirectLlm = isDirectLlmEnabled || settings.enable_direct_llm;
      
      // Filter only valid results (with similarity_score)
      const validResults = batchTestResult.results.filter(r => r.similarity_score != null);
      
      if (validResults.length === 0) {
        toast.error("Kaydedilecek geçerli sonuç bulunamadı");
        return;
      }

      const batchResponse = await api.saveSemanticSimilarityResultsBatch({
        course_id: selectedCourseId,
        group_name: saveGroupName || undefined,
        results: validResults.map(result => ({
          question: result.question,
          ground_truth: result.ground_truth,
          generated_answer: result.generated_answer,
          similarity_score: result.similarity_score,
          best_match_ground_truth: result.best_match_ground_truth || result.ground_truth,
          rouge1: result.rouge1 ?? undefined,
          rouge2: result.rouge2 ?? undefined,
          rougel: result.rougel ?? undefined,
          bertscore_precision: result.bertscore_precision ?? undefined,
          bertscore_recall: result.bertscore_recall ?? undefined,
          bertscore_f1: result.bertscore_f1 ?? undefined,
          original_bertscore_precision: result.original_bertscore_precision ?? undefined,
          original_bertscore_recall: result.original_bertscore_recall ?? undefined,
          original_bertscore_f1: result.original_bertscore_f1 ?? undefined,
          latency_ms: result.latency_ms || 0,
          embedding_model_used: isDirectLlm ? "N/A (Direct LLM)" : (batchTestResult.embedding_model_used || selectedEmbeddingModel || "unknown"),
          llm_model_used: batchTestResult.llm_model_used,
          retrieved_contexts: result.retrieved_contexts,
          system_prompt_used: isDirectLlm ? "Direct LLM - No system prompt" : result.system_prompt_used,
          search_top_k: isDirectLlm ? undefined : settings.search_top_k,
          search_alpha: isDirectLlm ? undefined : settings.search_alpha,
          reranker_used: isDirectLlm ? false : settings.enable_reranker,
          reranker_provider: isDirectLlm ? undefined : settings.reranker_provider,
          reranker_model: isDirectLlm ? undefined : settings.reranker_model,
        })),
      });

      if (batchResponse.saved_count > 0) {
        toast.success(`${batchResponse.saved_count} sonuç kaydedildi${batchResponse.failed_count > 0 ? `, ${batchResponse.failed_count} başarısız` : ""}`);
      } else {
        toast.error("Hiçbir sonuç kaydedilemedi");
      }

      setIsSaveDialogOpen(false);
      setSaveGroupName("");
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
        "Cosine Sim. (%)",
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
        r.similarity_score != null ? (r.similarity_score * 100).toFixed(2) : "-",
        r.rouge1 != null ? (r.rouge1 * 100).toFixed(2) : "-",
        r.rouge2 != null ? (r.rouge2 * 100).toFixed(2) : "-",
        r.rougel != null ? (r.rougel * 100).toFixed(2) : "-",
        r.original_bertscore_precision != null ? (r.original_bertscore_precision * 100).toFixed(2) : "-",
        r.original_bertscore_recall != null ? (r.original_bertscore_recall * 100).toFixed(2) : "-",
        r.original_bertscore_f1 != null ? (r.original_bertscore_f1 * 100).toFixed(2) : "-",
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

  const getMetricColor = (value?: number | null) => {
    if (value === undefined || value === null) return "text-slate-400";
    if (value >= 0.8) return "text-emerald-600";
    if (value >= 0.6) return "text-amber-600";
    return "text-red-600";
  };

  const getMetricBgColor = (value?: number | null) => {
    if (value === undefined || value === null) return "bg-slate-50 border-slate-200";
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
              <Button
                variant="secondary"
                size="sm"
                className="bg-white/20 hover:bg-white/30 text-white border-0 backdrop-blur-sm h-10"
                onClick={() => window.location.href = '/dashboard/semantic-similarity/results'}
              >
                <History className="w-4 h-4 mr-2" />Sonuçlar
              </Button>
              <Button
                variant="secondary"
                size="sm"
                className="bg-white/20 hover:bg-white/30 text-white border-0 backdrop-blur-sm h-10"
                onClick={() => window.location.href = '/dashboard/semantic-similarity/analysis'}
              >
                <BarChart3 className="w-4 h-4 mr-2" />Analiz
              </Button>
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
                          {getEmbeddingModels().map((model) => {
                            const [provider, modelName] = model.split('/');
                            const displayName = modelName.replace(/-/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
                            const dimensions = model.includes('v2') ? '768 dim' : 
                                             model.includes('v3') ? '1024 dim' :
                                             model.includes('v4') ? '1024 dim' :
                                             model.includes('small') ? '1536 dim' :
                                             model.includes('large') ? '3072 dim' :
                                             model.includes('ada') ? '1536 dim' :
                                             model.includes('light') ? '384 dim' :
                                             model.includes('8b') ? '1024 dim' : '';
                            
                            return (
                              <SelectItem key={model} value={model}>
                                {provider.charAt(0).toUpperCase() + provider.slice(1)} {displayName} ({dimensions})
                              </SelectItem>
                            );
                          })}
                        </SelectContent>
                      </Select>
                    </div>
                    <div className="space-y-2">
                      <Label className="text-sm font-medium">LLM Provider</Label>
                      <Select value={selectedLlmProvider || ""} onValueChange={(v) => { setSelectedLlmProvider(v); setSelectedLlmModel(""); }}>
                        <SelectTrigger className="h-11"><SelectValue placeholder="Provider seçin" /></SelectTrigger>
                        <SelectContent>
                          {getLLMProviders().map((provider) => (
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
                          {getLLMModels(selectedLlmProvider).map((model) => (
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
                    <div className="border-t pt-4 mt-2">
                      <div className="flex items-center justify-between">
                        <div className="space-y-1">
                          <Label className="text-sm font-medium">Direct LLM Modu</Label>
                          <p className="text-xs text-slate-500">RAG pipeline devre dışı, LLM doğrudan yanıt verir</p>
                        </div>
                        <label className="relative inline-flex items-center cursor-pointer">
                          <input
                            type="checkbox"
                            checked={isDirectLlmEnabled}
                            onChange={(e) => setIsDirectLlmEnabled(e.target.checked)}
                            className="sr-only peer"
                          />
                          <div className="w-11 h-6 bg-gray-200 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-amber-100 rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-amber-500"></div>
                        </label>
                      </div>
                      {isDirectLlmEnabled && (
                        <div className="mt-2 p-2 rounded bg-amber-50 border border-amber-200 text-xs text-amber-700">
                          Doküman bağlamı kullanılmayacak. RAG vs yalın LLM karşılaştırması için idealdir.
                        </div>
                      )}
                    </div>
                  </div>
                  <DialogFooter>
                    <Button variant="outline" onClick={() => setIsSettingsOpen(false)}>İptal</Button>
                    <Button onClick={handleSaveSettings} disabled={isSavingSettings} className="bg-teal-600 hover:bg-teal-700">{isSavingSettings ? <><Loader2 className="w-4 h-4 mr-2 animate-spin" />Kaydediliyor...</> : <><Save className="w-4 h-4 mr-2" />Kaydet</>}</Button>
                  </DialogFooter>
                </DialogContent>
              </Dialog>

              <Select value={selectedCourseId?.toString() || ""} onValueChange={(v) => {
                const courseId = Number(v);
                setSelectedCourseId(courseId);
                localStorage.setItem('semantic_similarity_selected_course_id', courseId.toString());
              }}>
                <SelectTrigger className="w-80 bg-white/20 border-0 text-white hover:bg-white/30 backdrop-blur-sm h-10"><BookOpen className="w-4 h-4 mr-2" /><SelectValue placeholder="Ders seçin" /></SelectTrigger>
                <SelectContent>{courses.map((course) => (<SelectItem key={course.id} value={course.id.toString()}>{course.name}</SelectItem>))}</SelectContent>
              </Select>
            </div>
          </div>
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
                        {(quickTestResult.similarity_score != null || quickTestResult.rouge1 != null || quickTestResult.rouge2 != null || quickTestResult.rougel != null || 
                          quickTestResult.bertscore_f1 != null || quickTestResult.original_bertscore_f1 != null) && (
                          <div>
                            <Label className="text-sm font-medium text-slate-700">
                              Ek Metrikler
                            </Label>
                            <div className="mt-2 grid grid-cols-2 gap-3">
                              {quickTestResult.similarity_score != null && (
                                <div className="p-4 bg-gradient-to-br from-blue-50 to-blue-100 rounded-xl border border-blue-200 shadow-sm">
                                  <p className="text-xs text-blue-600 font-medium">Cosine Similarity</p>
                                  <p className={`text-2xl font-bold ${getMetricColor(quickTestResult.similarity_score)}`}>
                                    {(quickTestResult.similarity_score * 100).toFixed(1)}%
                                  </p>
                                </div>
                              )}
                              {quickTestResult.rouge1 != null && (
                                <div className="p-4 bg-gradient-to-br from-purple-50 to-purple-100 rounded-xl border border-purple-200 shadow-sm">
                                  <p className="text-xs text-purple-600 font-medium">ROUGE-1</p>
                                  <p className={`text-2xl font-bold ${getMetricColor(quickTestResult.rouge1)}`}>
                                    {(quickTestResult.rouge1 * 100).toFixed(1)}%
                                  </p>
                                </div>
                              )}
                              {quickTestResult.rouge2 != null && (
                                <div className="p-4 bg-gradient-to-br from-purple-50 to-purple-100 rounded-xl border border-purple-200 shadow-sm">
                                  <p className="text-xs text-purple-600 font-medium">ROUGE-2</p>
                                  <p className={`text-2xl font-bold ${getMetricColor(quickTestResult.rouge2)}`}>
                                    {(quickTestResult.rouge2 * 100).toFixed(1)}%
                                  </p>
                                </div>
                              )}
                              {quickTestResult.rougel != null && (
                                <div className="p-4 bg-gradient-to-br from-purple-50 to-purple-100 rounded-xl border border-purple-200 shadow-sm">
                                  <p className="text-xs text-purple-600 font-medium">ROUGE-L</p>
                                  <p className={`text-2xl font-bold ${getMetricColor(quickTestResult.rougel)}`}>
                                    {(quickTestResult.rougel * 100).toFixed(1)}%
                                  </p>
                                </div>
                              )}
                              {quickTestResult.original_bertscore_f1 != null && (
                                <div className="p-4 bg-gradient-to-br from-emerald-50 to-emerald-100 rounded-xl border border-emerald-200 shadow-sm">
                                  <p className="text-xs text-emerald-700 font-medium">BERTScore F1</p>
                                  <p className={`text-2xl font-bold ${getMetricColor(quickTestResult.original_bertscore_f1)}`}>
                                    {(quickTestResult.original_bertscore_f1 * 100).toFixed(1)}%
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

                      {/* Dataset Management */}
                      <div className="flex gap-2 mt-3">
                        <Select value={selectedDataset} onValueChange={handleLoadDataset}>
                          <SelectTrigger className="flex-1 border-slate-200 focus:border-cyan-400 focus:ring-cyan-400">
                            <SelectValue placeholder="Kayıtlı veri seti seçin..." />
                          </SelectTrigger>
                          <SelectContent>
                            {testDatasets.map((dataset) => (
                              <SelectItem key={dataset.id} value={dataset.id.toString()}>
                                {dataset.name} ({dataset.total_test_cases} test)
                              </SelectItem>
                            ))}
                          </SelectContent>
                        </Select>
                        
                        <Button
                          onClick={() => setShowSaveDatasetDialog(true)}
                          disabled={!batchTestJson}
                          variant="outline"
                          className="border-cyan-300 text-cyan-700 hover:bg-cyan-50"
                        >
                          <Save className="w-4 h-4 mr-1" />
                          Kaydet
                        </Button>
                      </div>

                      {testDatasets.length > 0 && (
                        <div className="mt-3 p-3 bg-cyan-50 rounded-lg border border-cyan-200">
                          <p className="text-xs font-medium text-cyan-900 mb-2">Kayıtlı Veri Setleri:</p>
                          <div className="space-y-1">
                            {testDatasets.map((dataset) => (
                              <div key={dataset.id} className="flex items-center justify-between text-xs">
                                <span className="text-cyan-700">
                                  • {dataset.name} ({dataset.question_count} soru)
                                </span>
                              </div>
                            ))}
                          </div>
                        </div>
                      )}
                    </div>

                    <div>
                      <Label className="text-sm font-medium text-slate-700">Tekrar Sayısı (1-10)</Label>
                      <Input
                        type="number"
                        min={1}
                        max={10}
                        value={repeatCount}
                        onChange={(e) => {
                          const value = parseInt(e.target.value);
                          if (value >= 1 && value <= 10) {
                            setRepeatCount(value);
                            setTotalRuns(value);
                          }
                        }}
                        className="mt-1.5 border-slate-200 focus:border-cyan-400 focus:ring-cyan-400"
                      />
                      <p className="text-xs text-slate-500 mt-2">
                        Test kaç kez otomatik olarak tekrarlanacak. Her çalışma ayrı bir oturum olarak kaydedilecek.
                      </p>
                    </div>

                    <div className="flex gap-2">
                      <Button
                        onClick={handleBatchTest}
                        disabled={isBatchTesting || !batchTestJson}
                        className="flex-1 bg-gradient-to-r from-cyan-600 to-blue-600 hover:from-cyan-700 hover:to-blue-700 shadow-lg shadow-cyan-200"
                      >
                        {isBatchTesting ? (
                          <>
                            <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                            Test Ediliyor... ({currentRunNumber}/{totalRuns})
                          </>
                        ) : (
                          <>
                            <FileJson className="w-4 h-4 mr-2" />
                            {totalRuns > 1 ? `${totalRuns} Kez Test Başlat` : "Batch Test Başlat"}
                          </>
                        )}
                      </Button>
                      
                      {isBatchTesting && (
                        <Button
                          onClick={handleCancelCurrentTest}
                          disabled={isCancelling}
                          variant="destructive"
                          className="shadow-lg"
                          title="Testi Durdur"
                        >
                          {isCancelling ? (
                            <Loader2 className="w-4 h-4 animate-spin" />
                          ) : (
                            <Square className="w-4 h-4" />
                          )}
                        </Button>
                      )}
                    </div>
                  </div>

                  {/* Results Section */}
                  <div className="space-y-4">
                    {batchTestResult ? (
                      <>
                        {totalRuns > 1 && (
                          <div className="mb-4 p-3 bg-gradient-to-r from-indigo-50 to-purple-50 rounded-xl border border-indigo-200">
                            <div className="flex items-center justify-between">
                              <div>
                                <p className="text-sm font-semibold text-indigo-900">
                                  {totalRuns} Kez Çalıştırma Tamamlandı
                                </p>
                                <p className="text-xs text-indigo-600 mt-1">
                                  Her çalışma ayrı bir oturum olarak kaydedildi.
                                  Toplam {batchTestResult.aggregate.test_count} test sonucu birleştirildi.
                                </p>
                              </div>
                              <div className="text-right">
                                <p className="text-xs text-indigo-600">Run {currentRunNumber}/{totalRuns}</p>
                              </div>
                            </div>
                          </div>
                        )}

                        <div>
                          <Label className="text-sm font-medium text-slate-700">
                            {totalRuns > 1 ? "Birleştirilmiş İstatistikler" : "Özet İstatistikler"}
                          </Label>
                          <div className="mt-2 grid grid-cols-3 gap-3">
                            <div className="p-4 bg-white rounded-xl border border-slate-200 shadow-sm">
                              <p className="text-xs text-slate-500 font-medium">Test Sayısı</p>
                              <p className="text-2xl font-bold text-slate-900">
                                {batchTestResult.aggregate.test_count}
                              </p>
                            </div>
                            <div className="p-4 bg-gradient-to-br from-amber-50 to-orange-50 rounded-xl border border-amber-200 shadow-sm">
                              <p className="text-xs text-amber-600 font-medium">Geçen Süre</p>
                              <p className="text-2xl font-bold text-amber-700">
                                {batchTestElapsedTime}
                              </p>
                            </div>
                            {batchTestResult.aggregate.avg_similarity != null && (
                              <div className="p-4 bg-gradient-to-br from-blue-50 to-indigo-50 rounded-xl border border-blue-200 shadow-sm">
                                <p className="text-xs text-blue-600 font-medium">Ort. Cosine Sim.</p>
                                <p className="text-xl font-bold text-blue-700">
                                  {(batchTestResult.aggregate.avg_similarity * 100).toFixed(1)}%
                                </p>
                              </div>
                            )}
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
                            {batchTestResult.aggregate.avg_original_bertscore_f1 != null && (
                              <div className="p-4 bg-gradient-to-br from-emerald-50 to-teal-50 rounded-xl border border-emerald-200 shadow-sm">
                                <p className="text-xs text-emerald-700 font-medium">Ort. BERTScore F1</p>
                                <p className="text-xl font-bold text-emerald-800">
                                  {(batchTestResult.aggregate.avg_original_bertscore_f1 * 100).toFixed(1)}%
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
                              <Button
                                variant="outline"
                                size="sm"
                                onClick={async () => {
                                  if (!saveGroupName) {
                                    toast.error("Önce sonuçları kaydedin ve grup adı verin");
                                    return;
                                  }
                                  await exportSelectedGroupToWandb();
                                }}
                                disabled={!saveGroupName}
                                className="h-7 text-xs border-purple-200 text-purple-600 hover:bg-purple-50"
                                title="Sonuçları W&B'ye gönder"
                              >
                                <BarChart3 className="w-3 h-3 mr-1" /> W&B'ye Gönder
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
                                    Cos. Sim.
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
                                  <tr
                                    key={idx}
                                    className="hover:bg-slate-50"
                                  >
                                    <td className="px-3 py-2 text-slate-700 truncate max-w-[200px]">
                                      {result.question}
                                    </td>
                                    <td className="px-3 py-2 text-center">
                                      {result.similarity_score !== null && result.similarity_score !== undefined ? (
                                        <span className={`font-medium ${getMetricColor(result.similarity_score)}`}>
                                          {(result.similarity_score * 100).toFixed(0)}%
                                        </span>
                                      ) : (
                                        <span className="text-slate-400">-</span>
                                      )}
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
                                      {result.original_bertscore_f1 !== null && result.original_bertscore_f1 !== undefined ? (
                                        <span className={`font-medium ${getMetricColor(result.original_bertscore_f1)}`}>
                                          {(result.original_bertscore_f1 * 100).toFixed(0)}%
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

          {/* Batch Test Sessions Card */}
          <Card className="overflow-hidden border-0 shadow-lg bg-white">
            <button
              onClick={() => setIsBatchTestExpanded(!isBatchTestExpanded)}
              className="w-full px-6 py-5 flex items-center justify-between hover:bg-slate-50 transition-all duration-200"
            >
              <div className="flex items-center gap-4">
                <div className="p-3 bg-gradient-to-br from-indigo-500 to-purple-600 rounded-xl shadow-lg">
                  <FileJson className="w-6 h-6 text-white" />
                </div>
                <div className="text-left">
                  <h2 className="text-xl font-bold text-slate-900">Batch Test Oturumları</h2>
                  <p className="text-sm text-slate-600">
                    {batchTestSessions.length} kayıtlı oturum
                  </p>
                </div>
              </div>
              <div
                className={`p-2 rounded-full bg-slate-100 transition-transform duration-200 ${
                  isBatchTestExpanded ? "rotate-180" : ""
                }`}
              >
                <ChevronDown className="w-5 h-5 text-slate-600" />
              </div>
            </button>

            {isBatchTestExpanded && (
              <div className="px-6 pb-6 pt-2 border-t border-slate-100">
                {isSessionsLoading ? (
                  <div className="text-center py-8 text-slate-400">
                    <Loader2 className="w-6 h-6 animate-spin mx-auto mb-2" />
                    <p className="text-sm">Oturumlar yükleniyor...</p>
                  </div>
                ) : batchTestSessions.length === 0 ? (
                  <div className="text-center py-8 text-slate-400">
                    <FileJson className="w-12 h-12 mx-auto mb-3 opacity-50" />
                    <p className="font-medium">Henüz oturum yok</p>
                    <p className="text-xs mt-1">Batch test başlattığınızda oturumlar burada görünecek</p>
                  </div>
                ) : (
                  <div className="space-y-3">
                    {batchTestSessions.map((session) => (
                      <div
                        key={session.id}
                        className="p-4 bg-slate-50 rounded-xl border border-slate-200 hover:border-slate-300 transition-colors"
                      >
                        <div className="flex items-start justify-between">
                          <div className="flex-1 min-w-0">
                            <div className="flex items-center gap-2 mb-2">
                              <span className={`px-2 py-0.5 text-xs rounded-full font-medium ${getSessionStatusColor(session.status)}`}>
                                {getSessionStatusText(session.status)}
                              </span>
                              <span className="text-xs text-slate-500">
                                {new Date(session.started_at).toLocaleString("tr-TR")}
                              </span>
                            </div>
                            <p className="text-sm font-medium text-slate-900 truncate">
                              {session.group_name}
                            </p>
                            <div className="flex flex-wrap items-center gap-3 mt-2 text-xs">
                              <span className="text-slate-600">
                                Toplam: {session.total_tests}
                              </span>
                              <span className="text-slate-600">
                                Tamamlandı: {session.completed_tests}
                              </span>
                              {session.failed_tests > 0 && (
                                <span className="text-red-600">
                                  Başarısız: {session.failed_tests}
                                </span>
                              )}
                              {session.llm_model && (
                                <span className="text-purple-600">
                                  LLM: {session.llm_model}
                                </span>
                              )}
                            </div>
                            {session.status === 'in_progress' && (
                              <div className="mt-2">
                                <div className="flex items-center justify-between text-xs text-slate-600 mb-1">
                                  <span>İlerleme</span>
                                  <span>{Math.round((session.completed_tests / session.total_tests) * 100)}%</span>
                                </div>
                                <div className="w-full bg-slate-200 rounded-full h-2">
                                  <div
                                    className="bg-blue-600 h-2 rounded-full transition-all duration-300"
                                    style={{ width: `${(session.completed_tests / session.total_tests) * 100}%` }}
                                  />
                                </div>
                              </div>
                            )}
                          </div>
                          <div className="flex items-center gap-1 ml-2">
                            {(session.status === 'in_progress' || session.status === 'failed') && (
                              <Button
                                variant="ghost"
                                size="sm"
                                onClick={() => handleResumeSession(session.id)}
                                disabled={isResumingSession}
                                className="text-blue-600 hover:text-blue-700 hover:bg-blue-50"
                              >
                                {isResumingSession ? <Loader2 className="w-4 h-4 animate-spin" /> : <Target className="w-4 h-4" />}
                              </Button>
                            )}
                            {session.status === 'in_progress' && (
                              <Button
                                variant="ghost"
                                size="sm"
                                onClick={() => handleCancelSession(session.id)}
                                className="text-red-600 hover:text-red-700 hover:bg-red-50"
                              >
                                <X className="w-4 h-4" />
                              </Button>
                            )}
                            {session.status !== 'in_progress' && (
                              <Button
                                variant="ghost"
                                size="sm"
                                onClick={() => handleDeleteSession(session.id)}
                                className="text-slate-500 hover:text-red-600 hover:bg-red-50"
                                title="Oturumu Sil"
                              >
                                <Trash2 className="w-4 h-4" />
                              </Button>
                            )}
                          </div>
                        </div>
                      </div>
                    ))}
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

                  {/* Yapılandırma Ayarları */}
                  <div>
                    <Label className="text-sm font-medium flex items-center gap-2">
                      <Settings className="w-4 h-4 text-teal-600" />
                      Yapılandırma Ayarları
                    </Label>
                    <div className="mt-2 space-y-3">
                      {/* Embedding Model */}
                      <div className="p-4 bg-gradient-to-br from-teal-50 to-cyan-50 rounded-xl border border-teal-200">
                        <div className="flex items-start gap-3">
                          <div className="p-2 bg-teal-100 rounded-lg">
                            <Target className="w-5 h-5 text-teal-600" />
                          </div>
                          <div className="flex-1">
                            <p className="text-xs text-teal-600 font-medium mb-1">Embedding Model</p>
                            <p className="text-sm font-semibold text-slate-900">
                              {viewingResult.embedding_model_used || "Belirtilmemiş"}
                            </p>
                            {viewingResult.embedding_model_used && (
                              <p className="text-xs text-slate-500 mt-1">
                                {viewingResult.embedding_model_used.includes("openai") ? "OpenAI" :
                                 viewingResult.embedding_model_used.includes("alibaba") ? "Alibaba" :
                                 viewingResult.embedding_model_used.includes("cohere") ? "Cohere" :
                                 "Diğer"} sağlayıcısı kullanıldı
                              </p>
                            )}
                          </div>
                        </div>
                      </div>

                      {/* LLM Model */}
                      {viewingResult.llm_model_used && (
                        <div className="p-4 bg-gradient-to-br from-purple-50 to-pink-50 rounded-xl border border-purple-200">
                          <div className="flex items-start gap-3">
                            <div className="p-2 bg-purple-100 rounded-lg">
                              <Sparkles className="w-5 h-5 text-purple-600" />
                            </div>
                            <div className="flex-1">
                              <p className="text-xs text-purple-600 font-medium mb-1">LLM Model</p>
                              <p className="text-sm font-semibold text-slate-900">
                                {viewingResult.llm_model_used}
                              </p>
                              {viewingResult.llm_model_used.includes("/") && (
                                <p className="text-xs text-slate-500 mt-1">
                                  {viewingResult.llm_model_used.split("/")[0]} sağlayıcısı
                                </p>
                              )}
                            </div>
                          </div>
                        </div>
                      )}

                      {/* Sistem Prompt */}
                      {viewingResult.system_prompt_used && (
                        <div className="p-4 bg-gradient-to-br from-amber-50 to-orange-50 rounded-xl border border-amber-200">
                          <div className="flex items-start gap-3">
                            <div className="p-2 bg-amber-100 rounded-lg">
                              <FileText className="w-5 h-5 text-amber-600" />
                            </div>
                            <div className="flex-1">
                              <p className="text-xs text-amber-600 font-medium mb-1">Sistem Promptu</p>
                              <p className="text-xs text-slate-700 line-clamp-3 font-mono bg-white/50 rounded p-2 mt-1">
                                {viewingResult.system_prompt_used}
                              </p>
                            </div>
                          </div>
                        </div>
                      )}

                      {/* Gecikme (Latency) */}
                      <div className="p-4 bg-gradient-to-br from-slate-50 to-gray-100 rounded-xl border border-slate-200">
                        <div className="flex items-start gap-3">
                          <div className="p-2 bg-slate-100 rounded-lg">
                            <History className="w-5 h-5 text-slate-600" />
                          </div>
                          <div className="flex-1">
                            <p className="text-xs text-slate-600 font-medium mb-1">İşlem Süresi</p>
                            <p className="text-sm font-bold text-slate-900">
                              {viewingResult.latency_ms}ms
                            </p>
                            <p className="text-xs text-slate-500 mt-1">
                              {viewingResult.latency_ms < 500 ? "Çok hızlı" :
                               viewingResult.latency_ms < 1000 ? "Hızlı" :
                               viewingResult.latency_ms < 2000 ? "Normal" :
                               "Yavaş"}
                            </p>
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* ROUGE ve BERTScore Metrikleri */}
                  {(viewingResult.similarity_score != null || viewingResult.rouge1 != null || viewingResult.rouge2 != null ||
                    viewingResult.rougel != null || viewingResult.bertscore_f1 != null) && (
                    <div>
                      <Label className="text-sm font-medium">Metrikler</Label>
                      <div className="mt-2 grid grid-cols-2 gap-3">
                        {viewingResult.similarity_score != null && (
                          <div className="p-4 bg-gradient-to-br from-blue-50 to-indigo-50 rounded-xl border border-blue-200">
                            <p className="text-xs text-blue-600 font-medium mb-1">Cosine Similarity</p>
                            <p className={`text-2xl font-bold ${getMetricColor(viewingResult.similarity_score)}`}>
                              {(viewingResult.similarity_score * 100).toFixed(1)}%
                            </p>
                          </div>
                        )}
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

          {/* W&B Runs Management Modal */}
          <Dialog open={isWandbRunsModalOpen} onOpenChange={setIsWandbRunsModalOpen}>
            <DialogContent className="max-w-4xl max-h-[90vh] overflow-y-auto">
              <DialogHeader>
                <DialogTitle className="flex items-center gap-2">
                  <Settings className="w-5 h-5 text-indigo-600" />
                  W&B Run’ları Yönet
                </DialogTitle>
                <DialogDescription>
                  Eksik llm_model_used/embedding_model_used alanlarını DB’den güncelle.
                </DialogDescription>
              </DialogHeader>

              {isLoadingWandbRuns ? (
                <div className="flex items-center justify-center py-12">
                  <Loader2 className="w-6 h-6 animate-spin mr-2" />
                  <span>W&B run’ları yükleniyor...</span>
                </div>
              ) : wandbRuns.length === 0 ? (
                <div className="text-center py-12 text-slate-500">
                  <p className="text-sm">W&B run’ı bulunamadı.</p>
                </div>
              ) : (
                <div className="mt-4">
                  <div className="space-y-3">
                    {wandbRuns.map((run) => (
                      <div
                        key={run.id}
                        className="border border-slate-200 rounded-lg p-4 hover:bg-slate-50 transition-colors"
                      >
                        <div className="flex items-start justify-between gap-4">
                          <div className="flex-1 min-w-0">
                            <p className="font-semibold text-slate-900 truncate">{run.name}</p>
                            <p className="text-xs text-slate-500 mt-1">
                              ID: {run.id} | Durum: {run.state} |{" "}
                              {run.created_at
                                ? new Date(run.created_at).toLocaleString("tr-TR")
                                : "Tarih bilinmiyor"}
                            </p>
                            <p className="text-xs text-slate-500 mt-1">
                              Grup: {(run.config.group_name as string) || "Bilinmiyor"}
                            </p>
                            {run.missing_fields.length > 0 && (
                              <div className="mt-2">
                                <span className="text-xs font-medium text-amber-600">Eksik alanlar:</span>{" "}
                                <span className="text-xs text-amber-700">
                                  {run.missing_fields.join(", ")}
                                </span>
                              </div>
                            )}
                          </div>
                          <div className="flex flex-col gap-2">
                            {run.missing_fields.length > 0 ? (
                              <Button
                                variant="outline"
                                size="sm"
                                onClick={() => updateSelectedWandbRun(run)}
                                disabled={updatingRunIds.has(run.id)}
                                className="h-8 text-xs"
                              >
                                {updatingRunIds.has(run.id) ? (
                                  <>
                                    <Loader2 className="w-3 h-3 mr-1 animate-spin" />
                                    Güncelleniyor...
                                  </>
                                ) : (
                                  <>
                                    <RefreshCw className="w-3 h-3 mr-1" />
                                    Güncelle
                                  </>
                                )}
                              </Button>
                            ) : (
                              <div className="flex items-center justify-center h-8 px-3 text-xs text-green-700 bg-green-100 rounded border border-green-300">
                                Güncel
                              </div>
                            )}
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </DialogContent>
          </Dialog>

        {/* Save Dataset Dialog */}
        <Dialog open={showSaveDatasetDialog} onOpenChange={setShowSaveDatasetDialog}>
          <DialogContent className="sm:max-w-md">
            <DialogHeader>
              <DialogTitle>Veri Seti Kaydet</DialogTitle>
              <DialogDescription>
                Mevcut test verilerini kaydederek daha sonra kullanabilirsiniz.
              </DialogDescription>
            </DialogHeader>
            <div className="space-y-4 py-4">
              <div>
                <Label htmlFor="dataset-name">Veri Seti Adı</Label>
                <Input
                  id="dataset-name"
                  value={datasetName}
                  onChange={(e) => setDatasetName(e.target.value)}
                  placeholder="Örn: Veri Seti 1, Test Dataset 2"
                  className="mt-1"
                />
              </div>
              <div>
                <Label htmlFor="dataset-description">Açıklama (Opsiyonel)</Label>
                <Textarea
                  id="dataset-description"
                  value={datasetDescription}
                  onChange={(e) => setDatasetDescription(e.target.value)}
                  placeholder="Veri seti hakkında açıklama..."
                  rows={3}
                  className="mt-1"
                />
              </div>
              <div className="p-3 bg-slate-50 rounded-lg">
                <p className="text-xs font-medium text-slate-700 mb-1">Özet:</p>
                <p className="text-xs text-slate-600">
                  {batchTestJson ? `${JSON.parse(batchTestJson || "[]").length} test case` : "0 test case"}
                </p>
              </div>
            </div>
            <DialogFooter>
              <Button
                variant="outline"
                onClick={() => setShowSaveDatasetDialog(false)}
              >
                İptal
              </Button>
              <Button
                onClick={handleSaveDataset}
                disabled={!datasetName || !batchTestJson}
              >
                Kaydet
              </Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>

        {/* Statistics Modal Dialog */}
        <Dialog open={isStatisticsModalOpen} onOpenChange={setIsStatisticsModalOpen}>
          <DialogContent className="max-w-4xl max-h-[90vh] overflow-y-auto">
            <DialogHeader>
              <DialogTitle>
                {selectedGroup && selectedGroup !== "__all__"
                  ? `${selectedGroup === "__no_group__" ? "Grupsuz" : selectedGroup} grubunun detaylı istatistikleri`
                  : "Tüm sonuçların detaylı istatistikleri"}
              </DialogTitle>
              <DialogDescription>
                {selectedGroup && selectedGroup !== "__all__"
                  ? `"${selectedGroup === "__no_group__" ? "Grupsuz" : selectedGroup}" grubunun detaylı istatistikleri`
                  : "Tüm sonuçların detaylı istatistikleri"}
              </DialogDescription>
            </DialogHeader>
              
              {savedResultsAggregate && (
                <div className="space-y-6 py-4">
                  {/* Özet İstatistikler Tablosu */}
                  <div>
                    <h3 className="text-sm font-semibold text-slate-900 mb-3">Özet İstatistikler</h3>
                    <div className="border border-slate-200 rounded-lg overflow-hidden">
                      <table className="w-full text-sm">
                        <thead className="bg-slate-50">
                          <tr>
                            <th className="px-4 py-3 text-left font-medium text-slate-700">Metrik</th>
                            <th className="px-4 py-3 text-center font-medium text-slate-700">Değer</th>
                            <th className="px-4 py-3 text-left font-medium text-slate-700">Açıklama</th>
                          </tr>
                        </thead>
                        <tbody className="divide-y divide-slate-100">
                          <tr className="hover:bg-slate-50">
                            <td className="px-4 py-3 font-medium text-slate-900">Test Sayısı</td>
                            <td className="px-4 py-3 text-center">
                              <span className="px-3 py-1 bg-teal-100 text-teal-700 rounded-full font-bold">
                                {savedResultsAggregate.test_count}
                              </span>
                            </td>
                            <td className="px-4 py-3 text-slate-600">Toplam test sayısı</td>
                          </tr>
                          {savedResultsAggregate.avg_similarity != null && (
                            <tr className="hover:bg-slate-50">
                              <td className="px-4 py-3 font-medium text-slate-900">Ortalama Cosine Similarity</td>
                              <td className="px-4 py-3 text-center">
                                <span className={`px-3 py-1 rounded-full font-bold ${getMetricBgColor(savedResultsAggregate.avg_similarity)}`}>
                                  {(savedResultsAggregate.avg_similarity * 100).toFixed(2)}%
                                </span>
                              </td>
                              <td className="px-4 py-3 text-slate-600">Embedding tabanlı semantik benzerlik</td>
                            </tr>
                          )}
                          {savedResultsAggregate.avg_rouge1 != null && (
                            <tr className="hover:bg-slate-50">
                              <td className="px-4 py-3 font-medium text-slate-900">Ortalama ROUGE-1</td>
                              <td className="px-4 py-3 text-center">
                                <span className={`px-3 py-1 rounded-full font-bold ${getMetricBgColor(savedResultsAggregate.avg_rouge1)}`}>
                                  {(savedResultsAggregate.avg_rouge1 * 100).toFixed(2)}%
                                </span>
                              </td>
                              <td className="px-4 py-3 text-slate-600">1-gram örtüşme oranı</td>
                            </tr>
                          )}
                          {savedResultsAggregate.avg_rouge2 != null && (
                            <tr className="hover:bg-slate-50">
                              <td className="px-4 py-3 font-medium text-slate-900">Ortalama ROUGE-2</td>
                              <td className="px-4 py-3 text-center">
                                <span className={`px-3 py-1 rounded-full font-bold ${getMetricBgColor(savedResultsAggregate.avg_rouge2)}`}>
                                  {(savedResultsAggregate.avg_rouge2 * 100).toFixed(2)}%
                                </span>
                              </td>
                              <td className="px-4 py-3 text-slate-600">2-gram örtüşme oranı</td>
                            </tr>
                          )}
                          {savedResultsAggregate.avg_rougel != null && (
                            <tr className="hover:bg-slate-50">
                              <td className="px-4 py-3 font-medium text-slate-900">Ortalama ROUGE-L</td>
                              <td className="px-4 py-3 text-center">
                                <span className={`px-3 py-1 rounded-full font-bold ${getMetricBgColor(savedResultsAggregate.avg_rougel)}`}>
                                  {(savedResultsAggregate.avg_rougel * 100).toFixed(2)}%
                                </span>
                              </td>
                              <td className="px-4 py-3 text-slate-600">En uzun ortak alt dizi oranı</td>
                            </tr>
                          )}
                          {savedResultsAggregate.avg_original_bertscore_precision != null && (
                            <tr className="hover:bg-slate-50">
                              <td className="px-4 py-3 font-medium text-slate-900">Ortalama BERTScore Precision</td>
                              <td className="px-4 py-3 text-center">
                                <span className={`px-3 py-1 rounded-full font-bold ${getMetricBgColor(savedResultsAggregate.avg_original_bertscore_precision)}`}>
                                  {(savedResultsAggregate.avg_original_bertscore_precision * 100).toFixed(2)}%
                                </span>
                              </td>
                              <td className="px-4 py-3 text-slate-600">BERT hassasiyet skoru</td>
                            </tr>
                          )}
                          {savedResultsAggregate.avg_original_bertscore_recall != null && (
                            <tr className="hover:bg-slate-50">
                              <td className="px-4 py-3 font-medium text-slate-900">Ortalama BERTScore Recall</td>
                              <td className="px-4 py-3 text-center">
                                <span className={`px-3 py-1 rounded-full font-bold ${getMetricBgColor(savedResultsAggregate.avg_original_bertscore_recall)}`}>
                                  {(savedResultsAggregate.avg_original_bertscore_recall * 100).toFixed(2)}%
                                </span>
                              </td>
                              <td className="px-4 py-3 text-slate-600">BERT geri çağırma skoru</td>
                            </tr>
                          )}
                          {savedResultsAggregate.avg_original_bertscore_f1 != null && (
                            <tr className="hover:bg-slate-50">
                              <td className="px-4 py-3 font-medium text-slate-900">Ortalama BERTScore F1</td>
                              <td className="px-4 py-3 text-center">
                                <span className={`px-3 py-1 rounded-full font-bold ${getMetricBgColor(savedResultsAggregate.avg_original_bertscore_f1)}`}>
                                  {(savedResultsAggregate.avg_original_bertscore_f1 * 100).toFixed(2)}%
                                </span>
                              </td>
                              <td className="px-4 py-3 text-slate-600">BERT F1 skoru (precision ve recall'ın harmonik ortalaması)</td>
                            </tr>
                          )}
                        </tbody>
                      </table>
                    </div>
                  </div>

                  {/* Yapılandırma Ayarları */}
                  {statisticsModalResults.length > 0 && (
                  <div>
                    <h3 className="text-sm font-semibold text-slate-900 mb-3">Yapılandırma Ayarları</h3>
                    <div className="border border-slate-200 rounded-lg overflow-hidden">
                      <table className="w-full text-sm">
                        <tbody className="divide-y divide-slate-100">
                          {/* Embedding Model */}
                          {statisticsModalResults[0]?.embedding_model_used && (
                            <tr className="hover:bg-slate-50">
                              <td className="px-4 py-3 bg-gradient-to-r from-teal-50 to-cyan-50">
                                <div className="flex items-center gap-2">
                                  <Target className="w-4 h-4 text-teal-600" />
                                  <span className="text-xs text-teal-600 font-medium">Embedding Model</span>
                                </div>
                              </td>
                              <td className="px-4 py-3">
                                <p className="text-sm font-semibold text-slate-900">
                                  {statisticsModalResults[0].embedding_model_used}
                                </p>
                                <p className="text-xs text-slate-500 mt-1">
                                  {statisticsModalResults[0].embedding_model_used.includes("openai") ? "OpenAI" :
                                   statisticsModalResults[0].embedding_model_used.includes("alibaba") ? "Alibaba" :
                                   statisticsModalResults[0].embedding_model_used.includes("cohere") ? "Cohere" :
                                   "Diğer"} sağlayıcısı kullanıldı
                                </p>
                              </td>
                            </tr>
                          )}
                          {/* LLM Model */}
                          {statisticsModalResults[0]?.llm_model_used && (
                            <tr className="hover:bg-slate-50">
                              <td className="px-4 py-3 bg-gradient-to-r from-purple-50 to-pink-50">
                                <div className="flex items-center gap-2">
                                  <Sparkles className="w-4 h-4 text-purple-600" />
                                  <span className="text-xs text-purple-600 font-medium">LLM Model</span>
                                </div>
                              </td>
                              <td className="px-4 py-3">
                                <p className="text-sm font-semibold text-slate-900">
                                  {statisticsModalResults[0].llm_model_used}
                                </p>
                                <p className="text-xs text-slate-500 mt-1">
                                  {statisticsModalResults[0].llm_model_used.includes("/") && (
                                    <span>{statisticsModalResults[0].llm_model_used.split("/")[0]} sağlayıcısı</span>
                                  )}
                                </p>
                              </td>
                            </tr>
                          )}
                          {/* Ortalama Gecikme */}
                          <tr className="hover:bg-slate-50">
                            <td className="px-4 py-3 bg-gradient-to-r from-slate-50 to-gray-100">
                              <div className="flex items-center gap-2">
                                <History className="w-4 h-4 text-slate-600" />
                                <span className="text-xs text-slate-600 font-medium">Ortalama Gecikme</span>
                              </div>
                            </td>
                            <td className="px-4 py-3">
                                <p className="text-sm font-bold text-slate-900">
                                  {(savedResults.reduce((sum, r) => sum + r.latency_ms, 0) / savedResults.length).toFixed(0)}ms
                                </p>
                                <p className="text-xs text-slate-500 mt-1">
                                  {savedResults.reduce((sum, r) => sum + r.latency_ms, 0) / savedResults.length < 500 ? "Çok hızlı" :
                                   savedResults.reduce((sum, r) => sum + r.latency_ms, 0) / savedResults.length < 1000 ? "Hızlı" :
                                   savedResults.reduce((sum, r) => sum + r.latency_ms, 0) / savedResults.length < 2000 ? "Normal" :
                                   "Yavaş"}
                                </p>
                            </td>
                          </tr>
                          {/* Test Sayısı */}
                          <tr className="hover:bg-slate-50">
                            <td className="px-4 py-3 bg-gradient-to-r from-blue-50 to-indigo-50">
                              <div className="flex items-center gap-2">
                                <Target className="w-4 h-4 text-blue-600" />
                                <span className="text-xs text-blue-600 font-medium">Test Sayısı</span>
                              </div>
                            </td>
                            <td className="px-4 py-3">
                              <p className="text-sm font-bold text-slate-900">
                                {savedResultsAggregate?.test_count || savedResults.length}
                              </p>
                              <p className="text-xs text-slate-500 mt-1">
                                Toplam test sayısı
                              </p>
                            </td>
                          </tr>
                        </tbody>
                      </table>
                    </div>
                  </div>
                )}

                  {/* Detaylı Sonuçlar Tablosu */}
                  <div>
                    <div className="flex items-center justify-between mb-3">
                      <h3 className="text-sm font-semibold text-slate-900">Detaylı Sonuçlar</h3>
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={copyTableToExcel}
                        className="h-7 text-xs border-indigo-300 text-indigo-700 hover:bg-indigo-100"
                        disabled={isLoadingStatisticsResults || statisticsModalResults.length === 0}
                      >
                        <Download className="w-3 h-3 mr-1" /> Kopyala (Excel)
                      </Button>
                    </div>
                    <div className="border border-slate-200 rounded-lg overflow-hidden">
                      {isLoadingStatisticsResults ? (
                        <div className="p-8 text-center text-slate-500">
                          <Loader2 className="w-6 h-6 animate-spin mx-auto mb-2" />
                          <p className="text-sm">Sonuçlar yükleniyor...</p>
                        </div>
                      ) : statisticsModalResults.length === 0 ? (
                        <div className="p-8 text-center text-slate-500">
                          <p className="text-sm">Kayıtlı sonuç bulunamadı</p>
                        </div>
                      ) : (
                        <>
                          <div className="max-h-[400px] overflow-y-auto">
                            <table className="w-full text-xs">
                              <thead className="bg-slate-50 sticky top-0">
                                <tr>
                                  <th className="px-3 py-2 text-left font-medium text-slate-700">#</th>
                                  <th className="px-3 py-2 text-left font-medium text-slate-700">Soru</th>
                                  <th className="px-3 py-2 text-center font-medium text-slate-700">Cos. Sim.</th>
                                  <th className="px-3 py-2 text-center font-medium text-slate-700">ROUGE-1</th>
                                  <th className="px-3 py-2 text-center font-medium text-slate-700">ROUGE-2</th>
                                  <th className="px-3 py-2 text-center font-medium text-slate-700">ROUGE-L</th>
                                  <th className="px-3 py-2 text-center font-medium text-slate-700">BERTScore P</th>
                                  <th className="px-3 py-2 text-center font-medium text-slate-700">BERTScore R</th>
                                  <th className="px-3 py-2 text-center font-medium text-slate-700">BERTScore F1</th>
                                  <th className="px-3 py-2 text-center font-medium text-slate-700">Gecikme</th>
                                </tr>
                              </thead>
                              <tbody className="divide-y divide-slate-100">
                                {statisticsModalResults.map((result, idx) => (
                              <tr
                                key={result.id}
                                className="hover:bg-slate-50 cursor-pointer transition-colors"
                                onClick={() => {
                                  setViewingResult(result);
                                  setIsStatisticsModalOpen(false);
                                }}
                              >
                                <td className="px-3 py-2 text-slate-600 font-medium">{idx + 1}</td>
                                <td className="px-3 py-2 text-slate-900 truncate max-w-[200px]" title={result.question}>
                                  <div className="flex items-center gap-2">
                                    <Eye className="w-3 h-3 text-teal-600 flex-shrink-0" />
                                    <span>{result.question}</span>
                                  </div>
                                </td>
                                <td className="px-3 py-2 text-center">
                                  {result.similarity_score != null ? (
                                    <span className={`font-medium ${getMetricColor(result.similarity_score)}`}>
                                      {(result.similarity_score * 100).toFixed(1)}%
                                    </span>
                                  ) : (
                                    <span className="text-slate-400">-</span>
                                  )}
                                </td>
                                <td className="px-3 py-2 text-center">
                                  {result.rouge1 != null ? (
                                    <span className={`font-medium ${getMetricColor(result.rouge1)}`}>
                                      {(result.rouge1 * 100).toFixed(1)}%
                                    </span>
                                  ) : (
                                    <span className="text-slate-400">-</span>
                                  )}
                                </td>
                                <td className="px-3 py-2 text-center">
                                  {result.rouge2 != null ? (
                                    <span className={`font-medium ${getMetricColor(result.rouge2)}`}>
                                      {(result.rouge2 * 100).toFixed(1)}%
                                    </span>
                                  ) : (
                                    <span className="text-slate-400">-</span>
                                  )}
                                </td>
                                <td className="px-3 py-2 text-center">
                                  {result.rougel != null ? (
                                    <span className={`font-medium ${getMetricColor(result.rougel)}`}>
                                      {(result.rougel * 100).toFixed(1)}%
                                    </span>
                                  ) : (
                                    <span className="text-slate-400">-</span>
                                  )}
                                </td>
                                <td className="px-3 py-2 text-center">
                                  {result.original_bertscore_precision != null ? (
                                    <span className={`font-medium ${getMetricColor(result.original_bertscore_precision)}`}>
                                      {(result.original_bertscore_precision * 100).toFixed(1)}%
                                    </span>
                                  ) : (
                                    <span className="text-slate-400">-</span>
                                  )}
                                </td>
                                <td className="px-3 py-2 text-center">
                                  {result.original_bertscore_recall != null ? (
                                    <span className={`font-medium ${getMetricColor(result.original_bertscore_recall)}`}>
                                      {(result.original_bertscore_recall * 100).toFixed(1)}%
                                    </span>
                                  ) : (
                                    <span className="text-slate-400">-</span>
                                  )}
                                </td>
                                <td className="px-3 py-2 text-center">
                                  {result.original_bertscore_f1 != null ? (
                                    <span className={`font-medium ${getMetricColor(result.original_bertscore_f1)}`}>
                                      {(result.original_bertscore_f1 * 100).toFixed(1)}%
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
                        </>
                      )}
                    </div>
                  </div>

                  {/* Performans Analizi Tablosu */}
                  <div>
                    <h3 className="text-sm font-semibold text-slate-900 mb-3">Performans Analizi</h3>
                    <div className="border border-slate-200 rounded-lg overflow-hidden">
                      <table className="w-full text-sm">
                        <thead className="bg-slate-50">
                          <tr>
                            <th className="px-4 py-3 text-left font-medium text-slate-700">Metrik</th>
                            <th className="px-4 py-3 text-center font-medium text-slate-700">En İyi</th>
                            <th className="px-4 py-3 text-center font-medium text-slate-700">En Kötü</th>
                            <th className="px-4 py-3 text-center font-medium text-slate-700">Ortalama</th>
                          </tr>
                        </thead>
                        <tbody className="divide-y divide-slate-100">
                          {savedResultsAggregate.avg_similarity != null && (
                            <tr className="hover:bg-slate-50">
                              <td className="px-4 py-3 font-medium text-blue-700">Cosine Similarity</td>
                              <td className="px-4 py-3 text-center text-emerald-600 font-bold">
                                {(Math.max(...savedResults.filter(r => r.similarity_score != null).map(r => r.similarity_score!)) * 100).toFixed(1)}%
                              </td>
                              <td className="px-4 py-3 text-center text-red-600 font-bold">
                                {(Math.min(...savedResults.filter(r => r.similarity_score != null).map(r => r.similarity_score!)) * 100).toFixed(1)}%
                              </td>
                              <td className="px-4 py-3 text-center font-bold">
                                {(savedResultsAggregate.avg_similarity * 100).toFixed(1)}%
                              </td>
                            </tr>
                          )}
                          {savedResultsAggregate.avg_rouge1 != null && (
                            <tr className="hover:bg-slate-50">
                              <td className="px-4 py-3 font-medium text-purple-700">ROUGE-1</td>
                              <td className="px-4 py-3 text-center text-emerald-600 font-bold">
                                {(Math.max(...savedResults.filter(r => r.rouge1 != null).map(r => r.rouge1!)) * 100).toFixed(1)}%
                              </td>
                              <td className="px-4 py-3 text-center text-red-600 font-bold">
                                {(Math.min(...savedResults.filter(r => r.rouge1 != null).map(r => r.rouge1!)) * 100).toFixed(1)}%
                              </td>
                              <td className="px-4 py-3 text-center font-bold">
                                {(savedResultsAggregate.avg_rouge1 * 100).toFixed(1)}%
                              </td>
                            </tr>
                          )}
                          <tr className="hover:bg-slate-50">
                            <td className="px-4 py-3 font-medium text-slate-700">Gecikme</td>
                            <td className="px-4 py-3 text-center text-emerald-600 font-bold">
                              {Math.min(...savedResults.map(r => r.latency_ms))}ms
                            </td>
                            <td className="px-4 py-3 text-center text-red-600 font-bold">
                              {Math.max(...savedResults.map(r => r.latency_ms))}ms
                            </td>
                            <td className="px-4 py-3 text-center font-bold">
                              {(savedResults.reduce((sum, r) => sum + r.latency_ms, 0) / savedResults.length).toFixed(0)}ms
                            </td>
                          </tr>
                        </tbody>
                      </table>
                    </div>
                  </div>

                  {/* Ground Truth ve Üretilen Cevaplar Tablosu */}
                  <div>
                    <div className="flex items-center justify-between mb-3">
                      <h3 className="text-sm font-semibold text-slate-900">Ground Truth ve Üretilen Cevaplar</h3>
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() => {
                          const headers = ["#", "Soru", "Ground Truth", "Üretilen Cevap"];
                          const rows = statisticsModalResults.map((r, idx) => [
                            idx + 1,
                            `"${r.question.replace(/"/g, '""')}"`,
                            `"${r.ground_truth.replace(/"/g, '""')}"`,
                            `"${r.generated_answer.replace(/"/g, '""')}"`
                          ]);
                          const csv = [headers.join("\t"), ...rows.map(row => row.join("\t"))].join("\n");
                          navigator.clipboard.writeText(csv).then(() => {
                            toast.success("Tablo kopyalandı! Excel'e yapıştırabilirsiniz.");
                          }).catch(() => {
                            toast.error("Kopyalama başarısız");
                          });
                        }}
                        className="h-7 text-xs border-indigo-300 text-indigo-700 hover:bg-indigo-100"
                        disabled={isLoadingStatisticsResults || statisticsModalResults.length === 0}
                      >
                        <Download className="w-3 h-3 mr-1" /> Kopyala (Excel)
                      </Button>
                    </div>
                    <div className="border border-slate-200 rounded-lg overflow-hidden">
                      {isLoadingStatisticsResults ? (
                        <div className="p-8 text-center text-slate-500">
                          <Loader2 className="w-6 h-6 animate-spin mx-auto mb-2" />
                          <p className="text-sm">Sonuçlar yükleniyor...</p>
                        </div>
                      ) : statisticsModalResults.length === 0 ? (
                        <div className="p-8 text-center text-slate-500">
                          <p className="text-sm">Kayıtlı sonuç bulunamadı</p>
                        </div>
                      ) : (
                        <>
                          <div className="max-h-[500px] overflow-y-auto">
                            <table className="w-full text-xs">
                              <thead className="bg-slate-50 sticky top-0">
                                <tr>
                                  <th className="px-3 py-2 text-left font-medium text-slate-700 w-12">#</th>
                                  <th className="px-3 py-2 text-left font-medium text-slate-700 w-48">Soru</th>
                                  <th className="px-3 py-2 text-left font-medium text-slate-700 w-64">Ground Truth</th>
                                  <th className="px-3 py-2 text-left font-medium text-slate-700 w-64">Üretilen Cevap</th>
                                </tr>
                              </thead>
                              <tbody className="divide-y divide-slate-100">
                                {statisticsModalResults.map((result, idx) => (
                                  <tr key={result.id} className="hover:bg-slate-50">
                                    <td className="px-3 py-2 text-slate-600 font-medium">{idx + 1}</td>
                                    <td className="px-3 py-2 text-slate-900">
                                      <div className="max-h-20 overflow-y-auto whitespace-pre-wrap">
                                        {result.question}
                                      </div>
                                    </td>
                                    <td className="px-3 py-2 text-slate-700">
                                      <div className="max-h-32 overflow-y-auto whitespace-pre-wrap bg-purple-50 p-2 rounded border border-purple-100">
                                        {result.ground_truth}
                                      </div>
                                    </td>
                                    <td className="px-3 py-2 text-slate-700">
                                      <div className="max-h-32 overflow-y-auto whitespace-pre-wrap bg-blue-50 p-2 rounded border border-blue-100">
                                        {result.generated_answer}
                                      </div>
                                    </td>
                                  </tr>
                                ))}
                              </tbody>
                            </table>
                          </div>
                        </>
                      )}
                    </div>
                  </div>
                </div>
              )}

              <DialogFooter>
                <Button variant="outline" onClick={() => setIsStatisticsModalOpen(false)}>
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
