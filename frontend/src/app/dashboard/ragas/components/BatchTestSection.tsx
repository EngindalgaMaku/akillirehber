"use client";

import { useState, useRef, useEffect, useCallback } from "react";
import { api, RagasGroupInfo } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { Input } from "@/components/ui/input";
import { Card } from "@/components/ui/card";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Checkbox } from "@/components/ui/checkbox";
import { FileJson, ChevronDown, Loader2, Save, Download, Trash2, Zap, TrendingUp, Pause, Play, X } from "lucide-react";
import { toast } from "sonner";

interface BatchTestSectionProps {
  selectedCourseId: number;
  onBatchTestComplete: () => void;
  savedResultsGroups: RagasGroupInfo[];
  ragasEmbeddingModel?: string;
}

interface BatchTestResult {
  question: string;
  ground_truth: string;
  generated_answer: string;
  faithfulness?: number;
  answer_relevancy?: number;
  context_precision?: number;
  context_recall?: number;
  answer_correctness?: number;
  latency_ms: number;
  error_message?: string;
}

type BatchTestCase = {
  question: string;
  ground_truth: string;
  alternative_ground_truths?: string[];
  expected_contexts?: string[];
  [key: string]: unknown;
};

interface BatchResumeState {
  version: 1;
  courseId: number;
  groupName: string;
  enableWandb: boolean;
  startedAtMs: number;
  testCases: BatchTestCase[];
  completedIndices: number[];
  resultsByIndex: Record<number, BatchTestResult>;
  lastUpdatedAtMs: number;
}

export function BatchTestSection({ selectedCourseId, onBatchTestComplete, savedResultsGroups, ragasEmbeddingModel }: BatchTestSectionProps) {
  const [isExpanded, setIsExpanded] = useState(false);
  const [batchTestJson, setBatchTestJson] = useState("");
  const [isBatchTesting, setIsBatchTesting] = useState(false);
  const [batchTestElapsedTime, setBatchTestElapsedTime] = useState("00:00:00");
  const batchTestStartTimeRef = useRef<number | null>(null);
  const elapsedTimeIntervalRef = useRef<NodeJS.Timeout | null>(null);
  const completionCheckTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  
  // Realtime Results
  const [batchResults, setBatchResults] = useState<BatchTestResult[]>([]);
  const [currentTestIndex, setCurrentTestIndex] = useState(0);
  const [totalTests, setTotalTests] = useState(0);
  
  // W&B Integration
  const [enableWandbExport, setEnableWandbExport] = useState(true);
  const [wandbGroupName, setWandbGroupName] = useState("");

  // Resume
  const [resumeState, setResumeState] = useState<BatchResumeState | null>(null);
  const [retryCount, setRetryCount] = useState(0);
  const MAX_RETRY_ATTEMPTS = 3;
  
  // Pause/Cancel support
  const [currentTestId, setCurrentTestId] = useState<string | null>(null);
  const [isPaused, setIsPaused] = useState(false);
  
  // Dataset Management
  const [testDatasets, setTestDatasets] = useState<Array<{
    id: number;
    name: string;
    total_test_cases: number;
  }>>([]);
  const [selectedDataset, setSelectedDataset] = useState("");
  const [showSaveDatasetDialog, setShowSaveDatasetDialog] = useState(false);
  const [datasetName, setDatasetName] = useState("");
  const [datasetDescription, setDatasetDescription] = useState("");
  
  // Save Results Dialog
  const [isSaveDialogOpen, setIsSaveDialogOpen] = useState(false);
  const [saveGroupName, setSaveGroupName] = useState("");

  const getResumeStorageKey = (courseId: number, groupName: string) =>
    `ragas_batch_resume_v1:${courseId}:${groupName}`;

  const persistResumeState = (state: BatchResumeState | null) => {
    try {
      const groupName = state?.groupName || wandbGroupName;
      if (!groupName) return;
      const key = getResumeStorageKey(selectedCourseId, groupName);
      if (!state) {
        localStorage.removeItem(key);
      } else {
        localStorage.setItem(key, JSON.stringify(state));
      }
    } catch {
      // Ignore persistence errors
    }
  };

  const restoreResumeState = (groupName: string) => {
    try {
      if (!groupName) return;
      const raw = localStorage.getItem(getResumeStorageKey(selectedCourseId, groupName));
      if (!raw) return;
      const parsed = JSON.parse(raw) as BatchResumeState;
      if (!parsed || parsed.version !== 1) return;
      if (parsed.courseId !== selectedCourseId) return;
      if (!parsed.testCases || !Array.isArray(parsed.testCases)) return;
      if (!parsed.groupName) return;
      setResumeState(parsed);
    } catch {
      // Ignore restore errors
    }
  };

  const computeIndicesToRunFromDb = useCallback(async (opts: {
    groupName: string;
    testCases: BatchTestCase[];
  }) => {
    const existing = await api.getExistingQuickTestQuestions(
      selectedCourseId,
      opts.groupName
    );
    const existingSet = new Set(
      (existing?.questions || []).map((q) => String(q))
    );
    const indicesToRun: number[] = [];
    const completedIndices: number[] = [];
    for (let i = 0; i < opts.testCases.length; i++) {
      const q = String(opts.testCases[i]?.question || "");
      if (existingSet.has(q)) {
        completedIndices.push(i);
      } else {
        indicesToRun.push(i);
      }
    }
    return { indicesToRun, completedIndices };
  }, [selectedCourseId]);

  const buildResultsArrayFromResume = (state: BatchResumeState) => {
    const total = state.testCases.length;
    const arr: BatchTestResult[] = new Array(total);
    for (let i = 0; i < total; i++) {
      const existing = state.resultsByIndex[i];
      if (existing) {
        arr[i] = existing;
      } else {
        const tc = state.testCases[i];
        arr[i] = {
          question: String(tc?.question || ""),
          ground_truth: String(tc?.ground_truth || ""),
          generated_answer: "",
          latency_ms: 0,
        };
      }
    }
    return arr;
  };

  const normalizeTestCasesFromJson = (jsonStr: string) => {
    const parsedData = JSON.parse(jsonStr);
    let testCases = parsedData;
    if (parsedData.questions && Array.isArray(parsedData.questions)) {
      testCases = parsedData.questions;
    }
    if (!Array.isArray(testCases)) {
      throw new Error("Geçersiz JSON formatı: test case listesi bulunamadı");
    }

    return (testCases as unknown[]).map((tc, idx) => {
      if (!tc || typeof tc !== "object") {
        throw new Error(`Geçersiz test case (index=${idx}): object olmalı`);
      }
      const obj = tc as Record<string, unknown>;
      const question = String(obj.question ?? "");
      const ground_truth = String(obj.ground_truth ?? "");
      if (!question.trim() || !ground_truth.trim()) {
        throw new Error(
          `Geçersiz test case (index=${idx}): question ve ground_truth zorunlu`
        );
      }
      const alternative_ground_truths = Array.isArray(obj.alternative_ground_truths)
        ? (obj.alternative_ground_truths as unknown[])
            .filter((x) => typeof x === "string")
            .map((x) => String(x))
        : undefined;
      const expected_contexts = Array.isArray(obj.expected_contexts)
        ? (obj.expected_contexts as unknown[])
            .filter((x) => typeof x === "string")
            .map((x) => String(x))
        : undefined;

      return {
        ...obj,
        question,
        ground_truth,
        alternative_ground_truths,
        expected_contexts,
      } as BatchTestCase;
    });
  };

  const handlePauseBatchTest = async () => {
    if (!currentTestId) return;
    try {
      const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
      const token = localStorage.getItem("akilli_rehber_token");
      const response = await fetch(`${API_URL}/api/ragas/batch-test/${currentTestId}/pause`, {
        method: "POST",
        headers: { "Authorization": `Bearer ${token}` },
      });
      if (response.ok) {
        setIsPaused(true);
        toast.info("Test duraklatıldı");
      }
    } catch (error) {
      toast.error("Duraklatma başarısız");
    }
  };

  const handleResumePausedTest = async () => {
    if (!currentTestId) return;
    try {
      const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
      const token = localStorage.getItem("akilli_rehber_token");
      const response = await fetch(`${API_URL}/api/ragas/batch-test/${currentTestId}/resume`, {
        method: "POST",
        headers: { "Authorization": `Bearer ${token}` },
      });
      if (response.ok) {
        setIsPaused(false);
        toast.success("Test devam ediyor");
      }
    } catch (error) {
      toast.error("Devam ettirme başarısız");
    }
  };

  const handleCancelBatchTest = async () => {
    if (!currentTestId) return;
    if (!confirm("Testi iptal etmek istediğinizden emin misiniz?")) return;
    try {
      const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
      const token = localStorage.getItem("akilli_rehber_token");
      const response = await fetch(`${API_URL}/api/ragas/batch-test/${currentTestId}/cancel`, {
        method: "POST",
        headers: { "Authorization": `Bearer ${token}` },
      });
      if (response.ok) {
        toast.warning("Test iptal edildi");
        setCurrentTestId(null);
        setIsPaused(false);
      }
    } catch (error) {
      toast.error("İptal başarısız");
    }
  };

  const startBatchStream = async (opts: {
    testCases: BatchTestCase[];
    groupName: string;
    enableWandb: boolean;
    resumeBase?: BatchResumeState;
    onlyIndices?: number[];
  }) => {
    const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
    const token = localStorage.getItem("akilli_rehber_token");
    if (!token) {
      throw new Error("Oturum süresi dolmuş. Lütfen tekrar giriş yapın.");
    }

    const response = await fetch(`${API_URL}/api/ragas/quick-test-results/batch-stream`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "Authorization": `Bearer ${token}`,
      },
      body: JSON.stringify({
        course_id: selectedCourseId,
        test_cases: opts.testCases,
        group_name: opts.groupName,
        enable_wandb: false,
        only_indices: opts.onlyIndices,
        ragas_embedding_model: ragasEmbeddingModel || undefined,
      }),
    });

    if (response.status === 401) {
      throw new Error("Oturum süresi dolmuş. Lütfen sayfayı yenileyip tekrar deneyin.");
    }
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const reader = response.body?.getReader();
    const decoder = new TextDecoder();
    if (!reader) {
      throw new Error("No response body");
    }

    let wandbUrl: string | null = null;
    let buffer = "";

    let base = opts.resumeBase;
    const total = base ? base.testCases.length : opts.testCases.length;

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      const chunk = decoder.decode(value, { stream: true });
      buffer += chunk;

      const lines = buffer.split("\n");
      buffer = lines.pop() || "";

      for (const line of lines) {
        if (!line.trim()) continue;
        if (!line.startsWith("data: ")) continue;

        const jsonStr = line.slice(6).trim();
        if (!jsonStr) continue;

        try {
          const data = JSON.parse(jsonStr);

          // Handle init event - get test_id for pause/cancel
          if (data.event === "init") {
            if (data.test_id) {
              setCurrentTestId(data.test_id);
            }
            continue;
          }
          
          // Handle paused event
          if (data.event === "paused") {
            setIsPaused(true);
            continue;
          }
          
          // Handle cancelled event
          if (data.event === "cancelled") {
            toast.warning(`Test iptal edildi. ${data.completed}/${data.total} tamamlandı.`);
            setCurrentTestId(null);
            setIsPaused(false);
            return;
          }

          if (data.event === "progress") {
            setIsPaused(false); // Clear paused state on progress
            const idx = Number(data.index);
            const result: BatchTestResult = {
              question: data.result.question,
              ground_truth: data.result.ground_truth,
              generated_answer: data.result.generated_answer,
              faithfulness: data.result.faithfulness,
              answer_relevancy: data.result.answer_relevancy,
              context_precision: data.result.context_precision,
              context_recall: data.result.context_recall,
              answer_correctness: data.result.answer_correctness,
              latency_ms: data.result.latency_ms,
              error_message: data.result.error_message,
            };

            if (base) {
              const next: BatchResumeState = {
                ...base,
                completedIndices: Array.from(new Set([...(base.completedIndices || []), idx])).sort((a, b) => a - b),
                resultsByIndex: { ...base.resultsByIndex, [idx]: result },
                lastUpdatedAtMs: Date.now(),
              };
              base = next;
              setResumeState(next);
              persistResumeState(next);

              const arr = buildResultsArrayFromResume(next);
              setBatchResults(arr);
              setTotalTests(total);
              setCurrentTestIndex(next.completedIndices.length);
            } else {
              // Non-resume path; keep existing behaviour in caller
              throw new Error("Internal: resumeBase missing");
            }

            if (!data.result.error_message) {
              const retryInfo = data.result.retry_count > 0 ? ` (${data.result.retry_count} retry)` : '';
              const missingInfo = data.result.missing_metrics ? ` ⚠️ Eksik: ${data.result.missing_metrics.join(', ')}` : '';
              const lowScoreInfo = data.result.low_score_metrics ? ` ⚠️ Düşük skor: ${data.result.low_score_metrics.join(', ')}` : '';
              const hasIssues = missingInfo || lowScoreInfo;
              const toastMessage = `Test ${base.completedIndices.length}/${total} tamamlandı${retryInfo}${missingInfo}${lowScoreInfo}`;

              if (hasIssues) {
                toast.warning(toastMessage, {
                  id: `test-${idx}`,
                  duration: 4000,
                });
              } else {
                toast.success(toastMessage, {
                  id: `test-${idx}`,
                  duration: 1000,
                });
              }
            } else {
              toast.error(`Test başarısız: ${data.result.error_message}`, {
                id: `test-error-${idx}`,
                duration: 2000,
              });
            }
          } else if (data.event === "complete") {
            wandbUrl = data.wandb_url;
            
            // Clear pause/cancel state
            setCurrentTestId(null);
            setIsPaused(false);

            // Check if there are any failed tests that need retry
            if (base) {
              const allIndices = Array.from({ length: base.testCases.length }, (_, i) => i);
              const completedIndices = base.completedIndices || [];
              const missingIndices = allIndices.filter(i => !completedIndices.includes(i));
              
              // Also check for tests with errors
              const failedIndices = completedIndices.filter(i => {
                const result = base?.resultsByIndex[i];
                return result && result.error_message;
              });

              const indicesToRetry = [...new Set([...missingIndices, ...failedIndices])].sort((a, b) => a - b);

              if (indicesToRetry.length > 0 && retryCount < MAX_RETRY_ATTEMPTS) {
                const nextRetryCount = retryCount + 1;
                setRetryCount(nextRetryCount);
                
                // Capture base for the setTimeout callback
                const baseForRetry = base;
                
                toast.warning(
                  `${indicesToRetry.length} test başarısız oldu. Otomatik olarak tekrar deneniyor... (Deneme ${nextRetryCount}/${MAX_RETRY_ATTEMPTS})`,
                  { duration: 3000 }
                );

                // Automatically retry failed tests
                setTimeout(async () => {
                  try {
                    await startBatchStream({
                      testCases: baseForRetry.testCases,
                      groupName: opts.groupName,
                      enableWandb: opts.enableWandb,
                      resumeBase: baseForRetry,
                      onlyIndices: indicesToRetry,
                    });
                  } catch (retryError) {
                    console.error("Auto-retry failed:", retryError);
                    toast.error("Otomatik tekrar deneme başarısız oldu. Manuel olarak tekrar deneyin.");
                    setRetryCount(0); // Reset retry count on error
                  }
                }, 2000);
                return; // Don't show completion message yet
              } else if (indicesToRetry.length > 0 && retryCount >= MAX_RETRY_ATTEMPTS) {
                toast.error(
                  `${indicesToRetry.length} test ${MAX_RETRY_ATTEMPTS} denemeden sonra hala başarısız. Manuel olarak tekrar deneyin.`,
                  { duration: 5000 }
                );
              }

              const successCount = Object.values(base.resultsByIndex).filter(r => !r.error_message).length;
              const totalCount = base.testCases.length;
              
              if (successCount === totalCount) {
                toast.success(`🎉 Tüm testler başarıyla tamamlandı! ${successCount}/${totalCount}`, {
                  duration: 4000,
                });
              } else {
                toast.warning(`Testler tamamlandı: ${successCount}/${totalCount} başarılı, ${totalCount - successCount} başarısız`, {
                  duration: 4000,
                });
              }
              
              setRetryCount(0); // Reset retry count after completion
            } else {
              toast.success(`Tüm testler tamamlandı!`, { duration: 3000 });
            }

            onBatchTestComplete();

            if (wandbUrl) {
              const url = wandbUrl;
              toast.success(
                `W&B'ye başarıyla aktarıldı!`,
                {
                  duration: 8000,
                  action: {
                    label: "Aç",
                    onClick: () => window.open(url, "_blank")
                  }
                }
              );
            }

            setResumeState(null);
            persistResumeState(null);
          } else if (data.event === "error") {
            throw new Error(data.error);
          }
        } catch (e) {
          console.warn("SSE parse warning (will retry with next chunk):", {
            line: line.substring(0, 100) + "...",
            error: e instanceof Error ? e.message : String(e)
          });
        }
      }
    }
  };

  // Elapsed time counter
  useEffect(() => {
    if (isBatchTesting && batchTestStartTimeRef.current) {
      elapsedTimeIntervalRef.current = setInterval(() => {
        const elapsedMs = Date.now() - batchTestStartTimeRef.current!;
        const elapsedSeconds = Math.floor(elapsedMs / 1000);
        const hours = Math.floor(elapsedSeconds / 3600);
        const minutes = Math.floor((elapsedSeconds % 3600) / 60);
        const seconds = elapsedSeconds % 60;
        setBatchTestElapsedTime(
          `${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`
        );
      }, 1000);
    } else {
      if (elapsedTimeIntervalRef.current) {
        clearInterval(elapsedTimeIntervalRef.current);
        elapsedTimeIntervalRef.current = null;
      }
    }

    return () => {
      if (elapsedTimeIntervalRef.current) {
        clearInterval(elapsedTimeIntervalRef.current);
      }
    };
  }, [isBatchTesting]);

  useEffect(() => {
    if (!selectedCourseId) return;
    if (!wandbGroupName) return;
    restoreResumeState(wandbGroupName);
  }, [selectedCourseId, wandbGroupName]);

  useEffect(() => {
    if (!resumeState) return;
    setWandbGroupName(resumeState.groupName);
    setSaveGroupName(resumeState.groupName);
    setEnableWandbExport(resumeState.enableWandb);
    try {
      setBatchTestJson(JSON.stringify(resumeState.testCases, null, 2));
    } catch {
      // ignore
    }
    batchTestStartTimeRef.current = resumeState.startedAtMs;
    setTotalTests(resumeState.testCases.length);
    setCurrentTestIndex(resumeState.completedIndices.length);
    setBatchResults(buildResultsArrayFromResume(resumeState));
  }, [resumeState]);

  useEffect(() => {
    if (!isExpanded) return;
    if (!selectedCourseId) return;
    if (isBatchTesting) return;

    const groupName = (wandbGroupName || "").trim();
    if (!groupName) return;
    if (!batchTestJson.trim()) return;

    if (completionCheckTimeoutRef.current) {
      clearTimeout(completionCheckTimeoutRef.current);
      completionCheckTimeoutRef.current = null;
    }

    completionCheckTimeoutRef.current = setTimeout(async () => {
      try {
        const testCases = normalizeTestCasesFromJson(batchTestJson);
        const { completedIndices } = await computeIndicesToRunFromDb({
          groupName,
          testCases,
        });

        setResumeState((prev) => {
          if (
            prev &&
            prev.courseId === selectedCourseId &&
            prev.groupName === groupName &&
            prev.testCases.length === testCases.length
          ) {
            return {
              ...prev,
              completedIndices: Array.from(new Set(completedIndices)).sort(
                (a, b) => a - b
              ),
              lastUpdatedAtMs: Date.now(),
            };
          }

          const next: BatchResumeState = {
            version: 1,
            courseId: selectedCourseId,
            groupName,
            enableWandb: enableWandbExport,
            startedAtMs: Date.now(),
            testCases,
            completedIndices: Array.from(new Set(completedIndices)).sort(
              (a, b) => a - b
            ),
            resultsByIndex: prev?.resultsByIndex || {},
            lastUpdatedAtMs: Date.now(),
          };
          return next;
        });
      } catch {
        // ignore invalid JSON while typing
      }
    }, 350);

    return () => {
      if (completionCheckTimeoutRef.current) {
        clearTimeout(completionCheckTimeoutRef.current);
        completionCheckTimeoutRef.current = null;
      }
    };
  }, [
    isExpanded,
    selectedCourseId,
    wandbGroupName,
    batchTestJson,
    isBatchTesting,
    enableWandbExport,
    computeIndicesToRunFromDb,
  ]);

  const loadTestDatasets = async () => {
    try {
      // Use RAGAS test sets instead of old test_datasets
      const testSets = await api.getTestSets(selectedCourseId);
      // Convert to dataset format for compatibility
      const datasets = testSets.map(ts => ({
        id: ts.id,
        name: ts.name,
        total_test_cases: ts.question_count,
      }));
      setTestDatasets(datasets);
    } catch (error) {
      console.log("Failed to load test sets");
    }
  };

  const handleLoadDataset = async (datasetId: string) => {
    try {
      // Use RAGAS test set instead of old test_dataset
      const testSet = await api.getTestSet(parseInt(datasetId));
      
      // Convert test set questions to test cases format and sort alphabetically
      const testCases = testSet.questions
        .sort((a, b) => a.question.localeCompare(b.question, 'tr-TR'))
        .map(q => ({
          question: q.question,
          ground_truth: q.ground_truth,
          alternative_ground_truths: q.alternative_ground_truths,
          expected_contexts: q.expected_contexts,
        }));
      
      setBatchTestJson(JSON.stringify(testCases, null, 2));
      setSelectedDataset(datasetId);
      toast.success(`"${testSet.name}" test seti yüklendi (${testCases.length} soru)`);
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "Test seti yüklenemedi");
    }
  };

  const handleSaveDataset = async () => {
    if (!selectedCourseId || !batchTestJson || !datasetName) {
      toast.error("Lütfen JSON verisi ve veri seti adı girin");
      return;
    }

    try {
      const testCases = normalizeTestCasesFromJson(batchTestJson);

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
      toast.error(error instanceof Error ? error.message : "Veri seti kaydedilemedi");
    }
  };

  const handleDeleteDataset = async (datasetId: number) => {
    if (!confirm("Bu veri setini silmek istediğinizden emin misiniz?")) return;

    try {
      await api.deleteTestSet(datasetId);
      toast.success("Veri seti silindi");
      loadTestDatasets();
      if (selectedDataset === datasetId.toString()) {
        setSelectedDataset("");
      }
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "Veri seti silinemedi");
    }
  };

  const handleBatchTest = async () => {
    if (!batchTestJson) {
      toast.error("Lütfen JSON verisi girin");
      return;
    }

    try {
      const testCases = normalizeTestCasesFromJson(batchTestJson);

      batchTestStartTimeRef.current = Date.now();
      setBatchTestElapsedTime("00:00:00");
      setRetryCount(0); // Reset retry count at start

      const groupName = (wandbGroupName || "").trim();
      if (!groupName) {
        toast.error("Lütfen bir grup adı girin");
        return;
      }

      const { indicesToRun, completedIndices } = await computeIndicesToRunFromDb({
        groupName,
        testCases,
      });

      if (indicesToRun.length === 0) {
        toast.success("Bu grup için tüm sorular zaten kaydedilmiş. Çalıştırılacak soru yok.");
        setTotalTests(testCases.length);
        setCurrentTestIndex(testCases.length);
        const doneState: BatchResumeState = {
          version: 1,
          courseId: selectedCourseId,
          groupName,
          enableWandb: enableWandbExport,
          startedAtMs: Date.now(),
          testCases,
          completedIndices: Array.from(new Set(completedIndices)).sort((a, b) => a - b),
          resultsByIndex: {},
          lastUpdatedAtMs: Date.now(),
        };
        setResumeState(doneState);
        persistResumeState(doneState);
        return;
      }

      const initialResume: BatchResumeState = {
        version: 1,
        courseId: selectedCourseId,
        groupName,
        enableWandb: enableWandbExport,
        startedAtMs: batchTestStartTimeRef.current || Date.now(),
        testCases,
        completedIndices: Array.from(new Set(completedIndices)).sort((a, b) => a - b),
        resultsByIndex: {},
        lastUpdatedAtMs: Date.now(),
      };
      setResumeState(initialResume);
      persistResumeState(initialResume);

      setTotalTests(testCases.length);
      setWandbGroupName(groupName);
      setSaveGroupName(groupName);
      setCurrentTestIndex(initialResume.completedIndices.length);

      setBatchResults(buildResultsArrayFromResume(initialResume));

      setIsBatchTesting(true);

      await startBatchStream({
        testCases,
        groupName,
        enableWandb: enableWandbExport,
        resumeBase: initialResume,
        onlyIndices: indicesToRun,
      });
      
    } catch (error) {
      console.error("Batch test error:", error);
      toast.error(error instanceof Error ? error.message : "Test başarısız");
    } finally {
      setIsBatchTesting(false);
      if (elapsedTimeIntervalRef.current) {
        clearInterval(elapsedTimeIntervalRef.current);
      }
    }
  };

  const handleResumeBatchTest = async () => {
    if (!resumeState) {
      toast.error("Devam edilecek bir batch test bulunamadı");
      return;
    }
    if (isBatchTesting) return;

    const groupName = (resumeState.groupName || wandbGroupName || "").trim();
    if (!groupName) {
      toast.error("Grup adı bulunamadı");
      return;
    }

    setRetryCount(0); // Reset retry count on manual resume

    const { indicesToRun, completedIndices } = await computeIndicesToRunFromDb({
      groupName,
      testCases: resumeState.testCases,
    });

    if (indicesToRun.length === 0) {
      toast.success("Eksik soru yok. Sonuçlar tamamlanmış görünüyor.");
      setResumeState(null);
      persistResumeState(null);
      return;
    }

    setIsBatchTesting(true);
    batchTestStartTimeRef.current = resumeState.startedAtMs || Date.now();
    setBatchTestElapsedTime("00:00:00");
    setTotalTests(resumeState.testCases.length);
    setCurrentTestIndex(completedIndices.length);
    setBatchResults(buildResultsArrayFromResume(resumeState));

    try {
      await startBatchStream({
        testCases: resumeState.testCases,
        groupName,
        enableWandb: resumeState.enableWandb,
        resumeBase: {
          ...resumeState,
          groupName,
          completedIndices: Array.from(new Set(completedIndices)).sort((a, b) => a - b),
        },
        onlyIndices: indicesToRun,
      });
    } catch (error) {
      console.error("Resume batch test error:", error);
      toast.error(error instanceof Error ? error.message : "Resume başarısız");
    } finally {
      setIsBatchTesting(false);
      if (elapsedTimeIntervalRef.current) {
        clearInterval(elapsedTimeIntervalRef.current);
      }
    }
  };

  const handleSaveBatchResults = async () => {
    try {
      const settings = await api.getCourseSettings(selectedCourseId);
      let successCount = 0;
      for (const result of batchResults) {
        try {
          await api.saveQuickTestResult({
            course_id: selectedCourseId,
            group_name: saveGroupName || undefined,
            question: result.question,
            ground_truth: result.ground_truth,
            llm_provider: settings.llm_provider || "",
            llm_model: settings.llm_model || "",
            embedding_model: settings.default_embedding_model || undefined,
            search_top_k: settings.search_top_k ?? undefined,
            search_alpha: settings.search_alpha ?? undefined,
            reranker_used: settings.enable_reranker ?? undefined,
            reranker_provider: settings.reranker_provider || undefined,
            reranker_model: settings.reranker_model || undefined,
            generated_answer: result.generated_answer,
            faithfulness: result.faithfulness,
            answer_relevancy: result.answer_relevancy,
            context_precision: result.context_precision,
            context_recall: result.context_recall,
            answer_correctness: result.answer_correctness,
            latency_ms: result.latency_ms,
          });
          successCount++;
        } catch (error) {
          console.error("Failed to save result:", error);
        }
      }

      if (successCount > 0) {
        toast.success(`${successCount} sonuç kaydedildi`);
      }
      setIsSaveDialogOpen(false);
      setSaveGroupName("");
      onBatchTestComplete();
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "Kaydetme başarısız");
    }
  };

  const exportBatchResults = () => {
    if (batchResults.length === 0) return;

    const csv = [
      ["Question", "Ground Truth", "Generated Answer", "Faithfulness", "Answer Relevancy", "Context Precision", "Context Recall", "Answer Correctness", "Latency (ms)"].join(","),
      ...batchResults.map(r => [
        `"${r.question.replace(/"/g, '""')}"`,
        `"${r.ground_truth.replace(/"/g, '""')}"`,
        `"${r.generated_answer.replace(/"/g, '""')}"`,
        r.faithfulness != null ? (r.faithfulness * 100).toFixed(2) : "-",
        r.answer_relevancy != null ? (r.answer_relevancy * 100).toFixed(2) : "-",
        r.context_precision != null ? (r.context_precision * 100).toFixed(2) : "-",
        r.context_recall != null ? (r.context_recall * 100).toFixed(2) : "-",
        r.answer_correctness != null ? (r.answer_correctness * 100).toFixed(2) : "-",
        r.latency_ms
      ].join(","))
    ].join("\n");

    const blob = new Blob(["\ufeff" + csv], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `ragas-batch-${wandbGroupName || Date.now()}.csv`;
    a.click();
    URL.revokeObjectURL(url);
    toast.success("CSV indirildi");
  };

  const getMetricColor = (value?: number) => {
    if (value === undefined || value === null) return "text-slate-400";
    if (value >= 0.8) return "text-emerald-600";
    if (value >= 0.6) return "text-amber-600";
    return "text-red-600";
  };

  const calculateAggregate = () => {
    const validResults = batchResults.filter(r => r.faithfulness !== undefined);
    if (validResults.length === 0) return null;

    return {
      avg_faithfulness: validResults.reduce((sum, r) => sum + (r.faithfulness || 0), 0) / validResults.length,
      avg_answer_relevancy: validResults.reduce((sum, r) => sum + (r.answer_relevancy || 0), 0) / validResults.length,
      avg_context_precision: validResults.reduce((sum, r) => sum + (r.context_precision || 0), 0) / validResults.length,
      avg_context_recall: validResults.reduce((sum, r) => sum + (r.context_recall || 0), 0) / validResults.length,
      avg_answer_correctness: validResults.reduce((sum, r) => sum + (r.answer_correctness || 0), 0) / validResults.length,
      test_count: batchResults.length,
      success_count: validResults.length,
      error_count: batchResults.length - validResults.length,
    };
  };

  const aggregate = calculateAggregate();

  return (
    <>
      <Card className="overflow-hidden border-0 shadow-lg bg-gradient-to-br from-indigo-50 via-white to-purple-50">
        <button
          onClick={() => {
            setIsExpanded(!isExpanded);
            if (!isExpanded) loadTestDatasets();
          }}
          className="w-full px-6 py-5 flex items-center justify-between hover:bg-indigo-50/50 transition-all duration-200"
        >
          <div className="flex items-center gap-4">
            <div className="p-3 bg-gradient-to-br from-indigo-500 to-purple-600 rounded-xl shadow-lg shadow-indigo-200">
              <FileJson className="w-6 h-6 text-white" />
            </div>
            <div className="text-left">
              <h2 className="text-xl font-bold text-slate-900">Batch Test</h2>
              <p className="text-sm text-slate-600">JSON formatında toplu RAGAS değerlendirmesi</p>
            </div>
          </div>
          <div className={`p-2 rounded-full bg-indigo-100 transition-transform duration-200 ${isExpanded ? 'rotate-180' : ''}`}>
            <ChevronDown className="w-5 h-5 text-indigo-600" />
          </div>
        </button>

        {isExpanded && (
          <div className="px-6 pb-6 pt-2 border-t border-indigo-100">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Input Section */}
              <div className="space-y-4">
                <div>
                  <Label className="text-sm font-medium text-slate-700">Test Verileri (JSON)</Label>
                  <Textarea
                    value={batchTestJson}
                    onChange={(e) => setBatchTestJson(e.target.value)}
                    placeholder={`[\n  {\n    "question": "Soru 1",\n    "ground_truth": "Doğru cevap 1",\n    "alternative_ground_truths": []\n  }\n]`}
                    rows={12}
                    className="mt-1.5 border-slate-200 focus:border-indigo-400 focus:ring-indigo-400 font-mono text-xs resize-none h-64 overflow-y-auto"
                  />
                </div>

                <div className="flex gap-2">
                  <Select value={selectedDataset} onValueChange={handleLoadDataset}>
                    <SelectTrigger className="flex-1">
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
                    className="border-indigo-300 text-indigo-700 hover:bg-indigo-50"
                  >
                    <Save className="w-4 h-4 mr-1" />
                    Kaydet
                  </Button>
                </div>

                {testDatasets.length > 0 && (
                  <div className="p-3 bg-indigo-50 rounded-lg border border-indigo-200">
                    <p className="text-xs font-medium text-indigo-900 mb-2">Kayıtlı Veri Setleri:</p>
                    <div className="space-y-1 max-h-32 overflow-y-auto">
                      {testDatasets.map((dataset) => (
                        <div key={dataset.id} className="flex items-center justify-between text-xs">
                          <span className="text-indigo-700">• {dataset.name} ({dataset.total_test_cases} test)</span>
                          <button
                            onClick={() => handleDeleteDataset(dataset.id)}
                            className="text-red-600 hover:text-red-800"
                          >
                            <Trash2 className="w-3 h-3" />
                          </button>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* Grup Adı */}
                <div>
                  <Label className="text-sm font-medium text-slate-700">Grup Adı</Label>
                  <Input
                    value={wandbGroupName}
                    onChange={(e) => {
                      setWandbGroupName(e.target.value);
                      setSaveGroupName(e.target.value);
                    }}
                    placeholder="örn: RAGAS Test 1"
                    className="mt-1.5 border-slate-200 focus:border-indigo-400"
                  />
                  <p className="text-xs text-slate-500 mt-1">
                    Her test sonucu otomatik olarak kaydedilir
                  </p>
                </div>

                <Button
                  onClick={handleBatchTest}
                  disabled={isBatchTesting || !batchTestJson}
                  className="w-full bg-gradient-to-r from-indigo-600 to-purple-600 hover:from-indigo-700 hover:to-purple-700 shadow-lg shadow-indigo-200"
                >
                  {isBatchTesting ? (
                    <>
                      <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                      Test Ediliyor... ({currentTestIndex}/{totalTests})
                    </>
                  ) : (
                    <>
                      <FileJson className="w-4 h-4 mr-2" />
                      Batch Test Başlat
                    </>
                  )}
                </Button>

                {resumeState && !isBatchTesting && resumeState.completedIndices.length < resumeState.testCases.length && (
                  <div className="space-y-2">
                    <Button
                      onClick={handleResumeBatchTest}
                      variant="outline"
                      className="w-full border-amber-300 text-amber-800 hover:bg-amber-50"
                    >
                      Devam Et (Resume) ({resumeState.completedIndices.length}/{resumeState.testCases.length})
                    </Button>
                    <Button
                      onClick={() => {
                        setResumeState(null);
                        persistResumeState(null);
                        toast.success("Resume verisi temizlendi");
                      }}
                      variant="outline"
                      className="w-full border-slate-200 text-slate-700 hover:bg-slate-50"
                    >
                      Resume Bilgisini Temizle
                    </Button>
                  </div>
                )}
              </div>

              {/* Results Section */}
              <div className="space-y-4">
                {isBatchTesting || batchResults.length > 0 ? (
                  <>
                    {/* Progress & Stats */}
                    {isBatchTesting && (
                      <div className="p-4 bg-gradient-to-br from-blue-50 to-indigo-50 rounded-xl border border-blue-200">
                        <div className="flex items-center justify-between mb-2">
                          <span className="text-sm font-medium text-blue-900">
                            {isPaused ? "⏸️ Duraklatıldı" : "İlerleme"}
                          </span>
                          <span className="text-xs text-blue-600">{currentTestIndex}/{totalTests}</span>
                        </div>
                        <div className="w-full bg-blue-200 rounded-full h-2 mb-2">
                          <div
                            className={`h-2 rounded-full transition-all duration-300 ${isPaused ? 'bg-amber-500' : 'bg-blue-600'}`}
                            style={{ width: `${(currentTestIndex / totalTests) * 100}%` }}
                          />
                        </div>
                        <div className="flex items-center justify-between text-xs text-blue-600 mb-3">
                          <span>Geçen Süre: {batchTestElapsedTime}</span>
                          <span>{Math.round((currentTestIndex / totalTests) * 100)}%</span>
                        </div>
                        
                        {/* Pause/Resume/Cancel Buttons */}
                        {currentTestId && (
                          <div className="flex gap-2">
                            {isPaused ? (
                              <Button
                                onClick={handleResumePausedTest}
                                size="sm"
                                className="flex-1 bg-emerald-600 hover:bg-emerald-700"
                              >
                                <Play className="w-4 h-4 mr-1" />
                                Devam Et
                              </Button>
                            ) : (
                              <Button
                                onClick={handlePauseBatchTest}
                                size="sm"
                                variant="outline"
                                className="flex-1 border-amber-400 text-amber-700 hover:bg-amber-50"
                              >
                                <Pause className="w-4 h-4 mr-1" />
                                Duraklat
                              </Button>
                            )}
                            <Button
                              onClick={handleCancelBatchTest}
                              size="sm"
                              variant="outline"
                              className="border-red-400 text-red-700 hover:bg-red-50"
                            >
                              <X className="w-4 h-4 mr-1" />
                              İptal
                            </Button>
                          </div>
                        )}
                      </div>
                    )}

                    {aggregate && (
                      <div className="p-4 bg-gradient-to-br from-purple-50 to-pink-50 rounded-xl border border-purple-200">
                        <p className="text-sm font-semibold text-purple-900 mb-3 flex items-center gap-2">
                          <TrendingUp className="w-4 h-4" />
                          Özet İstatistikler
                        </p>
                        <div className="grid grid-cols-2 gap-2">
                          <div className="p-2 bg-white rounded-lg">
                            <p className="text-xs text-purple-600">Faithfulness</p>
                            <p className="text-lg font-bold text-purple-900">{(aggregate.avg_faithfulness * 100).toFixed(1)}%</p>
                          </div>
                          <div className="p-2 bg-white rounded-lg">
                            <p className="text-xs text-purple-600">Answer Relevancy</p>
                            <p className="text-lg font-bold text-purple-900">{(aggregate.avg_answer_relevancy * 100).toFixed(1)}%</p>
                          </div>
                          <div className="p-2 bg-white rounded-lg">
                            <p className="text-xs text-purple-600">Context Precision</p>
                            <p className="text-lg font-bold text-purple-900">{(aggregate.avg_context_precision * 100).toFixed(1)}%</p>
                          </div>
                          <div className="p-2 bg-white rounded-lg">
                            <p className="text-xs text-purple-600">Context Recall</p>
                            <p className="text-lg font-bold text-purple-900">{(aggregate.avg_context_recall * 100).toFixed(1)}%</p>
                          </div>
                          <div className="p-2 bg-white rounded-lg">
                            <p className="text-xs text-purple-600">Answer Correctness</p>
                            <p className="text-lg font-bold text-purple-900">{(aggregate.avg_answer_correctness * 100).toFixed(1)}%</p>
                          </div>
                        </div>
                        <div className="mt-2 flex items-center justify-between text-xs text-purple-600">
                          <span>Başarılı: {aggregate.success_count}</span>
                          {aggregate.error_count > 0 && <span className="text-red-600">Hata: {aggregate.error_count}</span>}
                        </div>
                      </div>
                    )}

                    {/* Results Table */}
                    <div>
                      <div className="flex items-center justify-between mb-2">
                        <Label className="text-sm font-medium text-slate-700">Canlı Sonuçlar</Label>
                        {batchResults.length > 0 && !isBatchTesting && (
                          <Button
                            variant="outline"
                            size="sm"
                            onClick={exportBatchResults}
                            className="h-7 text-xs"
                          >
                            <Download className="w-3 h-3 mr-1" /> CSV
                          </Button>
                        )}
                      </div>
                      <div className="max-h-[400px] overflow-y-auto border border-slate-200 rounded-xl">
                        <table className="w-full text-xs">
                          <thead className="bg-slate-50 sticky top-0">
                            <tr>
                              <th className="px-3 py-2 text-left font-medium text-slate-600">#</th>
                              <th className="px-3 py-2 text-left font-medium text-slate-600">Soru</th>
                              <th className="px-3 py-2 text-center font-medium text-slate-600">Faith.</th>
                              <th className="px-3 py-2 text-center font-medium text-slate-600">Rel.</th>
                              <th className="px-3 py-2 text-center font-medium text-slate-600">Prec.</th>
                              <th className="px-3 py-2 text-center font-medium text-slate-600">Corr.</th>
                            </tr>
                          </thead>
                          <tbody className="divide-y divide-slate-100">
                            {batchResults.map((result, idx) => (
                              <tr key={idx} className="hover:bg-slate-50">
                                <td className="px-3 py-2 text-slate-600">{idx + 1}</td>
                                <td
                                  className="px-3 py-2 text-slate-700 truncate max-w-[150px]"
                                  title={result?.question || ""}
                                >
                                  {result?.question || "-"}
                                </td>
                                <td className="px-3 py-2 text-center">
                                  {result.faithfulness != null ? (
                                    <span className={`font-medium ${getMetricColor(result.faithfulness)}`}>
                                      {(result.faithfulness * 100).toFixed(0)}%
                                    </span>
                                  ) : result.error_message ? (
                                    <span className="text-red-600 text-xs">✗</span>
                                  ) : (
                                    resumeState?.completedIndices?.includes(idx) ? (
                                      <span className="text-slate-400 text-xs">✓</span>
                                    ) : isBatchTesting ? (
                                      <Loader2 className="w-3 h-3 animate-spin mx-auto text-blue-600" />
                                    ) : (
                                      <span className="text-slate-300 text-xs">-</span>
                                    )
                                  )}
                                </td>
                                <td className="px-3 py-2 text-center">
                                  {result.answer_relevancy != null ? (
                                    <span className={`font-medium ${getMetricColor(result.answer_relevancy)}`}>
                                      {(result.answer_relevancy * 100).toFixed(0)}%
                                    </span>
                                  ) : result.error_message ? (
                                    <span className="text-red-600 text-xs">✗</span>
                                  ) : (
                                    resumeState?.completedIndices?.includes(idx) ? (
                                      <span className="text-slate-400 text-xs">✓</span>
                                    ) : isBatchTesting ? (
                                      <Loader2 className="w-3 h-3 animate-spin mx-auto text-blue-600" />
                                    ) : (
                                      <span className="text-slate-300 text-xs">-</span>
                                    )
                                  )}
                                </td>
                                <td className="px-3 py-2 text-center">
                                  {result.context_precision != null ? (
                                    <span className={`font-medium ${getMetricColor(result.context_precision)}`}>
                                      {(result.context_precision * 100).toFixed(0)}%
                                    </span>
                                  ) : result.error_message ? (
                                    <span className="text-red-600 text-xs">✗</span>
                                  ) : (
                                    resumeState?.completedIndices?.includes(idx) ? (
                                      <span className="text-slate-400 text-xs">✓</span>
                                    ) : isBatchTesting ? (
                                      <Loader2 className="w-3 h-3 animate-spin mx-auto text-blue-600" />
                                    ) : (
                                      <span className="text-slate-300 text-xs">-</span>
                                    )
                                  )}
                                </td>
                                <td className="px-3 py-2 text-center">
                                  {result.answer_correctness != null ? (
                                    <span className={`font-medium ${getMetricColor(result.answer_correctness)}`}>
                                      {(result.answer_correctness * 100).toFixed(0)}%
                                    </span>
                                  ) : result.error_message ? (
                                    <span className="text-red-600 text-xs">✗</span>
                                  ) : (
                                    resumeState?.completedIndices?.includes(idx) ? (
                                      <span className="text-slate-400 text-xs">✓</span>
                                    ) : isBatchTesting ? (
                                      <Loader2 className="w-3 h-3 animate-spin mx-auto text-blue-600" />
                                    ) : (
                                      <span className="text-slate-300 text-xs">-</span>
                                    )
                                  )}
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
                      <div className="w-20 h-20 bg-indigo-100 rounded-full flex items-center justify-center mx-auto mb-4">
                        <FileJson className="w-10 h-10 text-indigo-400" />
                      </div>
                      <p className="text-sm font-medium">Test sonuçları burada görünecek</p>
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

      {/* Save Dataset Dialog */}
      <Dialog open={showSaveDatasetDialog} onOpenChange={setShowSaveDatasetDialog}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Veri Seti Kaydet</DialogTitle>
            <DialogDescription>
              Test verilerini daha sonra kullanmak üzere kaydedin.
            </DialogDescription>
          </DialogHeader>
          <div className="space-y-4 py-4">
            <div>
              <Label>Veri Seti Adı</Label>
              <Input
                value={datasetName}
                onChange={(e) => setDatasetName(e.target.value)}
                placeholder="örn: Dataset 1"
              />
            </div>
            <div>
              <Label>Açıklama (Opsiyonel)</Label>
              <Textarea
                value={datasetDescription}
                onChange={(e) => setDatasetDescription(e.target.value)}
                placeholder="Veri seti hakkında..."
                rows={3}
              />
            </div>
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setShowSaveDatasetDialog(false)}>İptal</Button>
            <Button onClick={handleSaveDataset} disabled={!datasetName}>Kaydet</Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Save Results Dialog */}
      <Dialog open={isSaveDialogOpen} onOpenChange={setIsSaveDialogOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Batch Sonuçlarını Kaydet</DialogTitle>
            <DialogDescription>
              {batchResults.length} test sonucunu kaydetmek üzeresiniz.
            </DialogDescription>
          </DialogHeader>
          <div className="space-y-4 py-4">
            <div>
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
                    key={g.name}
                    type="button"
                    onClick={() => setSaveGroupName(g.name)}
                    className="px-2 py-1 text-xs bg-slate-100 hover:bg-slate-200 rounded"
                  >
                    {g.name}
                  </button>
                ))}
              </div>
            )}
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setIsSaveDialogOpen(false)}>İptal</Button>
            <Button onClick={handleSaveBatchResults}>Kaydet</Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </>
  );
}
