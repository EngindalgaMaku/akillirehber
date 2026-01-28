"use client";

import { useState, useRef, useEffect } from "react";
import { api } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { Input } from "@/components/ui/input";
import { Card } from "@/components/ui/card";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Checkbox } from "@/components/ui/checkbox";
import { FileJson, ChevronDown, Loader2, Save, Download, Trash2, Zap, TrendingUp } from "lucide-react";
import { toast } from "sonner";

interface BatchTestSectionProps {
  selectedCourseId: number;
  onBatchTestComplete: () => void;
  savedResultsGroups: string[];
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

export function BatchTestSection({ selectedCourseId, onBatchTestComplete, savedResultsGroups }: BatchTestSectionProps) {
  const [isExpanded, setIsExpanded] = useState(false);
  const [batchTestJson, setBatchTestJson] = useState("");
  const [isBatchTesting, setIsBatchTesting] = useState(false);
  const [batchTestElapsedTime, setBatchTestElapsedTime] = useState("00:00:00");
  const batchTestStartTimeRef = useRef<number | null>(null);
  const elapsedTimeIntervalRef = useRef<NodeJS.Timeout | null>(null);
  
  // Realtime Results
  const [batchResults, setBatchResults] = useState<BatchTestResult[]>([]);
  const [currentTestIndex, setCurrentTestIndex] = useState(0);
  const [totalTests, setTotalTests] = useState(0);
  
  // W&B Integration
  const [enableWandbExport, setEnableWandbExport] = useState(true);
  const [wandbGroupName, setWandbGroupName] = useState("");
  
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

  const loadTestDatasets = async () => {
    try {
      const data = await api.getTestDatasets(selectedCourseId);
      setTestDatasets(data.datasets);
    } catch (error) {
      console.log("Failed to load test datasets");
    }
  };

  const handleLoadDataset = async (datasetId: string) => {
    try {
      const dataset = await api.getTestDataset(parseInt(datasetId));
      setBatchTestJson(JSON.stringify(dataset.test_cases, null, 2));
      setSelectedDataset(datasetId);
      // Otomatik grup adını dataset'ten al
      if (dataset.name && !wandbGroupName) {
        setWandbGroupName(dataset.name);
        setSaveGroupName(dataset.name);
      }
      toast.success(`"${dataset.name}" veri seti yüklendi`);
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "Veri seti yüklenemedi");
    }
  };

  const handleSaveDataset = async () => {
    if (!selectedCourseId || !batchTestJson || !datasetName) {
      toast.error("Lütfen JSON verisi ve veri seti adı girin");
      return;
    }

    try {
      const parsedData = JSON.parse(batchTestJson);
      await api.saveTestDataset({
        course_id: selectedCourseId,
        name: datasetName,
        description: datasetDescription,
        test_cases: parsedData
      });

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
      await api.deleteTestDataset(datasetId);
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

    setIsBatchTesting(true);
    setBatchResults([]);
    setCurrentTestIndex(0);
    batchTestStartTimeRef.current = Date.now();
    setBatchTestElapsedTime("00:00:00");
    
    try {
      const parsedData = JSON.parse(batchTestJson);
      let testCases = parsedData;

      // RAGAS test set formatı kontrolü
      if (parsedData.questions && Array.isArray(parsedData.questions)) {
        testCases = parsedData.questions;
        if (parsedData.name && !wandbGroupName) {
          setWandbGroupName(parsedData.name);
          setSaveGroupName(parsedData.name);
        }
      }

      setTotalTests(testCases.length);
      
      // Streaming endpoint'e bağlan
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
          test_cases: testCases,
          group_name: wandbGroupName || `RAGAS Batch ${new Date().toISOString()}`,
          enable_wandb: enableWandbExport,
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
      const results: BatchTestResult[] = [];

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value);
        const lines = chunk.split("\n");

        for (const line of lines) {
          if (line.startsWith("data: ")) {
            try {
              const data = JSON.parse(line.slice(6));

              if (data.event === "progress") {
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

                results.push(result);
                setBatchResults([...results]);
                setCurrentTestIndex(data.completed);

                if (!data.result.error_message) {
                  toast.success(`Test ${data.completed}/${data.total} tamamlandı`, {
                    id: `test-${data.index}`,
                    duration: 1000,
                  });
                } else {
                  toast.error(`Test ${data.completed} başarısız`, {
                    id: `test-error-${data.index}`,
                    duration: 2000,
                  });
                }
              } else if (data.event === "complete") {
                wandbUrl = data.wandb_url;
                
                const successCount = results.filter(r => !r.error_message).length;
                toast.success(`Tüm testler tamamlandı! ${successCount}/${results.length} başarılı`, {
                  duration: 3000,
                });
                
                // Saved Results'ı refresh et
                onBatchTestComplete();
                
                if (wandbUrl) {
                  const url = wandbUrl; // TypeScript için non-null garantisi
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
              } else if (data.event === "error") {
                throw new Error(data.error);
              }
            } catch (e) {
              console.error("Error parsing SSE data:", e);
            }
          }
        }
      }
      
      onBatchTestComplete();
      
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

  const handleSaveBatchResults = async () => {
    try {
      let successCount = 0;
      for (const result of batchResults) {
        try {
          await api.saveQuickTestResult({
            course_id: selectedCourseId,
            group_name: saveGroupName || undefined,
            question: result.question,
            ground_truth: result.ground_truth,
            llm_provider: "",
            llm_model: "",
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
              </div>

              {/* Results Section */}
              <div className="space-y-4">
                {isBatchTesting || batchResults.length > 0 ? (
                  <>
                    {/* Progress & Stats */}
                    {isBatchTesting && (
                      <div className="p-4 bg-gradient-to-br from-blue-50 to-indigo-50 rounded-xl border border-blue-200">
                        <div className="flex items-center justify-between mb-2">
                          <span className="text-sm font-medium text-blue-900">İlerleme</span>
                          <span className="text-xs text-blue-600">{currentTestIndex}/{totalTests}</span>
                        </div>
                        <div className="w-full bg-blue-200 rounded-full h-2 mb-2">
                          <div
                            className="bg-blue-600 h-2 rounded-full transition-all duration-300"
                            style={{ width: `${(currentTestIndex / totalTests) * 100}%` }}
                          />
                        </div>
                        <div className="flex items-center justify-between text-xs text-blue-600">
                          <span>Geçen Süre: {batchTestElapsedTime}</span>
                          <span>{Math.round((currentTestIndex / totalTests) * 100)}%</span>
                        </div>
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
                                <td className="px-3 py-2 text-slate-700 truncate max-w-[150px]" title={result.question}>
                                  {result.question}
                                </td>
                                <td className="px-3 py-2 text-center">
                                  {result.faithfulness != null ? (
                                    <span className={`font-medium ${getMetricColor(result.faithfulness)}`}>
                                      {(result.faithfulness * 100).toFixed(0)}%
                                    </span>
                                  ) : result.error_message ? (
                                    <span className="text-red-600 text-xs">✗</span>
                                  ) : (
                                    <Loader2 className="w-3 h-3 animate-spin mx-auto text-blue-600" />
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
                                    <Loader2 className="w-3 h-3 animate-spin mx-auto text-blue-600" />
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
                                    <Loader2 className="w-3 h-3 animate-spin mx-auto text-blue-600" />
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
                                    <Loader2 className="w-3 h-3 animate-spin mx-auto text-blue-600" />
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
            <Button variant="outline" onClick={() => setIsSaveDialogOpen(false)}>İptal</Button>
            <Button onClick={handleSaveBatchResults}>Kaydet</Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </>
  );
}
