"use client";

import { useState, useEffect } from "react";
import { QuickTestResult, api } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Loader2, Download, Eye } from "lucide-react";
import { toast } from "sonner";

interface StatisticsModalProps {
  isOpen: boolean;
  onClose: () => void;
  selectedCourseId: number;
  selectedGroup: string;
  aggregate: {
    avg_faithfulness?: number;
    avg_answer_relevancy?: number;
    avg_context_precision?: number;
    avg_context_recall?: number;
    avg_answer_correctness?: number;
    test_count?: number;
    test_parameters?: {
      llm_model?: string;
      llm_provider?: string;
      embedding_model?: string;
      evaluation_model?: string;
      search_alpha?: number;
      search_top_k?: number;
      reranker_used?: boolean;
      reranker_provider?: string | null;
      reranker_model?: string | null;
    };
  } | null;
}

export function StatisticsModal({ isOpen, onClose, selectedCourseId, selectedGroup, aggregate }: StatisticsModalProps) {
  const [results, setResults] = useState<QuickTestResult[]>([]);
  const [isLoading, setIsLoading] = useState(false);

  useEffect(() => {
    if (isOpen) {
      loadResults();
    }
  }, [isOpen, selectedCourseId, selectedGroup]);

  const loadResults = async () => {
    setIsLoading(true);
    try {
      let groupFilter: string | undefined;
      if (selectedGroup === "__all__" || selectedGroup === "") {
        groupFilter = undefined;
      } else if (selectedGroup === "__no_group__") {
        groupFilter = "";
      } else {
        groupFilter = selectedGroup;
      }
      
      const data = await api.getQuickTestResults(selectedCourseId, groupFilter, 0, 10000);
      const sortedResults = [...data.results].sort((a, b) => 
        a.question.localeCompare(b.question, 'tr')
      );
      setResults(sortedResults);
    } catch {
      console.log("Failed to load results for statistics");
    } finally {
      setIsLoading(false);
    }
  };

  const copyTableToExcel = () => {
    const headers = ["#", "Soru", "Faithfulness (%)", "Answer Relevancy (%)", "Context Precision (%)", "Context Recall (%)", "Answer Correctness (%)", "Gecikme (ms)"];
    const rows = results.map((r, idx) => [
      idx + 1,
      `"${r.question.replace(/"/g, '""')}"`,
      r.faithfulness != null ? (r.faithfulness * 100).toFixed(2) : "-",
      r.answer_relevancy != null ? (r.answer_relevancy * 100).toFixed(2) : "-",
      r.context_precision != null ? (r.context_precision * 100).toFixed(2) : "-",
      r.context_recall != null ? (r.context_recall * 100).toFixed(2) : "-",
      r.answer_correctness != null ? (r.answer_correctness * 100).toFixed(2) : "-",
      r.latency_ms
    ]);
    
    const csv = [headers.join("\t"), ...rows.map(row => row.join("\t"))].join("\n");
    navigator.clipboard.writeText(csv).then(() => {
      toast.success("Tablo kopyalandı! Excel'e yapıştırabilirsiniz.");
    }).catch(() => {
      toast.error("Kopyalama başarısız");
    });
  };

  const getMetricColor = (value?: number | null) => {
    if (value === undefined || value === null) return "text-slate-400";
    if (value >= 0.8) return "text-emerald-600";
    if (value >= 0.6) return "text-amber-600";
    return "text-red-600";
  };

  return (
    <Dialog open={isOpen} onOpenChange={onClose}>
      <DialogContent className="max-w-6xl max-h-[90vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle>
            {selectedGroup && selectedGroup !== "__all__"
              ? `${selectedGroup === "__no_group__" ? "Grupsuz" : selectedGroup} grubunun detaylı istatistikleri`
              : "Tüm sonuçların detaylı istatistikleri"}
          </DialogTitle>
          <DialogDescription>
            RAGAS metrikleri ve performans analizi
          </DialogDescription>
        </DialogHeader>

        {isLoading ? (
          <div className="p-8 text-center text-slate-500">
            <Loader2 className="w-6 h-6 animate-spin mx-auto mb-2" />
            <p className="text-sm">Sonuçlar yükleniyor...</p>
          </div>
        ) : results.length === 0 ? (
          <div className="p-8 text-center text-slate-500">
            <p className="text-sm">Kayıtlı sonuç bulunamadı</p>
          </div>
        ) : (
          <div className="space-y-6 py-4">
            {/* Detaylı Sonuçlar Tablosu */}
            <div>
              <div className="flex items-center justify-between mb-3">
                <h3 className="text-sm font-semibold text-slate-900">Detaylı Sonuçlar</h3>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={copyTableToExcel}
                  className="h-7 text-xs border-indigo-300 text-indigo-700 hover:bg-indigo-100"
                >
                  <Download className="w-3 h-3 mr-1" /> Kopyala (Excel)
                </Button>
              </div>
              <div className="border border-slate-200 rounded-lg overflow-hidden">
                <div className="max-h-[400px] overflow-y-auto">
                  <table className="w-full text-xs">
                    <thead className="bg-slate-50 sticky top-0">
                      <tr>
                        <th className="px-3 py-2 text-left font-medium text-slate-700">#</th>
                        <th className="px-3 py-2 text-left font-medium text-slate-700">Soru</th>
                        <th className="px-3 py-2 text-center font-medium text-slate-700">Faithfulness</th>
                        <th className="px-3 py-2 text-center font-medium text-slate-700">Relevancy</th>
                        <th className="px-3 py-2 text-center font-medium text-slate-700">Precision</th>
                        <th className="px-3 py-2 text-center font-medium text-slate-700">Recall</th>
                        <th className="px-3 py-2 text-center font-medium text-slate-700">Correctness</th>
                        <th className="px-3 py-2 text-center font-medium text-slate-700">Gecikme</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-100">
                      {results.map((result, idx) => (
                        <tr key={result.id} className="hover:bg-slate-50">
                          <td className="px-3 py-2 text-slate-600 font-medium">{idx + 1}</td>
                          <td className="px-3 py-2 text-slate-900 truncate max-w-[200px]" title={result.question}>
                            {result.question}
                          </td>
                          <td className="px-3 py-2 text-center">
                            {result.faithfulness != null ? (
                              <span className={`font-medium ${getMetricColor(result.faithfulness)}`}>
                                {(result.faithfulness * 100).toFixed(1)}%
                              </span>
                            ) : (
                              <span className="text-slate-400">-</span>
                            )}
                          </td>
                          <td className="px-3 py-2 text-center">
                            {result.answer_relevancy != null ? (
                              <span className={`font-medium ${getMetricColor(result.answer_relevancy)}`}>
                                {(result.answer_relevancy * 100).toFixed(1)}%
                              </span>
                            ) : (
                              <span className="text-slate-400">-</span>
                            )}
                          </td>
                          <td className="px-3 py-2 text-center">
                            {result.context_precision != null ? (
                              <span className={`font-medium ${getMetricColor(result.context_precision)}`}>
                                {(result.context_precision * 100).toFixed(1)}%
                              </span>
                            ) : (
                              <span className="text-slate-400">-</span>
                            )}
                          </td>
                          <td className="px-3 py-2 text-center">
                            {result.context_recall != null ? (
                              <span className={`font-medium ${getMetricColor(result.context_recall)}`}>
                                {(result.context_recall * 100).toFixed(1)}%
                              </span>
                            ) : (
                              <span className="text-slate-400">-</span>
                            )}
                          </td>
                          <td className="px-3 py-2 text-center">
                            {result.answer_correctness != null ? (
                              <span className={`font-medium ${getMetricColor(result.answer_correctness)}`}>
                                {(result.answer_correctness * 100).toFixed(1)}%
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
            </div>

            {/* Performans Analizi */}
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
                    {results.some(r => r.faithfulness != null) && (
                      <tr className="hover:bg-slate-50">
                        <td className="px-4 py-3 font-medium text-purple-700">Faithfulness</td>
                        <td className="px-4 py-3 text-center text-emerald-600 font-bold">
                          {(Math.max(...results.filter(r => r.faithfulness != null).map(r => r.faithfulness!)) * 100).toFixed(1)}%
                        </td>
                        <td className="px-4 py-3 text-center text-red-600 font-bold">
                          {(Math.min(...results.filter(r => r.faithfulness != null).map(r => r.faithfulness!)) * 100).toFixed(1)}%
                        </td>
                        <td className="px-4 py-3 text-center font-bold">
                          {(results.filter(r => r.faithfulness != null).reduce((sum, r) => sum + r.faithfulness!, 0) / results.filter(r => r.faithfulness != null).length * 100).toFixed(1)}%
                        </td>
                      </tr>
                    )}
                    {results.some(r => r.answer_correctness != null) && (
                      <tr className="hover:bg-slate-50">
                        <td className="px-4 py-3 font-medium text-purple-700">Answer Correctness</td>
                        <td className="px-4 py-3 text-center text-emerald-600 font-bold">
                          {(Math.max(...results.filter(r => r.answer_correctness != null).map(r => r.answer_correctness!)) * 100).toFixed(1)}%
                        </td>
                        <td className="px-4 py-3 text-center text-red-600 font-bold">
                          {(Math.min(...results.filter(r => r.answer_correctness != null).map(r => r.answer_correctness!)) * 100).toFixed(1)}%
                        </td>
                        <td className="px-4 py-3 text-center font-bold">
                          {(results.filter(r => r.answer_correctness != null).reduce((sum, r) => sum + r.answer_correctness!, 0) / results.filter(r => r.answer_correctness != null).length * 100).toFixed(1)}%
                        </td>
                      </tr>
                    )}
                    <tr className="hover:bg-slate-50">
                      <td className="px-4 py-3 font-medium text-slate-700">Gecikme</td>
                      <td className="px-4 py-3 text-center text-emerald-600 font-bold">
                        {Math.min(...results.map(r => r.latency_ms))}ms
                      </td>
                      <td className="px-4 py-3 text-center text-red-600 font-bold">
                        {Math.max(...results.map(r => r.latency_ms))}ms
                      </td>
                      <td className="px-4 py-3 text-center font-bold">
                        {(results.reduce((sum, r) => sum + r.latency_ms, 0) / results.length).toFixed(0)}ms
                      </td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </div>

            {/* Test Parametreleri */}
            {aggregate?.test_parameters && (
              <div>
                <h3 className="text-sm font-semibold text-slate-900 mb-3">Test Parametreleri</h3>
                <div className="p-4 bg-gradient-to-br from-indigo-50 to-purple-50 rounded-xl border border-indigo-200">
                  {/* Test Tarihi */}
                  {results.length > 0 && results[0].created_at && (
                    <div className="mb-3 pb-3 border-b border-indigo-200">
                      <p className="text-xs text-indigo-600 font-medium mb-1">Test Tarihi</p>
                      <p className="text-sm font-bold text-slate-900">
                        {new Date(results[0].created_at).toLocaleString("tr-TR", {
                          year: "numeric",
                          month: "long",
                          day: "numeric",
                          hour: "2-digit",
                          minute: "2-digit",
                          second: "2-digit"
                        })}
                      </p>
                    </div>
                  )}
                  
                  <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
                    {aggregate.test_parameters.llm_model && (
                      <div className="p-3 bg-white rounded-lg border border-indigo-100">
                        <p className="text-xs text-indigo-600 font-medium mb-1">LLM Model (Cevap Üretimi)</p>
                        <p className="text-sm font-bold text-slate-900">{aggregate.test_parameters.llm_model}</p>
                      </div>
                    )}
                    {aggregate.test_parameters.evaluation_model && (
                      <div className="p-3 bg-white rounded-lg border border-purple-100">
                        <p className="text-xs text-purple-600 font-medium mb-1">Evaluation Model (RAGAS)</p>
                        <p className="text-sm font-bold text-slate-900">{aggregate.test_parameters.evaluation_model}</p>
                      </div>
                    )}
                    {aggregate.test_parameters.embedding_model && (
                      <div className="p-3 bg-white rounded-lg border border-indigo-100">
                        <p className="text-xs text-indigo-600 font-medium mb-1">Embedding Model (Retrieval)</p>
                        <p className="text-sm font-bold text-slate-900">{aggregate.test_parameters.embedding_model}</p>
                      </div>
                    )}
                    {aggregate.test_parameters.search_alpha != null && (
                      <div className="p-3 bg-white rounded-lg border border-indigo-100">
                        <p className="text-xs text-indigo-600 font-medium mb-1">Search Alpha</p>
                        <p className="text-sm font-bold text-slate-900">{aggregate.test_parameters.search_alpha}</p>
                      </div>
                    )}
                    {aggregate.test_parameters.search_top_k && (
                      <div className="p-3 bg-white rounded-lg border border-indigo-100">
                        <p className="text-xs text-indigo-600 font-medium mb-1">Top K</p>
                        <p className="text-sm font-bold text-slate-900">{aggregate.test_parameters.search_top_k}</p>
                      </div>
                    )}
                    {aggregate.test_parameters.reranker_used && (
                      <div className="p-3 bg-white rounded-lg border border-emerald-100">
                        <p className="text-xs text-emerald-600 font-medium mb-1">Reranker</p>
                        <p className="text-sm font-bold text-slate-900">
                          {aggregate.test_parameters.reranker_provider}/{aggregate.test_parameters.reranker_model}
                        </p>
                      </div>
                    )}
                  </div>
                </div>
              </div>
            )}
          </div>
        )}

        <DialogFooter>
          <Button variant="outline" onClick={onClose}>
            Kapat
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}