"use client";

import { useState } from "react";
import { QuickTestResult, api } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { History, ChevronDown, Eye, Trash2, Loader2, Download, Target, Settings, RefreshCw } from "lucide-react";
import { toast } from "sonner";
import { StatisticsModal } from "./StatisticsModal";

interface SavedResultsSectionProps {
  selectedCourseId: number;
  savedResults: QuickTestResult[];
  savedResultsGroups: string[];
  savedResultsTotal: number;
  savedResultsAggregate: any;
  selectedGroup: string;
  setSelectedGroup: (group: string) => void;
  resultsPage: number;
  onLoadMore: () => void;
  onDelete: () => void;
}

export function SavedResultsSection({
  selectedCourseId,
  savedResults,
  savedResultsGroups,
  savedResultsTotal,
  savedResultsAggregate,
  selectedGroup,
  setSelectedGroup,
  resultsPage,
  onLoadMore,
  onDelete
}: SavedResultsSectionProps) {
  const [isExpanded, setIsExpanded] = useState(false);
  const [isLoadingMore, setIsLoadingMore] = useState(false);
  const [viewingResult, setViewingResult] = useState<QuickTestResult | null>(null);
  const [isStatisticsModalOpen, setIsStatisticsModalOpen] = useState(false);
  
  // W&B States
  const [isWandbExporting, setIsWandbExporting] = useState(false);
  const [isWandbRunsModalOpen, setIsWandbRunsModalOpen] = useState(false);
  const [wandbRuns, setWandbRuns] = useState<any[]>([]);
  const [isLoadingWandbRuns, setIsLoadingWandbRuns] = useState(false);
  const [updatingRunIds, setUpdatingRunIds] = useState<Set<string>>(new Set());

  const handleLoadMore = async () => {
    setIsLoadingMore(true);
    await onLoadMore();
    setIsLoadingMore(false);
  };

  const handleDelete = async (id: number) => {
    if (!confirm("Bu sonucu silmek istediğinizden emin misiniz?")) return;
    try {
      await api.deleteQuickTestResult(id);
      toast.success("Sonuç silindi");
      onDelete();
      if (viewingResult?.id === id) {
        setViewingResult(null);
      }
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "Silme başarısız");
    }
  };

  const exportToCSV = async () => {
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

      if (data.results.length === 0) {
        toast.error("İndirilecek sonuç yok");
        return;
      }

      const headers = [
        "Soru No",
        "Faithfulness (%)",
        "Answer Relevancy (%)",
        "Context Precision (%)",
        "Context Recall (%)",
        "Answer Correctness (%)",
        "Gecikme (ms)"
      ];

      const rows = data.results.map((r, idx) => [
        idx + 1,
        r.faithfulness != null ? (r.faithfulness * 100).toFixed(2) : "-",
        r.answer_relevancy != null ? (r.answer_relevancy * 100).toFixed(2) : "-",
        r.context_precision != null ? (r.context_precision * 100).toFixed(2) : "-",
        r.context_recall != null ? (r.context_recall * 100).toFixed(2) : "-",
        r.answer_correctness != null ? (r.answer_correctness * 100).toFixed(2) : "-",
        r.latency_ms
      ]);

      const csv = [headers.join(","), ...rows.map(row => row.join(","))].join("\n");
      const blob = new Blob(["\ufeff" + csv], { type: "text/csv;charset=utf-8;" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      const groupName = selectedGroup && selectedGroup !== "__all__" 
        ? (selectedGroup === "__no_group__" ? "grupsuz" : selectedGroup)
        : "tum-sonuclar";
      a.download = `ragas-scores-${groupName}-${Date.now()}.csv`;
      a.click();
      URL.revokeObjectURL(url);

      toast.success(`${data.results.length} sonuç CSV olarak indirildi`);
    } catch (error) {
      toast.error("CSV indirme başarısız");
      console.error(error);
    }
  };

  // W&B Export Function - Placeholder for backend implementation
  const exportSelectedGroupToWandb = async () => {
    if (!selectedGroup || selectedGroup === "__all__" || selectedGroup === "__no_group__") {
      toast.error("W&B'ye göndermek için bir grup seçin");
      return;
    }

    toast.info("W&B entegrasyonu backend tarafında implement edilecek");
  };

  // Load W&B Runs - Placeholder
  const loadWandbRuns = async () => {
    toast.info("W&B run yönetimi backend tarafında implement edilecek");
    setWandbRuns([]);
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

  return (
    <>
      <Card className="overflow-hidden border-0 shadow-lg bg-white">
        <button
          onClick={() => setIsExpanded(!isExpanded)}
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
          <div className={`p-2 rounded-full bg-slate-100 transition-transform duration-200 ${isExpanded ? 'rotate-180' : ''}`}>
            <ChevronDown className="w-5 h-5 text-slate-600" />
          </div>
        </button>

        {isExpanded && (
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
            {savedResultsAggregate && savedResultsAggregate.test_count > 0 && (
              <div className="mb-6 p-4 bg-gradient-to-br from-purple-50 to-indigo-50 rounded-xl border border-purple-200">
                <div className="flex items-center justify-between mb-3">
                  <h3 className="text-sm font-semibold text-purple-900">
                    {selectedGroup && selectedGroup !== "__all__" 
                      ? `"${selectedGroup === "__no_group__" ? "Grupsuz" : selectedGroup}" Grup İstatistikleri (${savedResultsAggregate.test_count} Test)`
                      : `Genel İstatistikler (${savedResultsAggregate.test_count} Test)`}
                  </h3>
                  <div className="flex gap-2">
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => setIsStatisticsModalOpen(true)}
                      className="h-7 text-xs border-indigo-300 text-indigo-700 hover:bg-indigo-100"
                    >
                      <Target className="w-3 h-3 mr-1" /> İstatistik Tablosu
                    </Button>
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={exportToCSV}
                      className="h-7 text-xs border-purple-300 text-purple-700 hover:bg-purple-100"
                    >
                      <Download className="w-3 h-3 mr-1" /> CSV İndir
                    </Button>
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={exportSelectedGroupToWandb}
                      disabled={
                        isWandbExporting ||
                        !selectedGroup ||
                        selectedGroup === "__all__" ||
                        selectedGroup === "__no_group__"
                      }
                      className="h-7 text-xs border-emerald-300 text-emerald-700 hover:bg-emerald-100"
                    >
                      {isWandbExporting ? (
                        <>
                          <Loader2 className="w-3 h-3 mr-1 animate-spin" /> Aktarılıyor...
                        </>
                      ) : (
                        <>W&B'ye Gönder</>
                      )}
                    </Button>
                  </div>
                </div>
                <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
                  {savedResultsAggregate.avg_faithfulness != null && (
                    <div className="p-3 bg-white rounded-lg border border-purple-100">
                      <p className="text-xs text-purple-600 mb-1">Ort. Faithfulness</p>
                      <p className="text-xl font-bold text-purple-700">
                        {(savedResultsAggregate.avg_faithfulness * 100).toFixed(1)}%
                      </p>
                    </div>
                  )}
                  {savedResultsAggregate.avg_answer_relevancy != null && (
                    <div className="p-3 bg-white rounded-lg border border-purple-100">
                      <p className="text-xs text-purple-600 mb-1">Ort. Relevancy</p>
                      <p className="text-xl font-bold text-purple-700">
                        {(savedResultsAggregate.avg_answer_relevancy * 100).toFixed(1)}%
                      </p>
                    </div>
                  )}
                  {savedResultsAggregate.avg_context_precision != null && (
                    <div className="p-3 bg-white rounded-lg border border-purple-100">
                      <p className="text-xs text-purple-600 mb-1">Ort. Precision</p>
                      <p className="text-xl font-bold text-purple-700">
                        {(savedResultsAggregate.avg_context_precision * 100).toFixed(1)}%
                      </p>
                    </div>
                  )}
                  {savedResultsAggregate.avg_context_recall != null && (
                    <div className="p-3 bg-white rounded-lg border border-purple-100">
                      <p className="text-xs text-purple-600 mb-1">Ort. Recall</p>
                      <p className="text-xl font-bold text-purple-700">
                        {(savedResultsAggregate.avg_context_recall * 100).toFixed(1)}%
                      </p>
                    </div>
                  )}
                  {savedResultsAggregate.avg_answer_correctness != null && (
                    <div className="p-3 bg-white rounded-lg border border-purple-100">
                      <p className="text-xs text-purple-600 mb-1">Ort. Correctness</p>
                      <p className="text-xl font-bold text-purple-700">
                        {(savedResultsAggregate.avg_answer_correctness * 100).toFixed(1)}%
                      </p>
                    </div>
                  )}
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
                            <span className="px-2 py-0.5 text-xs bg-purple-100 text-purple-700 rounded-full font-medium">
                              {result.group_name}
                            </span>
                          )}
                          <span className="text-xs text-slate-500">
                            {new Date(result.created_at).toLocaleString("tr-TR")}
                          </span>
                        </div>
                        <p className="text-sm font-medium text-slate-900 truncate">{result.question}</p>
                        <div className="flex flex-wrap items-center gap-3 mt-2 text-xs">
                          <span className={getMetricColor(result.faithfulness)}>
                            Sadakat: {result.faithfulness != null ? `${(result.faithfulness * 100).toFixed(0)}%` : "N/A"}
                          </span>
                          <span className={getMetricColor(result.context_recall)}>
                            Recall: {result.context_recall != null ? `${(result.context_recall * 100).toFixed(0)}%` : "N/A"}
                          </span>
                          <span className={getMetricColor(result.answer_correctness)}>
                            Doğruluk: {result.answer_correctness != null ? `${(result.answer_correctness * 100).toFixed(0)}%` : "N/A"}
                          </span>
                          <span className="text-slate-400">{result.latency_ms}ms</span>
                        </div>
                      </div>
                      <div className="flex items-center gap-1 ml-2">
                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={() => setViewingResult(result)}
                          className="text-slate-600 hover:text-purple-600"
                        >
                          <Eye className="w-4 h-4" />
                        </Button>
                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={() => handleDelete(result.id)}
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
                    onClick={handleLoadMore}
                    disabled={isLoadingMore}
                    className="w-full"
                  >
                    {isLoadingMore ? (
                      <>
                        <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                        Yükleniyor...
                      </>
                    ) : (
                      <>Daha Fazla ({savedResults.length}/{savedResultsTotal})</>
                    )}
                  </Button>
                )}
              </div>
            )}
          </div>
        )}
      </Card>

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
                <p className="text-sm font-medium text-slate-700 mb-1">Soru</p>
                <p className="p-3 bg-slate-50 rounded-lg text-sm">{viewingResult.question}</p>
              </div>
              <div>
                <p className="text-sm font-medium text-slate-700 mb-1">Doğru Cevap</p>
                <p className="p-3 bg-slate-50 rounded-lg text-sm">{viewingResult.ground_truth}</p>
              </div>
              <div>
                <p className="text-sm font-medium text-slate-700 mb-1">Üretilen Cevap</p>
                <p className="p-3 bg-slate-50 rounded-lg text-sm">{viewingResult.generated_answer}</p>
              </div>
              <div className="grid grid-cols-3 gap-3">
                <div className={`p-3 rounded-lg border ${getMetricBgColor(viewingResult.faithfulness)}`}>
                  <p className="text-xs text-slate-600">Faithfulness</p>
                  <p className={`text-xl font-bold ${getMetricColor(viewingResult.faithfulness)}`}>
                    {viewingResult.faithfulness != null ? `${(viewingResult.faithfulness * 100).toFixed(1)}%` : "N/A"}
                  </p>
                </div>
                <div className={`p-3 rounded-lg border ${getMetricBgColor(viewingResult.answer_relevancy)}`}>
                  <p className="text-xs text-slate-600">Answer Relevancy</p>
                  <p className={`text-xl font-bold ${getMetricColor(viewingResult.answer_relevancy)}`}>
                    {viewingResult.answer_relevancy != null ? `${(viewingResult.answer_relevancy * 100).toFixed(1)}%` : "N/A"}
                  </p>
                </div>
                <div className={`p-3 rounded-lg border ${getMetricBgColor(viewingResult.context_precision)}`}>
                  <p className="text-xs text-slate-600">Context Precision</p>
                  <p className={`text-xl font-bold ${getMetricColor(viewingResult.context_precision)}`}>
                    {viewingResult.context_precision != null ? `${(viewingResult.context_precision * 100).toFixed(1)}%` : "N/A"}
                  </p>
                </div>
                <div className={`p-3 rounded-lg border ${getMetricBgColor(viewingResult.context_recall)}`}>
                  <p className="text-xs text-slate-600">Context Recall</p>
                  <p className={`text-xl font-bold ${getMetricColor(viewingResult.context_recall)}`}>
                    {viewingResult.context_recall != null ? `${(viewingResult.context_recall * 100).toFixed(1)}%` : "N/A"}
                  </p>
                </div>
                <div className={`p-3 rounded-lg border ${getMetricBgColor(viewingResult.answer_correctness)}`}>
                  <p className="text-xs text-slate-600">Answer Correctness</p>
                  <p className={`text-xl font-bold ${getMetricColor(viewingResult.answer_correctness)}`}>
                    {viewingResult.answer_correctness != null ? `${(viewingResult.answer_correctness * 100).toFixed(1)}%` : "N/A"}
                  </p>
                </div>
                <div className="p-3 rounded-lg bg-slate-50 border border-slate-200">
                  <p className="text-xs text-slate-600">Gecikme</p>
                  <p className="text-xl font-bold text-slate-900">{viewingResult.latency_ms}ms</p>
                </div>
              </div>
            </div>
          )}
          <DialogFooter>
            <Button variant="outline" onClick={() => setViewingResult(null)}>Kapat</Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Statistics Modal */}
      <StatisticsModal
        isOpen={isStatisticsModalOpen}
        onClose={() => setIsStatisticsModalOpen(false)}
        selectedCourseId={selectedCourseId}
        selectedGroup={selectedGroup}
      />
    </>
  );
}