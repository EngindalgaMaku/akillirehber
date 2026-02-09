"use client";

import { useEffect, useMemo, useState } from "react";
import Link from "next/link";
import { useParams } from "next/navigation";
import { toast } from "sonner";
import { ArrowLeft, Download, FileText, SquarePen, Trash2, BarChart3 } from "lucide-react";

import { api, QuickTestResult, Course } from "@/lib/api";
import { useAuth } from "@/lib/auth-context";
import { Button } from "@/components/ui/button";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Checkbox } from "@/components/ui/checkbox";

type RagasAggregate = {
  test_count?: number;
  avg_faithfulness?: number;
  avg_answer_relevancy?: number;
  avg_context_precision?: number;
  avg_context_recall?: number;
  avg_answer_correctness?: number;
  test_parameters?: Record<string, unknown>;
};

type TestParameters = {
  llm_model?: string;
  llm_provider?: string;
  evaluation_model?: string;
  embedding_model?: string;
  search_top_k?: number;
  search_alpha?: number;
  reranker_used?: boolean;
  reranker_provider?: string;
  reranker_model?: string;
};

export default function RagasGroupResultsPage() {
  const { user } = useAuth();
  const params = useParams<{ group: string }>();

  const groupName = useMemo(() => {
    const raw = params?.group;
    if (!raw) return "";
    try {
      return decodeURIComponent(String(raw));
    } catch {
      return String(raw);
    }
  }, [params]);

  const [courseId, setCourseId] = useState<number | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [results, setResults] = useState<QuickTestResult[]>([]);
  const [total, setTotal] = useState(0);
  const [aggregate, setAggregate] = useState<RagasAggregate | null>(null);

  const PAGE_SIZE = 10;
  const [page, setPage] = useState(1);
  const [selectedResult, setSelectedResult] = useState<QuickTestResult | null>(null);
  const [deletingResultId, setDeletingResultId] = useState<number | null>(null);
  const [isExportingToWandB, setIsExportingToWandB] = useState(false);
  
  // Multi-select for statistics
  const [selectedResultIds, setSelectedResultIds] = useState<Set<number>>(new Set());
  const [showStatisticsModal, setShowStatisticsModal] = useState(false);

  useEffect(() => {
    const raw = localStorage.getItem("ragas_selected_course_id");
    const parsed = raw ? Number(raw) : NaN;
    if (Number.isFinite(parsed)) {
      setCourseId(parsed);
    } else {
      // localStorage'da yoksa kursları yükleyip ilkini seç
      api.getCourses().then((courses) => {
        if (courses.length > 0) {
          setCourseId(courses[0].id);
          localStorage.setItem("ragas_selected_course_id", courses[0].id.toString());
        }
      }).catch(() => {
        // Kurslar yüklenemezse courseId null kalır
      });
    }
  }, []);

  useEffect(() => {
    const run = async () => {
      if (!courseId || !groupName) return;
      setIsLoading(true);
      try {
        const data = await api.getQuickTestResults(courseId, groupName, 0, 10000);
        setResults(data.results || []);
        setTotal(data.total || 0);
        setAggregate((data.aggregate as RagasAggregate | null | undefined) || null);
        setPage(1);
      } catch (error) {
        console.error("Failed to load group results:", error);
        toast.error("Sonuçlar yüklenemedi");
        setResults([]);
        setTotal(0);
        setAggregate(null);
      } finally {
        setIsLoading(false);
      }
    };

    run();
  }, [courseId, groupName]);

  const reloadResults = async () => {
    if (!courseId || !groupName) return;
    setIsLoading(true);
    try {
      const data = await api.getQuickTestResults(courseId, groupName, 0, 10000);
      setResults(data.results || []);
      setTotal(data.total || 0);
      setAggregate((data.aggregate as RagasAggregate | null | undefined) || null);
      setPage(1);
    } catch (error) {
      console.error("Failed to load group results:", error);
      toast.error("Sonuçlar yüklenemedi");
      setResults([]);
      setTotal(0);
      setAggregate(null);
    } finally {
      setIsLoading(false);
    }
  };

  const handleDeleteResult = async (result: QuickTestResult) => {
    if (!result?.id) return;
    const msg = "Bu sonucu silmek istediğinizden emin misiniz?";
    if (!confirm(msg)) return;

    setDeletingResultId(result.id);
    try {
      await api.deleteQuickTestResult(result.id);
      toast.success("Sonuç silindi");
      if (selectedResult?.id === result.id) {
        setSelectedResult(null);
      }
      await reloadResults();
    } catch (error) {
      console.error(error);
      toast.error(error instanceof Error ? error.message : "Silme başarısız");
    } finally {
      setDeletingResultId(null);
    }
  };

  const exportToCSV = async () => {
    if (!courseId || !groupName) return;
    try {
      const data = await api.getQuickTestResults(courseId, groupName, 0, 10000);
      if (data.results.length === 0) {
        toast.error("İndirilecek sonuç yok");
        return;
      }

      const headers = [
        "Soru No",
        "Soru",
        "Faithfulness (%)",
        "Answer Relevancy (%)",
        "Context Precision (%)",
        "Context Recall (%)",
        "Answer Correctness (%)",
        "Gecikme (ms)",
      ];

      const rows = data.results.map((r, idx) => [
        idx + 1,
        String(r.question ?? "").replaceAll("\"", "\"\""),
        r.faithfulness != null ? (r.faithfulness * 100).toFixed(2) : "-",
        r.answer_relevancy != null ? (r.answer_relevancy * 100).toFixed(2) : "-",
        r.context_precision != null ? (r.context_precision * 100).toFixed(2) : "-",
        r.context_recall != null ? (r.context_recall * 100).toFixed(2) : "-",
        r.answer_correctness != null ? (r.answer_correctness * 100).toFixed(2) : "-",
        r.latency_ms ?? "-",
      ]);

      const csv = [
        headers.join(","),
        ...rows.map((row) => row.map((cell) => `"${cell}"`).join(",")),
      ].join("\n");

      const blob = new Blob(["\ufeff" + csv], { type: "text/csv;charset=utf-8;" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `ragas-scores-${groupName}-${Date.now()}.csv`;
      a.click();
      URL.revokeObjectURL(url);
    } catch (error) {
      console.error(error);
      toast.error("CSV indirme başarısız");
    }
  };

  const exportToPdf = async () => {
    if (!courseId || !groupName) return;
    try {
      const data = await api.getQuickTestResults(courseId, groupName, 0, 10000);
      if (data.results.length === 0) {
        toast.error("İndirilecek sonuç yok");
        return;
      }

      const agg = (data.aggregate as RagasAggregate | null | undefined) || aggregate;
      const now = new Date().toLocaleString("tr-TR");

      const escapeHtml = (s: unknown) =>
        String(s ?? "")
          .replace(/&/g, "&amp;")
          .replace(/</g, "&lt;")
          .replace(/>/g, "&gt;")
          .replace(/\"/g, "&quot;")
          .replace(/'/g, "&#039;");

      const fmtPct = (v: number | null | undefined) => (v == null ? "-" : `${(v * 100).toFixed(1)}%`);

      const rowsHtml = data.results
        .map((r, i) => {
          return `
            <tr>
              <td class="n">${i + 1}</td>
              <td class="q">${escapeHtml(r.question)}</td>
              <td class="m">${fmtPct(r.faithfulness)}</td>
              <td class="m">${fmtPct(r.answer_relevancy)}</td>
              <td class="m">${fmtPct(r.context_precision)}</td>
              <td class="m">${fmtPct(r.context_recall)}</td>
              <td class="m">${fmtPct(r.answer_correctness)}</td>
              <td class="m">${escapeHtml(r.latency_ms)}ms</td>
            </tr>
          `;
        })
        .join("");

      const htmlContent = `
        <!DOCTYPE html>
        <html>
          <head>
            <meta charset="utf-8" />
            <title>RAGAS Sonuçları - ${escapeHtml(groupName)}</title>
            <style>
              body { font-family: Arial, sans-serif; padding: 24px; color: #0f172a; }
              h1 { font-size: 18px; margin: 0 0 6px 0; }
              .meta { color: #475569; font-size: 12px; margin-bottom: 16px; }
              .stats { display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: 8px; margin: 14px 0 16px 0; }
              .card { border: 1px solid #e2e8f0; border-radius: 10px; padding: 10px; }
              .label { font-size: 11px; color: #64748b; }
              .value { font-size: 14px; font-weight: 700; margin-top: 3px; }
              table { width: 100%; border-collapse: collapse; font-size: 10px; }
              thead th { text-align: left; background: #f8fafc; border: 1px solid #e2e8f0; padding: 8px; position: sticky; top: 0; }
              tbody td { border: 1px solid #e2e8f0; padding: 8px; vertical-align: top; }
              .n { width: 40px; color: #475569; }
              .q { width: 45%; }
              .m { width: 80px; text-align: right; }
              @media print { .no-print { display: none; } }
            </style>
          </head>
          <body>
            <div class="no-print" style="margin-bottom:12px;color:#475569;font-size:12px;">
              PDF olarak kaydetmek için yazdır penceresinde &quot;PDF olarak kaydet&quot; seçin.
            </div>
            <h1>RAGAS Kaydedilen Sonuçlar</h1>
            <div class="meta">Grup: <b>${escapeHtml(groupName)}</b> | Oluşturma: ${escapeHtml(now)} | Toplam: ${escapeHtml(data.total)}</div>

            ${agg && agg.test_count ? `
              <div class="stats">
                <div class="card"><div class="label">Ort. Faithfulness</div><div class="value">${fmtPct(agg.avg_faithfulness)}</div></div>
                <div class="card"><div class="label">Ort. Relevancy</div><div class="value">${fmtPct(agg.avg_answer_relevancy)}</div></div>
                <div class="card"><div class="label">Ort. Precision</div><div class="value">${fmtPct(agg.avg_context_precision)}</div></div>
                <div class="card"><div class="label">Ort. Recall</div><div class="value">${fmtPct(agg.avg_context_recall)}</div></div>
                <div class="card"><div class="label">Ort. Correctness</div><div class="value">${fmtPct(agg.avg_answer_correctness)}</div></div>
              </div>
            ` : ""}

            <table>
              <thead>
                <tr>
                  <th>#</th>
                  <th>Soru</th>
                  <th>Faith.</th>
                  <th>Rel.</th>
                  <th>Prec.</th>
                  <th>Recall</th>
                  <th>Corr.</th>
                  <th>Latency</th>
                </tr>
              </thead>
              <tbody>
                ${rowsHtml}
              </tbody>
            </table>
          </body>
        </html>
      `;

      const printWindow = window.open("", "_blank");
      if (!printWindow) {
        toast.error("PDF penceresi açılamadı. Popup engelleyici açık olabilir.");
        return;
      }

      printWindow.document.write(htmlContent);
      printWindow.document.close();
      printWindow.onload = () => {
        printWindow.print();
      };
    } catch (error) {
      console.error(error);
      toast.error("PDF indirme başarısız");
    }
  };

  const exportToWandB = async () => {
    if (!courseId || !groupName) return;
    
    setIsExportingToWandB(true);
    try {
      const data = await api.exportQuickTestResultsToWandB(courseId, groupName);
      
      toast.success(
        <div>
          <div className="font-semibold">W&B'ye başarıyla gönderildi!</div>
          <div className="text-xs mt-1">
            {data.exported_count} sonuç gönderildi
          </div>
          {data.run_url && (
            <a 
              href={data.run_url} 
              target="_blank" 
              rel="noopener noreferrer"
              className="text-xs underline mt-1 block"
            >
              W&B'de görüntüle →
            </a>
          )}
        </div>
      );
    } catch (error) {
      console.error(error);
      toast.error(error instanceof Error ? error.message : "W&B export başarısız");
    } finally {
      setIsExportingToWandB(false);
    }
  };

  const toggleSelectAll = () => {
    if (selectedResultIds.size === sortedResults.length) {
      setSelectedResultIds(new Set());
    } else {
      setSelectedResultIds(new Set(sortedResults.map(r => r.id).filter((id): id is number => id !== undefined)));
    }
  };

  const toggleSelectResult = (id: number) => {
    const newSet = new Set(selectedResultIds);
    if (newSet.has(id)) {
      newSet.delete(id);
    } else {
      newSet.add(id);
    }
    setSelectedResultIds(newSet);
  };

  const calculateStatistics = () => {
    const selectedResults = sortedResults.filter(r => r.id && selectedResultIds.has(r.id));
    
    if (selectedResults.length === 0) return null;

    const metrics = {
      faithfulness: selectedResults.map(r => r.faithfulness).filter((v): v is number => v != null),
      answer_relevancy: selectedResults.map(r => r.answer_relevancy).filter((v): v is number => v != null),
      context_precision: selectedResults.map(r => r.context_precision).filter((v): v is number => v != null),
      context_recall: selectedResults.map(r => r.context_recall).filter((v): v is number => v != null),
      answer_correctness: selectedResults.map(r => r.answer_correctness).filter((v): v is number => v != null),
      latency_ms: selectedResults.map(r => r.latency_ms).filter((v): v is number => v != null),
    };

    const calculateStats = (values: number[]) => {
      if (values.length === 0) return null;
      
      const mean = values.reduce((a, b) => a + b, 0) / values.length;
      const variance = values.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / values.length;
      const stdDev = Math.sqrt(variance);
      const sorted = [...values].sort((a, b) => a - b);
      const median = sorted.length % 2 === 0
        ? (sorted[sorted.length / 2 - 1] + sorted[sorted.length / 2]) / 2
        : sorted[Math.floor(sorted.length / 2)];
      const min = Math.min(...values);
      const max = Math.max(...values);

      return { mean, stdDev, median, min, max, count: values.length };
    };

    return {
      faithfulness: calculateStats(metrics.faithfulness),
      answer_relevancy: calculateStats(metrics.answer_relevancy),
      context_precision: calculateStats(metrics.context_precision),
      context_recall: calculateStats(metrics.context_recall),
      answer_correctness: calculateStats(metrics.answer_correctness),
      latency_ms: calculateStats(metrics.latency_ms),
      totalSelected: selectedResults.length,
    };
  };

  const metricBadge = (value: number | null | undefined) => {
    if (value == null) {
      return <span className="text-slate-400">-</span>;
    }
    const pct = Math.round(value * 100);
    const cls =
      value >= 0.8
        ? "bg-emerald-50 text-emerald-700 border-emerald-200"
        : value >= 0.6
          ? "bg-amber-50 text-amber-700 border-amber-200"
          : "bg-red-50 text-red-700 border-red-200";
    return (
      <span className={`inline-flex items-center rounded-full px-2 py-0.5 border text-[11px] ${cls}`}>
        {pct}%
      </span>
    );
  };

  const sortedResults = useMemo(() => {
    return [...results].sort((a, b) =>
      String(a.question || "").localeCompare(String(b.question || ""), "tr")
    );
  }, [results]);

  const summary = useMemo(() => {
    const values = {
      faithfulness: results.map((r) => r.faithfulness).filter((v): v is number => v != null),
      answer_relevancy: results.map((r) => r.answer_relevancy).filter((v): v is number => v != null),
      context_precision: results.map((r) => r.context_precision).filter((v): v is number => v != null),
      context_recall: results.map((r) => r.context_recall).filter((v): v is number => v != null),
      answer_correctness: results.map((r) => r.answer_correctness).filter((v): v is number => v != null),
      latency_ms: results.map((r) => r.latency_ms).filter((v): v is number => v != null),
    };

    const avg = (arr: number[]) => {
      if (arr.length === 0) return null;
      return arr.reduce((a, b) => a + b, 0) / arr.length;
    };

    return {
      count: results.length,
      avg_faithfulness: aggregate?.avg_faithfulness ?? avg(values.faithfulness),
      avg_answer_relevancy: aggregate?.avg_answer_relevancy ?? avg(values.answer_relevancy),
      avg_context_precision: aggregate?.avg_context_precision ?? avg(values.context_precision),
      avg_context_recall: aggregate?.avg_context_recall ?? avg(values.context_recall),
      avg_answer_correctness: aggregate?.avg_answer_correctness ?? avg(values.answer_correctness),
      avg_latency_ms: avg(values.latency_ms),
    };
  }, [aggregate, results]);

  const usedParams = useMemo<TestParameters>(() => {
    const tp = aggregate?.test_parameters as TestParameters | undefined;
    const first = results[0] as unknown as {
      llm_model?: string;
      llm_provider?: string;
      evaluation_model?: string | null;
      embedding_model?: string | null;
      search_top_k?: number | null;
      search_alpha?: number | null;
      reranker_used?: boolean | null;
      reranker_provider?: string | null;
      reranker_model?: string | null;
    } | null;

    return {
      llm_provider: tp?.llm_provider ?? first?.llm_provider,
      llm_model: tp?.llm_model ?? first?.llm_model,
      evaluation_model: tp?.evaluation_model ?? (first?.evaluation_model ?? undefined),
      embedding_model: tp?.embedding_model ?? (first?.embedding_model ?? undefined),
      search_top_k: tp?.search_top_k ?? (first?.search_top_k ?? undefined),
      search_alpha: tp?.search_alpha ?? (first?.search_alpha ?? undefined),
      reranker_used: tp?.reranker_used ?? (first?.reranker_used ?? undefined),
      reranker_provider: tp?.reranker_provider ?? (first?.reranker_provider ?? undefined),
      reranker_model: tp?.reranker_model ?? (first?.reranker_model ?? undefined),
    };
  }, [aggregate?.test_parameters, results]);

  const totalPages = Math.max(1, Math.ceil(sortedResults.length / PAGE_SIZE));
  const safePage = Math.min(Math.max(page, 1), totalPages);
  const start = (safePage - 1) * PAGE_SIZE;
  const end = Math.min(start + PAGE_SIZE, sortedResults.length);
  const pageItems = sortedResults.slice(start, end);

  if (!user) return null;

  return (
    <div className="space-y-6">
      <div className="bg-white rounded-2xl border border-slate-200 p-6 shadow-sm">
        <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
          <div>
            <div className="text-sm text-slate-500">Grup</div>
            <h1 className="text-2xl font-bold text-slate-900 break-all">{groupName || "-"}</h1>
            <div className="text-xs text-slate-500 mt-1">
              Toplam: <span className="font-medium text-slate-700">{total || results.length}</span>
            </div>
          </div>

          <div className="flex flex-wrap items-center gap-2">
            <Link href="/dashboard/ragas/results">
              <Button variant="outline">
                <ArrowLeft className="w-4 h-4 mr-2" />
                Listeye Dön
              </Button>
            </Link>

            {selectedResultIds.size > 0 && (
              <Button 
                variant="outline"
                onClick={() => setShowStatisticsModal(true)}
                className="bg-blue-50 hover:bg-blue-100 text-blue-700 border-blue-200"
              >
                <BarChart3 className="w-4 h-4 mr-2" />
                İstatistikler ({selectedResultIds.size})
              </Button>
            )}

            <Button variant="outline" onClick={exportToCSV} disabled={!courseId || !groupName}>
              <Download className="w-4 h-4 mr-2" /> CSV
            </Button>

            <Button variant="outline" onClick={exportToPdf} disabled={!courseId || !groupName}>
              <FileText className="w-4 h-4 mr-2" /> PDF
            </Button>

            <Button 
              variant="outline" 
              onClick={exportToWandB} 
              disabled={!courseId || !groupName || isExportingToWandB}
              className="bg-amber-50 hover:bg-amber-100 text-amber-700 border-amber-200"
            >
              {isExportingToWandB ? (
                <>
                  <span className="animate-spin mr-2">⏳</span>
                  W&B'ye Gönderiliyor...
                </>
              ) : (
                <>
                  <svg className="w-4 h-4 mr-2" viewBox="0 0 24 24" fill="currentColor">
                    <path d="M12 2L2 7v10c0 5.55 3.84 10.74 9 12 5.16-1.26 9-6.45 9-12V7l-10-5z"/>
                  </svg>
                  W&B'ye Gönder
                </>
              )}
            </Button>
          </div>
        </div>
      </div>

      {courseId && !isLoading && results.length > 0 && (
        <>
          <div className="bg-white rounded-2xl border border-slate-200 p-6 shadow-sm">
            <div className="flex items-center justify-between mb-4">
              <div>
                <div className="text-sm font-semibold text-slate-900">Kullanılan Ayarlar</div>
                <div className="text-xs text-slate-500">Bu grubun test parametreleri</div>
              </div>
            </div>

            <div className="flex flex-wrap gap-2">
              <span className="inline-flex items-center rounded-full bg-indigo-50 px-3 py-1 text-xs text-indigo-700 border border-indigo-200">
                LLM: {usedParams.llm_provider && usedParams.llm_model ? `${usedParams.llm_provider}/${usedParams.llm_model}` : (usedParams.llm_model || usedParams.llm_provider || "-")}
              </span>
              <span className="inline-flex items-center rounded-full bg-purple-50 px-3 py-1 text-xs text-purple-700 border border-purple-200">
                Eval: {usedParams.evaluation_model || "-"}
              </span>
              <span className="inline-flex items-center rounded-full bg-slate-50 px-3 py-1 text-xs text-slate-700 border border-slate-200">
                Embedding: {usedParams.embedding_model || "-"}
              </span>
              <span className="inline-flex items-center rounded-full bg-slate-50 px-3 py-1 text-xs text-slate-700 border border-slate-200">
                TopK: {usedParams.search_top_k ?? "-"}
              </span>
              <span className="inline-flex items-center rounded-full bg-slate-50 px-3 py-1 text-xs text-slate-700 border border-slate-200">
                Alpha: {usedParams.search_alpha ?? "-"}
              </span>
              <span
                className={`inline-flex items-center rounded-full px-3 py-1 text-xs border ${
                  usedParams.reranker_used
                    ? "bg-emerald-50 text-emerald-700 border-emerald-200"
                    : "bg-slate-100 text-slate-600 border-slate-200"
                }`}
              >
                Reranker: {usedParams.reranker_used ? `${usedParams.reranker_provider || "-"}/${usedParams.reranker_model || "-"}` : "Kapalı"}
              </span>
            </div>
          </div>

          <div className="bg-white rounded-2xl border border-slate-200 p-6 shadow-sm">
            <div className="flex items-center justify-between mb-4">
              <div>
                <div className="text-sm font-semibold text-slate-900">Ortalama Sonuçlar</div>
                <div className="text-xs text-slate-500">Metrikler 0-100% aralığına çevrilmiştir</div>
              </div>
            </div>

            <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-3">
              <div className="p-3 rounded-xl border border-slate-200 bg-slate-50">
                <div className="text-[11px] text-slate-500">Faithfulness</div>
                <div className="mt-1">{metricBadge(summary.avg_faithfulness)}</div>
              </div>
              <div className="p-3 rounded-xl border border-slate-200 bg-slate-50">
                <div className="text-[11px] text-slate-500">Answer Relevancy</div>
                <div className="mt-1">{metricBadge(summary.avg_answer_relevancy)}</div>
              </div>
              <div className="p-3 rounded-xl border border-slate-200 bg-slate-50">
                <div className="text-[11px] text-slate-500">Context Precision</div>
                <div className="mt-1">{metricBadge(summary.avg_context_precision)}</div>
              </div>
              <div className="p-3 rounded-xl border border-slate-200 bg-slate-50">
                <div className="text-[11px] text-slate-500">Context Recall</div>
                <div className="mt-1">{metricBadge(summary.avg_context_recall)}</div>
              </div>
              <div className="p-3 rounded-xl border border-slate-200 bg-slate-50">
                <div className="text-[11px] text-slate-500">Answer Correctness</div>
                <div className="mt-1">{metricBadge(summary.avg_answer_correctness)}</div>
              </div>
              <div className="p-3 rounded-xl border border-slate-200 bg-slate-50">
                <div className="text-[11px] text-slate-500">Ort. Latency</div>
                <div className="mt-1 text-sm font-semibold text-slate-800">
                  {summary.avg_latency_ms == null ? "-" : `${Math.round(summary.avg_latency_ms)}ms`}
                </div>
              </div>
            </div>
          </div>
        </>
      )}

      {!courseId ? (
        <div className="bg-white rounded-2xl border border-slate-200 p-12 text-center shadow-sm">
          <p className="text-slate-500">Ders seçimi bulunamadı. Önce RAGAS sayfasından ders seçin.</p>
        </div>
      ) : isLoading ? (
        <div className="bg-white rounded-2xl border border-slate-200 p-12 text-center shadow-sm">
          <p className="text-slate-500">Sonuçlar yükleniyor...</p>
        </div>
      ) : results.length === 0 ? (
        <div className="bg-white rounded-2xl border border-slate-200 p-12 text-center shadow-sm">
          <p className="text-slate-500">Bu grup için sonuç bulunamadı.</p>
        </div>
      ) : (
        <div className="bg-white rounded-2xl border border-slate-200 p-6 shadow-sm">
          <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-3 mb-3">
            <div className="text-xs text-slate-500">
              Gösterilen: <span className="font-medium text-slate-700">{sortedResults.length === 0 ? 0 : start + 1}-{end}</span>
              <span className="text-slate-400"> / </span>
              <span className="font-medium text-slate-700">{sortedResults.length}</span>
              <span className="text-slate-400"> (alfabetik)</span>
            </div>

            <div className="flex items-center gap-2">
              <Button
                type="button"
                variant="outline"
                size="sm"
                className="h-8"
                onClick={() => setPage((p) => Math.max(1, p - 1))}
                disabled={safePage <= 1}
              >
                Önceki
              </Button>
              <div className="text-xs text-slate-600 min-w-[80px] text-center">
                {safePage} / {totalPages}
              </div>
              <Button
                type="button"
                variant="outline"
                size="sm"
                className="h-8"
                onClick={() => setPage((p) => Math.min(totalPages, p + 1))}
                disabled={safePage >= totalPages}
              >
                Sonraki
              </Button>
            </div>
          </div>

          <div className="overflow-auto border border-slate-200 rounded-xl">
            <table className="w-full text-xs">
              <thead className="bg-slate-50 sticky top-0">
                <tr>
                  <th className="px-3 py-2 text-left font-medium text-slate-600 w-[40px]">
                    <Checkbox
                      checked={selectedResultIds.size === sortedResults.length && sortedResults.length > 0}
                      onCheckedChange={toggleSelectAll}
                    />
                  </th>
                  <th className="px-3 py-2 text-left font-medium text-slate-600 w-[60px]">#</th>
                  <th className="px-3 py-2 text-left font-medium text-slate-600">Soru</th>
                  <th className="px-3 py-2 text-right font-medium text-slate-600">Faith.</th>
                  <th className="px-3 py-2 text-right font-medium text-slate-600">Rel.</th>
                  <th className="px-3 py-2 text-right font-medium text-slate-600">Prec.</th>
                  <th className="px-3 py-2 text-right font-medium text-slate-600">Recall</th>
                  <th className="px-3 py-2 text-right font-medium text-slate-600">Corr.</th>
                  <th className="px-3 py-2 text-right font-medium text-slate-600 w-[60px]"></th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100">
                {pageItems.map((r, idx) => (
                  <tr key={r.id ?? `${start}-${idx}`} className={idx % 2 === 0 ? "bg-white hover:bg-slate-50" : "bg-slate-50/40 hover:bg-slate-50"}>
                    <td className="px-3 py-2">
                      {r.id && (
                        <Checkbox
                          checked={selectedResultIds.has(r.id)}
                          onCheckedChange={() => r.id && toggleSelectResult(r.id)}
                        />
                      )}
                    </td>
                    <td className="px-3 py-2 text-slate-500">{start + idx + 1}</td>
                    <td className="px-3 py-2 text-slate-800 min-w-[420px]">
                      <div className="whitespace-pre-wrap break-words">
                        {r.question}
                      </div>
                    </td>
                    <td className="px-3 py-2 text-right">{metricBadge(r.faithfulness)}</td>
                    <td className="px-3 py-2 text-right">{metricBadge(r.answer_relevancy)}</td>
                    <td className="px-3 py-2 text-right">{metricBadge(r.context_precision)}</td>
                    <td className="px-3 py-2 text-right">{metricBadge(r.context_recall)}</td>
                    <td className="px-3 py-2 text-right">{metricBadge(r.answer_correctness)}</td>
                    <td className="px-3 py-2 text-right">
                      <div className="inline-flex items-center gap-1">
                        <Button
                          variant="ghost"
                          size="sm"
                          className="h-8 w-8 p-0"
                          type="button"
                          title="Soru / Ground Truth / Cevap"
                          onClick={() => setSelectedResult(r)}
                        >
                          <SquarePen className="w-4 h-4" />
                        </Button>
                        <Button
                          variant="ghost"
                          size="sm"
                          className="h-8 w-8 p-0 text-red-600 hover:text-red-700"
                          type="button"
                          title="Sil"
                          onClick={() => handleDeleteResult(r)}
                          disabled={deletingResultId === r.id}
                        >
                          <Trash2 className={`w-4 h-4 ${deletingResultId === r.id ? "opacity-50" : ""}`} />
                        </Button>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      <Dialog open={!!selectedResult} onOpenChange={(open) => (!open ? setSelectedResult(null) : null)}>
        <DialogContent className="max-w-4xl">
          <DialogHeader>
            <div className="flex items-center justify-between gap-3">
              <DialogTitle>Detay</DialogTitle>
              <Button
                variant="destructive"
                type="button"
                onClick={() => (selectedResult ? handleDeleteResult(selectedResult) : null)}
                disabled={!selectedResult?.id || deletingResultId === selectedResult?.id}
              >
                Sil
              </Button>
            </div>
          </DialogHeader>

          {selectedResult && (
            <div className="space-y-4">
              <div>
                <div className="text-xs font-semibold text-slate-600 mb-1">Soru</div>
                <div className="p-3 rounded-lg border border-slate-200 bg-slate-50 whitespace-pre-wrap break-words max-h-[240px] overflow-auto">
                  {selectedResult.question}
                </div>
              </div>

              <div>
                <div className="text-xs font-semibold text-slate-600 mb-1">Ground Truth</div>
                <div className="p-3 rounded-lg border border-slate-200 bg-white whitespace-pre-wrap break-words max-h-[240px] overflow-auto">
                  {selectedResult.ground_truth || "-"}
                </div>
              </div>

              <div>
                <div className="text-xs font-semibold text-slate-600 mb-1">Cevap (Generated Answer)</div>
                <div className="p-3 rounded-lg border border-slate-200 bg-white whitespace-pre-wrap break-words max-h-[240px] overflow-auto">
                  {selectedResult.generated_answer || "-"}
                </div>
              </div>
            </div>
          )}
        </DialogContent>
      </Dialog>

      {/* Statistics Modal */}
      <Dialog open={showStatisticsModal} onOpenChange={setShowStatisticsModal}>
        <DialogContent className="max-w-6xl max-h-[90vh] overflow-y-auto">
          <DialogHeader>
            <DialogTitle>İstatistikler - {selectedResultIds.size} Sonuç Seçildi</DialogTitle>
          </DialogHeader>

          {(() => {
            const stats = calculateStatistics();
            if (!stats) return <p className="text-slate-500">İstatistik hesaplanamadı</p>;

            type MetricStats = { mean: number; stdDev: number; median: number; min: number; max: number; count: number; } | null;
            const StatCard = ({ title, data }: { title: string; data: MetricStats }) => {
              if (!data) return null;
              
              return (
                <div className="p-4 rounded-xl border border-slate-200 bg-gradient-to-br from-white to-slate-50">
                  <h3 className="text-sm font-semibold text-slate-900 mb-3">{title}</h3>
                  <div className="grid grid-cols-2 gap-3">
                    <div>
                      <div className="text-xs text-slate-500">Ortalama</div>
                      <div className="text-lg font-bold text-blue-600">
                        {(data.mean * 100).toFixed(1)}%
                      </div>
                    </div>
                    <div>
                      <div className="text-xs text-slate-500">Std. Sapma</div>
                      <div className="text-lg font-bold text-purple-600">
                        {(data.stdDev * 100).toFixed(1)}%
                      </div>
                    </div>
                    <div>
                      <div className="text-xs text-slate-500">Medyan</div>
                      <div className="text-base font-semibold text-slate-700">
                        {(data.median * 100).toFixed(1)}%
                      </div>
                    </div>
                    <div>
                      <div className="text-xs text-slate-500">Aralık</div>
                      <div className="text-base font-semibold text-slate-700">
                        {(data.min * 100).toFixed(1)}% - {(data.max * 100).toFixed(1)}%
                      </div>
                    </div>
                  </div>
                  
                  {/* Simple bar chart */}
                  <div className="mt-3 pt-3 border-t border-slate-200">
                    <div className="flex items-center gap-2 text-xs">
                      <div className="flex-1">
                        <div className="h-2 bg-slate-200 rounded-full overflow-hidden">
                          <div 
                            className="h-full bg-gradient-to-r from-blue-500 to-purple-500 rounded-full transition-all"
                            style={{ width: `${data.mean * 100}%` }}
                          />
                        </div>
                      </div>
                      <span className="text-slate-600 font-medium">{data.count} test</span>
                    </div>
                  </div>
                </div>
              );
            };

            const LatencyCard = ({ data }: { data: MetricStats }) => {
              if (!data) return null;
              
              return (
                <div className="p-4 rounded-xl border border-slate-200 bg-gradient-to-br from-white to-slate-50">
                  <h3 className="text-sm font-semibold text-slate-900 mb-3">Latency (ms)</h3>
                  <div className="grid grid-cols-2 gap-3">
                    <div>
                      <div className="text-xs text-slate-500">Ortalama</div>
                      <div className="text-lg font-bold text-emerald-600">
                        {Math.round(data.mean)}ms
                      </div>
                    </div>
                    <div>
                      <div className="text-xs text-slate-500">Std. Sapma</div>
                      <div className="text-lg font-bold text-orange-600">
                        {Math.round(data.stdDev)}ms
                      </div>
                    </div>
                    <div>
                      <div className="text-xs text-slate-500">Medyan</div>
                      <div className="text-base font-semibold text-slate-700">
                        {Math.round(data.median)}ms
                      </div>
                    </div>
                    <div>
                      <div className="text-xs text-slate-500">Aralık</div>
                      <div className="text-base font-semibold text-slate-700">
                        {Math.round(data.min)}ms - {Math.round(data.max)}ms
                      </div>
                    </div>
                  </div>
                  
                  <div className="mt-3 pt-3 border-t border-slate-200">
                    <div className="flex items-center gap-2 text-xs">
                      <div className="flex-1">
                        <div className="h-2 bg-slate-200 rounded-full overflow-hidden">
                          <div 
                            className="h-full bg-gradient-to-r from-emerald-500 to-teal-500 rounded-full transition-all"
                            style={{ width: `${Math.min((data.mean / 5000) * 100, 100)}%` }}
                          />
                        </div>
                      </div>
                      <span className="text-slate-600 font-medium">{data.count} test</span>
                    </div>
                  </div>
                </div>
              );
            };

            return (
              <div className="space-y-6">
                {/* Summary */}
                <div className="p-4 rounded-xl bg-gradient-to-r from-blue-50 to-purple-50 border border-blue-200">
                  <div className="flex items-center justify-between">
                    <div>
                      <h3 className="text-lg font-bold text-slate-900">Genel Özet</h3>
                      <p className="text-sm text-slate-600 mt-1">
                        {stats.totalSelected} sonuç üzerinden hesaplanmıştır
                      </p>
                    </div>
                    <BarChart3 className="w-12 h-12 text-blue-500 opacity-50" />
                  </div>
                </div>

                {/* RAGAS Metrics */}
                <div>
                  <h3 className="text-base font-semibold text-slate-900 mb-3">RAGAS Metrikleri</h3>
                  <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                    <StatCard title="Faithfulness" data={stats.faithfulness} />
                    <StatCard title="Answer Relevancy" data={stats.answer_relevancy} />
                    <StatCard title="Context Precision" data={stats.context_precision} />
                    <StatCard title="Context Recall" data={stats.context_recall} />
                    <StatCard title="Answer Correctness" data={stats.answer_correctness} />
                    <LatencyCard data={stats.latency_ms} />
                  </div>
                </div>

                {/* Comparison Chart */}
                <div className="p-4 rounded-xl border border-slate-200 bg-white">
                  <h3 className="text-base font-semibold text-slate-900 mb-4">Metrik Karşılaştırması</h3>
                  <div className="space-y-3">
                    {stats.faithfulness && (
                      <div>
                        <div className="flex items-center justify-between text-xs mb-1">
                          <span className="font-medium text-slate-700">Faithfulness</span>
                          <span className="text-slate-600">{(stats.faithfulness.mean * 100).toFixed(1)}%</span>
                        </div>
                        <div className="h-3 bg-slate-100 rounded-full overflow-hidden">
                          <div 
                            className="h-full bg-blue-500 rounded-full transition-all"
                            style={{ width: `${stats.faithfulness.mean * 100}%` }}
                          />
                        </div>
                      </div>
                    )}
                    {stats.answer_relevancy && (
                      <div>
                        <div className="flex items-center justify-between text-xs mb-1">
                          <span className="font-medium text-slate-700">Answer Relevancy</span>
                          <span className="text-slate-600">{(stats.answer_relevancy.mean * 100).toFixed(1)}%</span>
                        </div>
                        <div className="h-3 bg-slate-100 rounded-full overflow-hidden">
                          <div 
                            className="h-full bg-purple-500 rounded-full transition-all"
                            style={{ width: `${stats.answer_relevancy.mean * 100}%` }}
                          />
                        </div>
                      </div>
                    )}
                    {stats.context_precision && (
                      <div>
                        <div className="flex items-center justify-between text-xs mb-1">
                          <span className="font-medium text-slate-700">Context Precision</span>
                          <span className="text-slate-600">{(stats.context_precision.mean * 100).toFixed(1)}%</span>
                        </div>
                        <div className="h-3 bg-slate-100 rounded-full overflow-hidden">
                          <div 
                            className="h-full bg-emerald-500 rounded-full transition-all"
                            style={{ width: `${stats.context_precision.mean * 100}%` }}
                          />
                        </div>
                      </div>
                    )}
                    {stats.context_recall && (
                      <div>
                        <div className="flex items-center justify-between text-xs mb-1">
                          <span className="font-medium text-slate-700">Context Recall</span>
                          <span className="text-slate-600">{(stats.context_recall.mean * 100).toFixed(1)}%</span>
                        </div>
                        <div className="h-3 bg-slate-100 rounded-full overflow-hidden">
                          <div 
                            className="h-full bg-amber-500 rounded-full transition-all"
                            style={{ width: `${stats.context_recall.mean * 100}%` }}
                          />
                        </div>
                      </div>
                    )}
                    {stats.answer_correctness && (
                      <div>
                        <div className="flex items-center justify-between text-xs mb-1">
                          <span className="font-medium text-slate-700">Answer Correctness</span>
                          <span className="text-slate-600">{(stats.answer_correctness.mean * 100).toFixed(1)}%</span>
                        </div>
                        <div className="h-3 bg-slate-100 rounded-full overflow-hidden">
                          <div 
                            className="h-full bg-rose-500 rounded-full transition-all"
                            style={{ width: `${stats.answer_correctness.mean * 100}%` }}
                          />
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              </div>
            );
          })()}
        </DialogContent>
      </Dialog>
    </div>
  );
}
