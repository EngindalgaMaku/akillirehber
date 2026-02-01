"use client";

import { useEffect, useMemo, useState } from "react";
import Link from "next/link";
import { useParams } from "next/navigation";
import { toast } from "sonner";
import { ArrowLeft, Download, FileText, SquarePen, Trash2 } from "lucide-react";

import { api, QuickTestResult } from "@/lib/api";
import { useAuth } from "@/lib/auth-context";
import { Button } from "@/components/ui/button";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";

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

  useEffect(() => {
    const raw = localStorage.getItem("ragas_selected_course_id");
    const parsed = raw ? Number(raw) : null;
    setCourseId(Number.isFinite(parsed as number) ? (parsed as number) : null);
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

            <Button variant="outline" onClick={exportToCSV} disabled={!courseId || !groupName}>
              <Download className="w-4 h-4 mr-2" /> CSV
            </Button>

            <Button variant="outline" onClick={exportToPdf} disabled={!courseId || !groupName}>
              <FileText className="w-4 h-4 mr-2" /> PDF
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
    </div>
  );
}
