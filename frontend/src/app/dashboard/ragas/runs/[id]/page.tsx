"use client";

import { useEffect, useState, useCallback, useRef } from "react";
import { useParams, useRouter } from "next/navigation";
import { useAuth } from "@/lib/auth-context";
import { api, EvaluationRunDetail } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { toast } from "sonner";
import { PageHeader } from "@/components/ui/page-header";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import {
  BarChart3,
  Loader2,
  ArrowLeft,
  CheckCircle,
  XCircle,
  ChevronDown,
  ChevronUp,
  FileJson,
  FileText,
  Wrench,
} from "lucide-react";
import Link from "next/link";

export default function RunResultsPage() {
  const { id } = useParams();
  const router = useRouter();
  const { user } = useAuth();
  const [run, setRun] = useState<EvaluationRunDetail | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [isPdfGenerating, setIsPdfGenerating] = useState(false);
  const [isFixing, setIsFixing] = useState(false);
  const [expandedResults, setExpandedResults] = useState<Set<number>>(new Set());

  const sseAbortRef = useRef<AbortController | null>(null);
  const sseRunningRef = useRef(false);

  const loadRun = useCallback(async () => {
    try {
      const data = await api.getEvaluationRun(Number(id));
      setRun(data);
    } catch {
      toast.error("Değerlendirme yüklenirken hata oluştu");
      router.push("/dashboard/ragas");
    } finally {
      setIsLoading(false);
    }
  }, [id, router]);

  useEffect(() => {
    loadRun();
  }, [loadRun]);

  // Live SSE stream for running evaluations (preferred)
  useEffect(() => {
    if (!run) return;
    if (run.status !== "running" && run.status !== "pending") return;
    if (sseRunningRef.current) return;

    const token = localStorage.getItem("akilli_rehber_token");
    if (!token) return;

    sseRunningRef.current = true;
    const abort = new AbortController();
    sseAbortRef.current = abort;

    const start = async () => {
      try {
        const baseUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
        const res = await fetch(`${baseUrl}/api/ragas/runs/${run.id}/stream`, {
          method: "GET",
          headers: {
            Authorization: `Bearer ${token}`,
            Accept: "text/event-stream",
          },
          signal: abort.signal,
        });

        if (!res.ok) {
          throw new Error(`Stream failed (${res.status})`);
        }

        const reader = res.body?.getReader();
        if (!reader) {
          throw new Error("No stream reader available");
        }

        const decoder = new TextDecoder();
        let buffer = "";

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          if (!value) continue;

          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split(/\r?\n/);
          buffer = lines.pop() ?? "";

          for (const line of lines) {
            if (!line.startsWith("data: ")) continue;
            const payload = line.slice(6);

            try {
              const data = JSON.parse(payload);
              if (data.event === "status") {
                setRun((prev) => {
                  if (!prev) return prev;
                  return {
                    ...prev,
                    status: data.status ?? prev.status,
                    total_questions: data.total_questions ?? prev.total_questions,
                    processed_questions:
                      data.processed_questions ?? prev.processed_questions,
                    error_message: data.error_message ?? prev.error_message,
                    wandb_run_url: data.wandb_run_url ?? prev.wandb_run_url,
                    wandb_run_id: data.wandb_run_id ?? prev.wandb_run_id,
                  } as EvaluationRunDetail;
                });
              } else if (data.event === "result") {
                const result = data.result;
                if (!result?.id) continue;

                setRun((prev) => {
                  if (!prev) return prev;
                  const exists = prev.results.some((r) => r.id === result.id);
                  if (exists) return prev;

                  const nextResults = [...prev.results, result].sort(
                    (a, b) => a.id - b.id
                  );
                  return { ...prev, results: nextResults } as EvaluationRunDetail;
                });
              } else if (data.event === "complete") {
                setRun((prev) => {
                  if (!prev) return prev;
                  return {
                    ...prev,
                    status: data.status ?? prev.status,
                  } as EvaluationRunDetail;
                });
                break;
              } else if (data.event === "error") {
                break;
              }
            } catch {
              // ignore malformed chunk
            }
          }
        }
      } catch {
        // SSE is best-effort; polling will keep UI updated.
      } finally {
        sseRunningRef.current = false;
        sseAbortRef.current = null;
      }
    };

    start();

    return () => {
      abort.abort();
      sseRunningRef.current = false;
      sseAbortRef.current = null;
    };
  }, [run]);

  // Auto-refresh for running evaluations
  useEffect(() => {
    if (!run) return;
    if (run.status !== "running" && run.status !== "pending") return;

    const interval = setInterval(() => {
      loadRun();
    }, 5000); // Refresh every 5 seconds

    return () => clearInterval(interval);
  }, [run, loadRun]);

  const toggleExpand = (resultId: number) => {
    setExpandedResults(prev => {
      const next = new Set(prev);
      if (next.has(resultId)) {
        next.delete(resultId);
      } else {
        next.add(resultId);
      }
      return next;
    });
  };

  const formatScore = (score?: number) => {
    if (score === undefined || score === null) return "-";
    return (score * 100).toFixed(1) + "%";
  };

  const getScoreColor = (score?: number) => {
    if (score === undefined || score === null) return "text-slate-400";
    if (score >= 0.8) return "text-green-600";
    if (score >= 0.6) return "text-yellow-600";
    return "text-red-600";
  };

  const handleFixSummary = async () => {
    if (!run) return;
    
    setIsFixing(true);
    try {
      const result = await api.fixRunSummary(run.id);
      toast.success(`Özet düzeltildi: ${result.successful_questions} başarılı, ${result.failed_questions} başarısız`);
      // Reload the run data
      await loadRun();
    } catch (error) {
      toast.error(`Özet düzeltilirken hata: ${error instanceof Error ? error.message : 'Bilinmeyen hata'}`);
    } finally {
      setIsFixing(false);
    }
  };

  const handleExportJson = () => {
    if (!run) return;

    const exportData = {
      run_id: run.id,
      name: run.name,
      status: run.status,
      created_at: run.created_at,
      completed_at: run.completed_at,
      total_questions: run.total_questions,
      summary: run.summary ? {
        avg_faithfulness: run.summary.avg_faithfulness,
        avg_answer_relevancy: run.summary.avg_answer_relevancy,
        avg_context_precision: run.summary.avg_context_precision,
        avg_context_recall: run.summary.avg_context_recall,
        avg_answer_correctness: run.summary.avg_answer_correctness,
        avg_latency_ms: run.summary.avg_latency_ms,
        successful_questions: run.summary.successful_questions,
        failed_questions: run.summary.failed_questions,
      } : null,
      results: run.results.map(r => ({
        question: r.question_text,
        ground_truth: r.ground_truth_text,
        generated_answer: r.generated_answer,
        retrieved_contexts: r.retrieved_contexts,
        metrics: {
          faithfulness: r.faithfulness,
          answer_relevancy: r.answer_relevancy,
          context_precision: r.context_precision,
          context_recall: r.context_recall,
          answer_correctness: r.answer_correctness,
        },
        latency_ms: r.latency_ms,
        error: r.error_message,
      })),
    };

    const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `ragas_evaluation_${run.id}_${new Date().toISOString().split("T")[0]}.json`;
    a.click();
    URL.revokeObjectURL(url);
    toast.success("JSON olarak indirildi");
  };

  const handleExportPdf = async () => {
    if (!run) return;

    try {
      // Set loading state
      setIsPdfGenerating(true);
      
      // Show loading toast
      toast.loading("Generating enhanced PDF report...", { id: "pdf-generation" });

      console.log('Starting PDF generation...');

      // ===== SUBTASK 7.1: Calculate all statistics before rendering =====
      
      // Extract metric values from results (convert to percentages)
      const faithfulnessValues = run.results
        .map(r => r.faithfulness)
        .filter((v): v is number => v !== null && v !== undefined)
        .map(v => v * 100);
      
      const answerRelevancyValues = run.results
        .map(r => r.answer_relevancy)
        .filter((v): v is number => v !== null && v !== undefined)
        .map(v => v * 100);
      
      const contextPrecisionValues = run.results
        .map(r => r.context_precision)
        .filter((v): v is number => v !== null && v !== undefined)
        .map(v => v * 100);
      
      const contextRecallValues = run.results
        .map(r => r.context_recall)
        .filter((v): v is number => v !== null && v !== undefined)
        .map(v => v * 100);
      
      const answerCorrectnessValues = run.results
        .map(r => r.answer_correctness)
        .filter((v): v is number => v !== null && v !== undefined)
        .map(v => v * 100);

      console.log('Metric values extracted:', {
        faithfulness: faithfulnessValues.length,
        answerRelevancy: answerRelevancyValues.length,
        contextPrecision: contextPrecisionValues.length,
        contextRecall: contextRecallValues.length,
        answerCorrectness: answerCorrectnessValues.length,
      });

      // Calculate statistics for each metric
      const { calculateStatistics, calculateOverallAverage } = await import('@/lib/statisticsCalculator');
      
      const faithfulnessStats = calculateStatistics(faithfulnessValues);
      const answerRelevancyStats = calculateStatistics(answerRelevancyValues);
      const contextPrecisionStats = calculateStatistics(contextPrecisionValues);
      const contextRecallStats = calculateStatistics(contextRecallValues);
      const answerCorrectnessStats = calculateStatistics(answerCorrectnessValues);

      // Calculate overall average from summary metrics
      const overallAverage = run.summary ? calculateOverallAverage({
        faithfulness: (run.summary.avg_faithfulness ?? 0) * 100,
        answer_relevancy: (run.summary.avg_answer_relevancy ?? 0) * 100,
        context_precision: (run.summary.avg_context_precision ?? 0) * 100,
        context_recall: (run.summary.avg_context_recall ?? 0) * 100,
        answer_correctness: (run.summary.avg_answer_correctness ?? 0) * 100,
      }) : 0;

      console.log('Statistics calculated, overall average:', overallAverage);

      // ===== SUBTASK 7.2: Generate all charts asynchronously =====
      
      console.log('Loading chart generator...');
      const { 
        generateMetricsBarChart, 
        generateMetricsLineChart, 
        generateBoxPlot 
      } = await import('@/lib/chartGenerator');

      console.log('Generating bar chart...');
      // Generate bar chart with average metrics
      const barChart = await generateMetricsBarChart({
        faithfulness: (run.summary?.avg_faithfulness ?? 0) * 100,
        answer_relevancy: (run.summary?.avg_answer_relevancy ?? 0) * 100,
        context_precision: (run.summary?.avg_context_precision ?? 0) * 100,
        context_recall: (run.summary?.avg_context_recall ?? 0) * 100,
        answer_correctness: (run.summary?.avg_answer_correctness ?? 0) * 100,
      });
      console.log('Bar chart generated:', barChart.substring(0, 50) + '...');

      console.log('Generating line chart...');
      // Generate line chart with results across questions
      const lineChartData = run.results.map((r, index) => ({
        question_number: index + 1,
        faithfulness: r.faithfulness ? r.faithfulness * 100 : null,
        answer_relevancy: r.answer_relevancy ? r.answer_relevancy * 100 : null,
        context_precision: r.context_precision ? r.context_precision * 100 : null,
        context_recall: r.context_recall ? r.context_recall * 100 : null,
        answer_correctness: r.answer_correctness ? r.answer_correctness * 100 : null,
      }));
      const lineChart = await generateMetricsLineChart(lineChartData);
      console.log('Line chart generated:', lineChart.substring(0, 50) + '...');

      console.log('Generating box plots...');
      // Generate box plots for each metric
      const faithfulnessBoxPlot = await generateBoxPlot(
        'Faithfulness',
        faithfulnessValues,
        {
          min: faithfulnessStats.min,
          q1: faithfulnessStats.q1,
          median: faithfulnessStats.median,
          q3: faithfulnessStats.q3,
          max: faithfulnessStats.max,
        }
      );
      console.log('Faithfulness box plot result type:', faithfulnessBoxPlot.substring(0, 30));

      const answerRelevancyBoxPlot = await generateBoxPlot(
        'Answer Relevancy',
        answerRelevancyValues,
        {
          min: answerRelevancyStats.min,
          q1: answerRelevancyStats.q1,
          median: answerRelevancyStats.median,
          q3: answerRelevancyStats.q3,
          max: answerRelevancyStats.max,
        }
      );
      console.log('Answer Relevancy box plot result type:', answerRelevancyBoxPlot.substring(0, 30));

      const contextPrecisionBoxPlot = await generateBoxPlot(
        'Context Precision',
        contextPrecisionValues,
        {
          min: contextPrecisionStats.min,
          q1: contextPrecisionStats.q1,
          median: contextPrecisionStats.median,
          q3: contextPrecisionStats.q3,
          max: contextPrecisionStats.max,
        }
      );

      const contextRecallBoxPlot = await generateBoxPlot(
        'Context Recall',
        contextRecallValues,
        {
          min: contextRecallStats.min,
          q1: contextRecallStats.q1,
          median: contextRecallStats.median,
          q3: contextRecallStats.q3,
          max: contextRecallStats.max,
        }
      );

      const answerCorrectnessBoxPlot = await generateBoxPlot(
        'Answer Correctness',
        answerCorrectnessValues,
        {
          min: answerCorrectnessStats.min,
          q1: answerCorrectnessStats.q1,
          median: answerCorrectnessStats.median,
          q3: answerCorrectnessStats.q3,
          max: answerCorrectnessStats.max,
        }
      );

      console.log('All box plots generated');

      // ===== SUBTASK 7.3: Build complete HTML with all sections =====
      
      console.log('Building PDF sections...');
      const { extractConfiguration } = await import('@/lib/configurationExtractor');
      const { 
        buildReportHeader,
        buildTableOfContents,
        buildMetricsExplanation,
        buildSummaryStatistics,
        buildConfigurationSection,
        buildStatisticalAnalysisSection,
        buildVisualizationsSection,
        buildQuestionAnalysisTable,
        buildDetailedResultsSection,
      } = await import('@/lib/pdfReportBuilder');
      const { getPdfStyles } = await import('@/lib/pdfStyles');

      // Extract configuration
      const configuration = extractConfiguration(run.results);

      // Build all sections
      const headerHtml = buildReportHeader(run);
      const tocHtml = buildTableOfContents();
      const metricsExplanationHtml = buildMetricsExplanation();
      const summaryHtml = buildSummaryStatistics(run, overallAverage);
      const configHtml = buildConfigurationSection(configuration);
      const statisticsHtml = buildStatisticalAnalysisSection({
        faithfulness: faithfulnessStats,
        answer_relevancy: answerRelevancyStats,
        context_precision: contextPrecisionStats,
        context_recall: contextRecallStats,
        answer_correctness: answerCorrectnessStats,
      });
      const visualizationsHtml = buildVisualizationsSection({
        barChart,
        lineChart,
        boxPlots: {
          faithfulness: faithfulnessBoxPlot,
          answer_relevancy: answerRelevancyBoxPlot,
          context_precision: contextPrecisionBoxPlot,
          context_recall: contextRecallBoxPlot,
          answer_correctness: answerCorrectnessBoxPlot,
        },
      });
      const questionAnalysisHtml = buildQuestionAnalysisTable(run.results);
      const detailedResultsHtml = buildDetailedResultsSection(run.results);

      console.log('All sections built');

      // Combine all sections into complete HTML document
      const htmlContent = `
        <!DOCTYPE html>
        <html>
        <head>
          <meta charset="utf-8">
          <title>RAGAS Evaluation Report - ${run.name || `Evaluation #${run.id}`}</title>
          ${getPdfStyles()}
        </head>
        <body>
          ${headerHtml}
          ${tocHtml}
          ${metricsExplanationHtml}
          ${summaryHtml}
          ${configHtml}
          ${statisticsHtml}
          ${visualizationsHtml}
          ${questionAnalysisHtml}
          ${detailedResultsHtml}
        </body>
        </html>
      `;

      // ===== SUBTASK 7.4: Open print dialog =====
      
      console.log('Opening print window...');
      const printWindow = window.open('', '_blank');
      if (!printWindow) {
        toast.error("Failed to open print window. Please check your popup blocker.", { id: "pdf-generation" });
        return;
      }

      printWindow.document.write(htmlContent);
      printWindow.document.close();
      
      printWindow.onload = () => {
        console.log('Print window loaded, triggering print dialog');
        printWindow.print();
      };

      toast.success("Enhanced PDF report generated successfully!", { id: "pdf-generation" });
      
    } catch (error) {
      console.error("Error generating PDF report:", error);
      toast.error(`Failed to generate PDF report: ${error instanceof Error ? error.message : 'Unknown error'}`, { id: "pdf-generation" });
    } finally {
      // Reset loading state
      setIsPdfGenerating(false);
    }
  };

  if (!user) return null;

  if (isLoading) {
    return (
      <div className="flex justify-center py-12">
        <Loader2 className="w-6 h-6 text-slate-400 animate-spin" />
      </div>
    );
  }

  if (!run) return null;

  return (
    <div>
      <Link href="/dashboard/ragas" className="flex items-center gap-2 text-sm text-muted-foreground hover:text-foreground mb-4">
        <ArrowLeft className="h-4 w-4" />
        RAGAS Değerlendirmesi&apos;ne Dön
      </Link>
      <PageHeader
        icon={BarChart3}
        title={run.name || `Değerlendirme #${run.id}`}
        description={`${run.total_questions} soru değerlendirildi`}
      >
        <div className="flex items-center gap-2">
          {run.wandb_run_url && (
            <a
              href={run.wandb_run_url}
              target="_blank"
              rel="noreferrer"
              className="inline-flex"
            >
              <Button variant="outline" size="sm">
                W&B
              </Button>
            </a>
          )}
          {/* Fix Summary Button - show if there might be issues or mismatch */}
          {run.summary && (
            run.summary.failed_questions !== 0 ||
            run.status === "failed" ||
            run.results.length !== run.summary.total_questions
          ) && (
            <TooltipProvider>
              <Tooltip>
                <TooltipTrigger asChild>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={handleFixSummary}
                    disabled={isFixing}
                    className="text-orange-600 border-orange-300 hover:bg-orange-50"
                  >
                    {isFixing ? (
                      <>
                        <Loader2 className="w-4 h-4 mr-1 animate-spin" /> Düzeltiliyor...
                      </>
                    ) : (
                      <>
                        <Wrench className="w-4 h-4 mr-1" /> Özeti Düzelt
                      </>
                    )}
                  </Button>
                </TooltipTrigger>
                <TooltipContent side="bottom" className="max-w-xs">
                  <p className="text-sm">
                    Silinen sorulardan kaynaklanan yanlış başarısız sayısını düzeltir
                  </p>
                </TooltipContent>
              </Tooltip>
            </TooltipProvider>
          )}
          <Button variant="outline" size="sm" onClick={handleExportJson}>
            <FileJson className="w-4 h-4 mr-1" /> JSON
          </Button>
          <TooltipProvider>
            <Tooltip>
              <TooltipTrigger asChild>
                <Button 
                  variant="outline" 
                  size="sm" 
                  onClick={handleExportPdf}
                  disabled={isPdfGenerating}
                >
                  {isPdfGenerating ? (
                    <>
                      <Loader2 className="w-4 h-4 mr-1 animate-spin" /> Generating...
                    </>
                  ) : (
                    <>
                      <FileText className="w-4 h-4 mr-1" /> Enhanced PDF
                    </>
                  )}
                </Button>
              </TooltipTrigger>
              <TooltipContent side="bottom" className="max-w-xs">
                <p className="font-semibold mb-1">Enhanced PDF Report</p>
                <ul className="text-xs space-y-1">
                  <li>• Statistical analysis (mean, median, std dev)</li>
                  <li>• Visual charts (bar, line, box plots)</li>
                  <li>• Configuration details</li>
                  <li>• Question ranking & analysis</li>
                </ul>
              </TooltipContent>
            </Tooltip>
          </TooltipProvider>
          <Link href="/dashboard/ragas">
            <Button variant="outline" size="sm">
              <ArrowLeft className="w-4 h-4 mr-1" /> Geri
            </Button>
          </Link>
        </div>
      </PageHeader>

      {/* Summary Cards */}
      {run.summary && (
        <div className="grid grid-cols-2 md:grid-cols-5 gap-4 mb-6">
          <div className="bg-white rounded-xl border border-slate-200 p-4">
            <p className="text-xs text-slate-500 mb-1">Faithfulness</p>
            <p className={`text-2xl font-bold ${getScoreColor(run.summary.avg_faithfulness)}`}>
              {formatScore(run.summary.avg_faithfulness)}
            </p>
          </div>
          <div className="bg-white rounded-xl border border-slate-200 p-4">
            <p className="text-xs text-slate-500 mb-1">Answer Relevancy</p>
            <p className={`text-2xl font-bold ${getScoreColor(run.summary.avg_answer_relevancy)}`}>
              {formatScore(run.summary.avg_answer_relevancy)}
            </p>
          </div>
          <div className="bg-white rounded-xl border border-slate-200 p-4">
            <p className="text-xs text-slate-500 mb-1">Context Precision</p>
            <p className={`text-2xl font-bold ${getScoreColor(run.summary.avg_context_precision)}`}>
              {formatScore(run.summary.avg_context_precision)}
            </p>
          </div>
          <div className="bg-white rounded-xl border border-slate-200 p-4">
            <p className="text-xs text-slate-500 mb-1">Context Recall</p>
            <p className={`text-2xl font-bold ${getScoreColor(run.summary.avg_context_recall)}`}>
              {formatScore(run.summary.avg_context_recall)}
            </p>
          </div>
          <div className="bg-white rounded-xl border border-slate-200 p-4">
            <p className="text-xs text-slate-500 mb-1">Answer Correctness</p>
            <p className={`text-2xl font-bold ${getScoreColor(run.summary.avg_answer_correctness)}`}>
              {formatScore(run.summary.avg_answer_correctness)}
            </p>
          </div>
        </div>
      )}

      {/* Stats Row */}
      <div className="flex items-center gap-4 mb-6">
        <div className="flex items-center gap-2 text-sm">
          <CheckCircle className="w-4 h-4 text-green-500" />
          <span className="text-slate-600">{run.summary?.successful_questions || 0} başarılı</span>
        </div>
        <div className="flex items-center gap-2 text-sm">
          <XCircle className="w-4 h-4 text-red-500" />
          <span className="text-slate-600">{run.summary?.failed_questions || 0} başarısız</span>
        </div>
        {run.summary?.avg_latency_ms && (
          <div className="text-sm text-slate-500">
            Ortalama yanıt: {run.summary.avg_latency_ms.toFixed(0)}ms
          </div>
        )}
      </div>

      {/* Results Table */}
      <div className="bg-white rounded-xl border border-slate-200 overflow-hidden">
        <div className="px-6 py-4 border-b border-slate-200 bg-slate-50">
          <h2 className="font-semibold text-slate-900">Detaylı Sonuçlar</h2>
        </div>
        
        <div className="divide-y divide-slate-100">
          {run.results.map((result, index) => (
            <div key={result.id} className="hover:bg-slate-50">
              <div
                className="px-6 py-4 cursor-pointer"
                onClick={() => toggleExpand(result.id)}
              >
                <div className="flex items-start justify-between gap-4">
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2 mb-1">
                      <span className="text-xs font-medium text-slate-400">#{index + 1}</span>
                      {result.error_message ? (
                        <XCircle className="w-4 h-4 text-red-500" />
                      ) : (
                        <CheckCircle className="w-4 h-4 text-green-500" />
                      )}
                    </div>
                    <p className="text-sm font-medium text-slate-900 line-clamp-2">
                      {result.question_text}
                    </p>
                  </div>
                  
                  <div className="flex items-center gap-4">
                    {!result.error_message && (
                      <div className="flex items-center gap-3 text-xs">
                        <div className="text-center">
                          <p className="text-slate-400">Faith</p>
                          <p className={`font-medium ${getScoreColor(result.faithfulness)}`}>
                            {formatScore(result.faithfulness)}
                          </p>
                        </div>
                        <div className="text-center">
                          <p className="text-slate-400">Rel</p>
                          <p className={`font-medium ${getScoreColor(result.answer_relevancy)}`}>
                            {formatScore(result.answer_relevancy)}
                          </p>
                        </div>
                        <div className="text-center">
                          <p className="text-slate-400">Prec</p>
                          <p className={`font-medium ${getScoreColor(result.context_precision)}`}>
                            {formatScore(result.context_precision)}
                          </p>
                        </div>
                        <div className="text-center">
                          <p className="text-slate-400">Rec</p>
                          <p className={`font-medium ${getScoreColor(result.context_recall)}`}>
                            {formatScore(result.context_recall)}
                          </p>
                        </div>
                        <div className="text-center">
                          <p className="text-slate-400">Corr</p>
                          <p className={`font-medium ${getScoreColor(result.answer_correctness)}`}>
                            {formatScore(result.answer_correctness)}
                          </p>
                        </div>
                      </div>
                    )}
                    {expandedResults.has(result.id) ? (
                      <ChevronUp className="w-5 h-5 text-slate-400" />
                    ) : (
                      <ChevronDown className="w-5 h-5 text-slate-400" />
                    )}
                  </div>
                </div>
              </div>

              {expandedResults.has(result.id) && (
                <div className="px-6 pb-4 space-y-4 bg-slate-50">
                  {result.error_message ? (
                    <div className="p-3 bg-red-50 border border-red-200 rounded-lg">
                      <p className="text-sm text-red-700">{result.error_message}</p>
                    </div>
                  ) : (
                    <>
                      <div>
                        <p className="text-xs font-medium text-slate-500 mb-1">Beklenen Cevap (Ground Truth)</p>
                        <p className="text-sm text-slate-700 bg-white p-3 rounded border border-slate-200">
                          {result.ground_truth_text}
                        </p>
                      </div>
                      <div>
                        <p className="text-xs font-medium text-slate-500 mb-1">Üretilen Cevap</p>
                        <p className="text-sm text-slate-700 bg-white p-3 rounded border border-slate-200">
                          {result.generated_answer || "-"}
                        </p>
                      </div>
                      {result.retrieved_contexts && result.retrieved_contexts.length > 0 && (
                        <div>
                          <p className="text-xs font-medium text-slate-500 mb-1">
                            Getirilen Bağlamlar ({result.retrieved_contexts.length})
                          </p>
                          <div className="space-y-2">
                            {result.retrieved_contexts.map((ctx, i) => (
                              <p key={i} className="text-xs text-slate-600 bg-white p-2 rounded border border-slate-200">
                                {ctx}
                              </p>
                            ))}
                          </div>
                        </div>
                      )}
                      {result.latency_ms && (
                        <p className="text-xs text-slate-400">Yanıt süresi: {result.latency_ms}ms</p>
                      )}
                      {/* Model Information */}
                      {(result.llm_model || result.embedding_model || result.evaluation_model) && (
                        <div className="border-t border-slate-200 pt-3 mt-3">
                          <p className="text-xs font-medium text-slate-500 mb-2">Kullanılan Modeller</p>
                          <div className="grid grid-cols-2 gap-2 text-xs">
                            {result.llm_provider && result.llm_model && (
                              <div className="bg-white p-2 rounded border border-slate-200">
                                <span className="text-slate-400">LLM:</span>{" "}
                                <span className="text-slate-700 font-medium">{result.llm_provider}/{result.llm_model}</span>
                              </div>
                            )}
                            {result.embedding_model && (
                              <div className="bg-white p-2 rounded border border-slate-200">
                                <span className="text-slate-400">Embedding:</span>{" "}
                                <span className="text-slate-700 font-medium">{result.embedding_model}</span>
                              </div>
                            )}
                            {result.evaluation_model && (
                              <div className="bg-white p-2 rounded border border-slate-200">
                                <span className="text-slate-400">Evaluation:</span>{" "}
                                <span className="text-slate-700 font-medium">{result.evaluation_model}</span>
                              </div>
                            )}
                            {result.search_alpha !== undefined && result.search_alpha !== null && (
                              <div className="bg-white p-2 rounded border border-slate-200">
                                <span className="text-slate-400">Search Alpha:</span>{" "}
                                <span className="text-slate-700 font-medium">{result.search_alpha.toFixed(2)}</span>
                              </div>
                            )}
                            {result.search_top_k && (
                              <div className="bg-white p-2 rounded border border-slate-200">
                                <span className="text-slate-400">Top K:</span>{" "}
                                <span className="text-slate-700 font-medium">{result.search_top_k}</span>
                              </div>
                            )}
                          </div>
                        </div>
                      )}
                    </>
                  )}
                </div>
              )}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
