"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import Link from "next/link";
import { useParams, useRouter } from "next/navigation";
import { toast } from "sonner";
import { ArrowLeft, BarChart3, Loader2 } from "lucide-react";

import { useAuth } from "@/lib/auth-context";
import { api, GiskardEvaluationRun, GiskardResult, GiskardSummary } from "@/lib/api";

import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { PageHeader } from "@/components/ui/page-header";

export default function GiskardRunPage() {
  const { id } = useParams();
  const router = useRouter();
  const { user } = useAuth();

  const runId = useMemo(() => Number(id), [id]);

  const [run, setRun] = useState<GiskardEvaluationRun | null>(null);
  const [summary, setSummary] = useState<GiskardSummary | null>(null);
  const [results, setResults] = useState<GiskardResult[]>([]);
  const [isLoading, setIsLoading] = useState(true);

  const load = useCallback(async () => {
    if (!Number.isFinite(runId)) return;
    setIsLoading(true);
    try {
      const [r, s, res] = await Promise.all([
        api.getGiskardEvaluationRun(runId),
        api.getGiskardRunSummary(runId),
        api.getGiskardRunResults(runId),
      ]);
      setRun(r);
      setSummary(s);
      setResults(res);
    } catch {
      toast.error("Run yüklenemedi");
      router.push("/dashboard/giskard");
    } finally {
      setIsLoading(false);
    }
  }, [router, runId]);

  useEffect(() => {
    if (!user) return;
    load();
  }, [load, user]);

  if (!user) return null;

  if (isLoading) {
    return (
      <div className="min-h-[60vh] flex items-center justify-center">
        <div className="text-center">
          <Loader2 className="w-8 h-8 animate-spin mx-auto text-slate-600" />
          <p className="mt-3 text-sm text-slate-600">Yükleniyor...</p>
        </div>
      </div>
    );
  }

  if (!run) return null;

  return (
    <div>
      <PageHeader
        icon={BarChart3}
        title={run.name || `Run #${run.id}`}
        description={`Durum: ${run.status}`}
        iconColor="text-blue-600"
        iconBg="bg-blue-100"
      >
        <Link href="/dashboard/giskard">
          <Button variant="outline" size="sm">
            <ArrowLeft className="w-4 h-4 mr-2" />
            Geri
          </Button>
        </Link>
      </PageHeader>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <Card className="p-6 lg:col-span-1">
          <h2 className="text-lg font-semibold text-slate-900">Özet</h2>
          {summary ? (
            <div className="mt-4 space-y-2 text-sm text-slate-700">
              <div>Toplam: {summary.total_questions}</div>
              <div>Başarılı: {summary.successful_questions}</div>
              <div>Başarısız: {summary.failed_questions}</div>
              <div>Halüsinasyon Oranı: {summary.hallucination_rate ?? "-"}</div>
              <div>Doğru Reddetme Oranı: {summary.correct_refusal_rate ?? "-"}</div>
              <div>Türkçe Oranı: {summary.turkish_response_rate ?? "-"}</div>
            </div>
          ) : (
            <p className="mt-4 text-sm text-slate-500">Özet bulunamadı</p>
          )}
        </Card>

        <Card className="p-6 lg:col-span-2">
          <h2 className="text-lg font-semibold text-slate-900">Sonuçlar</h2>
          <p className="text-sm text-slate-500 mt-1">{results.length} sonuç</p>

          <div className="mt-4 space-y-3">
            {results.length === 0 ? (
              <div className="text-center py-10 text-slate-400">
                <p className="text-sm">Henüz sonuç yok</p>
              </div>
            ) : (
              results.map((r) => (
                <div key={r.id} className="p-4 rounded-lg border border-slate-200">
                  <p className="text-sm font-medium text-slate-900 break-words">{r.question_text}</p>
                  <div className="mt-2 text-xs text-slate-600 grid grid-cols-2 gap-2">
                    <div>Skor: {r.score ?? "-"}</div>
                    <div>Halüsinasyon: {r.hallucinated === undefined ? "-" : r.hallucinated ? "Evet" : "Hayır"}</div>
                    <div>Doğru Reddetme: {r.correct_refusal === undefined ? "-" : r.correct_refusal ? "Evet" : "Hayır"}</div>
                    <div>Dil: {r.language ?? "-"}</div>
                  </div>
                </div>
              ))
            )}
          </div>
        </Card>
      </div>
    </div>
  );
}
