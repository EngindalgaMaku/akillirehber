"use client";

import { useEffect, useState, useCallback } from "react";
import { useSearchParams, useRouter } from "next/navigation";
import { useAuth } from "@/lib/auth-context";
import { api, Course, TestSet, EvaluationRun } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Slider } from "@/components/ui/slider";
import { toast } from "sonner";
import { PageHeader } from "@/components/ui/page-header";
import {
  Play,
  Loader2,
  ArrowLeft,
  CheckCircle,
  XCircle,
  Clock,
  FlaskConical,
} from "lucide-react";
import Link from "next/link";

export default function EvaluatePage() {
  const searchParams = useSearchParams();
  const router = useRouter();
  const { user } = useAuth();
  const [courses, setCourses] = useState<Course[]>([]);
  const [testSets, setTestSets] = useState<TestSet[]>([]);
  const [selectedCourseId, setSelectedCourseId] = useState<number | null>(null);
  const [selectedTestSetId, setSelectedTestSetId] = useState<number | null>(null);
  const [runName, setRunName] = useState("");
  const [searchType, setSearchType] = useState("hybrid");
  const [alpha, setAlpha] = useState(0.5);
  const [topK, setTopK] = useState(5);
  const [isLoading, setIsLoading] = useState(true);
  const [isStarting, setIsStarting] = useState(false);
  const [currentRun, setCurrentRun] = useState<EvaluationRun | null>(null);
  const [pollingInterval, setPollingInterval] = useState<NodeJS.Timeout | null>(null);

  const loadCourses = useCallback(async () => {
    try {
      const data = await api.getCourses();
      setCourses(data);
      if (data.length > 0 && !selectedCourseId) {
        setSelectedCourseId(data[0].id);
      }
    } catch {
      toast.error("Dersler yüklenirken hata oluştu");
    } finally {
      setIsLoading(false);
    }
  }, [selectedCourseId]);

  useEffect(() => {
    loadCourses();
    const courseId = searchParams.get("course");
    if (courseId) {
      setSelectedCourseId(Number(courseId));
    }
  }, [searchParams, loadCourses]);

  const loadTestSets = useCallback(async () => {
    if (!selectedCourseId) return;
    try {
      const data = await api.getTestSets(selectedCourseId);
      setTestSets(data);
    } catch {
      toast.error("Test setleri yüklenirken hata oluştu");
    }
  }, [selectedCourseId]);

  useEffect(() => {
    if (selectedCourseId) {
      loadTestSets();
    }
  }, [selectedCourseId, loadTestSets]);

  useEffect(() => {
    return () => {
      if (pollingInterval) {
        clearInterval(pollingInterval);
      }
    };
  }, [pollingInterval]);

  const pollRunStatus = useCallback(async (runId: number) => {
    try {
      const status = await api.getRunStatus(runId);
      setCurrentRun(prev => {
        if (!prev) return null;
        return {
          ...prev,
          status: status.status as EvaluationRun["status"],
          total_questions: status.total_questions,
          processed_questions: status.processed_questions,
          error_message: status.error_message,
        };
      });
      
      if (status.status === "completed" || status.status === "failed") {
        if (pollingInterval) {
          clearInterval(pollingInterval);
          setPollingInterval(null);
        }
        if (status.status === "completed") {
          toast.success("Değerlendirme tamamlandı!");
          router.push(`/dashboard/ragas/runs/${runId}`);
        } else {
          toast.error("Değerlendirme başarısız oldu");
        }
      }
    } catch (error) {
      console.error("Polling error:", error);
    }
  }, [pollingInterval, router]);

  const handleStartEvaluation = async () => {
    if (!selectedCourseId || !selectedTestSetId) {
      toast.error("Lütfen ders ve test seti seçin");
      return;
    }

    setIsStarting(true);
    try {
      const run = await api.startEvaluation({
        test_set_id: selectedTestSetId,
        course_id: selectedCourseId,
        name: runName || undefined,
        config: {
          search_type: searchType,
          search_alpha: alpha,
          top_k: topK,
        },
      });
      
      setCurrentRun(run);
      toast.success("Değerlendirme başlatıldı");
      
      // Start polling
      const interval = setInterval(() => pollRunStatus(run.id), 2000);
      setPollingInterval(interval);
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "Hata oluştu");
    } finally {
      setIsStarting(false);
    }
  };

  const getStatusBadge = (status: string) => {
    switch (status) {
      case "completed":
        return (
          <span className="flex items-center gap-1 px-3 py-1.5 text-sm rounded-full bg-green-100 text-green-700">
            <CheckCircle className="w-4 h-4" /> Tamamlandı
          </span>
        );
      case "running":
        return (
          <span className="flex items-center gap-1 px-3 py-1.5 text-sm rounded-full bg-blue-100 text-blue-700">
            <Loader2 className="w-4 h-4 animate-spin" /> Çalışıyor
          </span>
        );
      case "failed":
        return (
          <span className="flex items-center gap-1 px-3 py-1.5 text-sm rounded-full bg-red-100 text-red-700">
            <XCircle className="w-4 h-4" /> Başarısız
          </span>
        );
      default:
        return (
          <span className="flex items-center gap-1 px-3 py-1.5 text-sm rounded-full bg-slate-100 text-slate-600">
            <Clock className="w-4 h-4" /> Bekliyor
          </span>
        );
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

  return (
    <div>
      <PageHeader
        icon={FlaskConical}
        title="Değerlendirme Başlat"
        description="RAG sisteminizi test edin ve metriklerini ölçün"
      >
        <Link href="/dashboard/ragas">
          <Button variant="outline" size="sm">
            <ArrowLeft className="w-4 h-4 mr-1" /> Geri
          </Button>
        </Link>
      </PageHeader>

      {currentRun ? (
        <div className="bg-white rounded-xl border border-slate-200 p-8">
          <div className="text-center">
            <div className="mb-4">{getStatusBadge(currentRun.status)}</div>
            <h2 className="text-xl font-semibold text-slate-900 mb-2">
              {currentRun.name || `Değerlendirme #${currentRun.id}`}
            </h2>
            <p className="text-slate-500 mb-6">
              {currentRun.processed_questions} / {currentRun.total_questions} soru işlendi
            </p>
            
            {/* Progress bar */}
            <div className="w-full max-w-md mx-auto bg-slate-100 rounded-full h-3 mb-6">
              <div
                className="bg-indigo-600 h-3 rounded-full transition-all duration-500"
                style={{
                  width: `${currentRun.total_questions > 0 
                    ? (currentRun.processed_questions / currentRun.total_questions) * 100 
                    : 0}%`
                }}
              />
            </div>

            {currentRun.status === "running" && (
              <p className="text-sm text-slate-400">
                Değerlendirme devam ediyor, lütfen bekleyin...
              </p>
            )}

            {currentRun.status === "completed" && (
              <Link href={`/dashboard/ragas/runs/${currentRun.id}`}>
                <Button className="bg-emerald-600 hover:bg-emerald-700">
                  Sonuçları Görüntüle
                </Button>
              </Link>
            )}

            {currentRun.error_message && (
              <p className="text-sm text-red-600 mt-4">{currentRun.error_message}</p>
            )}
          </div>
        </div>
      ) : (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Configuration */}
          <div className="bg-white rounded-xl border border-slate-200 p-6">
            <h2 className="font-semibold text-slate-900 mb-4">Değerlendirme Ayarları</h2>
            
            <div className="space-y-4">
              <div className="space-y-2">
                <Label>Ders</Label>
                <Select
                  value={selectedCourseId?.toString() || ""}
                  onValueChange={(v) => { setSelectedCourseId(Number(v)); setSelectedTestSetId(null); }}
                >
                  <SelectTrigger>
                    <SelectValue placeholder="Ders seçin" />
                  </SelectTrigger>
                  <SelectContent>
                    {courses.map((course) => (
                      <SelectItem key={course.id} value={course.id.toString()}>
                        {course.name}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              <div className="space-y-2">
                <Label>Test Seti</Label>
                <Select
                  value={selectedTestSetId?.toString() || ""}
                  onValueChange={(v) => setSelectedTestSetId(Number(v))}
                  disabled={!selectedCourseId}
                >
                  <SelectTrigger>
                    <SelectValue placeholder="Test seti seçin" />
                  </SelectTrigger>
                  <SelectContent>
                    {testSets.map((ts) => (
                      <SelectItem key={ts.id} value={ts.id.toString()}>
                        {ts.name} ({ts.question_count} soru)
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              <div className="space-y-2">
                <Label>Değerlendirme Adı (Opsiyonel)</Label>
                <Input
                  value={runName}
                  onChange={(e) => setRunName(e.target.value)}
                  placeholder="Örn: Baseline Test"
                />
              </div>
            </div>
          </div>

          {/* Search Settings */}
          <div className="bg-white rounded-xl border border-slate-200 p-6">
            <h2 className="font-semibold text-slate-900 mb-4">Arama Ayarları</h2>
            
            <div className="space-y-6">
              <div className="space-y-2">
                <Label>Arama Tipi</Label>
                <Select value={searchType} onValueChange={setSearchType}>
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="hybrid">Hibrit (Vektör + Anahtar Kelime)</SelectItem>
                    <SelectItem value="vector">Sadece Vektör</SelectItem>
                    <SelectItem value="keyword">Sadece Anahtar Kelime</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              {searchType === "hybrid" && (
                <div className="space-y-2">
                  <div className="flex justify-between">
                    <Label>Alpha (Vektör Ağırlığı)</Label>
                    <span className="text-sm text-slate-500">{alpha.toFixed(2)}</span>
                  </div>
                  <Slider
                    value={[alpha]}
                    onValueChange={([v]) => setAlpha(v)}
                    min={0}
                    max={1}
                    step={0.1}
                  />
                  <p className="text-xs text-slate-400">
                    0 = Sadece anahtar kelime, 1 = Sadece vektör
                  </p>
                </div>
              )}

              <div className="space-y-2">
                <div className="flex justify-between">
                  <Label>Top K (Sonuç Sayısı)</Label>
                  <span className="text-sm text-slate-500">{topK}</span>
                </div>
                <Slider
                  value={[topK]}
                  onValueChange={([v]) => setTopK(v)}
                  min={1}
                  max={20}
                  step={1}
                />
              </div>
            </div>
          </div>

          {/* Start Button */}
          <div className="lg:col-span-2">
            <Button
              onClick={handleStartEvaluation}
              disabled={isStarting || !selectedTestSetId}
              className="w-full bg-indigo-600 hover:bg-indigo-700 h-12 text-lg"
            >
              {isStarting ? (
                <><Loader2 className="w-5 h-5 mr-2 animate-spin" /> Başlatılıyor...</>
              ) : (
                <><Play className="w-5 h-5 mr-2" /> Değerlendirmeyi Başlat</>
              )}
            </Button>
          </div>
        </div>
      )}
    </div>
  );
}
