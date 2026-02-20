
"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { toast } from "sonner";
import { api, type Course, type CourseSettings, type TestSet } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { ArrowLeft, Loader2, Plus, Sparkles } from "lucide-react";

export default function GenerateTestSetQuestionsPage() {
  const router = useRouter();
  const [courses, setCourses] = useState<Course[]>([]);
  const [selectedCourse, setSelectedCourse] = useState<number | null>(null);
  const [testSets, setTestSets] = useState<TestSet[]>([]);
  const [selectedTestSet, setSelectedTestSet] = useState<number | null>(null);

  const [numQuestions, setNumQuestions] = useState<number>(50);
  const [rememberingRatio, setRememberingRatio] = useState<number>(0.3);
  const [understandingApplyingRatio, setUnderstandingApplyingRatio] = useState<number>(0.4);
  const [analyzingEvaluatingRatio, setAnalyzingEvaluatingRatio] = useState<number>(0.3);

  const [isLoading, setIsLoading] = useState(true);
  const [isGenerating, setIsGenerating] = useState(false);
  
  // Quality filter states
  const [isQualityFilterOpen, setIsQualityFilterOpen] = useState(false);
  const [isQualityGenerating, setIsQualityGenerating] = useState(false);
  const [minRouge1Score, setMinRouge1Score] = useState<number>(0.60);
  const [qualityProgress, setQualityProgress] = useState<{
    accepted: number;
    rejected: number;
    target: number;
    events: Array<{
      type: 'accepted' | 'rejected';
      question: string;
      bloom_level: string;
      rouge1: number;
      reason?: string;
    }>;
  }>({
    accepted: 0,
    rejected: 0,
    target: 0,
    events: []
  });

  const [isNewDialogOpen, setIsNewDialogOpen] = useState(false);
  const [newTestSetName, setNewTestSetName] = useState("");
  const [newTestSetDescription, setNewTestSetDescription] = useState("");
  const [isCreating, setIsCreating] = useState(false);

  const [courseSettings, setCourseSettings] = useState<CourseSettings | null>(null);
  const [isLoadingSettings, setIsLoadingSettings] = useState(false);
  const [isSavingSettings, setIsSavingSettings] = useState(false);

  const [promptRemembering, setPromptRemembering] = useState<string>("");
  const [promptUnderstandingApplying, setPromptUnderstandingApplying] = useState<string>("");
  const [promptAnalyzingEvaluating, setPromptAnalyzingEvaluating] = useState<string>("");
  const [showPrompts, setShowPrompts] = useState<boolean>(false);

  useEffect(() => {
    const loadCourses = async () => {
      setIsLoading(true);
      try {
        const data = await api.getCourses();
        setCourses(data);

        if (data.length > 0) {
          const savedCourseId = localStorage.getItem("ragas_selected_course_id");
          const parsed = savedCourseId ? Number.parseInt(savedCourseId) : null;
          if (parsed && data.some((c) => c.id === parsed)) {
            setSelectedCourse(parsed);
          } else {
            setSelectedCourse(data[0].id);
            localStorage.setItem("ragas_selected_course_id", data[0].id.toString());
          }
        } else {
          setSelectedCourse(null);
        }
      } catch (error) {
        console.error("Failed to load courses:", error);
        setCourses([]);
        setSelectedCourse(null);
      } finally {
        setIsLoading(false);
      }
    };

    loadCourses();
  }, []);

  useEffect(() => {
    const loadTestSets = async () => {
      if (!selectedCourse) return;
      try {
        const data = await api.getTestSets(selectedCourse);
        setTestSets(data);
        setSelectedTestSet(data[0]?.id ?? null);
      } catch (error) {
        console.error("Failed to load test sets:", error);
        setTestSets([]);
        setSelectedTestSet(null);
      }
    };

    loadTestSets();
  }, [selectedCourse]);

  useEffect(() => {
    const loadSettings = async () => {
      if (!selectedCourse) {
        setCourseSettings(null);
        return;
      }
      setIsLoadingSettings(true);
      try {
        const settings = await api.getCourseSettings(selectedCourse);
        setCourseSettings(settings);
        setPromptRemembering(settings.system_prompt_remembering || "");
        setPromptUnderstandingApplying(settings.system_prompt_understanding_applying || "");
        setPromptAnalyzingEvaluating(settings.system_prompt_analyzing_evaluating || "");
      } catch (error) {
        console.error("Failed to load course settings:", error);
        setCourseSettings(null);
      } finally {
        setIsLoadingSettings(false);
      }
    };

    loadSettings();
  }, [selectedCourse]);

  const handleSavePrompts = async () => {
    if (!selectedCourse) return;
    setIsSavingSettings(true);
    try {
      const updated = await api.updateCourseSettings(selectedCourse, {
        system_prompt_remembering: promptRemembering,
        system_prompt_understanding_applying: promptUnderstandingApplying,
        system_prompt_analyzing_evaluating: promptAnalyzingEvaluating,
      });
      setCourseSettings(updated);
      toast.success("Bloom promptları kaydedildi");
    } catch (error) {
      console.error("Failed to save course settings:", error);
      toast.error(error instanceof Error ? error.message : "Ayarlar kaydedilemedi");
    } finally {
      setIsSavingSettings(false);
    }
  };

  const handleCreateTestSet = async () => {
    if (!selectedCourse || !newTestSetName.trim()) return;
    setIsCreating(true);
    try {
      const created = await api.createTestSet({
        course_id: selectedCourse,
        name: newTestSetName.trim(),
        description: newTestSetDescription.trim() || undefined,
      });

      const updated = await api.getTestSets(selectedCourse);
      setTestSets(updated);
      setSelectedTestSet(created.id);

      setIsNewDialogOpen(false);
      setNewTestSetName("");
      setNewTestSetDescription("");
      toast.success("Test seti oluşturuldu");
    } catch (error) {
      console.error("Failed to create test set:", error);
      toast.error("Test seti oluşturulamadı");
    } finally {
      setIsCreating(false);
    }
  };

  const handleGenerate = async () => {
    if (!selectedTestSet) return;

    const safeNum = Number.isFinite(numQuestions) ? Math.max(1, Math.min(200, numQuestions)) : 50;
    const totalRatio = rememberingRatio + understandingApplyingRatio + analyzingEvaluatingRatio;
    if (Math.abs(totalRatio - 1.0) > 0.01) {
      toast.error(`Bloom oranları toplamı 1.0 olmalı (şu an: ${totalRatio.toFixed(2)})`);
      return;
    }
    setIsGenerating(true);
    try {
      const res = await api.generateFromCourse({
        test_set_id: selectedTestSet,
        total_questions: safeNum,
        remembering_ratio: rememberingRatio,
        understanding_applying_ratio: understandingApplyingRatio,
        analyzing_evaluating_ratio: analyzingEvaluatingRatio,
      });

      toast.success(`Sorular üretildi: ${res.saved_count}/${res.generated_count}`);
      router.push(`/dashboard/ragas/test-sets/${selectedTestSet}`);
    } catch (error) {
      console.error("Generate error:", error);
      toast.error(error instanceof Error ? error.message : "Soru üretimi başarısız");
    } finally {
      setIsGenerating(false);
    }
  };

  const handleQualityFilterGenerate = async () => {
    if (!selectedTestSet) return;

    const safeNum = Number.isFinite(numQuestions) ? Math.max(1, Math.min(200, numQuestions)) : 10;
    const totalRatio = rememberingRatio + understandingApplyingRatio + analyzingEvaluatingRatio;
    if (Math.abs(totalRatio - 1.0) > 0.01) {
      toast.error(`Bloom oranları toplamı 1.0 olmalı (şu an: ${totalRatio.toFixed(2)})`);
      return;
    }

    setIsQualityGenerating(true);
    setQualityProgress({
      accepted: 0,
      rejected: 0,
      target: safeNum,
      events: []
    });

    try {
      const generator = api.generateWithQualityFilter({
        test_set_id: selectedTestSet,
        target_questions: safeNum,
        min_rouge1_score: minRouge1Score,
        remembering_ratio: rememberingRatio,
        understanding_applying_ratio: understandingApplyingRatio,
        analyzing_evaluating_ratio: analyzingEvaluatingRatio,
      });

      for await (const event of generator) {
        if (event.event === 'start') {
          setQualityProgress(prev => ({
            ...prev,
            target: event.target || safeNum
          }));
        } else if (event.event === 'accepted') {
          setQualityProgress(prev => ({
            accepted: event.accepted || prev.accepted,
            rejected: event.rejected || prev.rejected,
            target: prev.target,
            events: [...prev.events, {
              type: 'accepted',
              question: event.question || '',
              bloom_level: event.bloom_level || '',
              rouge1: event.rouge1 || 0,
            }]
          }));
        } else if (event.event === 'rejected') {
          setQualityProgress(prev => ({
            accepted: event.accepted || prev.accepted,
            rejected: event.rejected || prev.rejected,
            target: prev.target,
            events: [...prev.events, {
              type: 'rejected',
              question: event.question || '',
              bloom_level: event.bloom_level || '',
              rouge1: event.rouge1 || 0,
              reason: event.reason
            }]
          }));
        } else if (event.event === 'complete') {
          toast.success(`${event.accepted} kaliteli soru oluşturuldu!`);
          setTimeout(() => {
            router.push(`/dashboard/ragas/test-sets/${selectedTestSet}`);
          }, 2000);
        } else if (event.event === 'error') {
          toast.error(event.error || 'Hata oluştu');
        }
      }
    } catch (error) {
      console.error("Quality filter generate error:", error);
      toast.error(error instanceof Error ? error.message : "Soru üretimi başarısız");
    } finally {
      setIsQualityGenerating(false);
    }
  };

  return (
    <div className="container mx-auto py-8 px-4">
      <Link href="/dashboard/ragas/test-sets" className="inline-flex items-center gap-2 text-sm text-muted-foreground hover:text-foreground mb-4">
        <ArrowLeft className="h-4 w-4" />
        Test Setlerine Dön
      </Link>

      <div className="mb-8">
        <h1 className="text-3xl font-bold">Yeni Soru Üret</h1>
        <p className="text-muted-foreground mt-2">Ders dokümanlarından RAGAS ile otomatik test soruları üretin</p>
      </div>

      <div className="grid gap-6 lg:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle>Hedef</CardTitle>
            <CardDescription>Hangi ders ve test seti içine ekleneceğini seçin</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-2">
              <Label>Ders</Label>
              <select
                className="w-full px-3 py-2 border rounded-md"
                value={selectedCourse || ""}
                onChange={(e) => {
                  const nextId = Number(e.target.value);
                  setSelectedCourse(nextId);
                  localStorage.setItem("ragas_selected_course_id", nextId.toString());
                }}
                disabled={isLoading || courses.length === 0}
              >
                <option value="">Seçin...</option>
                {courses.map((c) => (
                  <option key={c.id} value={c.id}>
                    {c.name}
                  </option>
                ))}
              </select>
            </div>

            <div className="space-y-2">
              <div className="flex items-center justify-between gap-2">
                <Label>Test Seti</Label>
                <Dialog open={isNewDialogOpen} onOpenChange={setIsNewDialogOpen}>
                  <DialogTrigger asChild>
                    <Button variant="outline" size="sm" disabled={!selectedCourse}>
                      <Plus className="h-4 w-4 mr-1" />
                      Yeni
                    </Button>
                  </DialogTrigger>
                  <DialogContent>
                    <DialogHeader>
                      <DialogTitle>Yeni Test Seti Oluştur</DialogTitle>
                      <DialogDescription>Üretilen sorular bu test setine eklenecek</DialogDescription>
                    </DialogHeader>
                    <div className="space-y-4 py-4">
                      <div className="space-y-2">
                        <Label htmlFor="new_testset_name">Test Seti Adı *</Label>
                        <Input
                          id="new_testset_name"
                          value={newTestSetName}
                          onChange={(e) => setNewTestSetName(e.target.value)}
                          disabled={isCreating}
                        />
                      </div>
                      <div className="space-y-2">
                        <Label htmlFor="new_testset_desc">Açıklama (Opsiyonel)</Label>
                        <Textarea
                          id="new_testset_desc"
                          value={newTestSetDescription}
                          onChange={(e) => setNewTestSetDescription(e.target.value)}
                          disabled={isCreating}
                          rows={3}
                        />
                      </div>
                    </div>
                    <DialogFooter>
                      <Button
                        variant="outline"
                        onClick={() => {
                          setIsNewDialogOpen(false);
                          setNewTestSetName("");
                          setNewTestSetDescription("");
                        }}
                        disabled={isCreating}
                      >
                        İptal
                      </Button>
                      <Button onClick={handleCreateTestSet} disabled={!newTestSetName.trim() || isCreating}>
                        {isCreating ? (
                          <>
                            <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                            Oluşturuluyor...
                          </>
                        ) : (
                          "Oluştur"
                        )}
                      </Button>
                    </DialogFooter>
                  </DialogContent>
                </Dialog>
              </div>

              <select
                className="w-full px-3 py-2 border rounded-md"
                value={selectedTestSet || ""}
                onChange={(e) => setSelectedTestSet(Number(e.target.value))}
                disabled={!selectedCourse || testSets.length === 0}
              >
                <option value="">Seçin...</option>
                {testSets.map((ts) => (
                  <option key={ts.id} value={ts.id}>
                    {ts.name} ({ts.question_count})
                  </option>
                ))}
              </select>
            </div>

            {selectedCourse && testSets.length === 0 && (
              <div className="text-sm text-muted-foreground">
                Bu ders için test seti yok. Önce bir test seti oluşturmalısın.
              </div>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Ayarlar</CardTitle>
            <CardDescription>Soru sayısı ve Bloom taksonomisi dağılımı</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-2">
              <Label htmlFor="num_questions">Soru Sayısı</Label>
              <Input
                id="num_questions"
                type="number"
                min={1}
                max={200}
                value={numQuestions}
                onChange={(e) => setNumQuestions(Number(e.target.value))}
              />
              <div className="text-xs text-muted-foreground">1-200 arası önerilir</div>
            </div>

            <div className="space-y-4">
              <Label>Bloom Dağılımı</Label>
              
              {/* Hatırlama */}
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <span className="text-sm font-medium">🧠 Hatırlama</span>
                  <span className="text-sm font-bold text-blue-600">{(rememberingRatio * 100).toFixed(0)}%</span>
                </div>
                <input
                  type="range"
                  min={0}
                  max={100}
                  step={5}
                  value={rememberingRatio * 100}
                  onChange={(e) => {
                    const newValue = Number(e.target.value) / 100;
                    const remaining = 1 - newValue;
                    const currentOtherTotal = understandingApplyingRatio + analyzingEvaluatingRatio;
                    
                    if (currentOtherTotal > 0) {
                      // Diğer ikisini orantılı olarak ayarla
                      const ratio = remaining / currentOtherTotal;
                      setRememberingRatio(newValue);
                      setUnderstandingApplyingRatio(understandingApplyingRatio * ratio);
                      setAnalyzingEvaluatingRatio(analyzingEvaluatingRatio * ratio);
                    } else {
                      // Eğer diğerleri 0 ise, eşit dağıt
                      setRememberingRatio(newValue);
                      setUnderstandingApplyingRatio(remaining / 2);
                      setAnalyzingEvaluatingRatio(remaining / 2);
                    }
                  }}
                  className="w-full h-2 bg-blue-200 rounded-lg appearance-none cursor-pointer accent-blue-600"
                />
              </div>

              {/* Anlama/Uygulama */}
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <span className="text-sm font-medium">🔧 Anlama/Uygulama</span>
                  <span className="text-sm font-bold text-purple-600">{(understandingApplyingRatio * 100).toFixed(0)}%</span>
                </div>
                <input
                  type="range"
                  min={0}
                  max={100}
                  step={5}
                  value={understandingApplyingRatio * 100}
                  onChange={(e) => {
                    const newValue = Number(e.target.value) / 100;
                    const remaining = 1 - newValue;
                    const currentOtherTotal = rememberingRatio + analyzingEvaluatingRatio;
                    
                    if (currentOtherTotal > 0) {
                      const ratio = remaining / currentOtherTotal;
                      setUnderstandingApplyingRatio(newValue);
                      setRememberingRatio(rememberingRatio * ratio);
                      setAnalyzingEvaluatingRatio(analyzingEvaluatingRatio * ratio);
                    } else {
                      setUnderstandingApplyingRatio(newValue);
                      setRememberingRatio(remaining / 2);
                      setAnalyzingEvaluatingRatio(remaining / 2);
                    }
                  }}
                  className="w-full h-2 bg-purple-200 rounded-lg appearance-none cursor-pointer accent-purple-600"
                />
              </div>

              {/* Analiz/Değerlendirme */}
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <span className="text-sm font-medium">⭐ Analiz/Değerlendirme</span>
                  <span className="text-sm font-bold text-orange-600">{(analyzingEvaluatingRatio * 100).toFixed(0)}%</span>
                </div>
                <input
                  type="range"
                  min={0}
                  max={100}
                  step={5}
                  value={analyzingEvaluatingRatio * 100}
                  onChange={(e) => {
                    const newValue = Number(e.target.value) / 100;
                    const remaining = 1 - newValue;
                    const currentOtherTotal = rememberingRatio + understandingApplyingRatio;
                    
                    if (currentOtherTotal > 0) {
                      const ratio = remaining / currentOtherTotal;
                      setAnalyzingEvaluatingRatio(newValue);
                      setRememberingRatio(rememberingRatio * ratio);
                      setUnderstandingApplyingRatio(understandingApplyingRatio * ratio);
                    } else {
                      setAnalyzingEvaluatingRatio(newValue);
                      setRememberingRatio(remaining / 2);
                      setUnderstandingApplyingRatio(remaining / 2);
                    }
                  }}
                  className="w-full h-2 bg-orange-200 rounded-lg appearance-none cursor-pointer accent-orange-600"
                />
              </div>

              {/* Görsel Bar */}
              <div className="mt-4">
                <div className="flex h-8 rounded-lg overflow-hidden border border-slate-200">
                  <div 
                    className="bg-blue-500 flex items-center justify-center text-white text-xs font-bold transition-all duration-300"
                    style={{ width: `${rememberingRatio * 100}%` }}
                  >
                    {rememberingRatio > 0.1 && `${(rememberingRatio * 100).toFixed(0)}%`}
                  </div>
                  <div 
                    className="bg-purple-500 flex items-center justify-center text-white text-xs font-bold transition-all duration-300"
                    style={{ width: `${understandingApplyingRatio * 100}%` }}
                  >
                    {understandingApplyingRatio > 0.1 && `${(understandingApplyingRatio * 100).toFixed(0)}%`}
                  </div>
                  <div 
                    className="bg-orange-500 flex items-center justify-center text-white text-xs font-bold transition-all duration-300"
                    style={{ width: `${analyzingEvaluatingRatio * 100}%` }}
                  >
                    {analyzingEvaluatingRatio > 0.1 && `${(analyzingEvaluatingRatio * 100).toFixed(0)}%`}
                  </div>
                </div>
                <div className="flex justify-between text-xs text-muted-foreground mt-2">
                  <span>🧠 Hatırlama</span>
                  <span>🔧 Anlama/Uygulama</span>
                  <span>⭐ Analiz/Değerlendirme</span>
                </div>
              </div>

              <div className="text-xs text-center text-muted-foreground bg-slate-50 p-2 rounded">
                Toplam: {((rememberingRatio + understandingApplyingRatio + analyzingEvaluatingRatio) * 100).toFixed(0)}%
              </div>
            </div>

            <div className="space-y-3">
              <div className="flex items-center justify-between gap-2">
                <button
                  onClick={() => setShowPrompts(!showPrompts)}
                  className="flex items-center gap-2 text-sm font-medium text-slate-700 hover:text-slate-900"
                >
                  <span>{showPrompts ? "▼" : "▶"}</span>
                  <span>Bloom Sistem Promptları (Gelişmiş)</span>
                </button>
                {showPrompts && (
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={handleSavePrompts}
                    disabled={!selectedCourse || isLoadingSettings || isSavingSettings}
                  >
                    {isSavingSettings ? (
                      <>
                        <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                        Kaydediliyor...
                      </>
                    ) : (
                      "Kaydet"
                    )}
                  </Button>
                )}
              </div>

              {showPrompts && (
                <div className="space-y-3 pt-2 border-t">
                  <div className="space-y-2">
                    <Label htmlFor="prompt_remembering" className="text-xs">Hatırlama Promptu</Label>
                    <Textarea
                      id="prompt_remembering"
                      value={promptRemembering}
                      onChange={(e) => setPromptRemembering(e.target.value)}
                      rows={4}
                      disabled={!selectedCourse || isLoadingSettings}
                      className="text-xs"
                    />
                  </div>

                  <div className="space-y-2">
                    <Label htmlFor="prompt_understanding_applying" className="text-xs">Anlama/Uygulama Promptu</Label>
                    <Textarea
                      id="prompt_understanding_applying"
                      value={promptUnderstandingApplying}
                      onChange={(e) => setPromptUnderstandingApplying(e.target.value)}
                      rows={4}
                      disabled={!selectedCourse || isLoadingSettings}
                      className="text-xs"
                    />
                  </div>

                  <div className="space-y-2">
                    <Label htmlFor="prompt_analyzing_evaluating" className="text-xs">Analiz/Değerlendirme Promptu</Label>
                    <Textarea
                      id="prompt_analyzing_evaluating"
                      value={promptAnalyzingEvaluating}
                      onChange={(e) => setPromptAnalyzingEvaluating(e.target.value)}
                      rows={4}
                      disabled={!selectedCourse || isLoadingSettings}
                      className="text-xs"
                    />
                  </div>
                </div>
              )}
            </div>

            <Button
              className="w-full"
              onClick={handleGenerate}
              disabled={!selectedTestSet || isGenerating}
            >
              {isGenerating ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  Üretiliyor...
                </>
              ) : (
                <>
                  <Sparkles className="mr-2 h-4 w-4" />
                  Soruları Üret
                </>
              )}
            </Button>

            <Button
              className="w-full mt-3"
              variant="outline"
              onClick={() => setIsQualityFilterOpen(true)}
              disabled={!selectedTestSet || isQualityGenerating}
            >
              <Sparkles className="mr-2 h-4 w-4" />
              🎯 Kalite Filtreli Üretim (Önerilen)
            </Button>
          </CardContent>
        </Card>
      </div>

      {/* Quality Filter Modal */}
      <Dialog open={isQualityFilterOpen} onOpenChange={setIsQualityFilterOpen}>
        <DialogContent className="max-w-4xl max-h-[90vh] overflow-y-auto">
          <DialogHeader>
            <DialogTitle>🎯 Kalite Filtreli Soru Üretimi</DialogTitle>
            <DialogDescription>
              Her soru test edilir, kalitesiz olanlar otomatik elenir. Sadece ROUGE-1 ≥ {(minRouge1Score * 100).toFixed(0)}% olan sorular kabul edilir.
            </DialogDescription>
          </DialogHeader>

          <div className="space-y-4">
            <div>
              <Label>Minimum ROUGE-1 Skoru (%)</Label>
              <Input
                type="number"
                min="0"
                max="100"
                value={minRouge1Score * 100}
                onChange={(e) => setMinRouge1Score(Number(e.target.value) / 100)}
                disabled={isQualityGenerating}
              />
              <p className="text-xs text-muted-foreground mt-1">
                Önerilen: 60% (Daha yüksek = daha kaliteli ama daha yavaş)
              </p>
            </div>

            {isQualityGenerating && (
              <div className="space-y-3">
                <div className="flex items-center justify-between p-3 bg-slate-100 rounded">
                  <div>
                    <p className="font-medium">İlerleme</p>
                    <p className="text-sm text-muted-foreground">
                      {qualityProgress.accepted} / {qualityProgress.target} kabul edildi
                      {qualityProgress.rejected > 0 && ` (${qualityProgress.rejected} reddedildi)`}
                    </p>
                  </div>
                  <Loader2 className="h-6 w-6 animate-spin text-indigo-600" />
                </div>

                <div className="max-h-96 overflow-y-auto space-y-2">
                  {qualityProgress.events.slice().reverse().map((event, idx) => (
                    <div
                      key={idx}
                      className={`p-3 rounded border ${
                        event.type === 'accepted'
                          ? 'bg-green-50 border-green-200'
                          : 'bg-red-50 border-red-200'
                      }`}
                    >
                      <div className="flex items-start justify-between gap-2">
                        <div className="flex-1 min-w-0">
                          <p className="text-sm font-medium truncate">
                            {event.type === 'accepted' ? '✅' : '❌'} {event.question}
                          </p>
                          <p className="text-xs text-muted-foreground mt-1">
                            {event.bloom_level} • ROUGE-1: {event.rouge1.toFixed(1)}%
                            {event.reason && ` • ${event.reason}`}
                          </p>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>

          <DialogFooter>
            <Button
              variant="outline"
              onClick={() => setIsQualityFilterOpen(false)}
              disabled={isQualityGenerating}
            >
              {isQualityGenerating ? 'Devam Ediyor...' : 'İptal'}
            </Button>
            <Button
              onClick={handleQualityFilterGenerate}
              disabled={isQualityGenerating || !selectedTestSet}
            >
              {isQualityGenerating ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  Üretiliyor...
                </>
              ) : (
                'Başlat'
              )}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}
