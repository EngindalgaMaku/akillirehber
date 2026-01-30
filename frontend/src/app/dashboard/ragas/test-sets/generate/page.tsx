
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

            <div className="space-y-2">
              <Label>Bloom Dağılımı (toplam 1.0)</Label>
              <div className="grid gap-3">
                <div className="grid grid-cols-3 items-center gap-3">
                  <div className="text-sm">Hatırlama</div>
                  <Input
                    type="number"
                    min={0}
                    max={1}
                    step={0.05}
                    value={rememberingRatio}
                    onChange={(e) => setRememberingRatio(Number(e.target.value))}
                  />
                  <div className="text-xs text-muted-foreground text-right">{(rememberingRatio * 100).toFixed(0)}%</div>
                </div>
                <div className="grid grid-cols-3 items-center gap-3">
                  <div className="text-sm">Anlama/Uygulama</div>
                  <Input
                    type="number"
                    min={0}
                    max={1}
                    step={0.05}
                    value={understandingApplyingRatio}
                    onChange={(e) => setUnderstandingApplyingRatio(Number(e.target.value))}
                  />
                  <div className="text-xs text-muted-foreground text-right">{(understandingApplyingRatio * 100).toFixed(0)}%</div>
                </div>
                <div className="grid grid-cols-3 items-center gap-3">
                  <div className="text-sm">Analiz/Değerlendirme</div>
                  <Input
                    type="number"
                    min={0}
                    max={1}
                    step={0.05}
                    value={analyzingEvaluatingRatio}
                    onChange={(e) => setAnalyzingEvaluatingRatio(Number(e.target.value))}
                  />
                  <div className="text-xs text-muted-foreground text-right">{(analyzingEvaluatingRatio * 100).toFixed(0)}%</div>
                </div>
                <div className="text-xs text-muted-foreground">
                  Toplam: {(rememberingRatio + understandingApplyingRatio + analyzingEvaluatingRatio).toFixed(2)}
                </div>
              </div>
            </div>

            <div className="space-y-3">
              <div className="flex items-center justify-between gap-2">
                <Label>Bloom Sistem Promptları</Label>
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
              </div>

              <div className="space-y-2">
                <Label htmlFor="prompt_remembering">Hatırlama</Label>
                <Textarea
                  id="prompt_remembering"
                  value={promptRemembering}
                  onChange={(e) => setPromptRemembering(e.target.value)}
                  rows={4}
                  disabled={!selectedCourse || isLoadingSettings}
                />
              </div>

              <div className="space-y-2">
                <Label htmlFor="prompt_understanding_applying">Anlama/Uygulama</Label>
                <Textarea
                  id="prompt_understanding_applying"
                  value={promptUnderstandingApplying}
                  onChange={(e) => setPromptUnderstandingApplying(e.target.value)}
                  rows={4}
                  disabled={!selectedCourse || isLoadingSettings}
                />
              </div>

              <div className="space-y-2">
                <Label htmlFor="prompt_analyzing_evaluating">Analiz/Değerlendirme</Label>
                <Textarea
                  id="prompt_analyzing_evaluating"
                  value={promptAnalyzingEvaluating}
                  onChange={(e) => setPromptAnalyzingEvaluating(e.target.value)}
                  rows={4}
                  disabled={!selectedCourse || isLoadingSettings}
                />
              </div>
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
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
