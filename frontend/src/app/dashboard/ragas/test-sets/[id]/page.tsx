"use client";

import { useEffect, useState, useCallback } from "react";
import { useParams, useRouter } from "next/navigation";
import { useAuth } from "@/lib/auth-context";
import { api, TestSetDetail, TestQuestion, EvaluationRun } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { toast } from "sonner";
import { PageHeader } from "@/components/ui/page-header";
import {
  FileText,
  Plus,
  Trash2,
  Edit2,
  Loader2,
  ArrowLeft,
  Download,
  Upload,
  Save,
  Play,
  BarChart3,
  Settings,
} from "lucide-react";
import Link from "next/link";

export default function TestSetEditorPage() {
  const { id } = useParams();
  const router = useRouter();
  const { user } = useAuth();
  const [testSet, setTestSet] = useState<TestSetDetail | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [isAddOpen, setIsAddOpen] = useState(false);
  const [isEditOpen, setIsEditOpen] = useState(false);
  const [isImportOpen, setIsImportOpen] = useState(false);
  const [isSaving, setIsSaving] = useState(false);
  const [editingQuestion, setEditingQuestion] = useState<TestQuestion | null>(null);
  const [newQuestion, setNewQuestion] = useState({
    question: "",
    ground_truth: "",
    alternative_ground_truths: [] as string[]
  });
  const [newAltAnswer, setNewAltAnswer] = useState("");
  const [editAltAnswer, setEditAltAnswer] = useState("");
  const [importJson, setImportJson] = useState("");
  const [isStartingEval, setIsStartingEval] = useState(false);
  const [selectedQuestions, setSelectedQuestions] = useState<Set<number>>(new Set());
  const [isDuplicating, setIsDuplicating] = useState(false);
  const [evaluationRun, setEvaluationRun] = useState<EvaluationRun | null>(null);
  const [isLoadingRun, setIsLoadingRun] = useState(false);
  const [isRenameOpen, setIsRenameOpen] = useState(false);
  const [newName, setNewName] = useState("");
  const [isRenaming, setIsRenaming] = useState(false);
  const [courseSettings, setCourseSettings] = useState<any>(null);
  const [isLoadingSettings, setIsLoadingSettings] = useState(false);

  const [testDatasets, setTestDatasets] = useState<any[]>([]);
  const [selectedDataset, setSelectedDataset] = useState<string>("");
  const [isDatasetsLoading, setIsDatasetsLoading] = useState(false);

  const toggleQuestionSelection = (questionId: number) => {
    const newSelected = new Set(selectedQuestions);
    if (newSelected.has(questionId)) {
      newSelected.delete(questionId);
    } else {
      newSelected.add(questionId);
    }
    setSelectedQuestions(newSelected);
  };

  const toggleSelectAll = () => {
    if (selectedQuestions.size === testSet?.questions.length) {
      setSelectedQuestions(new Set());
    } else {
      setSelectedQuestions(new Set(testSet?.questions.map(q => q.id) || []));
    }
  };

  const loadTestSet = useCallback(async () => {
    try {
      const data = await api.getTestSet(Number(id));
      setTestSet(data);
    } catch {
      toast.error("Test seti yüklenirken hata oluştu");
      router.push("/dashboard/ragas");
    } finally {
      setIsLoading(false);
    }
  }, [id, router]);

  const loadEvaluationRun = useCallback(async () => {
    if (!testSet) return;
    setIsLoadingRun(true);
    try {
      const runs = await api.getEvaluationRuns(testSet.course_id, testSet.id);
      const latestRun = [...runs].sort((a, b) => {
        const at = a.created_at ? Date.parse(a.created_at) : 0;
        const bt = b.created_at ? Date.parse(b.created_at) : 0;
        if (bt !== at) return bt - at;
        return (b.id ?? 0) - (a.id ?? 0);
      })[0];
      setEvaluationRun(latestRun ?? null);
    } catch {
      console.error("Değerlendirme yüklenirken hata oluştu");
    } finally {
      setIsLoadingRun(false);
    }
  }, [testSet]);

  const loadCourseSettings = useCallback(async () => {
    if (!testSet) return;
    setIsLoadingSettings(true);
    try {
      const settings = await api.getCourseSettings(testSet.course_id);
      setCourseSettings(settings);
    } catch {
      console.error("Ders ayarları yüklenirken hata oluştu");
    } finally {
      setIsLoadingSettings(false);
    }
  }, [testSet]);

  const loadTestDatasets = useCallback(async () => {
    if (!testSet) return;
    setIsDatasetsLoading(true);
    try {
      const data = await api.getTestDatasets(testSet.course_id);
      setTestDatasets(data.datasets);
    } catch {
      console.error("Test veri setleri yüklenirken hata oluştu");
    } finally {
      setIsDatasetsLoading(false);
    }
  }, [testSet]);

  useEffect(() => {
    loadTestSet();
  }, [loadTestSet]);

  useEffect(() => {
    if (testSet) {
      loadEvaluationRun();
      loadCourseSettings();
      loadTestDatasets();
    }
  }, [testSet, loadEvaluationRun, loadCourseSettings, loadTestDatasets]);

  const handleLoadDataset = async (datasetId: string) => {
    if (!datasetId) return;
    try {
      const dataset = await api.getTestDataset(parseInt(datasetId));
      setImportJson(JSON.stringify(dataset.test_cases, null, 2));
      setSelectedDataset(datasetId);
      toast.success(`"${dataset.name}" veri seti yüklendi`);
    } catch (error) {
      console.error("Load dataset error:", error);
      toast.error(error instanceof Error ? error.message : "Veri seti yüklenemedi");
    }
  };

  const handleAddQuestion = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!testSet) return;
    
    setIsSaving(true);
    try {
      await api.addQuestion(testSet.id, {
        question: newQuestion.question,
        ground_truth: newQuestion.ground_truth,
        alternative_ground_truths: newQuestion.alternative_ground_truths.length > 0
          ? newQuestion.alternative_ground_truths
          : undefined
      });
      setNewQuestion({ question: "", ground_truth: "", alternative_ground_truths: [] });
      setNewAltAnswer("");
      setIsAddOpen(false);
      loadTestSet();
      toast.success("Soru eklendi");
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "Hata oluştu");
    } finally {
      setIsSaving(false);
    }
  };

  const handleEditQuestion = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!editingQuestion) return;
    
    setIsSaving(true);
    try {
      await api.updateQuestion(editingQuestion.id, {
        question: editingQuestion.question,
        ground_truth: editingQuestion.ground_truth,
        alternative_ground_truths: editingQuestion.alternative_ground_truths && editingQuestion.alternative_ground_truths.length > 0
          ? editingQuestion.alternative_ground_truths
          : undefined
      });
      setEditingQuestion(null);
      setEditAltAnswer("");
      setIsEditOpen(false);
      loadTestSet();
      toast.success("Soru güncellendi");
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "Hata oluştu");
    } finally {
      setIsSaving(false);
    }
  };

  const handleDeleteQuestion = async (questionId: number) => {
    if (!confirm("Bu soruyu silmek istediğinizden emin misiniz?")) return;
    try {
      await api.deleteQuestion(questionId);
      loadTestSet();
      toast.success("Soru silindi");
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "Hata oluştu");
    }
  };

  const handleImport = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!testSet) return;
    
    setIsSaving(true);
    try {
      const data = JSON.parse(importJson);
      const questions = Array.isArray(data) ? data : data.questions;
      await api.importQuestions(testSet.id, questions);
      setImportJson("");
      setIsImportOpen(false);
      loadTestSet();
      toast.success("Sorular içe aktarıldı");
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "Geçersiz JSON formatı");
    } finally {
      setIsSaving(false);
    }
  };

  const handleExport = async () => {
    if (!testSet) return;
    try {
      const data = await api.exportTestSet(testSet.id);
      const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `${testSet.name.replaceAll(/\s+/g, "_")}_export.json`;
      a.click();
      URL.revokeObjectURL(url);
      toast.success("Test seti dışa aktarıldı");
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "Hata oluştu");
    }
  };

  const handleStartEvaluation = async () => {
    if (!testSet) return;
    
    const questionsToEvaluate = selectedQuestions.size > 0 
      ? testSet.questions.filter(q => selectedQuestions.has(q.id))
      : testSet.questions;

    if (questionsToEvaluate.length === 0) {
      toast.error("Değerlendirilecek soru seçilmedi");
      return;
    }

    setIsStartingEval(true);
    try {
      const run = await api.startEvaluation({
        test_set_id: testSet.id,
        course_id: testSet.course_id,
        name: `${testSet.name} - ${new Date().toLocaleString("tr-TR")}`,
        question_ids: selectedQuestions.size > 0 ? Array.from(selectedQuestions) : undefined,
      });
      toast.success(`Değerlendirme başlatıldı (${questionsToEvaluate.length} soru)`);
      router.push(`/dashboard/ragas/runs/${run.id}`);
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "Değerlendirme başlatılamadı");
    } finally {
      setIsStartingEval(false);
    }
  };

  const handleDuplicateTestSet = async () => {
    if (!testSet) return;
    
    if (!confirm(`"${testSet.name}" test setini kopyalamak istediğinizden emin misiniz?`)) return;
    
    setIsDuplicating(true);
    try {
      const duplicated = await api.duplicateTestSet(testSet.id);
      toast.success("Test seti kopyalandı");
      router.push(`/dashboard/ragas/test-sets/${duplicated.id}`);
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "Hata oluştu");
    } finally {
      setIsDuplicating(false);
    }
  };

  const handleRenameTestSet = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!testSet || !newName.trim()) return;
    
    setIsRenaming(true);
    try {
      await api.updateTestSet(testSet.id, { name: newName.trim() });
      toast.success("Test seti adı güncellendi");
      setIsRenameOpen(false);
      loadTestSet();
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "Hata oluştu");
    } finally {
      setIsRenaming(false);
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

  if (!testSet) return null;

  return (
    <div>
      <Link href="/dashboard/ragas" className="flex items-center gap-2 text-sm text-muted-foreground hover:text-foreground mb-4">
        <ArrowLeft className="h-4 w-4" />
        RAGAS Değerlendirmesi&apos;ne Dön
      </Link>
      <PageHeader
        icon={FileText}
        title={testSet.name}
        description={testSet.description || "Test seti düzenleme"}
      >
        <div className="flex items-center gap-2">
          {evaluationRun && (
            <Link href={`/dashboard/ragas/runs/${evaluationRun.id}`}>
              <Button variant="outline" size="sm" className="border-indigo-200 text-indigo-600 hover:bg-indigo-50">
                <BarChart3 className="w-4 h-4 mr-1" />
                {evaluationRun.status === "completed" ? "Sonuçları Gör" : 
                 evaluationRun.status === "running" ? "Çalışıyor..." : 
                 "Değerlendirme"}
              </Button>
            </Link>
          )}
          <Link href="/dashboard/ragas">
            <Button variant="outline" size="sm">
              <ArrowLeft className="w-4 h-4 mr-1" /> Geri
            </Button>
          </Link>
          <Button 
            variant="outline" 
            size="sm" 
            onClick={() => {
              setNewName(testSet.name);
              setIsRenameOpen(true);
            }}
          >
            ✏️ İsim Değiştir
          </Button>
          <Button 
            variant="outline" 
            size="sm" 
            onClick={handleDuplicateTestSet}
            disabled={isDuplicating}
          >
            {isDuplicating ? (
              <><Loader2 className="w-4 h-4 mr-1 animate-spin" /> Kopyalanıyor...</>
            ) : (
              <>📋 Kopyala</>
            )}
          </Button>
          <Button variant="outline" size="sm" onClick={() => setIsImportOpen(true)}>
            <Upload className="w-4 h-4 mr-1" /> İçe Aktar
          </Button>
          <Button variant="outline" size="sm" onClick={handleExport}>
            <Download className="w-4 h-4 mr-1" /> Dışa Aktar
          </Button>
          <Button size="sm" onClick={() => setIsAddOpen(true)}>
            <Plus className="w-4 h-4 mr-1" /> Soru Ekle
          </Button>
          <Button
            size="sm"
            onClick={handleStartEvaluation}
            disabled={isStartingEval || testSet.questions.length === 0}
            className="bg-green-600 hover:bg-green-700"
          >
            {isStartingEval ? (
              <><Loader2 className="w-4 h-4 mr-1 animate-spin" /> Başlatılıyor...</>
            ) : (
              <><Play className="w-4 h-4 mr-1" /> 
                {selectedQuestions.size > 0 
                  ? `Seçili ${selectedQuestions.size} Soruyu Değerlendir` 
                  : evaluationRun ? 'Devam Ettir' : 'Değerlendir'}
              </>
            )}
          </Button>
        </div>
      </PageHeader>

      {/* Course Settings Info Card */}
      {courseSettings && (
        <div className="bg-gradient-to-r from-blue-50 to-indigo-50 rounded-xl border border-blue-200 p-4 mb-4">
          <div className="flex items-start gap-3">
            <div className="p-2 bg-blue-100 rounded-lg">
              <Settings className="w-5 h-5 text-blue-600" />
            </div>
            <div className="flex-1">
              <h3 className="font-semibold text-slate-900 mb-2">Ders Ayarları (Test için kullanılacak)</h3>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
                <div className="bg-white rounded-lg p-3 border border-blue-100">
                  <p className="text-xs text-slate-500 mb-1">Alpha (Benzerlik Eşiği)</p>
                  <p className="font-bold text-blue-600 text-lg">{courseSettings.search_alpha ?? 0.7}</p>
                </div>
                <div className="bg-white rounded-lg p-3 border border-blue-100">
                  <p className="text-xs text-slate-500 mb-1">Top K</p>
                  <p className="font-bold text-slate-900 text-lg">{courseSettings.search_top_k ?? 5}</p>
                </div>
                <div className="bg-white rounded-lg p-3 border border-blue-100">
                  <p className="text-xs text-slate-500 mb-1">Embedding Model</p>
                  <p className="font-medium text-slate-900 text-xs truncate">{courseSettings.default_embedding_model || "Varsayılan"}</p>
                </div>
              </div>
              {courseSettings.system_prompt && (
                <div className="mt-3 bg-white rounded-lg p-3 border border-blue-100">
                  <p className="text-xs text-slate-500 mb-1">Sistem Promptu</p>
                  <p className="text-xs text-slate-700 line-clamp-2">{courseSettings.system_prompt}</p>
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      <div className="bg-white rounded-xl border border-slate-200 overflow-hidden">
        {testSet.questions.length === 0 ? (
          <div className="p-12 text-center">
            <FileText className="w-10 h-10 text-slate-300 mx-auto mb-3" />
            <p className="text-slate-500">Henüz soru eklenmemiş</p>
            <Button className="mt-4" onClick={() => setIsAddOpen(true)}>
              <Plus className="w-4 h-4 mr-1" /> İlk Soruyu Ekle
            </Button>
          </div>
        ) : (
          <div>
            {/* Select All Header */}
            <div className="p-4 bg-slate-50 border-b border-slate-200 flex items-center gap-3">
              <input
                type="checkbox"
                checked={selectedQuestions.size === testSet.questions.length && testSet.questions.length > 0}
                onChange={toggleSelectAll}
                className="w-4 h-4 text-indigo-600 rounded border-slate-300 focus:ring-indigo-500"
              />
              <span className="text-sm font-medium text-slate-700">
                {selectedQuestions.size > 0 
                  ? `${selectedQuestions.size} soru seçildi` 
                  : 'Tümünü seç'}
              </span>
              {selectedQuestions.size > 0 && (
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => setSelectedQuestions(new Set())}
                  className="text-xs text-slate-500 hover:text-slate-700"
                >
                  Seçimi Temizle
                </Button>
              )}
            </div>
            
            <div className="divide-y divide-slate-100">
              {testSet.questions.map((q, index) => (
                <div key={q.id} className="p-4 hover:bg-slate-50">
                  <div className="flex items-start gap-3">
                    <input
                      type="checkbox"
                      checked={selectedQuestions.has(q.id)}
                      onChange={() => toggleQuestionSelection(q.id)}
                      className="mt-1 w-4 h-4 text-indigo-600 rounded border-slate-300 focus:ring-indigo-500"
                    />
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2 mb-2">
                        <span className="text-xs font-medium text-slate-400">#{index + 1}</span>
                        <span className="text-sm font-medium text-slate-900">{q.question}</span>
                      </div>
                      <div className="space-y-2">
                        <p className="text-sm text-slate-600 bg-slate-50 p-2 rounded">
                          <span className="text-xs text-slate-400 block mb-1">Beklenen Cevap:</span>
                          {q.ground_truth}
                        </p>
                        {q.alternative_ground_truths && q.alternative_ground_truths.length > 0 && (
                          <div className="text-sm text-slate-600 bg-blue-50 p-2 rounded">
                            <span className="text-xs text-blue-600 block mb-1">
                              Alternatif Cevaplar ({q.alternative_ground_truths.length}):
                            </span>
                            <ul className="list-disc list-inside space-y-1">
                              {q.alternative_ground_truths.map((alt, idx) => (
                                <li key={idx} className="text-xs">{alt}</li>
                              ))}
                            </ul>
                          </div>
                        )}
                      </div>
                    </div>
                    <div className="flex items-center gap-1">
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={() => { setEditingQuestion(q); setIsEditOpen(true); }}
                        className="text-slate-400 hover:text-indigo-600"
                      >
                        <Edit2 className="w-4 h-4" />
                      </Button>
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={() => handleDeleteQuestion(q.id)}
                        className="text-slate-400 hover:text-red-600"
                      >
                        <Trash2 className="w-4 h-4" />
                      </Button>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* Add Question Dialog */}
      <Dialog open={isAddOpen} onOpenChange={setIsAddOpen}>
        <DialogContent>
          <form onSubmit={handleAddQuestion}>
            <DialogHeader>
              <DialogTitle>Yeni Soru Ekle</DialogTitle>
              <DialogDescription>
                Test setine yeni bir soru-cevap çifti ekleyin.
              </DialogDescription>
            </DialogHeader>
            <div className="space-y-4 py-4">
              <div className="space-y-2">
                <Label>Soru</Label>
                <Textarea
                  value={newQuestion.question}
                  onChange={(e) => setNewQuestion({ ...newQuestion, question: e.target.value })}
                  placeholder="Kullanıcının soracağı soru"
                  rows={3}
                  required
                />
              </div>
              <div className="space-y-2">
                <Label>Beklenen Cevap (Ground Truth)</Label>
                <Textarea
                  value={newQuestion.ground_truth}
                  onChange={(e) => setNewQuestion({ ...newQuestion, ground_truth: e.target.value })}
                  placeholder="Doğru/beklenen cevap"
                  rows={4}
                  required
                />
              </div>
              <div className="space-y-2">
                <Label>Alternatif Cevaplar (Opsiyonel)</Label>
                <div className="space-y-2">
                  {newQuestion.alternative_ground_truths.map((alt, idx) => (
                    <div key={idx} className="flex items-center gap-2 p-2 bg-slate-50 rounded">
                      <span className="flex-1 text-sm">{alt}</span>
                      <Button
                        type="button"
                        variant="ghost"
                        size="sm"
                        onClick={() => {
                          const newAlts = [...newQuestion.alternative_ground_truths];
                          newAlts.splice(idx, 1);
                          setNewQuestion({ ...newQuestion, alternative_ground_truths: newAlts });
                        }}
                        className="text-red-600 hover:text-red-700"
                      >
                        <Trash2 className="w-4 h-4" />
                      </Button>
                    </div>
                  ))}
                  <div className="flex gap-2">
                    <Textarea
                      value={newAltAnswer}
                      onChange={(e) => setNewAltAnswer(e.target.value)}
                      placeholder="Alternatif cevap ekle"
                      rows={2}
                      className="flex-1"
                    />
                    <Button
                      type="button"
                      variant="outline"
                      size="sm"
                      onClick={() => {
                        if (newAltAnswer.trim()) {
                          setNewQuestion({
                            ...newQuestion,
                            alternative_ground_truths: [...newQuestion.alternative_ground_truths, newAltAnswer.trim()]
                          });
                          setNewAltAnswer("");
                        }
                      }}
                      disabled={!newAltAnswer.trim()}
                    >
                      <Plus className="w-4 h-4" />
                    </Button>
                  </div>
                </div>
              </div>
            </div>
            <DialogFooter>
              <Button type="button" variant="outline" onClick={() => setIsAddOpen(false)}>
                İptal
              </Button>
              <Button type="submit" disabled={isSaving}>
                {isSaving ? <><Loader2 className="w-4 h-4 mr-1 animate-spin" /> Ekleniyor...</> : <><Save className="w-4 h-4 mr-1" /> Ekle</>}
              </Button>
            </DialogFooter>
          </form>
        </DialogContent>
      </Dialog>

      {/* Edit Question Dialog */}
      <Dialog open={isEditOpen} onOpenChange={setIsEditOpen}>
        <DialogContent>
          <form onSubmit={handleEditQuestion}>
            <DialogHeader>
              <DialogTitle>Soruyu Düzenle</DialogTitle>
            </DialogHeader>
            <div className="space-y-4 py-4">
              <div className="space-y-2">
                <Label>Soru</Label>
                <Textarea
                  value={editingQuestion?.question || ""}
                  onChange={(e) => setEditingQuestion(prev => prev ? { ...prev, question: e.target.value } : null)}
                  rows={3}
                  required
                />
              </div>
              <div className="space-y-2">
                <Label>Beklenen Cevap (Ground Truth)</Label>
                <Textarea
                  value={editingQuestion?.ground_truth || ""}
                  onChange={(e) => setEditingQuestion(prev => prev ? { ...prev, ground_truth: e.target.value } : null)}
                  rows={4}
                  required
                />
              </div>
              <div className="space-y-2">
                <Label>Alternatif Cevaplar (Opsiyonel)</Label>
                <div className="space-y-2">
                  {(editingQuestion?.alternative_ground_truths || []).map((alt, idx) => (
                    <div key={idx} className="flex items-center gap-2 p-2 bg-slate-50 rounded">
                      <span className="flex-1 text-sm">{alt}</span>
                      <Button
                        type="button"
                        variant="ghost"
                        size="sm"
                        onClick={() => {
                          if (!editingQuestion) return;
                          const newAlts = [...(editingQuestion.alternative_ground_truths || [])];
                          newAlts.splice(idx, 1);
                          setEditingQuestion({ ...editingQuestion, alternative_ground_truths: newAlts });
                        }}
                        className="text-red-600 hover:text-red-700"
                      >
                        <Trash2 className="w-4 h-4" />
                      </Button>
                    </div>
                  ))}
                  <div className="flex gap-2">
                    <Textarea
                      value={editAltAnswer}
                      onChange={(e) => setEditAltAnswer(e.target.value)}
                      placeholder="Alternatif cevap ekle"
                      rows={2}
                      className="flex-1"
                    />
                    <Button
                      type="button"
                      variant="outline"
                      size="sm"
                      onClick={() => {
                        if (editAltAnswer.trim() && editingQuestion) {
                          setEditingQuestion({
                            ...editingQuestion,
                            alternative_ground_truths: [...(editingQuestion.alternative_ground_truths || []), editAltAnswer.trim()]
                          });
                          setEditAltAnswer("");
                        }
                      }}
                      disabled={!editAltAnswer.trim()}
                    >
                      <Plus className="w-4 h-4" />
                    </Button>
                  </div>
                </div>
              </div>
            </div>
            <DialogFooter>
              <Button type="button" variant="outline" onClick={() => setIsEditOpen(false)}>
                İptal
              </Button>
              <Button type="submit" disabled={isSaving}>
                {isSaving ? "Kaydediliyor..." : "Kaydet"}
              </Button>
            </DialogFooter>
          </form>
        </DialogContent>
      </Dialog>

      {/* Import Dialog */}
      <Dialog open={isImportOpen} onOpenChange={setIsImportOpen}>
        <DialogContent className="max-w-2xl max-h-[90vh] flex flex-col">
          <form onSubmit={handleImport} className="flex flex-col flex-1 overflow-hidden">
            <DialogHeader>
              <DialogTitle>Soruları İçe Aktar</DialogTitle>
              <DialogDescription>
                JSON formatında soru-cevap çiftlerini yapıştırın.
              </DialogDescription>
            </DialogHeader>
            <div className="flex-1 overflow-y-auto py-4">
              <div className="mb-3">
                <Label>Kayıtlı Veri Setleri (Opsiyonel)</Label>
                <Select
                  value={selectedDataset}
                  onValueChange={handleLoadDataset}
                  disabled={isDatasetsLoading || testDatasets.length === 0}
                >
                  <SelectTrigger className="mt-2">
                    <SelectValue
                      placeholder={
                        isDatasetsLoading
                          ? "Yükleniyor..."
                          : testDatasets.length === 0
                            ? "Kayıtlı veri seti yok"
                            : "Kayıtlı veri seti seçin..."
                      }
                    />
                  </SelectTrigger>
                  <SelectContent>
                    {testDatasets.map((dataset) => (
                      <SelectItem key={dataset.id} value={dataset.id.toString()}>
                        {dataset.name} ({dataset.total_test_cases} soru)
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
              <Textarea
                value={importJson}
                onChange={(e) => setImportJson(e.target.value)}
                placeholder={`[
  {
    "question": "Soru metni",
    "ground_truth": "Beklenen cevap",
    "alternative_ground_truths": ["Alternatif cevap 1", "Alternatif cevap 2"]
  }
]`}
                rows={15}
                className="font-mono text-sm min-h-[300px]"
                required
              />
            </div>
            <DialogFooter className="flex-shrink-0 pt-4 border-t">
              <Button type="button" variant="outline" onClick={() => setIsImportOpen(false)}>
                İptal
              </Button>
              <Button type="submit" disabled={isSaving}>
                {isSaving ? "İçe Aktarılıyor..." : "İçe Aktar"}
              </Button>
            </DialogFooter>
          </form>
        </DialogContent>
      </Dialog>

      {/* Rename Dialog */}
      <Dialog open={isRenameOpen} onOpenChange={setIsRenameOpen}>
        <DialogContent>
          <form onSubmit={handleRenameTestSet}>
            <DialogHeader>
              <DialogTitle>Test Seti Adını Değiştir</DialogTitle>
              <DialogDescription>
                Test setinin yeni adını girin.
              </DialogDescription>
            </DialogHeader>
            <div className="space-y-4 py-4">
              <div className="space-y-2">
                <Label>Yeni İsim</Label>
                <Input
                  value={newName}
                  onChange={(e) => setNewName(e.target.value)}
                  placeholder="Test seti adı"
                  required
                  autoFocus
                />
              </div>
            </div>
            <DialogFooter>
              <Button type="button" variant="outline" onClick={() => setIsRenameOpen(false)}>
                İptal
              </Button>
              <Button type="submit" disabled={isRenaming}>
                {isRenaming ? <><Loader2 className="w-4 h-4 mr-1 animate-spin" /> Güncelleniyor...</> : "Güncelle"}
              </Button>
            </DialogFooter>
          </form>
        </DialogContent>
      </Dialog>
    </div>
  );
}
