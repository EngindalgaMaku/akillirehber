"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import Link from "next/link";
import { useParams, useRouter } from "next/navigation";
import { toast } from "sonner";
import { ArrowLeft, Loader2, Plus, Play, Trash2 } from "lucide-react";

import { useAuth } from "@/lib/auth-context";
import { api, GiskardQuestion, GiskardTestSet } from "@/lib/api";

import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Textarea } from "@/components/ui/textarea";
import { PageHeader } from "@/components/ui/page-header";

export default function GiskardTestSetPage() {
  const { id } = useParams();
  const router = useRouter();
  const { user } = useAuth();

  const testSetId = useMemo(() => Number(id), [id]);

  const [testSet, setTestSet] = useState<GiskardTestSet | null>(null);
  const [questions, setQuestions] = useState<GiskardQuestion[]>([]);
  const [isLoading, setIsLoading] = useState(true);

  const [isAddOpen, setIsAddOpen] = useState(false);
  const [isAdding, setIsAdding] = useState(false);
  const [newQuestion, setNewQuestion] = useState({
    question: "",
    question_type: "relevant" as "relevant" | "irrelevant",
    expected_answer: "",
    question_metadata: "",
  });

  const [isStartingRun, setIsStartingRun] = useState(false);

  const [isGenerateOpen, setIsGenerateOpen] = useState(false);
  const [isGenerating, setIsGenerating] = useState(false);
  const [ragetConfig, setRagetConfig] = useState({
    num_questions: 30,
    language: "tr" as "tr" | "en",
    agent_description: "",
  });
  const [ragetSamples, setRagetSamples] = useState<Array<Record<string, unknown>>>([]);

  const load = useCallback(async () => {
    if (!Number.isFinite(testSetId)) return;
    setIsLoading(true);
    try {
      const [ts, qs] = await Promise.all([
        api.getGiskardTestSet(testSetId),
        api.getGiskardQuestions(testSetId),
      ]);
      setTestSet(ts);
      setQuestions(qs);
    } catch {
      toast.error("Test set yüklenemedi");
      router.push("/dashboard/giskard");
    } finally {
      setIsLoading(false);
    }
  }, [router, testSetId]);

  useEffect(() => {
    if (!user) return;
    load();
  }, [load, user]);

  const handleAddQuestion = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!testSet) return;

    setIsAdding(true);
    try {
      let question_metadata: Record<string, unknown> | undefined;
      if (newQuestion.question_metadata.trim()) {
        question_metadata = JSON.parse(newQuestion.question_metadata);
      }

      await api.addGiskardQuestion(testSet.id, {
        question: newQuestion.question,
        question_type: newQuestion.question_type,
        expected_answer: newQuestion.expected_answer,
        question_metadata,
      });

      setNewQuestion({
        question: "",
        question_type: "relevant",
        expected_answer: "",
        question_metadata: "",
      });

      setIsAddOpen(false);
      await load();
      toast.success("Soru eklendi");
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Soru eklenemedi");
    } finally {
      setIsAdding(false);
    }
  };

  const handleGenerateQuestions = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!testSet) return;

    setIsGenerating(true);
    try {
      const res = await api.generateGiskardRagetTestset({
        course_id: testSet.course_id,
        num_questions: ragetConfig.num_questions,
        language: ragetConfig.language,
        agent_description: ragetConfig.agent_description || undefined,
      });
      setRagetSamples(res.samples || []);
      toast.success(`Toplam ${res.num_questions} soru üretildi (Giskard RAGET)`);
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Sorular üretilemedi");
    } finally {
      setIsGenerating(false);
    }
  };

  const handleDeleteQuestion = async (questionId: number) => {
    if (!confirm("Bu soruyu silmek istediğinizden emin misiniz?")) return;
    try {
      await api.deleteGiskardQuestion(questionId);
      await load();
      toast.success("Soru silindi");
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Soru silinemedi");
    }
  };

  const handleStartRun = async () => {
    if (!testSet) return;
    if (questions.length === 0) {
      toast.error("Değerlendirme başlatmadan önce en az 1 soru ekleyin");
      return;
    }

    setIsStartingRun(true);
    try {
      const run = await api.startGiskardEvaluation({
        test_set_id: testSet.id,
        course_id: testSet.course_id,
        name: `${testSet.name} Run`,
        total_questions: questions.length,
      });

      toast.success("Değerlendirme başlatıldı");
      router.push(`/dashboard/giskard/runs/${run.id}`);
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Değerlendirme başlatılamadı");
    } finally {
      setIsStartingRun(false);
    }
  };

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

  if (!testSet) return null;

  return (
    <div>
      <PageHeader
        icon={ArrowLeft}
        title={testSet.name}
        description={"Giskard test seti düzenleme"}
        iconColor="text-emerald-600"
        iconBg="bg-emerald-100"
      >
        <Link href="/dashboard/giskard">
          <Button variant="outline" size="sm">
            <ArrowLeft className="w-4 h-4 mr-2" />
            Geri
          </Button>
        </Link>
        <Button onClick={handleStartRun} disabled={isStartingRun} size="sm">
          {isStartingRun ? (
            <>
              <Loader2 className="w-4 h-4 mr-2 animate-spin" />
              Başlatılıyor...
            </>
          ) : (
            <>
              <Play className="w-4 h-4 mr-2" />
              Değerlendirme Başlat
            </>
          )}
        </Button>

        <Dialog open={isGenerateOpen} onOpenChange={setIsGenerateOpen}>
          <DialogTrigger asChild>
            <Button variant="outline" size="sm">
              Giskard (RAGET) ile üret
            </Button>
          </DialogTrigger>
          <DialogContent className="max-w-3xl">
            <form onSubmit={handleGenerateQuestions}>
              <DialogHeader>
                <DialogTitle>Giskard (RAGET) ile test set üret</DialogTitle>
                <DialogDescription>
                  Giskard RAG Evaluation Toolkit ile native testset üretir.
                </DialogDescription>
              </DialogHeader>

              <div className="space-y-4 py-2">
                <div className="grid grid-cols-2 gap-4">
                  <div className="space-y-2">
                    <Label>Soru sayısı</Label>
                    <Input
                      type="number"
                      min={1}
                      max={300}
                      value={ragetConfig.num_questions}
                      onChange={(e) =>
                        setRagetConfig({
                          ...ragetConfig,
                          num_questions: Number(e.target.value),
                        })
                      }
                    />
                  </div>
                  <div className="space-y-2">
                    <Label>Dil</Label>
                    <Select
                      value={ragetConfig.language}
                      onValueChange={(v) =>
                        setRagetConfig({
                          ...ragetConfig,
                          language: v as "tr" | "en",
                        })
                      }
                    >
                      <SelectTrigger>
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="tr">Türkçe</SelectItem>
                        <SelectItem value="en">English</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                </div>

                <div className="space-y-2">
                  <Label>Agent açıklaması (opsiyonel)</Label>
                  <Input
                    value={ragetConfig.agent_description}
                    onChange={(e) =>
                      setRagetConfig({
                        ...ragetConfig,
                        agent_description: e.target.value,
                      })
                    }
                    placeholder="Örn: BTT dersi için bir eğitim asistanı"
                  />
                </div>

                {ragetSamples.length > 0 ? (
                  <div className="space-y-2">
                    <Label>Üretilen örnekler</Label>
                    <div className="max-h-80 overflow-auto rounded-md border p-2 text-sm">
                      {ragetSamples.map((s, idx) => (
                        <div key={idx} className="border-b py-2 last:border-b-0">
                          <div className="font-medium">
                            {String(s.question ?? "")}
                          </div>
                          <div className="mt-1 text-muted-foreground">
                            {String(s.reference_answer ?? "")}
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                ) : null}
              </div>

              <DialogFooter>
                <Button type="button" variant="outline" onClick={() => setIsGenerateOpen(false)}>
                  Kapat
                </Button>
                <Button type="submit" disabled={isGenerating}>
                  {isGenerating ? "Üretiliyor..." : "Üret"}
                </Button>
              </DialogFooter>
            </form>
          </DialogContent>
        </Dialog>
      </PageHeader>

      <Card className="p-6">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-lg font-semibold text-slate-900">Sorular</h2>
            <p className="text-sm text-slate-500">{questions.length} soru</p>
          </div>

          <Dialog open={isAddOpen} onOpenChange={setIsAddOpen}>
            <DialogTrigger asChild>
              <Button size="sm">
                <Plus className="w-4 h-4 mr-2" />
                Soru Ekle
              </Button>
            </DialogTrigger>
            <DialogContent>
              <form onSubmit={handleAddQuestion}>
                <DialogHeader>
                  <DialogTitle>Yeni Soru</DialogTitle>
                  <DialogDescription>Test setine yeni bir soru ekleyin.</DialogDescription>
                </DialogHeader>

                <div className="space-y-4 py-4">
                  <div className="space-y-2">
                    <Label>Soru</Label>
                    <Textarea
                      value={newQuestion.question}
                      onChange={(e) => setNewQuestion({ ...newQuestion, question: e.target.value })}
                      required
                      rows={3}
                    />
                  </div>

                  <div className="space-y-2">
                    <Label>Soru Tipi</Label>
                    <Select
                      value={newQuestion.question_type}
                      onValueChange={(v) => setNewQuestion({
                        ...newQuestion,
                        question_type: v as "relevant" | "irrelevant",
                      })}
                    >
                      <SelectTrigger>
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="relevant">İlgili</SelectItem>
                        <SelectItem value="irrelevant">İlgisiz</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>

                  <div className="space-y-2">
                    <Label>Beklenen Cevap</Label>
                    <Textarea
                      value={newQuestion.expected_answer}
                      onChange={(e) => setNewQuestion({ ...newQuestion, expected_answer: e.target.value })}
                      required
                      rows={3}
                    />
                  </div>

                  <div className="space-y-2">
                    <Label>Metadata (opsiyonel JSON)</Label>
                    <Input
                      value={newQuestion.question_metadata}
                      onChange={(e) => setNewQuestion({ ...newQuestion, question_metadata: e.target.value })}
                      placeholder='{"topic":"..."}'
                    />
                  </div>
                </div>

                <DialogFooter>
                  <Button type="button" variant="outline" onClick={() => setIsAddOpen(false)}>
                    İptal
                  </Button>
                  <Button type="submit" disabled={isAdding}>
                    {isAdding ? "Ekleniyor..." : "Ekle"}
                  </Button>
                </DialogFooter>
              </form>
            </DialogContent>
          </Dialog>
        </div>

        <div className="mt-6 space-y-3">
          {questions.length === 0 ? (
            <div className="text-center py-10 text-slate-400">
              <p className="text-sm">Henüz soru yok</p>
            </div>
          ) : (
            questions.map((q) => (
              <div
                key={q.id}
                className="p-4 rounded-lg border border-slate-200 flex items-start justify-between gap-4"
              >
                <div className="min-w-0">
                  <p className="text-sm font-medium text-slate-900 break-words">{q.question}</p>
                  <p className="text-xs text-slate-500 mt-1">Tip: {q.question_type}</p>
                </div>

                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => handleDeleteQuestion(q.id)}
                  className="text-slate-400 hover:text-red-600 hover:bg-red-50"
                >
                  <Trash2 className="w-4 h-4" />
                </Button>
              </div>
            ))
          )}
        </div>
      </Card>
    </div>
  );
}
