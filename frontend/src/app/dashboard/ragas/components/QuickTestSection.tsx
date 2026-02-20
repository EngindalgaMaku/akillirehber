"use client";

import { useState, useEffect } from "react";
import { QuickTestResponse, RagasGroupInfo, api } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { Card } from "@/components/ui/card";
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog";
import { Sparkles, ChevronDown, Plus, X, Play, Loader2, Save, Bookmark } from "lucide-react";
import { toast } from "sonner";

interface QuickTestSectionProps {
  selectedCourseId: number;
  quickTestResult: QuickTestResponse | null;
  setQuickTestResult: (result: QuickTestResponse | null) => void;
  onResultSaved: () => void;
  savedResultsGroups: RagasGroupInfo[];
  ragasEmbeddingModel?: string;
}

export function QuickTestSection({ 
  selectedCourseId, 
  quickTestResult, 
  setQuickTestResult,
  onResultSaved,
  savedResultsGroups,
  ragasEmbeddingModel
}: QuickTestSectionProps) {
  const [isExpanded, setIsExpanded] = useState(false);
  const [question, setQuestion] = useState("");
  const [groundTruth, setGroundTruth] = useState("");
  const [alternatives, setAlternatives] = useState<string[]>([]);
  const [systemPrompt, setSystemPrompt] = useState("");
  const [llmModel, setLlmModel] = useState("");
  const [isTesting, setIsTesting] = useState(false);
  
  // Save Dialog
  const [isSaveDialogOpen, setIsSaveDialogOpen] = useState(false);
  const [saveGroupName, setSaveGroupName] = useState("");
  const [isSaving, setIsSaving] = useState(false);

  useEffect(() => {
    if (selectedCourseId) {
      loadSystemPrompt();
    }
  }, [selectedCourseId]);

  const loadSystemPrompt = async () => {
    try {
      const settings = await api.getCourseSettings(selectedCourseId);
      setSystemPrompt(settings.system_prompt || "");
    } catch {
      /* ignore */
    }
  };

  const handleTest = async () => {
    if (!question || !groundTruth) {
      toast.error("Lütfen soru ve doğru cevap alanlarını doldurun");
      return;
    }

    setIsTesting(true);
    setQuickTestResult(null);

    try {
      const result = await api.quickTest({
        course_id: selectedCourseId,
        question,
        ground_truth: groundTruth,
        alternative_ground_truths: alternatives.filter(a => a.trim() !== ""),
        system_prompt: systemPrompt || undefined,
        llm_provider: llmModel ? "openrouter" : undefined,
        llm_model: llmModel || undefined,
        ragas_embedding_model: ragasEmbeddingModel || undefined
      });
      setQuickTestResult(result);
      toast.success("Test tamamlandı");
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "Test başarısız");
    } finally {
      setIsTesting(false);
    }
  };

  const handleSaveResult = async () => {
    if (!quickTestResult) return;
    setIsSaving(true);
    try {
      await api.saveQuickTestResult({
        course_id: selectedCourseId,
        group_name: saveGroupName || undefined,
        question: quickTestResult.question,
        ground_truth: quickTestResult.ground_truth,
        alternative_ground_truths: alternatives.filter(a => a.trim() !== ""),
        system_prompt: quickTestResult.system_prompt_used,
        llm_provider: quickTestResult.llm_provider_used,
        llm_model: quickTestResult.llm_model_used,
        evaluation_model: quickTestResult.evaluation_model_used,
        embedding_model: quickTestResult.embedding_model_used,
        search_top_k: quickTestResult.search_top_k_used,
        search_alpha: quickTestResult.search_alpha_used,
        reranker_used: quickTestResult.reranker_used,
        reranker_provider: quickTestResult.reranker_provider,
        reranker_model: quickTestResult.reranker_model,
        generated_answer: quickTestResult.generated_answer,
        retrieved_contexts: quickTestResult.retrieved_contexts,
        faithfulness: quickTestResult.faithfulness,
        answer_relevancy: quickTestResult.answer_relevancy,
        context_precision: quickTestResult.context_precision,
        context_recall: quickTestResult.context_recall,
        answer_correctness: quickTestResult.answer_correctness,
        latency_ms: quickTestResult.latency_ms
      });
      toast.success("Sonuç kaydedildi");
      setIsSaveDialogOpen(false);
      setSaveGroupName("");
      onResultSaved();
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "Kaydetme başarısız");
    } finally {
      setIsSaving(false);
    }
  };

  const handleSaveSystemPrompt = async () => {
    if (!systemPrompt) {
      toast.error("Sistem promptu boş olamaz");
      return;
    }
    try {
      await api.updateCourseSettings(selectedCourseId, { system_prompt: systemPrompt });
      toast.success("Sistem promptu kaydedildi");
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "Kaydetme başarısız");
    }
  };

  const addAlternative = () => setAlternatives([...alternatives, ""]);
  const removeAlternative = (index: number) => setAlternatives(alternatives.filter((_, i) => i !== index));
  const updateAlternative = (index: number, value: string) => {
    const newAlternatives = [...alternatives];
    newAlternatives[index] = value;
    setAlternatives(newAlternatives);
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
    <Card className="overflow-hidden border-0 shadow-lg bg-gradient-to-br from-purple-50 via-white to-indigo-50">
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className="w-full px-6 py-5 flex items-center justify-between hover:bg-purple-50/50 transition-all duration-200"
      >
        <div className="flex items-center gap-4">
          <div className="p-3 bg-gradient-to-br from-purple-500 to-indigo-600 rounded-xl shadow-lg shadow-purple-200">
            <Sparkles className="w-6 h-6 text-white" />
          </div>
          <div className="text-left">
            <h2 className="text-xl font-bold text-slate-900">Hızlı Test</h2>
            <p className="text-sm text-slate-600">Tek bir soru için anında RAGAS değerlendirmesi</p>
          </div>
        </div>
        <div className={`p-2 rounded-full bg-purple-100 transition-transform duration-200 ${isExpanded ? 'rotate-180' : ''}`}>
          <ChevronDown className="w-5 h-5 text-purple-600" />
        </div>
      </button>

      {isExpanded && (
        <div className="px-6 pb-6 pt-2 border-t border-purple-100">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Input Section */}
            <div className="space-y-4">
              <div>
                <Label className="text-sm font-medium text-slate-700">Soru</Label>
                <Textarea
                  value={question}
                  onChange={(e) => setQuestion(e.target.value)}
                  placeholder="Test etmek istediğiniz soruyu girin..."
                  rows={3}
                  className="mt-1.5 border-slate-200 focus:border-purple-400 focus:ring-purple-400"
                />
              </div>

              <div>
                <Label className="text-sm font-medium text-slate-700">Doğru Cevap (Ground Truth)</Label>
                <Textarea
                  value={groundTruth}
                  onChange={(e) => setGroundTruth(e.target.value)}
                  placeholder="Beklenen doğru cevabı girin..."
                  rows={3}
                  className="mt-1.5 border-slate-200 focus:border-purple-400 focus:ring-purple-400"
                />
              </div>

              <div>
                <div className="flex items-center justify-between mb-2">
                  <Label className="text-sm font-medium text-slate-700">Alternatif Doğru Cevaplar</Label>
                  <Button
                    type="button"
                    variant="outline"
                    size="sm"
                    onClick={addAlternative}
                    className="h-7 text-xs border-purple-200 text-purple-600 hover:bg-purple-50"
                  >
                    <Plus className="w-3 h-3 mr-1" /> Ekle
                  </Button>
                </div>
                {alternatives.map((alt, index) => (
                  <div key={index} className="flex gap-2 mb-2">
                    <Input
                      value={alt}
                      onChange={(e) => updateAlternative(index, e.target.value)}
                      placeholder={`Alternatif ${index + 1}`}
                      className="flex-1 border-slate-200"
                    />
                    <Button
                      type="button"
                      variant="ghost"
                      size="sm"
                      onClick={() => removeAlternative(index)}
                      className="text-red-500 hover:text-red-700 hover:bg-red-50"
                    >
                      <X className="w-4 h-4" />
                    </Button>
                  </div>
                ))}
              </div>

              <div>
                <div className="flex items-center justify-between mb-2">
                  <Label className="text-sm font-medium text-slate-700">Sistem Promptu</Label>
                  <Button
                    type="button"
                    variant="outline"
                    size="sm"
                    onClick={handleSaveSystemPrompt}
                    disabled={!systemPrompt}
                    className="h-7 text-xs border-purple-200 text-purple-600 hover:bg-purple-50"
                  >
                    <Save className="w-3 h-3 mr-1" /> Kaydet
                  </Button>
                </div>
                <Textarea
                  value={systemPrompt}
                  onChange={(e) => setSystemPrompt(e.target.value)}
                  placeholder="Sistem promptu (boş bırakılırsa ders ayarlarından alınır)"
                  rows={3}
                  className="mt-1 border-slate-200 focus:border-purple-400 focus:ring-purple-400"
                />
              </div>

              <div>
                <Label className="text-sm font-medium text-slate-700">OpenRouter Model (Opsiyonel)</Label>
                <Input
                  value={llmModel}
                  onChange={(e) => setLlmModel(e.target.value)}
                  placeholder="örn: openai/gpt-4o-mini"
                  className="mt-1.5 border-slate-200"
                />
              </div>

              <Button
                onClick={handleTest}
                disabled={isTesting || !question || !groundTruth}
                className="w-full bg-gradient-to-r from-purple-600 to-indigo-600 hover:from-purple-700 hover:to-indigo-700 shadow-lg shadow-purple-200"
              >
                {isTesting ? (
                  <>
                    <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                    Test Ediliyor...
                  </>
                ) : (
                  <>
                    <Play className="w-4 h-4 mr-2" />
                    Test Et
                  </>
                )}
              </Button>
            </div>

            {/* Results Section */}
            <div className="space-y-4">
              {quickTestResult ? (
                <>
                  <div>
                    <Label className="text-sm font-medium text-slate-700">Üretilen Cevap</Label>
                    <div className="mt-1.5 p-4 bg-white rounded-xl border border-slate-200 text-sm shadow-sm">
                      {quickTestResult.generated_answer}
                    </div>
                  </div>

                  <div>
                    <Label className="text-sm font-medium text-slate-700">Metrikler</Label>
                    <div className="grid grid-cols-2 gap-3 mt-2">
                      <div className={`p-4 rounded-xl border ${getMetricBgColor(quickTestResult.faithfulness)}`}>
                        <p className="text-xs text-slate-600 font-medium">Faithfulness</p>
                        <p className={`text-2xl font-bold ${getMetricColor(quickTestResult.faithfulness)}`}>
                          {quickTestResult.faithfulness != null ? `${(quickTestResult.faithfulness * 100).toFixed(1)}%` : "N/A"}
                        </p>
                      </div>
                      <div className={`p-4 rounded-xl border ${getMetricBgColor(quickTestResult.answer_relevancy)}`}>
                        <p className="text-xs text-slate-600 font-medium">Answer Relevancy</p>
                        <p className={`text-2xl font-bold ${getMetricColor(quickTestResult.answer_relevancy)}`}>
                          {quickTestResult.answer_relevancy != null ? `${(quickTestResult.answer_relevancy * 100).toFixed(1)}%` : "N/A"}
                        </p>
                      </div>
                      <div className={`p-4 rounded-xl border ${getMetricBgColor(quickTestResult.context_precision)}`}>
                        <p className="text-xs text-slate-600 font-medium">Context Precision</p>
                        <p className={`text-2xl font-bold ${getMetricColor(quickTestResult.context_precision)}`}>
                          {quickTestResult.context_precision != null ? `${(quickTestResult.context_precision * 100).toFixed(1)}%` : "N/A"}
                        </p>
                      </div>
                      <div className={`p-4 rounded-xl border ${getMetricBgColor(quickTestResult.context_recall)}`}>
                        <p className="text-xs text-slate-600 font-medium">Context Recall</p>
                        <p className={`text-2xl font-bold ${getMetricColor(quickTestResult.context_recall)}`}>
                          {quickTestResult.context_recall != null ? `${(quickTestResult.context_recall * 100).toFixed(1)}%` : "N/A"}
                        </p>
                      </div>
                      <div className={`p-4 rounded-xl border col-span-2 ${getMetricBgColor(quickTestResult.answer_correctness)}`}>
                        <p className="text-xs text-slate-600 font-medium">Answer Correctness</p>
                        <p className={`text-2xl font-bold ${getMetricColor(quickTestResult.answer_correctness)}`}>
                          {quickTestResult.answer_correctness != null ? `${(quickTestResult.answer_correctness * 100).toFixed(1)}%` : "N/A"}
                        </p>
                      </div>
                    </div>
                  </div>

                  <div className="grid grid-cols-2 gap-3">
                    <div className="p-4 bg-white rounded-xl border border-slate-200 shadow-sm">
                      <p className="text-xs text-slate-500 font-medium">Gecikme</p>
                      <p className="text-lg font-bold text-slate-900">{quickTestResult.latency_ms}ms</p>
                    </div>
                    <div className="p-4 bg-white rounded-xl border border-slate-200 shadow-sm">
                      <p className="text-xs text-slate-500 font-medium">Model</p>
                      <p className="text-sm font-medium text-slate-900 truncate">{quickTestResult.llm_model_used}</p>
                    </div>
                  </div>

                  <Dialog open={isSaveDialogOpen} onOpenChange={setIsSaveDialogOpen}>
                    <DialogTrigger asChild>
                      <Button variant="outline" className="w-full border-purple-200 text-purple-700 hover:bg-purple-50">
                        <Bookmark className="w-4 h-4 mr-2" />
                        Sonucu Kaydet
                      </Button>
                    </DialogTrigger>
                    <DialogContent>
                      <DialogHeader>
                        <DialogTitle>Sonucu Kaydet</DialogTitle>
                        <DialogDescription>Bu test sonucunu daha sonra görüntülemek için kaydedin.</DialogDescription>
                      </DialogHeader>
                      <div className="space-y-4 py-4">
                        <div className="space-y-2">
                          <Label>Grup Adı (Opsiyonel)</Label>
                          <Input
                            value={saveGroupName}
                            onChange={(e) => setSaveGroupName(e.target.value)}
                            placeholder="örn: Deneme 1"
                          />
                        </div>
                        {savedResultsGroups.length > 0 && (
                          <div className="flex flex-wrap gap-1">
                            {savedResultsGroups.map((g) => (
                              <button
                                key={g.name}
                                type="button"
                                onClick={() => setSaveGroupName(g.name)}
                                className="px-2 py-1 text-xs bg-slate-100 hover:bg-slate-200 rounded"
                              >
                                {g.name}
                              </button>
                            ))}
                          </div>
                        )}
                      </div>
                      <DialogFooter>
                        <Button variant="outline" onClick={() => setIsSaveDialogOpen(false)}>İptal</Button>
                        <Button onClick={handleSaveResult} disabled={isSaving}>
                          {isSaving ? "Kaydediliyor..." : "Kaydet"}
                        </Button>
                      </DialogFooter>
                    </DialogContent>
                  </Dialog>
                </>
              ) : (
                <div className="flex items-center justify-center h-full min-h-[400px] text-slate-400">
                  <div className="text-center">
                    <div className="w-20 h-20 bg-purple-100 rounded-full flex items-center justify-center mx-auto mb-4">
                      <Sparkles className="w-10 h-10 text-purple-400" />
                    </div>
                    <p className="text-sm font-medium">Test sonuçları burada görünecek</p>
                    <p className="text-xs text-slate-400 mt-1">Soru ve doğru cevabı girerek test başlatın</p>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </Card>
  );
}