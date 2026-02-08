"use client";

import { useEffect, useState, useCallback } from "react";
import { useParams, useRouter } from "next/navigation";
import { useAuth } from "@/lib/auth-context";
import { api, TestSetDetail, TestQuestion, TestSet } from "@/lib/api";
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
  const [selectedQuestions, setSelectedQuestions] = useState<Set<number>>(new Set());
  const [isDuplicating, setIsDuplicating] = useState(false);
  const [isRenameOpen, setIsRenameOpen] = useState(false);
  const [newName, setNewName] = useState("");
  const [isRenaming, setIsRenaming] = useState(false);

  const [isMergeOpen, setIsMergeOpen] = useState(false);
  const [sourceTestSetId, setSourceTestSetId] = useState<string>("");
  const [isMerging, setIsMerging] = useState(false);
  const [availableTestSets, setAvailableTestSets] = useState<TestSet[]>([]);

  const [testDatasets, setTestDatasets] = useState<Array<Record<string, unknown>>>([]);
  const [selectedDataset, setSelectedDataset] = useState<string>("");
  const [isDatasetsLoading, setIsDatasetsLoading] = useState(false);

  // Filtering and pagination
  const [bloomFilter, setBloomFilter] = useState<string>("all");
  const [searchQuery, setSearchQuery] = useState<string>("");
  const [currentPage, setCurrentPage] = useState<number>(1);
  const [questionsPerPage] = useState<number>(10);
  const [isDeletingSelected, setIsDeletingSelected] = useState(false);

  // Quick test state
  const [isQuickTesting, setIsQuickTesting] = useState(false);
  const [testResults, setTestResults] = useState<Map<number, any>>(() => {
    // Load from localStorage
    if (typeof window !== 'undefined') {
      const saved = localStorage.getItem(`test_results_${id}`);
      if (saved) {
        try {
          const parsed = JSON.parse(saved);
          // Convert string keys back to numbers
          const entries: [number, any][] = Object.entries(parsed).map(([key, value]) => [Number(key), value]);
          return new Map(entries);
        } catch (e) {
          console.error('Failed to parse saved test results:', e);
        }
      }
    }
    return new Map();
  });
  const [selectedResultId, setSelectedResultId] = useState<number | null>(null);
  const [isResultModalOpen, setIsResultModalOpen] = useState(false);

  // Duplicate detection state
  const [isDuplicateModalOpen, setIsDuplicateModalOpen] = useState(false);
  const [isFindingDuplicates, setIsFindingDuplicates] = useState(false);
  const [duplicateGroups, setDuplicateGroups] = useState<any[]>([]);
  const [similarityThreshold, setSimilarityThreshold] = useState(0.85);
  const [selectedDuplicates, setSelectedDuplicates] = useState<Set<number>>(new Set());

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

  const loadTestDatasets = useCallback(async () => {
    if (!testSet) return;
    setIsDatasetsLoading(true);
    try {
      const testSets = await api.getTestSets(testSet.course_id);
      const datasets = testSets.map(ts => ({
        id: ts.id,
        name: ts.name,
        total_test_cases: ts.question_count,
      }));
      setTestDatasets(datasets);
    } catch {
      console.error("Test veri setleri yüklenirken hata oluştu");
    } finally {
      setIsDatasetsLoading(false);
    }
  }, [testSet]);

  useEffect(() => {
    loadTestSet();
  }, [loadTestSet]);

  // Save test results to localStorage when they change
  useEffect(() => {
    if (testResults.size > 0) {
      const resultsObj = Object.fromEntries(testResults);
      localStorage.setItem(`test_results_${id}`, JSON.stringify(resultsObj));
    }
  }, [testResults, id]);

  useEffect(() => {
    if (testSet) {
      loadTestDatasets();
    }
  }, [testSet, loadTestDatasets]);

  const handleLoadDataset = async (datasetId: string) => {
    if (!datasetId) return;
    try {
      const testSetData = await api.getTestSet(parseInt(datasetId));
      const testCases = testSetData.questions
        .sort((a, b) => a.question.localeCompare(b.question, 'tr-TR'))
        .map(q => ({
          question: q.question,
          ground_truth: q.ground_truth,
          alternative_ground_truths: q.alternative_ground_truths || [],
          expected_contexts: q.expected_contexts || [],
        }));
      setImportJson(JSON.stringify(testCases, null, 2));
      setSelectedDataset(datasetId);
      toast.success(`"${testSetData.name}" test seti yüklendi (${testCases.length} soru)`);
    } catch (error) {
      console.error("Load dataset error:", error);
      toast.error(error instanceof Error ? error.message : "Test seti yüklenemedi");
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

  const handleDeleteSelected = async () => {
    if (selectedQuestions.size === 0) return;
    if (!confirm(`${selectedQuestions.size} soruyu silmek istediğinizden emin misiniz?`)) return;
    
    setIsDeletingSelected(true);
    try {
      const deletePromises = Array.from(selectedQuestions).map(id => api.deleteQuestion(id));
      await Promise.all(deletePromises);
      setSelectedQuestions(new Set());
      loadTestSet();
      toast.success(`${deletePromises.length} soru silindi`);
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "Silme işlemi başarısız");
    } finally {
      setIsDeletingSelected(false);
    }
  };

  const handleQuickTest = async () => {
    if (!testSet || selectedQuestions.size === 0) return;
    
    const questionsToTest = testSet.questions.filter(q => selectedQuestions.has(q.id));
    if (questionsToTest.length === 0) return;

    setIsQuickTesting(true);
    toast.info(`${questionsToTest.length} soru test ediliyor...`);

    try {
      const results = new Map(testResults);
      
      for (const question of questionsToTest) {
        try {
          // Semantic similarity quick test
          const result = await api.semanticSimilarityQuickTest({
            course_id: testSet.course_id,
            question: question.question,
            ground_truth: question.ground_truth,
            alternative_ground_truths: question.alternative_ground_truths || [],
          });
          
          results.set(question.id, {
            ...result,
            tested_at: new Date().toISOString(),
          });
        } catch (error) {
          console.error(`Test failed for question ${question.id}:`, error);
          results.set(question.id, {
            error: error instanceof Error ? error.message : "Test başarısız",
            tested_at: new Date().toISOString(),
          });
        }
      }
      
      setTestResults(results);
      toast.success(`${questionsToTest.length} soru test edildi`);
    } catch (error) {
      toast.error("Test işlemi başarısız");
    } finally {
      setIsQuickTesting(false);
    }
  };

  const getQuestionTestResult = (questionId: number) => {
    return testResults.get(questionId);
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
      console.log(`Exporting test set ID: ${testSet.id}, Name: ${testSet.name}`);
      const data = await api.exportTestSet(testSet.id);
      console.log(`Export data received:`, data);
      const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `${testSet.name.replaceAll(/\s+/g, "_")}_export.json`;
      a.click();
      URL.revokeObjectURL(url);
      toast.success(`Test seti dışa aktarıldı: ${testSet.name} (ID: ${testSet.id})`);
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "Hata oluştu");
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

  const handleMergeTestSets = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!testSet || !sourceTestSetId) return;
    
    const sourceSet = availableTestSets.find(ts => ts.id === Number(sourceTestSetId));
    if (!confirm(`"${sourceSet?.name}" test setindeki tüm soruları bu test setine eklemek istediğinizden emin misiniz?`)) return;
    
    setIsMerging(true);
    try {
      await api.mergeTestSets(testSet.id, Number(sourceTestSetId));
      toast.success(`${sourceSet?.name} test seti birleştirildi`);
      setIsMergeOpen(false);
      setSourceTestSetId("");
      loadTestSet();
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "Birleştirme başarısız");
    } finally {
      setIsMerging(false);
    }
  };

  const loadAvailableTestSets = useCallback(async () => {
    if (!testSet) return;
    try {
      const sets = await api.getTestSets(testSet.course_id);
      // Mevcut test setini listeden çıkar
      setAvailableTestSets(sets.filter(ts => ts.id !== testSet.id));
    } catch (error) {
      console.error("Test setleri yüklenemedi:", error);
    }
  }, [testSet]);

  const handleFindDuplicates = async () => {
    if (!testSet) return;
    
    setIsFindingDuplicates(true);
    setDuplicateGroups([]);
    setSelectedDuplicates(new Set());
    
    try {
      const result = await api.findDuplicateQuestions(testSet.id, similarityThreshold);
      setDuplicateGroups(result.duplicate_groups);
      
      if (result.duplicate_groups.length === 0) {
        toast.info("Benzer soru bulunamadı");
      } else {
        toast.success(`${result.duplicate_groups.length} benzer soru grubu bulundu`);
      }
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "Benzer sorular bulunamadı");
    } finally {
      setIsFindingDuplicates(false);
    }
  };

  const handleDeleteDuplicates = async () => {
    if (!testSet || selectedDuplicates.size === 0) return;
    
    if (!confirm(`${selectedDuplicates.size} soruyu silmek istediğinizden emin misiniz?`)) return;
    
    try {
      await api.deleteMultipleQuestions(testSet.id, Array.from(selectedDuplicates));
      toast.success(`${selectedDuplicates.size} soru silindi`);
      setIsDuplicateModalOpen(false);
      setDuplicateGroups([]);
      setSelectedDuplicates(new Set());
      loadTestSet();
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "Silme işlemi başarısız");
    }
  };

  useEffect(() => {
    if (isMergeOpen && testSet) {
      loadAvailableTestSets();
    }
  }, [isMergeOpen, testSet, loadAvailableTestSets]);

  if (!user) return null;

  if (isLoading) {
    return (
      <div className="flex justify-center py-12">
        <Loader2 className="w-6 h-6 text-slate-400 animate-spin" />
      </div>
    );
  }

  if (!testSet) return null;

  // Filter questions by Bloom level and search query
  const filteredQuestions = testSet.questions.filter(q => {
    // Bloom filter
    if (bloomFilter !== "all" && q.question_metadata?.bloom_level !== bloomFilter) {
      return false;
    }
    
    // Search filter
    if (searchQuery.trim()) {
      const query = searchQuery.toLowerCase();
      const questionMatch = q.question.toLowerCase().includes(query);
      const answerMatch = q.ground_truth.toLowerCase().includes(query);
      const topicMatch = q.question_metadata?.topic?.toLowerCase().includes(query);
      return questionMatch || answerMatch || topicMatch;
    }
    
    return true;
  }).sort((a, b) => {
    // Alfabetik sıralama (Türkçe karakterlere duyarlı)
    return a.question.localeCompare(b.question, 'tr-TR');
  });

  // Pagination
  const totalPages = Math.ceil(filteredQuestions.length / questionsPerPage);
  const startIndex = (currentPage - 1) * questionsPerPage;
  const endIndex = startIndex + questionsPerPage;
  const paginatedQuestions = filteredQuestions.slice(startIndex, endIndex);

  // Reset to page 1 when filter or search changes
  const handleBloomFilterChange = (value: string) => {
    setBloomFilter(value);
    setCurrentPage(1);
  };

  const handleSearchChange = (value: string) => {
    setSearchQuery(value);
    setCurrentPage(1);
  };

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
          <Button variant="outline" size="sm" onClick={() => setIsMergeOpen(true)}>
            🔀 Birleştir
          </Button>
          <Button variant="outline" size="sm" onClick={() => setIsDuplicateModalOpen(true)}>
            🔍 Benzer Sorular
          </Button>
          <Link href="/dashboard/ragas/test-sets/generate">
            <Button variant="outline" size="sm">
              ✨ Soru Üret
            </Button>
          </Link>
          <Button variant="outline" size="sm" onClick={() => setIsImportOpen(true)}>
            <Upload className="w-4 h-4 mr-1" /> İçe Aktar
          </Button>
          <Button variant="outline" size="sm" onClick={handleExport}>
            <Download className="w-4 h-4 mr-1" /> Dışa Aktar
          </Button>
          <Button size="sm" onClick={() => setIsAddOpen(true)}>
            <Plus className="w-4 h-4 mr-1" /> Soru Ekle
          </Button>
        </div>
      </PageHeader>

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
            {/* Filters and Actions */}
            <div className="p-4 bg-slate-50 border-b border-slate-200 space-y-3">
              {/* Search Bar */}
              <div className="flex items-center gap-3">
                <div className="flex-1 relative">
                  <Input
                    type="text"
                    placeholder="Soru, cevap veya konuda ara..."
                    value={searchQuery}
                    onChange={(e) => handleSearchChange(e.target.value)}
                    className="pl-8"
                  />
                  <svg
                    className="absolute left-2.5 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-400"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
                  </svg>
                  {searchQuery && (
                    <button
                      onClick={() => handleSearchChange("")}
                      className="absolute right-2.5 top-1/2 -translate-y-1/2 text-slate-400 hover:text-slate-600"
                    >
                      ✕
                    </button>
                  )}
                </div>
                
                <select
                  value={bloomFilter}
                  onChange={(e) => handleBloomFilterChange(e.target.value)}
                  className="text-sm px-3 py-2 border border-slate-300 rounded-md"
                >
                  <option value="all">Tüm Seviyeler ({testSet.questions.length})</option>
                  <option value="remembering">🧠 Hatırlama ({testSet.questions.filter(q => q.question_metadata?.bloom_level === 'remembering').length})</option>
                  <option value="understanding_applying">🔧 Anlama/Uygulama ({testSet.questions.filter(q => q.question_metadata?.bloom_level === 'understanding_applying').length})</option>
                  <option value="analyzing_evaluating">⭐ Analiz/Değerlendirme ({testSet.questions.filter(q => q.question_metadata?.bloom_level === 'analyzing_evaluating').length})</option>
                </select>
              </div>

              {/* Selection and Actions */}
              <div className="flex items-center justify-between gap-3">
                <div className="flex items-center gap-3">
                  <input
                    type="checkbox"
                    checked={selectedQuestions.size === filteredQuestions.length && filteredQuestions.length > 0}
                    onChange={toggleSelectAll}
                    className="w-4 h-4 text-indigo-600 rounded border-slate-300 focus:ring-indigo-500"
                  />
                  <span className="text-sm font-medium text-slate-700">
                    {selectedQuestions.size > 0 
                      ? `${selectedQuestions.size} soru seçildi` 
                      : 'Tümünü seç'}
                  </span>
                  {selectedQuestions.size > 0 && (
                    <>
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={() => setSelectedQuestions(new Set())}
                        className="text-xs text-slate-500 hover:text-slate-700"
                      >
                        Seçimi Temizle
                      </Button>
                      <Button
                        variant="destructive"
                        size="sm"
                        onClick={handleDeleteSelected}
                        disabled={isDeletingSelected}
                        className="text-xs"
                      >
                        {isDeletingSelected ? (
                          <><Loader2 className="w-3 h-3 mr-1 animate-spin" /> Siliniyor...</>
                        ) : (
                          <><Trash2 className="w-3 h-3 mr-1" /> Seçilenleri Sil</>
                        )}
                      </Button>
                      <Button
                        variant="default"
                        size="sm"
                        onClick={handleQuickTest}
                        disabled={isQuickTesting}
                        className="text-xs bg-indigo-600 hover:bg-indigo-700"
                      >
                        {isQuickTesting ? (
                          <><Loader2 className="w-3 h-3 mr-1 animate-spin" /> Test Ediliyor...</>
                        ) : (
                          <>🧪 Hızlı Test</>
                        )}
                      </Button>
                    </>
                  )}
                </div>
                
                {/* Results info */}
                <div className="text-xs text-slate-500">
                  {filteredQuestions.length} soru bulundu
                  {searchQuery && ` (arama: "${searchQuery}")`}
                </div>
              </div>
              
              {/* Pagination info */}
              {filteredQuestions.length > questionsPerPage && (
                <div className="text-xs text-slate-500 text-center">
                  Sayfa {currentPage} / {totalPages}
                </div>
              )}
            </div>
            
            <div className="divide-y divide-slate-100">
              {paginatedQuestions.map((q, index) => (
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
                        <span className="text-xs font-medium text-slate-400">#{startIndex + index + 1}</span>
                        {q.question_metadata?.bloom_level && (
                          <span className={`text-xs px-2 py-0.5 rounded-full font-medium ${
                            q.question_metadata.bloom_level === 'remembering' ? 'bg-blue-100 text-blue-700' :
                            q.question_metadata.bloom_level === 'understanding_applying' ? 'bg-purple-100 text-purple-700' :
                            q.question_metadata.bloom_level === 'analyzing_evaluating' ? 'bg-orange-100 text-orange-700' :
                            // Legacy support for old names
                            q.question_metadata.bloom_level === 'applying_analyzing' ? 'bg-purple-100 text-purple-700' :
                            q.question_metadata.bloom_level === 'evaluating_creating' ? 'bg-orange-100 text-orange-700' :
                            'bg-gray-100 text-gray-700'
                          }`}>
                            {q.question_metadata.bloom_level === 'remembering' ? '🧠 Hatırlama' :
                             q.question_metadata.bloom_level === 'understanding_applying' ? '🔧 Anlama/Uygulama' :
                             q.question_metadata.bloom_level === 'analyzing_evaluating' ? '⭐ Analiz/Değerlendirme' :
                             // Legacy support for old names
                             q.question_metadata.bloom_level === 'applying_analyzing' ? '🔧 Uygulama/Analiz' :
                             q.question_metadata.bloom_level === 'evaluating_creating' ? '⭐ Değerlendirme/Sentez' :
                             q.question_metadata.bloom_level}
                          </span>
                        )}
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
                        {q.question_metadata && (
                          <div className="text-xs text-slate-500 bg-slate-50 p-2 rounded border border-slate-200">
                            <div className="flex flex-wrap gap-3">
                              {q.question_metadata.topic && (
                                <span className="flex items-center gap-1">
                                  <span className="font-medium">📚 Konu:</span>
                                  <span className="text-slate-700">{q.question_metadata.topic}</span>
                                </span>
                              )}
                              {q.question_metadata.document_name && (
                                <span className="flex items-center gap-1">
                                  <span className="font-medium">📄 Döküman:</span>
                                  <span className="text-slate-700">{q.question_metadata.document_name}</span>
                                </span>
                              )}
                              {q.question_metadata.chunk_id && (
                                <span className="flex items-center gap-1">
                                  <span className="font-medium">🔖 Chunk:</span>
                                  <span className="text-slate-700">#{q.question_metadata.chunk_id}</span>
                                </span>
                              )}
                              {q.question_metadata.generated_at && (
                                <span className="flex items-center gap-1">
                                  <span className="font-medium">🕐 Oluşturulma:</span>
                                  <span className="text-slate-700">{new Date(q.question_metadata.generated_at).toLocaleDateString('tr-TR')}</span>
                                </span>
                              )}
                            </div>
                          </div>
                        )}
                        {(() => {
                          const result = getQuestionTestResult(q.id);
                          if (result && !result.error) {
                            // Check for warning condition: BERTScore < 80% OR ROUGE-1 < 60%
                            const bertscoreLow = result.original_bertscore_f1 !== undefined && result.original_bertscore_f1 < 0.80;
                            const rouge1Low = result.rouge1 !== undefined && result.rouge1 < 0.60;
                            const hasWarning = bertscoreLow || rouge1Low;
                            
                            return (
                              <button
                                onClick={() => {
                                  setSelectedResultId(q.id);
                                  setIsResultModalOpen(true);
                                }}
                                className={`text-xs px-3 py-1.5 rounded border hover:opacity-80 transition-all cursor-pointer w-full text-left ${
                                  hasWarning 
                                    ? 'bg-yellow-50 border-yellow-400 border-2' 
                                    : 'bg-indigo-50 border-indigo-200'
                                }`}
                              >
                                <div className="flex items-center gap-3 flex-wrap">
                                  <span className={`font-medium ${hasWarning ? 'text-yellow-900' : 'text-indigo-900'}`}>
                                    {hasWarning ? '⚠️ Test:' : '🧪 Test:'}
                                  </span>
                                  {result.rouge1 !== undefined && (
                                    <span className={`font-bold ${rouge1Low ? 'text-orange-700' : 'text-green-700'}`}>
                                      R1: {(result.rouge1 * 100).toFixed(0)}%
                                    </span>
                                  )}
                                  {result.rouge2 !== undefined && (
                                    <span className="font-bold text-green-700">R2: {(result.rouge2 * 100).toFixed(0)}%</span>
                                  )}
                                  {result.rougel !== undefined && (
                                    <span className="font-bold text-green-700">RL: {(result.rougel * 100).toFixed(0)}%</span>
                                  )}
                                  {result.original_bertscore_f1 !== undefined && (
                                    <span className={`font-bold ${bertscoreLow ? 'text-orange-700' : 'text-purple-700'}`}>
                                      B: {(result.original_bertscore_f1 * 100).toFixed(0)}%
                                    </span>
                                  )}
                                  {result.llm_model_used && (
                                    <span className="text-xs px-1.5 py-0.5 bg-slate-200 text-slate-700 rounded font-medium">
                                      {result.llm_model_used}
                                    </span>
                                  )}
                                  <span className={hasWarning ? 'text-yellow-600 ml-auto' : 'text-indigo-500 ml-auto'}>
                                    → Detay
                                  </span>
                                </div>
                              </button>
                            );
                          } else if (result && result.error) {
                            return (
                              <div className="text-xs bg-red-50 px-3 py-1.5 rounded border border-red-200">
                                <span className="text-red-700 font-medium">❌ Test Hatası: {result.error}</span>
                              </div>
                            );
                          }
                          return null;
                        })()}
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

            {/* Pagination */}
            {totalPages > 1 && (
              <div className="p-4 bg-slate-50 border-t border-slate-200 flex items-center justify-center gap-2">
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => setCurrentPage(p => Math.max(1, p - 1))}
                  disabled={currentPage === 1}
                >
                  ← Önceki
                </Button>
                
                <div className="flex items-center gap-1">
                  {Array.from({ length: totalPages }, (_, i) => i + 1).map(page => (
                    <Button
                      key={page}
                      variant={currentPage === page ? "default" : "outline"}
                      size="sm"
                      onClick={() => setCurrentPage(page)}
                      className="w-8 h-8 p-0"
                    >
                      {page}
                    </Button>
                  ))}
                </div>

                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => setCurrentPage(p => Math.min(totalPages, p + 1))}
                  disabled={currentPage === totalPages}
                >
                  Sonraki →
                </Button>
              </div>
            )}
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

      {/* Merge Dialog */}
      <Dialog open={isMergeOpen} onOpenChange={setIsMergeOpen}>
        <DialogContent>
          <form onSubmit={handleMergeTestSets}>
            <DialogHeader>
              <DialogTitle>Test Setlerini Birleştir</DialogTitle>
              <DialogDescription>
                Başka bir test setindeki tüm soruları bu test setine ekleyin.
              </DialogDescription>
            </DialogHeader>
            <div className="space-y-4 py-4">
              <div className="space-y-2">
                <Label>Kaynak Test Seti (soruları buradan alınacak)</Label>
                <Select value={sourceTestSetId} onValueChange={setSourceTestSetId}>
                  <SelectTrigger>
                    <SelectValue placeholder="Test seti seçin..." />
                  </SelectTrigger>
                  <SelectContent>
                    {availableTestSets.length === 0 ? (
                      <SelectItem value="none" disabled>Başka test seti yok</SelectItem>
                    ) : (
                      availableTestSets.map(ts => (
                        <SelectItem key={ts.id} value={ts.id.toString()}>
                          {ts.name} ({ts.question_count} soru)
                        </SelectItem>
                      ))
                    )}
                  </SelectContent>
                </Select>
                <p className="text-xs text-muted-foreground">
                  ⚠️ Kaynak test seti değişmez, soruları kopyalanır.
                </p>
              </div>
            </div>
            <DialogFooter>
              <Button type="button" variant="outline" onClick={() => setIsMergeOpen(false)}>
                İptal
              </Button>
              <Button type="submit" disabled={isMerging || !sourceTestSetId}>
                {isMerging ? <><Loader2 className="w-4 h-4 mr-1 animate-spin" /> Birleştiriliyor...</> : "🔀 Birleştir"}
              </Button>
            </DialogFooter>
          </form>
        </DialogContent>
      </Dialog>

      {/* Duplicate Detection Dialog */}
      <Dialog open={isDuplicateModalOpen} onOpenChange={setIsDuplicateModalOpen}>
        <DialogContent className="max-w-4xl max-h-[90vh] flex flex-col">
          <DialogHeader>
            <DialogTitle>Benzer Soruları Bul ve Sil</DialogTitle>
            <DialogDescription>
              Cosine similarity kullanarak benzer soruları tespit edin ve temizleyin.
            </DialogDescription>
          </DialogHeader>
          
          <div className="flex-1 overflow-y-auto space-y-4 py-4">
            {/* Threshold Selector */}
            <div className="space-y-2 p-4 bg-slate-50 rounded-lg">
              <Label>Benzerlik Eşiği: {(similarityThreshold * 100).toFixed(0)}%</Label>
              <input
                type="range"
                min="0.7"
                max="0.95"
                step="0.05"
                value={similarityThreshold}
                onChange={(e) => setSimilarityThreshold(parseFloat(e.target.value))}
                className="w-full"
              />
              <p className="text-xs text-slate-500">
                Daha yüksek değer = daha benzer sorular bulunur
              </p>
              <Button
                onClick={handleFindDuplicates}
                disabled={isFindingDuplicates}
                className="w-full mt-2"
              >
                {isFindingDuplicates ? (
                  <><Loader2 className="w-4 h-4 mr-2 animate-spin" /> Aranıyor...</>
                ) : (
                  <>🔍 Benzer Soruları Bul</>
                )}
              </Button>
            </div>

            {/* Results */}
            {duplicateGroups.length > 0 && (
              <div className="space-y-3">
                <div className="flex items-center justify-between p-3 bg-blue-50 rounded-lg border border-blue-200">
                  <div>
                    <p className="font-medium text-blue-900">
                      {duplicateGroups.length} benzer soru grubu bulundu
                    </p>
                    <p className="text-sm text-blue-700">
                      Toplam {duplicateGroups.reduce((sum, g) => sum + g.questions.length - 1, 0)} tekrar eden soru
                    </p>
                  </div>
                  {selectedDuplicates.size > 0 && (
                    <Button
                      variant="destructive"
                      size="sm"
                      onClick={handleDeleteDuplicates}
                    >
                      <Trash2 className="w-4 h-4 mr-1" />
                      {selectedDuplicates.size} Soruyu Sil
                    </Button>
                  )}
                </div>

                {duplicateGroups.map((group, groupIdx) => (
                  <div key={groupIdx} className="p-4 bg-white rounded-lg border border-slate-200">
                    <div className="flex items-center justify-between mb-3">
                      <div className="flex items-center gap-2">
                        <span className="text-sm font-medium text-slate-700">
                          Grup {groupIdx + 1}
                        </span>
                        <span className="text-xs px-2 py-1 bg-purple-100 text-purple-700 rounded-full font-medium">
                          {(group.similarity_score * 100).toFixed(1)}% benzer
                        </span>
                        <span className="text-xs text-slate-500">
                          {group.questions.length} soru
                        </span>
                      </div>
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() => {
                          // Select all except first (keep one)
                          const newSelected = new Set(selectedDuplicates);
                          group.questions.slice(1).forEach((q: any) => {
                            if (newSelected.has(q.id)) {
                              newSelected.delete(q.id);
                            } else {
                              newSelected.add(q.id);
                            }
                          });
                          setSelectedDuplicates(newSelected);
                        }}
                        className="text-xs"
                      >
                        {group.questions.slice(1).every((q: any) => selectedDuplicates.has(q.id))
                          ? "Seçimi Kaldır"
                          : "Tekrarları Seç"}
                      </Button>
                    </div>

                    <div className="space-y-2">
                      {group.questions.map((q: any, qIdx: number) => (
                        <div
                          key={q.id}
                          className={`p-3 rounded border ${
                            qIdx === 0
                              ? "bg-green-50 border-green-200"
                              : selectedDuplicates.has(q.id)
                              ? "bg-red-50 border-red-300"
                              : "bg-slate-50 border-slate-200"
                          }`}
                        >
                          <div className="flex items-start gap-3">
                            {qIdx > 0 && (
                              <input
                                type="checkbox"
                                checked={selectedDuplicates.has(q.id)}
                                onChange={() => {
                                  const newSelected = new Set(selectedDuplicates);
                                  if (newSelected.has(q.id)) {
                                    newSelected.delete(q.id);
                                  } else {
                                    newSelected.add(q.id);
                                  }
                                  setSelectedDuplicates(newSelected);
                                }}
                                className="mt-1 w-4 h-4 text-red-600 rounded border-slate-300 focus:ring-red-500"
                              />
                            )}
                            <div className="flex-1 min-w-0">
                              <div className="flex items-center gap-2 mb-1">
                                {qIdx === 0 && (
                                  <span className="text-xs px-2 py-0.5 bg-green-600 text-white rounded-full font-medium">
                                    ✓ Tutulacak
                                  </span>
                                )}
                                {qIdx > 0 && selectedDuplicates.has(q.id) && (
                                  <span className="text-xs px-2 py-0.5 bg-red-600 text-white rounded-full font-medium">
                                    ✕ Silinecek
                                  </span>
                                )}
                                <span className="text-xs text-slate-400">ID: {q.id}</span>
                              </div>
                              <p className="text-sm font-medium text-slate-900 mb-1">
                                {q.question}
                              </p>
                              <p className="text-xs text-slate-600 bg-white p-2 rounded border border-slate-200">
                                {q.ground_truth}
                              </p>
                            </div>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            )}

            {duplicateGroups.length === 0 && !isFindingDuplicates && (
              <div className="p-8 text-center text-slate-500">
                <p>Benzer soruları bulmak için yukarıdaki butona tıklayın</p>
              </div>
            )}
          </div>

          <DialogFooter className="flex-shrink-0 pt-4 border-t">
            <Button
              variant="outline"
              onClick={() => {
                setIsDuplicateModalOpen(false);
                setDuplicateGroups([]);
                setSelectedDuplicates(new Set());
              }}
            >
              Kapat
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Test Result Modal */}
      <Dialog open={isResultModalOpen} onOpenChange={setIsResultModalOpen}>
        <DialogContent className="max-w-2xl">
          <DialogHeader>
            <DialogTitle>Test Sonucu</DialogTitle>
            <DialogDescription>
              Semantic Similarity test sonuçları
            </DialogDescription>
          </DialogHeader>
          {selectedResultId && (() => {
            const result = testResults.get(selectedResultId);
            const question = testSet?.questions.find(q => q.id === selectedResultId);
            if (!result || !question) return null;

            if (result.error) {
              return (
                <div className="p-4 bg-red-50 rounded-lg border border-red-200">
                  <p className="text-red-700 font-medium">❌ Test Hatası</p>
                  <p className="text-sm text-red-600 mt-2">{result.error}</p>
                </div>
              );
            }

            return (
              <div className="space-y-4">
                <div className="p-3 bg-slate-50 rounded-lg">
                  <p className="text-xs text-slate-500 mb-1">Soru:</p>
                  <p className="text-sm font-medium">{question.question}</p>
                </div>

                <div className="grid grid-cols-2 gap-3">
                  <div className="p-3 bg-blue-50 rounded-lg border border-blue-200">
                    <p className="text-xs text-blue-600 mb-1">Similarity Score</p>
                    <p className="text-2xl font-bold text-blue-700">
                      {(result.similarity_score * 100).toFixed(1)}%
                    </p>
                  </div>
                  {result.rouge1 && (
                    <div className="p-3 bg-green-50 rounded-lg border border-green-200">
                      <p className="text-xs text-green-600 mb-1">ROUGE-1</p>
                      <p className="text-2xl font-bold text-green-700">
                        {(result.rouge1 * 100).toFixed(1)}%
                      </p>
                    </div>
                  )}
                  {result.original_bertscore_f1 && (
                    <div className="p-3 bg-purple-50 rounded-lg border border-purple-200">
                      <p className="text-xs text-purple-600 mb-1">BERTScore F1</p>
                      <p className="text-2xl font-bold text-purple-700">
                        {(result.original_bertscore_f1 * 100).toFixed(1)}%
                      </p>
                    </div>
                  )}
                  <div className="p-3 bg-slate-50 rounded-lg border border-slate-200">
                    <p className="text-xs text-slate-600 mb-1">Latency</p>
                    <p className="text-2xl font-bold text-slate-700">
                      {result.latency_ms}ms
                    </p>
                  </div>
                </div>

                <div className="p-3 bg-slate-50 rounded-lg">
                  <p className="text-xs text-slate-500 mb-1">Üretilen Cevap:</p>
                  <p className="text-sm">{result.generated_answer}</p>
                </div>

                <div className="p-3 bg-slate-50 rounded-lg">
                  <p className="text-xs text-slate-500 mb-1">Beklenen Cevap:</p>
                  <p className="text-sm">{question.ground_truth}</p>
                </div>

                <div className="text-xs text-slate-400">
                  Test tarihi: {new Date(result.tested_at).toLocaleString('tr-TR')}
                </div>
              </div>
            );
          })()}
          <DialogFooter>
            <Button variant="outline" onClick={() => setIsResultModalOpen(false)}>
              Kapat
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}
