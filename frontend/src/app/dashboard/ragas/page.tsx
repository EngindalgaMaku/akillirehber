"use client";

import { useEffect, useState, useCallback } from "react";
import { useAuth } from "@/lib/auth-context";
import { api, Course, QuickTestResponse, RagasSettings, RagasProvider, RagasGroupInfo } from "@/lib/api";
import { toast } from "sonner";
import { FlaskConical, BookOpen, History } from "lucide-react";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import Link from "next/link";
import { Button } from "@/components/ui/button";

import { SettingsDialog } from "./components/SettingsDialog";
import { QuickTestSection } from "./components/QuickTestSection";
import { BatchTestSection } from "./components/BatchTestSection";

export default function RagasPage() {
  const { user } = useAuth();
  const [courses, setCourses] = useState<Course[]>([]);
  const [selectedCourseId, setSelectedCourseId] = useState<number | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  // Settings State
  const [ragasSettings, setRagasSettings] = useState<RagasSettings | null>(null);
  const [ragasProviders, setRagasProviders] = useState<RagasProvider[]>([]);
  const [ragasEmbeddingModel, setRagasEmbeddingModel] = useState<string>(() => {
    if (typeof window !== 'undefined') {
      return localStorage.getItem('ragas_embedding_model') || "";
    }
    return "";
  });

  // Quick Test Result
  const [quickTestResult, setQuickTestResult] = useState<QuickTestResponse | null>(null);

  const [savedResultsGroups, setSavedResultsGroups] = useState<RagasGroupInfo[]>([]);

  useEffect(() => {
    loadCourses();
    loadRagasSettings();
  }, []);

  useEffect(() => {
    if (selectedCourseId) {
      loadSavedResultsGroups();
    }
  }, [selectedCourseId]);

  const loadCourses = async () => {
    try {
      const data = await api.getCourses();
      setCourses(data);
      if (data.length > 0) {
        const savedCourseId = localStorage.getItem('ragas_selected_course_id');
        if (savedCourseId && data.find(c => c.id === parseInt(savedCourseId))) {
          setSelectedCourseId(parseInt(savedCourseId));
        } else {
          setSelectedCourseId(data[0].id);
          localStorage.setItem('ragas_selected_course_id', data[0].id.toString());
        }
      }
    } catch {
      toast.error("Dersler yüklenirken hata oluştu");
    } finally {
      setIsLoading(false);
    }
  };

  const loadRagasSettings = async () => {
    try {
      const [settings, providersData] = await Promise.all([
        api.getRagasSettings(),
        api.getRagasProviders(),
      ]);
      setRagasSettings(settings);
      setRagasProviders(providersData.providers);
    } catch {
      console.log("RAGAS settings not available");
    }
  };

  const loadSavedResultsGroups = useCallback(async () => {
    if (!selectedCourseId) return;
    try {
      const data = await api.getQuickTestResults(selectedCourseId, undefined, 0, 1);
      setSavedResultsGroups(data.groups);
    } catch (error) {
      console.error("Failed to load saved results groups:", error);
      setSavedResultsGroups([]);
    }
  }, [selectedCourseId]);

  if (!user) return null;

  if (isLoading) {
    return (
      <div className="min-h-[60vh] flex items-center justify-center">
        <div className="text-center">
          <div className="relative">
            <div className="w-16 h-16 border-4 border-purple-200 border-t-purple-600 rounded-full animate-spin mx-auto"></div>
            <FlaskConical className="w-6 h-6 text-purple-600 absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2" />
          </div>
          <p className="mt-4 text-slate-600 font-medium">RAGAS yükleniyor...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-8">
      {/* Hero Header */}
      <div className="relative overflow-hidden bg-gradient-to-br from-purple-600 via-indigo-600 to-blue-700 rounded-2xl p-8 text-white shadow-xl">
        <div className="absolute inset-0 opacity-30" style={{backgroundImage: "url(\"data:image/svg+xml,%3Csvg width='60' height='60' viewBox='0 0 60 60' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' fill-rule='evenodd'%3E%3Cg fill='%23fff' fill-opacity='0.1'%3E%3Ccircle cx='30' cy='30' r='2'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E\")"}}></div>
        <div className="absolute -top-24 -right-24 w-96 h-96 bg-white/10 rounded-full blur-3xl"></div>
        <div className="absolute -bottom-24 -left-24 w-96 h-96 bg-purple-500/20 rounded-full blur-3xl"></div>
        
        <div className="relative z-10">
          <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between gap-6">
            <div className="flex items-center gap-4">
              <div className="p-3 bg-white/20 rounded-xl backdrop-blur-sm">
                <FlaskConical className="w-8 h-8" />
              </div>
              <div>
                <h1 className="text-3xl font-bold">RAGAS Değerlendirme</h1>
                <p className="text-purple-200 mt-1">RAG sisteminizin kalitesini test edin ve ölçün</p>
              </div>
            </div>
            
            <div className="flex flex-wrap items-center gap-3">
              <SettingsDialog 
                ragasSettings={ragasSettings}
                ragasProviders={ragasProviders}
                onSettingsUpdate={loadRagasSettings}
                selectedEmbeddingModel={ragasEmbeddingModel}
                onEmbeddingModelChange={(model) => {
                  setRagasEmbeddingModel(model);
                  localStorage.setItem('ragas_embedding_model', model);
                }}
              />

              <Link href="/dashboard/ragas/results">
                <Button
                  type="button"
                  variant="secondary"
                  className="h-10 bg-white/20 text-white border-0 hover:bg-white/30 backdrop-blur-sm"
                >
                  <History className="w-4 h-4 mr-2" />
                  Sonuçlar
                </Button>
              </Link>

              <Select 
                value={selectedCourseId?.toString() || ""} 
                onValueChange={(v) => {
                  const courseId = Number(v);
                  setSelectedCourseId(courseId);
                  localStorage.setItem('ragas_selected_course_id', courseId.toString());
                }}
              >
                <SelectTrigger className="w-56 bg-white/20 border-0 text-white hover:bg-white/30 backdrop-blur-sm h-10">
                  <BookOpen className="w-4 h-4 mr-2" />
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
          </div>

          {selectedCourseId && (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-8">
              <div className="bg-white/10 backdrop-blur-sm rounded-xl p-4 border border-white/20">
                <div className="flex items-center gap-3">
                  <div className="p-2 bg-white/20 rounded-lg">
                    <BookOpen className="w-5 h-5" />
                  </div>
                  <div>
                    <p className="text-purple-200 text-sm">Seçili Ders</p>
                    <p className="text-lg font-bold truncate">
                      {courses.find(c => c.id === selectedCourseId)?.name || "-"}
                    </p>
                  </div>
                </div>
              </div>
              <div className="bg-white/10 backdrop-blur-sm rounded-xl p-4 border border-white/20">
                <div className="flex items-center gap-3">
                  <div className="p-2 bg-white/20 rounded-lg">
                    <History className="w-5 h-5" />
                  </div>
                  <div>
                    <p className="text-purple-200 text-sm">Gruplar</p>
                    <p className="text-2xl font-bold">{savedResultsGroups.length}</p>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>

      {!selectedCourseId ? (
        <div className="bg-white rounded-2xl border border-slate-200 p-16 text-center shadow-sm">
          <div className="w-20 h-20 bg-purple-100 rounded-full flex items-center justify-center mx-auto mb-6">
            <FlaskConical className="w-10 h-10 text-purple-600" />
          </div>
          <h3 className="text-xl font-semibold text-slate-900 mb-2">Ders Seçin</h3>
          <p className="text-slate-500 max-w-md mx-auto">
            RAGAS değerlendirmesi yapmak için yukarıdan bir ders seçin.
          </p>
        </div>
      ) : (
        <div className="space-y-6">
          <QuickTestSection 
            selectedCourseId={selectedCourseId}
            quickTestResult={quickTestResult}
            setQuickTestResult={setQuickTestResult}
            onResultSaved={loadSavedResultsGroups}
            savedResultsGroups={savedResultsGroups}
            ragasEmbeddingModel={ragasEmbeddingModel}
          />

          <BatchTestSection
            selectedCourseId={selectedCourseId}
            onBatchTestComplete={loadSavedResultsGroups}
            savedResultsGroups={savedResultsGroups}
            ragasEmbeddingModel={ragasEmbeddingModel}
          />
        </div>
      )}
    </div>
  );
}