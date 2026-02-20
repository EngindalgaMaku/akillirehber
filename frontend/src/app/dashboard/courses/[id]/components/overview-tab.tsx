"use client";

import { useEffect, useState } from "react";
import { api } from "@/lib/api";
import { Loader2, CheckCircle2, AlertCircle, ArrowRight, Settings, FileText, Zap, MessageSquare } from "lucide-react";
import { Button } from "@/components/ui/button";

interface OverviewTabProps {
  courseId: number;
  isOwner: boolean;
  onTabChange: (tab: string) => void;
}

interface CourseSettings {
  llm_provider: string;
  llm_model: string;
  llm_temperature: number;
  llm_max_tokens: number;
  system_prompt: string | null;
  search_alpha: number;
  search_top_k: number;
  default_embedding_model: string;
  min_relevance_score: number;
}

interface DocumentStats {
  total_documents: number;
  total_chunks: number;
  has_embeddings: boolean;
  total_vectors: number;
}

export function OverviewTab({ courseId, isOwner, onTabChange }: OverviewTabProps) {
  const [isLoading, setIsLoading] = useState(true);
  const [settings, setSettings] = useState<CourseSettings | null>(null);
  const [docStats, setDocStats] = useState<DocumentStats | null>(null);

  useEffect(() => {
    loadData();
  }, [courseId]);

  const loadData = async () => {
    try {
      setIsLoading(true);
      const [settingsData, documents] = await Promise.all([
        api.getCourseSettings(courseId),
        api.getCourseDocuments(courseId),
      ]);
      
      setSettings(settingsData);
      
      // Calculate document stats
      const totalChunks = documents.reduce((sum, doc) => sum + (doc.chunk_count || 0), 0);
      const totalVectors = documents.reduce((sum, doc) => sum + (doc.vector_count || 0), 0);
      const hasEmbeddings = documents.some(doc => 
        doc.embedding_status === "completed" && doc.vector_count > 0
      );
      
      setDocStats({
        total_documents: documents.length,
        total_chunks: totalChunks,
        has_embeddings: hasEmbeddings,
        total_vectors: totalVectors,
      });
    } catch (error) {
      console.error("Error loading overview data:", error);
    } finally {
      setIsLoading(false);
    }
  };

  if (isLoading) {
    return (
      <div className="flex justify-center py-12">
        <Loader2 className="w-6 h-6 text-slate-400 animate-spin" />
      </div>
    );
  }

  if (!settings || !docStats) {
    return (
      <div className="text-center py-12 text-slate-500">
        Veri yüklenemedi
      </div>
    );
  }

  // Check completion status
  const hasDocuments = docStats.total_documents > 0;
  const hasChunks = docStats.total_chunks > 0;
  const hasEmbeddings = docStats.has_embeddings;
  const hasLLMConfig = settings.llm_provider && settings.llm_model;
  const hasEmbeddingConfig = settings.default_embedding_model;
  
  const isFullyConfigured = hasDocuments && hasChunks && hasEmbeddings && hasLLMConfig && hasEmbeddingConfig;

  const issues: { type: "error" | "warning"; message: string; action?: { label: string; tab: string } }[] = [];

  if (!hasDocuments) {
    issues.push({
      type: "error",
      message: "Henüz doküman yüklenmemiş",
      action: { label: "Doküman Yükle", tab: "documents" }
    });
  }

  if (hasDocuments && !hasChunks) {
    issues.push({
      type: "error",
      message: "Dokümanlar parçalanmamış (chunking yapılmamış)",
      action: { label: "İşleme Git", tab: "processing" }
    });
  }

  if (hasChunks && !hasEmbeddings) {
    issues.push({
      type: "error",
      message: "Embedding işlemi yapılmamış",
      action: { label: "İşleme Git", tab: "processing" }
    });
  }

  if (!hasLLMConfig) {
    issues.push({
      type: "warning",
      message: "LLM ayarları eksik",
      action: { label: "Ayarlara Git", tab: "settings" }
    });
  }

  if (!hasEmbeddingConfig) {
    issues.push({
      type: "warning",
      message: "Embedding model ayarı eksik",
      action: { label: "Ayarlara Git", tab: "settings" }
    });
  }

  return (
    <div className="space-y-6">
      {/* Status Card */}
      <div className={`rounded-xl p-4 sm:p-6 border-2 ${
        isFullyConfigured 
          ? "bg-gradient-to-br from-green-50 to-emerald-50 border-green-200" 
          : "bg-gradient-to-br from-amber-50 to-orange-50 border-amber-200"
      }`}>
        <div className="flex items-start gap-3 sm:gap-4">
          {isFullyConfigured ? (
            <div className="p-2 sm:p-3 bg-green-100 rounded-xl shrink-0">
              <CheckCircle2 className="w-6 h-6 sm:w-8 sm:h-8 text-green-600" />
            </div>
          ) : (
            <div className="p-2 sm:p-3 bg-amber-100 rounded-xl shrink-0">
              <AlertCircle className="w-6 h-6 sm:w-8 sm:h-8 text-amber-600" />
            </div>
          )}
          <div className="flex-1 min-w-0">
            <h2 className="text-lg sm:text-xl font-bold text-slate-800 mb-1 sm:mb-2">
              {isFullyConfigured ? "Ders Hazır! 🎉" : "Kurulum Devam Ediyor"}
            </h2>
            <p className="text-slate-600 text-sm">
              {isFullyConfigured 
                ? "Tüm ayarlar tamamlandı. Artık sohbet edebilirsiniz!" 
                : "Dersi kullanmaya başlamak için aşağıdaki adımları tamamlayın."}
            </p>
          </div>
        </div>
      </div>

      {/* Issues List */}
      {issues.length > 0 && (
        <div className="space-y-3">
          <h3 className="text-sm font-semibold text-slate-700 uppercase tracking-wide">
            Yapılması Gerekenler
          </h3>
          {issues.map((issue, index) => (
            <div
              key={index}
              className={`flex flex-col sm:flex-row sm:items-center justify-between gap-3 p-3 sm:p-4 rounded-lg border ${
                issue.type === "error"
                  ? "bg-red-50 border-red-200"
                  : "bg-amber-50 border-amber-200"
              }`}
            >
              <div className="flex items-center gap-3 min-w-0">
                <AlertCircle className={`w-5 h-5 shrink-0 ${
                  issue.type === "error" ? "text-red-600" : "text-amber-600"
                }`} />
                <span className="text-slate-700 font-medium text-sm sm:text-base">{issue.message}</span>
              </div>
              {issue.action && (
                <Button
                  size="sm"
                  onClick={() => onTabChange(issue.action!.tab)}
                  className="bg-white hover:bg-slate-50 text-slate-700 border border-slate-300 shrink-0 self-end sm:self-auto"
                >
                  {issue.action.label}
                  <ArrowRight className="w-4 h-4 ml-1" />
                </Button>
              )}
            </div>
          ))}
        </div>
      )}

      {/* Settings Overview */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* Documents Card */}
        <div className="bg-white rounded-xl p-5 border border-slate-200 shadow-sm">
          <div className="flex items-center gap-3 mb-4">
            <div className="p-2 bg-blue-100 rounded-lg">
              <FileText className="w-5 h-5 text-blue-600" />
            </div>
            <h3 className="font-semibold text-slate-800">Dokümanlar</h3>
          </div>
          <div className="space-y-2">
            <div className="flex justify-between text-sm">
              <span className="text-slate-600">Toplam Doküman:</span>
              <span className="font-semibold text-slate-800">{docStats.total_documents}</span>
            </div>
            <div className="flex justify-between text-sm">
              <span className="text-slate-600">Toplam Parça:</span>
              <span className="font-semibold text-slate-800">{docStats.total_chunks}</span>
            </div>
            <div className="flex justify-between text-sm">
              <span className="text-slate-600">Vektör Sayısı:</span>
              <span className="font-semibold text-slate-800">{docStats.total_vectors}</span>
            </div>
            <div className="flex justify-between text-sm">
              <span className="text-slate-600">Embedding:</span>
              <span className={`font-semibold ${hasEmbeddings ? "text-green-600" : "text-red-600"}`}>
                {hasEmbeddings ? "✓ Hazır" : "✗ Yok"}
              </span>
            </div>
          </div>
        </div>

        {/* LLM Settings Card */}
        <div className="bg-white rounded-xl p-5 border border-slate-200 shadow-sm">
          <div className="flex items-center gap-3 mb-4">
            <div className="p-2 bg-purple-100 rounded-lg">
              <Zap className="w-5 h-5 text-purple-600" />
            </div>
            <h3 className="font-semibold text-slate-800">LLM Ayarları</h3>
          </div>
          <div className="space-y-2">
            <div className="flex justify-between text-sm">
              <span className="text-slate-600">Provider:</span>
              <span className="font-semibold text-slate-800">{settings.llm_provider || "-"}</span>
            </div>
            <div className="flex justify-between text-sm">
              <span className="text-slate-600">Model:</span>
              <span className="font-semibold text-slate-800">{settings.llm_model || "-"}</span>
            </div>
            <div className="flex justify-between text-sm">
              <span className="text-slate-600">Temperature:</span>
              <span className="font-semibold text-slate-800">{settings.llm_temperature}</span>
            </div>
          </div>
        </div>

        {/* Search Settings Card */}
        <div className="bg-white rounded-xl p-5 border border-slate-200 shadow-sm">
          <div className="flex items-center gap-3 mb-4">
            <div className="p-2 bg-green-100 rounded-lg">
              <Settings className="w-5 h-5 text-green-600" />
            </div>
            <h3 className="font-semibold text-slate-800">Arama Ayarları</h3>
          </div>
          <div className="space-y-2">
            <div className="flex justify-between text-sm">
              <span className="text-slate-600">Alpha (Hybrid):</span>
              <span className="font-semibold text-slate-800">{settings.search_alpha}</span>
            </div>
            <div className="flex justify-between text-sm">
              <span className="text-slate-600">Top K:</span>
              <span className="font-semibold text-slate-800">{settings.search_top_k}</span>
            </div>
            <div className="flex justify-between text-sm">
              <span className="text-slate-600">Min Relevance:</span>
              <span className="font-semibold text-slate-800">{settings.min_relevance_score}</span>
            </div>
          </div>
        </div>

        {/* Embedding Settings Card */}
        <div className="bg-white rounded-xl p-5 border border-slate-200 shadow-sm">
          <div className="flex items-center gap-3 mb-4">
            <div className="p-2 bg-indigo-100 rounded-lg">
              <Zap className="w-5 h-5 text-indigo-600" />
            </div>
            <h3 className="font-semibold text-slate-800">Embedding</h3>
          </div>
          <div className="space-y-2">
            <div className="flex justify-between text-sm">
              <span className="text-slate-600">Model:</span>
              <span className="font-semibold text-slate-800 text-xs">
                {settings.default_embedding_model || "-"}
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* System Prompt */}
      {settings.system_prompt && (
        <div className="bg-white rounded-xl p-5 border border-slate-200 shadow-sm">
          <div className="flex items-center gap-3 mb-3">
            <div className="p-2 bg-gradient-to-br from-violet-100 to-purple-100 rounded-lg">
              <MessageSquare className="w-5 h-5 text-violet-600" />
            </div>
            <h3 className="font-semibold text-slate-800">Sistem Promptu</h3>
          </div>
          <div className="bg-gradient-to-br from-slate-50 to-slate-100 rounded-lg p-4 border border-slate-200">
            <pre className="text-sm text-slate-700 whitespace-pre-wrap font-mono leading-relaxed">
              {settings.system_prompt}
            </pre>
          </div>
        </div>
      )}

      {/* Action Button */}
      {isFullyConfigured && (
        <div className="flex justify-center pt-4">
          <Button
            size="lg"
            onClick={() => onTabChange("chat")}
            className="bg-gradient-to-r from-indigo-600 to-purple-600 hover:from-indigo-700 hover:to-purple-700 text-white shadow-lg"
          >
            <MessageSquare className="w-5 h-5 mr-2" />
            Sohbete Başla
          </Button>
        </div>
      )}
    </div>
  );
}
