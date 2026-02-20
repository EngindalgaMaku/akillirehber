"use client";

import { useEffect, useState, useCallback } from "react";
import { useRouter } from "next/navigation";
import { api, CustomLLMModel, CoursePromptTemplate } from "@/lib/api";
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
import { Switch } from "@/components/ui/switch";
import { Slider } from "@/components/ui/slider";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { toast } from "sonner";
import {
  Loader2,
  Save,
  MessageSquare,
  Settings2,
  Search,
  Scissors,
  Trash2,
  AlertTriangle,
  X,
  Plus,
  Settings,
  Sparkles,
  Info,
  ChevronDown,
  Zap,
  Shield,
} from "lucide-react";

interface SettingsTabProps {
  readonly courseId: number;
  readonly isOwner: boolean;
  readonly courseName: string;
}

// Consistent card styling for all sections
const sectionCardStyles = "bg-white rounded-xl border border-slate-200 shadow-sm hover:shadow-md transition-shadow duration-200 overflow-hidden";
const sectionHeaderStyles = "px-4 sm:px-6 py-4 sm:py-5 border-b border-slate-100 bg-gradient-to-r from-slate-50/50 to-transparent";
const sectionContentStyles = "p-4 sm:p-6";

const EMBEDDING_MODELS = [
  { value: "openai/text-embedding-3-small", label: "OpenAI text-embedding-3-small (1536 dim)" },
  { value: "openai/text-embedding-3-large", label: "OpenAI text-embedding-3-large (3072 dim)" },
  { value: "alibaba/text-embedding-v4", label: "Alibaba text-embedding-v4 (1024 dim)" },
  { value: "cohere/embed-multilingual-v3.0", label: "Cohere embed-multilingual-v3.0 (1024 dim)" },
  { value: "cohere/embed-multilingual-light-v3.0", label: "Cohere embed-multilingual-light-v3.0 (384 dim)" },
  { value: "jina/jina-embeddings-v2", label: "Jina AI jina-embeddings-v2 (768 dim)" },
  { value: "jina/jina-embeddings-v3", label: "Jina AI jina-embeddings-v3 (1024 dim)" },
  { value: "qwen/qwen3-embedding-8b", label: "Qwen qwen3-embedding-8b (1024 dim)" },
  { value: "ollama/bge-m3", label: "Ollama BGE-M3 (1024 dim)", description: "Local BGE-M3 via Ollama" },
  { value: "ollama/nomic-embed-text", label: "Ollama Nomic Embed Text (768 dim)", description: "Local Nomic via Ollama" },
  { value: "voyage/voyage-4-large", label: "Voyage voyage-4-large (1536 dim)", description: "VoyageAI large model" },
  { value: "voyage/voyage-3-large", label: "Voyage voyage-3-large (1024 dim)", description: "VoyageAI large model" },
  { value: "voyage/voyage-3-lite", label: "Voyage voyage-3-lite (512 dim)", description: "VoyageAI lite model" },
  { value: "voyage/voyage-2", label: "Voyage voyage-2 (1024 dim)", description: "VoyageAI v2 model" },
];

const RERANKER_MODELS = {
  cohere: [
    { value: "rerank-english-v3.0", label: "Rerank English v3.0", description: "İngilizce için optimize edilmiş" },
    { value: "rerank-multilingual-v3.0", label: "Rerank Multilingual v3.0", description: "100+ dil desteği" },
  ],
  alibaba: [
    { value: "gte-rerank-v2", label: "GTE Rerank v2", description: "Çok dilli destek" },
  ],
  jina: [
    { value: "jina-reranker-v1-base-en", label: "Jina Reranker v1 Base EN", description: "İngilizce için" },
    { value: "jina-reranker-v2-base-multilingual", label: "Jina Reranker v2 Base Multilingual", description: "Çok dilli" },
  ],
  bge: [
    { value: "ollama-bge-reranker-v2-m3", label: "BGE Reranker v2-M3 (Ollama)", description: "Local BGE reranker" },
  ],
  zeroentropy: [
    { value: "zerank-2", label: "ZeRank-2", description: "ZeroEntropy hosted reranker" },
  ],
  voyage: [
    { value: "rerank-2", label: "Voyage Rerank-2", description: "VoyageAI reranker v2" },
    { value: "rerank-2.5", label: "Voyage Rerank-2.5", description: "VoyageAI reranker v2.5" },
    { value: "rerank-2.5-lite", label: "Voyage Rerank-2.5-Lite", description: "VoyageAI lite reranker" },
  ],
};

export function SettingsTab({ courseId, isOwner, courseName }: SettingsTabProps) {
  const router = useRouter();
  const [isLoading, setIsLoading] = useState(true);
  const [isSaving, setIsSaving] = useState(false);
  const [availableModels, setAvailableModels] = useState<string[]>([]);
  const [showDeleteModal, setShowDeleteModal] = useState(false);
  const [deleteConfirmText, setDeleteConfirmText] = useState("");
  const [isDeleting, setIsDeleting] = useState(false);
  const [validationErrors, setValidationErrors] = useState<string[]>([]);
  
  // Course name editing state
  const [isEditingName, setIsEditingName] = useState(false);
  const [editedCourseName, setEditedCourseName] = useState(courseName);
  const [editedCourseDescription, setEditedCourseDescription] = useState("");
  const [isSavingName, setIsSavingName] = useState(false);
  
  // Accordion state (only one section open at a time)
  const [expandedSection, setExpandedSection] = useState<
    "course_info" | "system_prompt" | "chunking" | "search" | "reranker" | "llm" | "vector_store" | "direct_llm" | "pii_filter" | null
  >(null);
  
  // Model management state
  const [showModelManager, setShowModelManager] = useState(false);
  const [customModels, setCustomModels] = useState<CustomLLMModel[]>([]);
  const [isLoadingModels, setIsLoadingModels] = useState(false);
  const [newModel, setNewModel] = useState({
    provider: "openrouter",
    model_id: "",
    display_name: "",
  });
  const [isAddingModel, setIsAddingModel] = useState(false);

  // Prompt template state
  const [promptTemplates, setPromptTemplates] = useState<CoursePromptTemplate[]>([]);
  const [isLoadingPromptTemplates, setIsLoadingPromptTemplates] = useState(false);
  const [newPromptTemplateName, setNewPromptTemplateName] = useState("");
  const [isCreatingPromptTemplate, setIsCreatingPromptTemplate] = useState(false);
  const [isDeletingPromptTemplateId, setIsDeletingPromptTemplateId] = useState<number | null>(null);

  const [formData, setFormData] = useState({
    default_chunk_strategy: "recursive",
    default_chunk_size: 500,
    default_overlap: 50,
    default_embedding_model: "openai/text-embedding-3-small",
    search_alpha: 0.5,
    search_top_k: 5,
    min_relevance_score: 0,
    llm_provider: "openrouter",
    llm_model: "openai/gpt-4o-mini",
    llm_temperature: 0.7,
    llm_max_tokens: 1000,
    system_prompt: "",
    active_prompt_template_id: null as number | null,
    enable_reranker: false,
    reranker_provider: null as string | null,
    reranker_model: null as string | null,
    reranker_top_k: 10,
    vector_store: "weaviate" as string,
    enable_direct_llm: false,
    enable_pii_filter: false,
  });

  const loadSettings = useCallback(async () => {
    try {
      const data = await api.getCourseSettings(courseId);
      console.log("API Response:", data);
      const newFormData = {
        default_chunk_strategy: data.default_chunk_strategy,
        default_chunk_size: data.default_chunk_size,
        default_overlap: data.default_overlap,
        default_embedding_model: data.default_embedding_model,
        search_alpha: data.search_alpha,
        search_top_k: data.search_top_k,
        min_relevance_score: data.min_relevance_score || 0,
        llm_provider: data.llm_provider,
        llm_model: data.llm_model,
        llm_temperature: data.llm_temperature,
        llm_max_tokens: data.llm_max_tokens,
        system_prompt: data.system_prompt || "",
        active_prompt_template_id: data.active_prompt_template_id ?? null,
        enable_reranker: data.enable_reranker || false,
        reranker_provider: data.reranker_provider || null,
        reranker_model: data.reranker_model || null,
        reranker_top_k: data.reranker_top_k || 10,
        vector_store: data.vector_store || "weaviate",
        enable_direct_llm: data.enable_direct_llm || false,
        enable_pii_filter: data.enable_pii_filter || false,
      };
      console.log("New Form Data:", newFormData);
      setFormData(newFormData);
      
      // Load course info
      const courseData = await api.getCourse(courseId);
      setEditedCourseName(courseData.name);
      setEditedCourseDescription(courseData.description || "");
    } catch (error) {
      console.error("Settings load error:", error);
      toast.error("Ayarlar yüklenirken hata oluştu");
    } finally {
      setIsLoading(false);
    }
  }, [courseId]);

  useEffect(() => {
    loadSettings();
  }, [loadSettings]);

  const loadPromptTemplates = useCallback(async () => {
    setIsLoadingPromptTemplates(true);
    try {
      const res = await api.getCoursePromptTemplates(courseId);
      setPromptTemplates(res.templates);
    } catch {
      toast.error("Prompt template listesi yüklenirken hata oluştu");
    } finally {
      setIsLoadingPromptTemplates(false);
    }
  }, [courseId]);

  useEffect(() => {
    loadPromptTemplates();
  }, [loadPromptTemplates]);

  const handleActivatePromptTemplate = async (templateId: number | null) => {
    try {
      const res = await api.activateCoursePromptTemplate(courseId, templateId);
      setFormData((prev) => ({
        ...prev,
        active_prompt_template_id: res.active_prompt_template_id,
      }));
      toast.success("Aktif prompt template güncellendi");
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "İşlem başarısız");
    }
  };

  const handleCreatePromptTemplateFromCurrent = async () => {
    if (!newPromptTemplateName.trim()) {
      toast.error("Template adı zorunludur");
      return;
    }
    if (!formData.system_prompt.trim()) {
      toast.error("Sistem promptu boşken template kaydedemezsiniz");
      return;
    }

    setIsCreatingPromptTemplate(true);
    try {
      const created = await api.createCoursePromptTemplate(courseId, {
        name: newPromptTemplateName.trim(),
        content: formData.system_prompt,
      });
      setNewPromptTemplateName("");
      await loadPromptTemplates();
      toast.success("Prompt template kaydedildi");

      if (formData.active_prompt_template_id === null) {
        await handleActivatePromptTemplate(created.id);
      }
    } catch (error) {
      toast.error(
        error instanceof Error ? error.message : "Template kaydedilemedi"
      );
    } finally {
      setIsCreatingPromptTemplate(false);
    }
  };

  const handleDeletePromptTemplate = async (templateId: number) => {
    setIsDeletingPromptTemplateId(templateId);
    try {
      await api.deleteCoursePromptTemplate(courseId, templateId);
      toast.success("Template silindi");
      await loadPromptTemplates();

      if (formData.active_prompt_template_id === templateId) {
        await handleActivatePromptTemplate(null);
      }
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "Silme hatası");
    } finally {
      setIsDeletingPromptTemplateId(null);
    }
  };

  useEffect(() => {
    const loadModels = async () => {
      try {
        const providers = await api.getLLMProviders();
        const models = providers[formData.llm_provider] || [];
        setAvailableModels(models);
      } catch {
        toast.error("Model listesi yüklenirken hata oluştu");
      }
    };
    loadModels();
  }, [formData.llm_provider]);

  const loadCustomModels = useCallback(async () => {
    setIsLoadingModels(true);
    try {
      const response = await api.getCustomLLMModels();
      setCustomModels(response.models);
    } catch {
      toast.error("Özel modeller yüklenirken hata oluştu");
    } finally {
      setIsLoadingModels(false);
    }
  }, []);

  useEffect(() => {
    if (showModelManager) {
      loadCustomModels();
    }
  }, [showModelManager, loadCustomModels]);

  const handleAddModel = async () => {
    if (!newModel.model_id.trim() || !newModel.display_name.trim()) {
      toast.error("Model ID ve görünen ad zorunludur");
      return;
    }

    setIsAddingModel(true);
    try {
      await api.createCustomLLMModel(newModel);
      toast.success("Model başarıyla eklendi");
      setNewModel({ provider: "openrouter", model_id: "", display_name: "" });
      loadCustomModels();
      // Refresh available models
      const providers = await api.getLLMProviders();
      const models = providers[formData.llm_provider] || [];
      setAvailableModels(models);
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "Model eklenirken hata oluştu");
    } finally {
      setIsAddingModel(false);
    }
  };

  const handleDeleteModel = async (modelId: number) => {
    try {
      await api.deleteCustomLLMModel(modelId);
      toast.success("Model başarıyla silindi");
      loadCustomModels();
      // Refresh available models
      const providers = await api.getLLMProviders();
      const models = providers[formData.llm_provider] || [];
      setAvailableModels(models);
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "Model silinirken hata oluştu");
    }
  };

  const validateRerankerSettings = (): string[] => {
    const errors: string[] = [];
    
    if (formData.enable_reranker) {
      if (!formData.reranker_provider) {
        errors.push("Reranker etkinleştirildiğinde provider seçimi zorunludur");
      }
      if (!formData.reranker_model) {
        errors.push("Reranker etkinleştirildiğinde model seçimi zorunludur");
      }
      if (formData.reranker_top_k < 5 || formData.reranker_top_k > 20) {
        errors.push("Reranker Top-K değeri 5-20 arasında olmalıdır");
      }
    }
    
    return errors;
  };

  const handleSave = async () => {
    // Validate reranker settings
    const errors = validateRerankerSettings();
    setValidationErrors(errors);
    
    if (errors.length > 0) {
      toast.error("Lütfen tüm zorunlu alanları doldurun");
      return;
    }
    
    setIsSaving(true);
    try {
      // Convert null to undefined for API compatibility
      const settingsToSave = {
        ...formData,
        reranker_provider: formData.reranker_provider || undefined,
        reranker_model: formData.reranker_model || undefined,
      };
      console.log("[SAVE] enable_direct_llm:", settingsToSave.enable_direct_llm);
      console.log("[SAVE] Full payload:", JSON.stringify(settingsToSave, null, 2));
      await api.updateCourseSettings(courseId, settingsToSave);
      toast.success("Ayarlar kaydedildi");
      setValidationErrors([]);
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "Kaydetme hatası");
    } finally {
      setIsSaving(false);
    }
  };

  const handleDeleteCourse = async () => {
    if (deleteConfirmText !== courseName) return;
    
    setIsDeleting(true);
    try {
      await api.deleteCourse(courseId);
      toast.success("Ders başarıyla silindi");
      router.push("/dashboard/courses");
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "Silme hatası");
    } finally {
      setIsDeleting(false);
      setShowDeleteModal(false);
    }
  };

  const handleSaveCourseName = async () => {
    if (!editedCourseName.trim()) {
      toast.error("Ders adı boş olamaz");
      return;
    }
    
    setIsSavingName(true);
    try {
      await api.updateCourse(courseId, {
        name: editedCourseName.trim(),
        description: editedCourseDescription.trim() || undefined,
      });
      toast.success("Ders bilgileri güncellendi");
      setIsEditingName(false);
      // Reload page to update header
      window.location.reload();
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "Güncelleme hatası");
    } finally {
      setIsSavingName(false);
    }
  };

  if (isLoading) {
    return (
      <div className="flex justify-center py-12">
        <Loader2 className="w-6 h-6 text-slate-400 animate-spin" />
      </div>
    );
  }

  console.log("Settings Tab Rendered - isOwner:", isOwner, "formData:", formData, "availableModels:", availableModels);

  return (
    <div className="max-w-4xl space-y-8">
      {/* Course Info Section - Collapsible */}
      <div className={sectionCardStyles}>
        <div 
          className={`${sectionHeaderStyles} cursor-pointer hover:bg-slate-100/50 transition-colors`}
          onClick={() =>
            setExpandedSection((prev) =>
              prev === "course_info" ? null : "course_info"
            )
          }
        >
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-violet-500 to-violet-600 flex items-center justify-center shadow-sm">
                <Settings className="w-5 h-5 text-white" />
              </div>
              <div>
                <h3 className="font-semibold text-slate-900 text-lg">Ders Bilgileri</h3>
                <p className="text-sm text-slate-500 mt-0.5">
                  Ders adı ve açıklamasını düzenleyin
                </p>
              </div>
            </div>
            <ChevronDown 
              className={`w-5 h-5 text-slate-400 transition-transform duration-200 ${
                expandedSection === "course_info" ? 'rotate-180' : ''
              }`}
            />
          </div>
        </div>
        
        {expandedSection === "course_info" && (
          <div className={sectionContentStyles}>
            <div className="space-y-4">
              <div className="space-y-2">
                <Label className="text-sm font-medium text-slate-700">Ders Adı</Label>
                <Input
                  value={editedCourseName}
                  onChange={(e) => setEditedCourseName(e.target.value)}
                  placeholder="Ders adını girin"
                  className="h-11 border-slate-200 focus:border-violet-300 focus:ring-violet-200"
                  disabled={!isOwner}
                />
              </div>

              <div className="space-y-2">
                <Label className="text-sm font-medium text-slate-700">Açıklama (Opsiyonel)</Label>
                <Textarea
                  value={editedCourseDescription}
                  onChange={(e) => setEditedCourseDescription(e.target.value)}
                  placeholder="Ders hakkında kısa bir açıklama..."
                  className="min-h-[100px] resize-none border-slate-200 focus:border-violet-300 focus:ring-violet-200"
                  disabled={!isOwner}
                />
              </div>

              {isOwner && (
                <div className="flex gap-2 pt-2">
                  <Button
                    onClick={handleSaveCourseName}
                    disabled={isSavingName || !editedCourseName.trim()}
                    className="bg-gradient-to-r from-violet-600 to-violet-700 hover:from-violet-700 hover:to-violet-800"
                  >
                    {isSavingName ? (
                      <>
                        <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                        Kaydediliyor...
                      </>
                    ) : (
                      <>
                        <Save className="w-4 h-4 mr-2" />
                        Kaydet
                      </>
                    )}
                  </Button>
                  <Button
                    variant="outline"
                    onClick={() => {
                      setEditedCourseName(courseName);
                      setEditedCourseDescription("");
                      loadSettings();
                    }}
                    disabled={isSavingName}
                  >
                    İptal
                  </Button>
                </div>
              )}
            </div>
          </div>
        )}
      </div>

      {/* System Prompt Section - Collapsible */}
      <div className={sectionCardStyles}>
        <div 
          className={`${sectionHeaderStyles} cursor-pointer hover:bg-slate-100/50 transition-colors`}
          onClick={() =>
            setExpandedSection((prev) =>
              prev === "system_prompt" ? null : "system_prompt"
            )
          }
        >
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-indigo-500 to-indigo-600 flex items-center justify-center shadow-sm">
                <MessageSquare className="w-5 h-5 text-white" />
              </div>
              <div>
                <h3 className="font-semibold text-slate-900 text-lg">Sistem Promptu</h3>
                <p className="text-sm text-slate-500 mt-0.5">
                  AI asistanının davranışını ve kişiliğini belirleyin
                </p>
              </div>
            </div>
            <ChevronDown 
              className={`w-5 h-5 text-slate-400 transition-transform duration-200 ${
                expandedSection === "system_prompt" ? 'rotate-180' : ''
              }`}
            />
          </div>
        </div>
        
        {expandedSection === "system_prompt" && (
          <div className={sectionContentStyles}>
          <div className="space-y-4">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="space-y-2">
                <Label className="text-sm font-medium text-slate-700">
                  Prompt Template (Opsiyonel)
                </Label>
                <Select
                  value={
                    formData.active_prompt_template_id !== null
                      ? String(formData.active_prompt_template_id)
                      : "__manual__"
                  }
                  onValueChange={(v) => {
                    if (v === "__manual__") {
                      handleActivatePromptTemplate(null);
                      return;
                    }
                    const id = Number(v);
                    if (!Number.isNaN(id)) {
                      handleActivatePromptTemplate(id);
                    }
                  }}
                  disabled={!isOwner || isLoadingPromptTemplates}
                >
                  <SelectTrigger className="h-11 border-slate-200 focus:border-indigo-300 focus:ring-indigo-200">
                    <SelectValue placeholder="Template seçin" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="__manual__">Manuel (Sistem promptu)</SelectItem>
                    {promptTemplates.map((t) => (
                      <SelectItem key={t.id} value={String(t.id)}>
                        {t.name}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
                <p className="text-xs text-slate-500 bg-slate-50 px-3 py-2 rounded-lg">
                  Seçiliyse, sohbet/RAG bu template içeriğini kullanır.
                </p>

                {promptTemplates.length > 0 && (
                  <div className="space-y-2">
                    {promptTemplates.map((t) => (
                      <div
                        key={t.id}
                        className="flex items-center justify-between gap-2 rounded-lg border border-slate-200 bg-white px-3 py-2"
                      >
                        <div className="min-w-0">
                          <div className="text-sm font-medium text-slate-800 truncate">
                            {t.name}
                          </div>
                          <div className="text-xs text-slate-500 truncate">
                            {t.content}
                          </div>
                        </div>
                        {isOwner && (
                          <Button
                            type="button"
                            variant="ghost"
                            size="sm"
                            onClick={() => handleDeletePromptTemplate(t.id)}
                            disabled={isDeletingPromptTemplateId === t.id}
                            className="text-red-600 hover:text-red-700"
                          >
                            {isDeletingPromptTemplateId === t.id ? (
                              <Loader2 className="w-4 h-4 animate-spin" />
                            ) : (
                              <Trash2 className="w-4 h-4" />
                            )}
                          </Button>
                        )}
                      </div>
                    ))}
                  </div>
                )}
              </div>

              {isOwner && (
                <div className="space-y-2">
                  <Label className="text-sm font-medium text-slate-700">
                    Mevcut promptu template olarak kaydet
                  </Label>
                  <div className="flex gap-2">
                    <Input
                      value={newPromptTemplateName}
                      onChange={(e) => setNewPromptTemplateName(e.target.value)}
                      placeholder="Template adı"
                      className="h-11 border-slate-200"
                    />
                    <Button
                      type="button"
                      onClick={handleCreatePromptTemplateFromCurrent}
                      disabled={isCreatingPromptTemplate}
                      className="h-11 bg-gradient-to-r from-indigo-600 to-indigo-700"
                    >
                      {isCreatingPromptTemplate ? (
                        <Loader2 className="w-4 h-4 animate-spin" />
                      ) : (
                        "Kaydet"
                      )}
                    </Button>
                  </div>
                  <p className="text-xs text-slate-500 bg-slate-50 px-3 py-2 rounded-lg">
                    Template adları ders içinde benzersiz olmalı.
                  </p>
                </div>
              )}
            </div>

            <div className="flex items-center justify-between">
              <Label className="text-sm font-medium text-slate-700">
                Ders İçin Özel Sistem Promptu
              </Label>
              <span className={`text-xs font-medium px-2 py-1 rounded-full ${
                formData.system_prompt.length >= 2000 
                  ? 'bg-red-100 text-red-700' 
                  : formData.system_prompt.length >= 1900 
                    ? 'bg-amber-100 text-amber-700' 
                    : 'bg-slate-100 text-slate-500'
              }`}>
                {formData.system_prompt.length}/2000 karakter
              </span>
            </div>
            <Textarea
              value={formData.system_prompt}
              onChange={(e) => setFormData({ ...formData, system_prompt: e.target.value })}
              placeholder="AI asistanının bu ders için nasıl davranması gerektiğini açıklayın..."
              className="min-h-[140px] resize-none border-slate-200 focus:border-indigo-300 focus:ring-indigo-200"
              maxLength={2000}
              disabled={!isOwner}
            />
            <p className="text-xs text-slate-500 bg-slate-50 px-3 py-2 rounded-lg">
              💡 Boş bırakılırsa varsayılan eğitim asistanı promptu kullanılır.
            </p>
          </div>
        </div>
        )}
      </div>

      {/* Chunking Settings Section */}
      <div className={sectionCardStyles}>
        <div
          className={`${sectionHeaderStyles} cursor-pointer hover:bg-slate-100/50 transition-colors`}
          onClick={() =>
            setExpandedSection((prev) =>
              prev === "chunking" ? null : "chunking"
            )
          }
        >
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-emerald-500 to-emerald-600 flex items-center justify-center shadow-sm">
                <Scissors className="w-5 h-5 text-white" />
              </div>
              <div>
                <h3 className="font-semibold text-slate-900 text-lg">Chunking Ayarları</h3>
                <p className="text-sm text-slate-500 mt-0.5">
                  Metin parçalama stratejisi ve boyut ayarları
                </p>
              </div>
            </div>
            <ChevronDown
              className={`w-5 h-5 text-slate-400 transition-transform duration-200 ${
                expandedSection === "chunking" ? 'rotate-180' : ''
              }`}
            />
          </div>
        </div>
        {expandedSection === "chunking" && (
          <div className={sectionContentStyles}>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="space-y-2">
              <Label className="text-sm font-medium text-slate-700">Varsayılan Strateji</Label>
              <Select
                value={formData.default_chunk_strategy}
                onValueChange={(v) => setFormData({ ...formData, default_chunk_strategy: v })}
                disabled={!isOwner}
              >
                <SelectTrigger className="h-11 border-slate-200 focus:border-emerald-300 focus:ring-emerald-200">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="recursive">Recursive</SelectItem>
                  <SelectItem value="semantic">Semantic</SelectItem>
                </SelectContent>
              </Select>
            </div>

            <div className="space-y-2">
              <Label className="text-sm font-medium text-slate-700">Varsayılan Chunk Boyutu</Label>
              <Input
                type="number"
                value={formData.default_chunk_size}
                onChange={(e) => setFormData({ ...formData, default_chunk_size: Number(e.target.value) })}
                min={100}
                max={5000}
                className="h-11 border-slate-200 focus:border-emerald-300 focus:ring-emerald-200"
                disabled={!isOwner}
              />
            </div>

            <div className="space-y-2">
              <Label className="text-sm font-medium text-slate-700">Varsayılan Overlap</Label>
              <Input
                type="number"
                value={formData.default_overlap}
                onChange={(e) => setFormData({ ...formData, default_overlap: Number(e.target.value) })}
                min={0}
                max={500}
                className="h-11 border-slate-200 focus:border-emerald-300 focus:ring-emerald-200"
                disabled={!isOwner}
              />
            </div>

            <div className="space-y-2">
              <Label className="text-sm font-medium text-slate-700">Varsayılan Embedding Model</Label>
              <Select
                value={formData.default_embedding_model}
                onValueChange={(v) => setFormData({ ...formData, default_embedding_model: v })}
                disabled={!isOwner}
              >
                <SelectTrigger className="h-11 border-slate-200 focus:border-emerald-300 focus:ring-emerald-200">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {EMBEDDING_MODELS.map((model) => (
                    <SelectItem key={model.value} value={model.value}>
                      {model.label}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          </div>
          </div>
        )}
      </div>

      {/* Search Settings Section */}
      <div className={sectionCardStyles}>
        <div
          className={`${sectionHeaderStyles} cursor-pointer hover:bg-slate-100/50 transition-colors`}
          onClick={() =>
            setExpandedSection((prev) =>
              prev === "search" ? null : "search"
            )
          }
        >
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-blue-500 to-blue-600 flex items-center justify-center shadow-sm">
                <Search className="w-5 h-5 text-white" />
              </div>
              <div>
                <h3 className="font-semibold text-slate-900 text-lg">Arama Ayarları</h3>
                <p className="text-sm text-slate-500 mt-0.5">
                  Hibrit arama ve sonuç filtreleme ayarları
                </p>
              </div>
            </div>
            <ChevronDown
              className={`w-5 h-5 text-slate-400 transition-transform duration-200 ${
                expandedSection === "search" ? 'rotate-180' : ''
              }`}
            />
          </div>
        </div>
        {expandedSection === "search" && (
          <div className={sectionContentStyles}>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="space-y-2">
              <Label className="text-sm font-medium text-slate-700">
                Hybrid Alpha (0=Kelime, 1=Vektör)
              </Label>
              <Input
                type="number"
                value={formData.search_alpha}
                onChange={(e) => setFormData({ ...formData, search_alpha: Number(e.target.value) })}
                min={0}
                max={1}
                step={0.1}
                className="h-11 border-slate-200 focus:border-blue-300 focus:ring-blue-200"
                disabled={!isOwner}
              />
              <p className="text-xs text-slate-500 bg-slate-50 px-2 py-1.5 rounded">
                0.5 = Dengeli hibrit arama
              </p>
            </div>

            <div className="space-y-2">
              <Label className="text-sm font-medium text-slate-700">Top-K Sonuç Sayısı</Label>
              <Input
                type="number"
                value={formData.search_top_k}
                onChange={(e) => setFormData({ ...formData, search_top_k: Number(e.target.value) })}
                min={1}
                max={20}
                className="h-11 border-slate-200 focus:border-blue-300 focus:ring-blue-200"
                disabled={!isOwner}
              />
              <p className="text-xs text-slate-500 bg-slate-50 px-2 py-1.5 rounded">
                Sohbette kullanılacak chunk sayısı
              </p>
            </div>

            <div className="space-y-2">
              <Label className="text-sm font-medium text-slate-700">Minimum Alaka Skoru</Label>
              <Input
                type="number"
                value={formData.min_relevance_score}
                onChange={(e) => setFormData({ ...formData, min_relevance_score: Number(e.target.value) })}
                min={0}
                max={1}
                step={0.05}
                className="h-11 border-slate-200 focus:border-blue-300 focus:ring-blue-200"
                disabled={!isOwner}
              />
              <p className="text-xs text-slate-500 bg-slate-50 px-2 py-1.5 rounded">
                Bu skorun altındaki sonuçlar filtrelenir (0 = filtre yok, 0.3-0.5 önerilir)
              </p>
            </div>
          </div>
          </div>
        )}
      </div>

      {/* Reranker Settings Section */}
      <div className={sectionCardStyles}>
        <div
          className={`${sectionHeaderStyles} cursor-pointer hover:bg-slate-100/50 transition-colors`}
          onClick={() =>
            setExpandedSection((prev) =>
              prev === "reranker" ? null : "reranker"
            )
          }
        >
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-amber-500 to-amber-600 flex items-center justify-center shadow-sm">
                <Sparkles className="w-5 h-5 text-white" />
              </div>
              <div>
                <h3 className="font-semibold text-slate-900 text-lg">Reranker Ayarları</h3>
                <p className="text-sm text-slate-500 mt-0.5">
                  Arama sonuçlarını yeniden sıralayarak alakalılığı artırın
                </p>
              </div>
            </div>
            <ChevronDown
              className={`w-5 h-5 text-slate-400 transition-transform duration-200 ${
                expandedSection === "reranker" ? 'rotate-180' : ''
              }`}
            />
          </div>
        </div>
        {expandedSection === "reranker" && (
          <div className={sectionContentStyles}>
            {/* Validation Errors */}
            {validationErrors.length > 0 && (
              <div className="mb-6 bg-red-50 border border-red-200 rounded-lg p-4">
                <div className="flex items-start gap-3">
                  <AlertTriangle className="w-5 h-5 text-red-600 mt-0.5 flex-shrink-0" />
                  <div className="flex-1">
                    <p className="font-medium text-red-900 mb-2">Doğrulama Hataları:</p>
                    <ul className="list-disc list-inside space-y-1 text-sm text-red-800">
                      {validationErrors.map((error, index) => (
                        <li key={index}>{error}</li>
                      ))}
                    </ul>
                  </div>
                </div>
              </div>
            )}
            
            <div className="space-y-6">
            {/* Enable Reranker Toggle */}
            <div className="flex items-center justify-between p-4 bg-amber-50 rounded-lg border border-amber-100">
              <div className="flex items-start gap-3 flex-1">
                <Info className="w-5 h-5 text-amber-600 mt-0.5 flex-shrink-0" />
                <div className="flex-1">
                  <Label className="text-sm font-medium text-slate-900 cursor-pointer">
                    Reranker&apos;ı Etkinleştir
                  </Label>
                  <p className="text-xs text-slate-600 mt-1">
                    Reranker, arama sonuçlarını sorguya göre yeniden puanlayarak en alakalı dokümanları üste çıkarır. 
                    Bu özellik arama kalitesini önemli ölçüde artırabilir.
                  </p>
                </div>
              </div>
              <Switch
                checked={formData.enable_reranker}
                onCheckedChange={(checked) => setFormData({ ...formData, enable_reranker: checked })}
                disabled={!isOwner}
                className="ml-4"
              />
            </div>

            {/* Reranker Configuration - Only shown when enabled */}
            {formData.enable_reranker && (
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6 pt-2">
                <div className="space-y-2">
                  <div className="flex items-center gap-2">
                    <Label className="text-sm font-medium text-slate-700">
                      Reranker Provider
                    </Label>
                    <Tooltip>
                      <TooltipTrigger asChild>
                        <Info className="w-4 h-4 text-slate-400 cursor-help" />
                      </TooltipTrigger>
                      <TooltipContent>
                        <p className="max-w-xs">
                          Reranker sağlayıcısı seçin. Cohere çok dilli destek, Alibaba Çince için optimize edilmiştir.
                        </p>
                      </TooltipContent>
                    </Tooltip>
                  </div>
                  <Select
                    value={formData.reranker_provider || ""}
                    onValueChange={(v) => setFormData({ 
                      ...formData, 
                      reranker_provider: v,
                      reranker_model: null // Reset model when provider changes
                    })}
                    disabled={!isOwner}
                  >
                    <SelectTrigger className={`h-11 border-slate-200 focus:border-amber-300 focus:ring-amber-200 ${
                      validationErrors.some(e => e.includes("provider")) ? "border-red-300" : ""
                    }`}>
                      <SelectValue placeholder="Provider seçin" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="cohere">
                        <div className="flex flex-col">
                          <span className="font-medium">Cohere</span>
                          <span className="text-xs text-slate-500">Yüksek kaliteli çok dilli reranking</span>
                        </div>
                      </SelectItem>
                      <SelectItem value="alibaba">
                        <div className="flex flex-col">
                          <span className="font-medium">Alibaba</span>
                          <span className="text-xs text-slate-500">Çince içerik için optimize edilmiş</span>
                        </div>
                      </SelectItem>
                      <SelectItem value="jina">
                        <div className="flex flex-col">
                          <span className="font-medium">Jina</span>
                          <span className="text-xs text-slate-500">Açık kaynak çok dilli reranking</span>
                        </div>
                      </SelectItem>
                      <SelectItem value="bge">
                        <div className="flex flex-col">
                          <span className="font-medium">BGE</span>
                          <span className="text-xs text-slate-500">BAAI çok dilli reranking modeli</span>
                        </div>
                      </SelectItem>
                      <SelectItem value="zeroentropy">
                        <div className="flex flex-col">
                          <span className="font-medium">ZeroEntropy</span>
                          <span className="text-xs text-slate-500">Hosted reranking (zerank-2)</span>
                        </div>
                      </SelectItem>
                      <SelectItem value="voyage">
                        <div className="flex flex-col">
                          <span className="font-medium">VoyageAI</span>
                          <span className="text-xs text-slate-500">Yüksek performanslı reranking</span>
                        </div>
                      </SelectItem>
                    </SelectContent>
                  </Select>
                  <p className="text-xs text-slate-500 bg-slate-50 px-2 py-1.5 rounded">
                    Her provider farklı modeller ve dil desteği sunar
                  </p>
                </div>

                {/* Model Selection - Only shown when provider is selected */}
                {formData.reranker_provider && (
                  <div className="space-y-2">
                    <div className="flex items-center gap-2">
                      <Label className="text-sm font-medium text-slate-700">
                        Reranker Model
                      </Label>
                      <Tooltip>
                        <TooltipTrigger asChild>
                          <Info className="w-4 h-4 text-slate-400 cursor-help" />
                        </TooltipTrigger>
                        <TooltipContent>
                          <p className="max-w-xs">
                            Seçilen provider için uygun reranker modelini seçin. Her model farklı dil desteği ve performans özellikleri sunar.
                          </p>
                        </TooltipContent>
                      </Tooltip>
                    </div>
                    <Select
                      value={formData.reranker_model || ""}
                      onValueChange={(v) => setFormData({ ...formData, reranker_model: v })}
                      disabled={!isOwner}
                    >
                      <SelectTrigger className={`h-11 border-slate-200 focus:border-amber-300 focus:ring-amber-200 ${
                        validationErrors.some(e => e.includes("model")) ? "border-red-300" : ""
                      }`}>
                        <SelectValue placeholder="Model seçin" />
                      </SelectTrigger>
                      <SelectContent>
                        {RERANKER_MODELS[formData.reranker_provider as keyof typeof RERANKER_MODELS]?.map((model) => (
                          <SelectItem key={model.value} value={model.value}>
                            <div className="flex flex-col">
                              <span className="font-medium">{model.label}</span>
                              <span className="text-xs text-slate-500">{model.description}</span>
                            </div>
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                    <p className="text-xs text-slate-500 bg-slate-50 px-2 py-1.5 rounded">
                      Seçilen provider için uygun model
                    </p>
                  </div>
                )}

                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <Label className="text-sm font-medium text-slate-700">
                        Reranker Top-K
                      </Label>
                      <Tooltip>
                        <TooltipTrigger asChild>
                          <Info className="w-4 h-4 text-slate-400 cursor-help" />
                        </TooltipTrigger>
                        <TooltipContent>
                          <p className="max-w-xs">
                            İlk aramadan kaç sonuç alınıp yeniden sıralanacağını belirler. Daha yüksek değerler daha fazla sonuç arasından seçim yapar ancak daha yavaş olabilir. Önerilen: 10-15 arası.
                          </p>
                        </TooltipContent>
                      </Tooltip>
                    </div>
                    <span className="text-sm font-semibold text-amber-600 bg-amber-50 px-3 py-1 rounded-full">
                      {formData.reranker_top_k}
                    </span>
                  </div>
                  <Slider
                    value={[formData.reranker_top_k]}
                    onValueChange={(value) => setFormData({ ...formData, reranker_top_k: value[0] })}
                    min={5}
                    max={20}
                    step={1}
                    disabled={!isOwner}
                    className={`py-4 ${
                      validationErrors.some(e => e.includes("Top-K")) ? "border-red-300" : ""
                    }`}
                  />
                  <div className="flex justify-between text-xs text-slate-500">
                    <span>5</span>
                    <span>20</span>
                  </div>
                  <p className="text-xs text-slate-500 bg-slate-50 px-2 py-1.5 rounded">
                    İlk aramadan kaç sonuç alınıp yeniden sıralanacak. Daha yüksek değerler daha fazla sonuç arasından seçim yapar ancak daha yavaş olabilir.
                  </p>
                </div>
              </div>
            )}

            {/* Info Box */}
            <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
              <div className="flex items-start gap-3">
                <Info className="w-5 h-5 text-blue-600 mt-0.5 flex-shrink-0" />
                <div className="text-sm text-blue-900">
                  <p className="font-medium mb-1">Reranker Nasıl Çalışır?</p>
                  <p className="text-blue-800">
                    Reranker, hibrit aramadan gelen sonuçları alır ve her birini sorguya göre yeniden puanlar. 
                    Bu sayede semantik olarak daha alakalı dokümanlar üst sıralara çıkar. 
                    Reranker Top-K değeri, ilk aramadan kaç sonuç alınacağını belirler.
                  </p>
                </div>
              </div>
            </div>
          </div>
          </div>
        )}
      </div>

      {/* Vector Store Settings Section (EXPERIMENTAL) */}
      <div className={sectionCardStyles}>
        <div
          className={`${sectionHeaderStyles} cursor-pointer hover:bg-slate-100/50 transition-colors`}
          onClick={() =>
            setExpandedSection((prev) => (prev === "vector_store" ? null : "vector_store"))
          }
        >
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-cyan-500 to-cyan-600 flex items-center justify-center shadow-sm">
                <Settings2 className="w-5 h-5 text-white" />
              </div>
              <div>
                <h3 className="font-semibold text-slate-900 text-lg flex items-center gap-2">
                  Vector Database
                  <span className="text-xs font-normal bg-orange-100 text-orange-700 px-2 py-0.5 rounded">
                    EXPERIMENTAL
                  </span>
                </h3>
                <p className="text-sm text-slate-500 mt-0.5">
                  Vektör veritabanı seçimi (Benchmark için)
                </p>
              </div>
            </div>
            <ChevronDown
              className={`w-5 h-5 text-slate-400 transition-transform duration-200 ${
                expandedSection === "vector_store" ? 'rotate-180' : ''
              }`}
            />
          </div>
        </div>
        {expandedSection === "vector_store" && (
          <div className={sectionContentStyles}>
            <div className="space-y-6">
              {/* Vector Store Selection */}
              <div className="space-y-2">
                <div className="flex items-center gap-2">
                  <Label className="text-sm font-medium text-slate-700">
                    Vector Database
                  </Label>
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <Info className="w-4 h-4 text-slate-400 cursor-help" />
                    </TooltipTrigger>
                    <TooltipContent>
                      <p className="max-w-xs">
                        Weaviate: Hybrid search (vector + keyword).
                        ChromaDB: Pure vector search. Karşılaştırma için kullanın.
                      </p>
                    </TooltipContent>
                  </Tooltip>
                </div>
                <Select
                  value={formData.vector_store}
                  onValueChange={(v) => setFormData({ ...formData, vector_store: v })}
                  disabled={!isOwner}
                >
                  <SelectTrigger className="h-11 border-slate-200 focus:border-cyan-300 focus:ring-cyan-200">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="weaviate">
                      <div className="flex flex-col">
                        <span className="font-medium">Weaviate</span>
                        <span className="text-xs text-slate-500">Hybrid search (vector + keyword)</span>
                      </div>
                    </SelectItem>
                  </SelectContent>
                </Select>
                <p className="text-xs text-slate-500 bg-slate-50 px-2 py-1.5 rounded">
                  Weaviate: Hybrid search (vector + BM25 keyword)
                </p>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Direct LLM Mode Section */}
      <div className={sectionCardStyles}>
        <div
          className={`${sectionHeaderStyles} cursor-pointer hover:bg-slate-100/50 transition-colors`}
          onClick={() =>
            setExpandedSection((prev) => (prev === "direct_llm" ? null : "direct_llm"))
          }
        >
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className={`w-10 h-10 rounded-xl flex items-center justify-center shadow-sm ${
                formData.enable_direct_llm
                  ? 'bg-gradient-to-br from-amber-500 to-orange-600'
                  : 'bg-gradient-to-br from-slate-400 to-slate-500'
              }`}>
                <Zap className="w-5 h-5 text-white" />
              </div>
              <div>
                <h3 className="font-semibold text-slate-900 text-lg">Direct LLM Modu</h3>
                <p className="text-sm text-slate-500 mt-0.5">
                  RAG pipeline&apos;ı devre dışı bırakarak doğrudan LLM yanıtı al
                </p>
              </div>
            </div>
            <div className="flex items-center gap-2">
              {formData.enable_direct_llm && (
                <span className="text-xs font-medium px-2 py-1 rounded-full bg-amber-100 text-amber-700">
                  Aktif
                </span>
              )}
              <ChevronDown
                className={`w-5 h-5 text-slate-400 transition-transform duration-200 ${
                  expandedSection === "direct_llm" ? 'rotate-180' : ''
                }`}
              />
            </div>
          </div>
        </div>
        {expandedSection === "direct_llm" && (
          <div className={sectionContentStyles}>
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <div className="space-y-1">
                  <Label className="text-sm font-medium text-slate-700">Direct LLM Modunu Etkinleştir</Label>
                  <p className="text-xs text-slate-500">
                    Aktif olduğunda sistem promptu, embedding ve doküman araması devre dışı kalır.
                    LLM soruları kendi bilgisiyle yanıtlar.
                  </p>
                </div>
                <Switch
                  checked={formData.enable_direct_llm}
                  onCheckedChange={(checked) => setFormData({ ...formData, enable_direct_llm: checked })}
                  disabled={!isOwner}
                  className="ml-4"
                />
              </div>
              {formData.enable_direct_llm && (
                <div className="p-3 rounded-lg bg-amber-50 border border-amber-200">
                  <div className="flex items-start gap-2">
                    <AlertTriangle className="w-4 h-4 text-amber-600 mt-0.5 flex-shrink-0" />
                    <div className="text-xs text-amber-700 space-y-1">
                      <p className="font-medium">Direct LLM modu aktif</p>
                      <p>Sohbet ve testlerde doküman bağlamı kullanılmayacak. LLM yalnızca kendi eğitim verisine dayanarak yanıt verecek. Bu mod RAG vs yalın LLM karşılaştırması için idealdir.</p>
                    </div>
                  </div>
                </div>
              )}
            </div>
          </div>
        )}
      </div>

      {/* PII Filter Section */}
      <div className={sectionCardStyles}>
        <div
          className={`${sectionHeaderStyles} cursor-pointer hover:bg-slate-100/50 transition-colors`}
          onClick={() =>
            setExpandedSection((prev) => (prev === "pii_filter" ? null : "pii_filter"))
          }
        >
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className={`w-10 h-10 rounded-xl flex items-center justify-center shadow-sm ${
                formData.enable_pii_filter
                  ? 'bg-gradient-to-br from-rose-500 to-pink-600'
                  : 'bg-gradient-to-br from-slate-400 to-slate-500'
              }`}>
                <Shield className="w-5 h-5 text-white" />
              </div>
              <div>
                <h3 className="font-semibold text-slate-900 text-lg">Kişisel Bilgi Filtresi</h3>
                <p className="text-sm text-slate-500 mt-0.5">
                  Öğrenci mesajlarındaki kişisel bilgileri otomatik tespit et
                </p>
              </div>
            </div>
            <div className="flex items-center gap-2">
              {formData.enable_pii_filter && (
                <span className="text-xs font-medium px-2 py-1 rounded-full bg-rose-100 text-rose-700">
                  Aktif
                </span>
              )}
              <ChevronDown
                className={`w-5 h-5 text-slate-400 transition-transform duration-200 ${
                  expandedSection === "pii_filter" ? 'rotate-180' : ''
                }`}
              />
            </div>
          </div>
        </div>
        {expandedSection === "pii_filter" && (
          <div className={sectionContentStyles}>
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <div className="space-y-1">
                  <Label className="text-sm font-medium text-slate-700">Kişisel Bilgi Filtresini Etkinleştir</Label>
                  <p className="text-xs text-slate-500">
                    Aktif olduğunda öğrenci mesajları LLM&apos;e gönderilmeden önce kişisel bilgi
                    içerip içermediği kontrol edilir. Kişisel bilgi tespit edilirse mesaj engellenir.
                  </p>
                </div>
                <Switch
                  checked={formData.enable_pii_filter}
                  onCheckedChange={(checked) => setFormData({ ...formData, enable_pii_filter: checked })}
                  disabled={!isOwner}
                  className="ml-4"
                />
              </div>
              {formData.enable_pii_filter && (
                <div className="p-3 rounded-lg bg-rose-50 border border-rose-200">
                  <div className="flex items-start gap-2">
                    <Shield className="w-4 h-4 text-rose-600 mt-0.5 flex-shrink-0" />
                    <div className="text-xs text-rose-700 space-y-1">
                      <p className="font-medium">Kişisel bilgi filtresi aktif</p>
                      <p>
                        TC kimlik no, telefon, adres, e-posta, şifre, kredi kartı gibi kişisel bilgiler
                        embedding tabanlı zero-shot classification ile tespit edilir. Dersin embedding modeli
                        ({formData.default_embedding_model}) kullanılır.
                      </p>
                    </div>
                  </div>
                </div>
              )}
            </div>
          </div>
        )}
      </div>

      {/* LLM Settings Section */}
      <div className={sectionCardStyles}>
        <div
          className={`${sectionHeaderStyles} cursor-pointer hover:bg-slate-100/50 transition-colors`}
          onClick={() =>
            setExpandedSection((prev) => (prev === "llm" ? null : "llm"))
          }
        >
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-purple-500 to-purple-600 flex items-center justify-center shadow-sm">
                <Settings2 className="w-5 h-5 text-white" />
              </div>
              <div>
                <h3 className="font-semibold text-slate-900 text-lg">LLM Ayarları</h3>
                <p className="text-sm text-slate-500 mt-0.5">
                  Dil modeli sağlayıcısı ve parametreleri
                </p>
              </div>
            </div>
            {isOwner && (
              <Button
                variant="outline"
                size="sm"
                onClick={(e) => {
                  e.stopPropagation();
                  setShowModelManager(true);
                }}
                className="border-purple-200 text-purple-600 hover:bg-purple-50"
              >
                <Settings className="w-4 h-4 mr-2" />
                Model Yönetimi
              </Button>
            )}
            <ChevronDown
              className={`w-5 h-5 text-slate-400 transition-transform duration-200 ${
                expandedSection === "llm" ? 'rotate-180' : ''
              }`}
            />
          </div>
        </div>
        {expandedSection === "llm" && (
          <div className={sectionContentStyles}>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="space-y-2">
              <Label className="text-sm font-medium text-slate-700">LLM Provider</Label>
              <Select
                value={formData.llm_provider}
                onValueChange={(v) => setFormData({ ...formData, llm_provider: v, llm_model: "" })}
                disabled={!isOwner}
              >
                <SelectTrigger className="h-11 border-slate-200 focus:border-purple-300 focus:ring-purple-200">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="openrouter">OpenRouter</SelectItem>
                  <SelectItem value="claudegg">Claude.gg</SelectItem>
                  <SelectItem value="apiclaudegg">API Claude.gg</SelectItem>
                  <SelectItem value="groq">Groq</SelectItem>
                  <SelectItem value="openai">OpenAI</SelectItem>
                  <SelectItem value="deepseek">DeepSeek</SelectItem>
                  <SelectItem value="cohere">Cohere</SelectItem>
                  <SelectItem value="alibaba">Alibaba</SelectItem>
                  <SelectItem value="zai">Z.ai</SelectItem>
                </SelectContent>
              </Select>
            </div>

            <div className="space-y-2">
              <Label className="text-sm font-medium text-slate-700">Model</Label>
              <Select
                value={formData.llm_model}
                onValueChange={(v) => setFormData({ ...formData, llm_model: v })}
                disabled={!isOwner || availableModels.length === 0}
              >
                <SelectTrigger className="h-11 border-slate-200 focus:border-purple-300 focus:ring-purple-200">
                  <SelectValue placeholder="Model seçin" />
                </SelectTrigger>
                <SelectContent>
                  {availableModels.map((model) => (
                    <SelectItem key={model} value={model}>
                      {model}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            <div className="space-y-2">
              <Label className="text-sm font-medium text-slate-700">Temperature</Label>
              <Input
                type="number"
                value={formData.llm_temperature}
                onChange={(e) => setFormData({ ...formData, llm_temperature: Number(e.target.value) })}
                min={0}
                max={2}
                step={0.1}
                className="h-11 border-slate-200 focus:border-purple-300 focus:ring-purple-200"
                disabled={!isOwner}
              />
              <p className="text-xs text-slate-500 bg-slate-50 px-2 py-1.5 rounded">
                Yanıt çeşitliliği (0=deterministik, 2=yaratıcı)
              </p>
            </div>

            <div className="space-y-2">
              <Label className="text-sm font-medium text-slate-700">Max Tokens</Label>
              <Input
                type="number"
                value={formData.llm_max_tokens}
                onChange={(e) => setFormData({ ...formData, llm_max_tokens: Number(e.target.value) })}
                min={100}
                max={4000}
                step={100}
                className="h-11 border-slate-200 focus:border-purple-300 focus:ring-purple-200"
                disabled={!isOwner}
              />
              <p className="text-xs text-slate-500 bg-slate-50 px-2 py-1.5 rounded">
                Maksimum yanıt uzunluğu
              </p>
            </div>
            </div>
          </div>
        )}
      </div>

      {/* Save Button */}
      {isOwner && (
        <div className={`${sectionCardStyles} bg-gradient-to-r from-slate-50 to-white`}>
          <div className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <h4 className="font-semibold text-slate-900">Ayarları Kaydet</h4>
                <p className="text-sm text-slate-500 mt-1">
                  Değişiklikler tüm ders için geçerli olacak
                </p>
              </div>
              <Button 
                onClick={handleSave} 
                disabled={isSaving} 
                size="lg"
                className="bg-gradient-to-r from-indigo-600 to-indigo-700 hover:from-indigo-700 hover:to-indigo-800 shadow-md hover:shadow-lg transition-all duration-200"
              >
                {isSaving ? (
                  <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                ) : (
                  <Save className="w-4 h-4 mr-2" />
                )}
                Kaydet
              </Button>
            </div>
          </div>
        </div>
      )}

      {/* Danger Zone - Delete Course */}
      {isOwner && (
        <div className="bg-white rounded-xl border-2 border-red-200 shadow-sm">
          <div className="px-6 py-5 border-b border-red-100 bg-gradient-to-r from-red-50/50 to-transparent">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-red-500 to-red-600 flex items-center justify-center shadow-sm">
                <AlertTriangle className="w-5 h-5 text-white" />
              </div>
              <div>
                <h3 className="font-semibold text-red-900 text-lg">Tehlikeli Bölge</h3>
                <p className="text-sm text-red-600 mt-0.5">
                  Bu işlemler geri alınamaz
                </p>
              </div>
            </div>
          </div>
          <div className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <h4 className="font-semibold text-slate-900">Dersi Sil</h4>
                <p className="text-sm text-slate-500 mt-1">
                  Bu ders ve tüm içerikleri (dokümanlar, chunklar, vektörler) kalıcı olarak silinecek.
                </p>
              </div>
              <Button 
                onClick={() => setShowDeleteModal(true)}
                variant="outline"
                className="border-red-300 text-red-600 hover:bg-red-50 hover:text-red-700 hover:border-red-400"
              >
                <Trash2 className="w-4 h-4 mr-2" />
                Dersi Sil
              </Button>
            </div>
          </div>
        </div>
      )}

      {/* Model Manager Modal */}
      {showModelManager && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4">
          <div className="bg-white rounded-xl max-w-2xl w-full shadow-xl max-h-[90vh] overflow-hidden flex flex-col">
            <div className="flex items-center justify-between px-6 py-4 border-b border-slate-200">
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 bg-purple-100 rounded-full flex items-center justify-center">
                  <Settings className="w-5 h-5 text-purple-600" />
                </div>
                <h3 className="font-semibold text-slate-900">Model Yönetimi</h3>
              </div>
              <button
                onClick={() => setShowModelManager(false)}
                className="p-2 hover:bg-slate-100 rounded-lg transition-colors"
              >
                <X className="w-5 h-5 text-slate-500" />
              </button>
            </div>
            
            <div className="p-6 overflow-y-auto flex-1">
              {/* Add New Model Form */}
              <div className="bg-slate-50 rounded-lg p-4 mb-6">
                <h4 className="font-medium text-slate-900 mb-4 flex items-center gap-2">
                  <Plus className="w-4 h-4" />
                  Yeni Model Ekle
                </h4>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  <div className="space-y-2">
                    <Label className="text-sm text-slate-600">Provider</Label>
                    <Select
                      value={newModel.provider}
                      onValueChange={(v) => setNewModel({ ...newModel, provider: v })}
                    >
                      <SelectTrigger className="h-10">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="openrouter">OpenRouter</SelectItem>
                        <SelectItem value="claudegg">Claude.gg</SelectItem>
                        <SelectItem value="apiclaudegg">API Claude.gg</SelectItem>
                        <SelectItem value="groq">Groq</SelectItem>
                        <SelectItem value="openai">OpenAI</SelectItem>
                        <SelectItem value="deepseek">DeepSeek</SelectItem>
                        <SelectItem value="cohere">Cohere</SelectItem>
                        <SelectItem value="alibaba">Alibaba</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                  <div className="space-y-2">
                    <Label className="text-sm text-slate-600">Model ID</Label>
                    <Input
                      value={newModel.model_id}
                      onChange={(e) => setNewModel({ ...newModel, model_id: e.target.value })}
                      placeholder="örn: anthropic/claude-3.5-haiku"
                      className="h-10"
                    />
                  </div>
                  <div className="space-y-2">
                    <Label className="text-sm text-slate-600">Görünen Ad</Label>
                    <Input
                      value={newModel.display_name}
                      onChange={(e) => setNewModel({ ...newModel, display_name: e.target.value })}
                      placeholder="örn: Claude 3.5 Haiku"
                      className="h-10"
                    />
                  </div>
                </div>
                <div className="mt-4 flex justify-end">
                  <Button
                    onClick={handleAddModel}
                    disabled={isAddingModel || !newModel.model_id.trim() || !newModel.display_name.trim()}
                    className="bg-purple-600 hover:bg-purple-700"
                  >
                    {isAddingModel ? (
                      <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                    ) : (
                      <Plus className="w-4 h-4 mr-2" />
                    )}
                    Model Ekle
                  </Button>
                </div>
              </div>

              {/* Custom Models List */}
              <div>
                <h4 className="font-medium text-slate-900 mb-4">Özel Modeller</h4>
                {isLoadingModels ? (
                  <div className="flex justify-center py-8">
                    <Loader2 className="w-6 h-6 text-slate-400 animate-spin" />
                  </div>
                ) : customModels.length === 0 ? (
                  <div className="text-center py-8 text-slate-500">
                    <Settings className="w-12 h-12 mx-auto mb-3 text-slate-300" />
                    <p>Henüz özel model eklenmemiş</p>
                    <p className="text-sm mt-1">Yukarıdaki formu kullanarak yeni model ekleyebilirsiniz</p>
                  </div>
                ) : (
                  <div className="space-y-2">
                    {customModels.map((model) => (
                      <div
                        key={model.id}
                        className="flex items-center justify-between p-3 bg-white border border-slate-200 rounded-lg hover:border-slate-300 transition-colors"
                      >
                        <div className="flex-1">
                          <div className="flex items-center gap-2">
                            <span className="font-medium text-slate-900">{model.display_name}</span>
                            <span className="text-xs px-2 py-0.5 bg-purple-100 text-purple-700 rounded-full">
                              {model.provider}
                            </span>
                          </div>
                          <p className="text-sm text-slate-500 mt-0.5 font-mono">{model.model_id}</p>
                        </div>
                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={() => handleDeleteModel(model.id)}
                          className="text-red-500 hover:text-red-700 hover:bg-red-50"
                        >
                          <Trash2 className="w-4 h-4" />
                        </Button>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>

            <div className="px-6 py-4 border-t border-slate-200 bg-slate-50 flex justify-end rounded-b-xl">
              <Button variant="outline" onClick={() => setShowModelManager(false)}>
                Kapat
              </Button>
            </div>
          </div>
        </div>
      )}

      {/* Delete Confirmation Modal */}
      {showDeleteModal && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4">
          <div className="bg-white rounded-xl max-w-md w-full shadow-xl">
            <div className="flex items-center justify-between px-6 py-4 border-b border-slate-200">
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 bg-red-100 rounded-full flex items-center justify-center">
                  <AlertTriangle className="w-5 h-5 text-red-600" />
                </div>
                <h3 className="font-semibold text-slate-900">Dersi Sil</h3>
              </div>
              <button
                onClick={() => {
                  setShowDeleteModal(false);
                  setDeleteConfirmText("");
                }}
                className="p-2 hover:bg-slate-100 rounded-lg transition-colors"
              >
                <X className="w-5 h-5 text-slate-500" />
              </button>
            </div>
            <div className="p-6 space-y-4">
              <div className="bg-red-50 border border-red-200 rounded-lg p-4">
                <p className="text-sm text-red-800">
                  <strong>Uyarı:</strong> Bu işlem geri alınamaz. Ders ve tüm içerikleri kalıcı olarak silinecektir.
                </p>
              </div>
              <div className="space-y-2">
                <Label className="text-sm text-slate-700">
                  Onaylamak için ders adını yazın: <strong className="text-red-600">{courseName}</strong>
                </Label>
                <Input
                  value={deleteConfirmText}
                  onChange={(e) => setDeleteConfirmText(e.target.value)}
                  placeholder="Ders adını yazın..."
                  className="border-slate-200"
                />
              </div>
            </div>
            <div className="px-6 py-4 border-t border-slate-200 bg-slate-50 flex justify-end gap-3 rounded-b-xl">
              <Button
                variant="outline"
                onClick={() => {
                  setShowDeleteModal(false);
                  setDeleteConfirmText("");
                }}
              >
                İptal
              </Button>
              <Button
                onClick={handleDeleteCourse}
                disabled={deleteConfirmText !== courseName || isDeleting}
                className="bg-red-600 hover:bg-red-700 text-white"
              >
                {isDeleting ? (
                  <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                ) : (
                  <Trash2 className="w-4 h-4 mr-2" />
                )}
                Kalıcı Olarak Sil
              </Button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
