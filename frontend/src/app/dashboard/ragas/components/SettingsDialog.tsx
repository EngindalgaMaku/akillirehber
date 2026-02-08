"use client";

import { useState, useEffect } from "react";
import { RagasSettings, RagasProvider, api } from "@/lib/api";
import { useModelProviders } from "@/hooks/useModelProviders";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Settings, Save, Loader2, Zap, Activity } from "lucide-react";
import { toast } from "sonner";

interface SettingsDialogProps {
  ragasSettings: RagasSettings | null;
  ragasProviders: RagasProvider[];
  onSettingsUpdate: () => void;
  selectedEmbeddingModel: string;
  onEmbeddingModelChange: (model: string) => void;
}

export function SettingsDialog({ ragasSettings, ragasProviders, onSettingsUpdate, selectedEmbeddingModel, onEmbeddingModelChange }: SettingsDialogProps) {
  const { getLLMProviders, getLLMModels, getEmbeddingModels, isLoading: providersLoading } = useModelProviders();
  const [isOpen, setIsOpen] = useState(false);
  const [selectedProvider, setSelectedProvider] = useState<string>("");
  const [selectedModel, setSelectedModel] = useState<string>("");
  const [localEmbeddingModel, setLocalEmbeddingModel] = useState<string>("");
  const [isSaving, setIsSaving] = useState(false);

  useEffect(() => {
    if (ragasSettings) {
      setSelectedProvider(ragasSettings.current_provider || "");
      setSelectedModel(ragasSettings.current_model || "");
    }
  }, [ragasSettings]);

  useEffect(() => {
    setLocalEmbeddingModel(selectedEmbeddingModel);
  }, [selectedEmbeddingModel]);

  const handleSave = async () => {
    setIsSaving(true);
    try {
      await api.updateRagasSettings({
        provider: selectedProvider || "",
        model: selectedModel || ""
      });
      onEmbeddingModelChange(localEmbeddingModel);
      toast.success("RAGAS ayarları güncellendi");
      setIsOpen(false);
      onSettingsUpdate();
    } catch {
      toast.error("Ayarlar kaydedilirken hata oluştu");
    } finally {
      setIsSaving(false);
    }
  };

  return (
    <Dialog open={isOpen} onOpenChange={setIsOpen}>
      <DialogTrigger asChild>
        <Button variant="secondary" size="sm" className="bg-white/20 hover:bg-white/30 text-white border-0 backdrop-blur-sm h-10">
          <Settings className="w-4 h-4 mr-2" />
          Ayarlar
          {ragasSettings?.is_free && (
            <span className="ml-2 flex items-center gap-1 px-2 py-0.5 bg-green-500/30 rounded-full text-xs">
              <Zap className="w-3 h-3" /> Ücretsiz
            </span>
          )}
        </Button>
      </DialogTrigger>
      <DialogContent className="sm:max-w-lg">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <Settings className="w-5 h-5 text-purple-600" />
            RAGAS Değerlendirme Ayarları
          </DialogTitle>
          <DialogDescription>
            RAGAS değerlendirmesi için kullanılacak LLM ve embedding modelini seçin.
          </DialogDescription>
        </DialogHeader>
        <div className="space-y-4 py-4">
          {/* Embedding Model Selection */}
          <div className="space-y-2">
            <Label className="text-sm font-medium">RAGAS Embedding Model</Label>
            <p className="text-xs text-slate-500">answer_relevancy ve answer_correctness metrikleri için kullanılır</p>
            <Select
              value={localEmbeddingModel || "course_default"}
              onValueChange={(v) => setLocalEmbeddingModel(v === "course_default" ? "" : v)}
            >
              <SelectTrigger className="h-11">
                <SelectValue placeholder="Ders embedding modeli (varsayılan)" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="course_default">
                  <span className="flex items-center gap-2">
                    <Zap className="w-4 h-4 text-amber-500" />
                    Ders Embedding Modeli (Varsayılan)
                  </span>
                </SelectItem>
                {getEmbeddingModels().map((model) => {
                  const parts = model.split('/');
                  const provider = parts[0];
                  const modelName = parts.slice(1).join('/');
                  return (
                    <SelectItem key={model} value={model}>
                      {provider.charAt(0).toUpperCase() + provider.slice(1)} / {modelName}
                    </SelectItem>
                  );
                })}
              </SelectContent>
            </Select>
          </div>

          {/* LLM Provider Selection */}
          <div className="space-y-2">
            <Label className="text-sm font-medium">LLM Provider</Label>
            <Select 
              value={selectedProvider || "auto"} 
              onValueChange={(v) => {
                const newProvider = v === "auto" ? "" : v;
                setSelectedProvider(newProvider);
                setSelectedModel("");
              }}
            >
              <SelectTrigger className="h-11">
                <SelectValue placeholder="Otomatik seç" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="auto">
                  <span className="flex items-center gap-2">
                    <Zap className="w-4 h-4 text-amber-500" />
                    Otomatik Seçim
                  </span>
                </SelectItem>
                {getLLMProviders().map((provider) => (
                  <SelectItem key={provider} value={provider}>
                    {provider}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          {selectedProvider && selectedProvider !== "auto" && (
            <div className="space-y-2">
              <Label className="text-sm font-medium">LLM Model</Label>
              <Select 
                value={selectedModel || ""} 
                onValueChange={setSelectedModel}
                disabled={!selectedProvider || providersLoading}
              >
                <SelectTrigger className="h-11">
                  <SelectValue placeholder={selectedProvider ? "Model seçin" : "Önce provider seçin"} />
                </SelectTrigger>
                <SelectContent>
                  {getLLMModels(selectedProvider).map((model) => (
                    <SelectItem key={model} value={model}>
                      {model}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          )}

          {(selectedProvider || selectedModel || localEmbeddingModel) && (
            <div className="bg-gradient-to-r from-slate-50 to-slate-100 rounded-xl p-4 border border-slate-200">
              <p className="font-semibold text-slate-700 mb-2 flex items-center gap-2">
                <Activity className="w-4 h-4" />
                Seçili Ayarlar
              </p>
              {localEmbeddingModel && (
                <p className="text-sm text-slate-600">
                  RAGAS Embedding: <span className="font-medium text-slate-900">{localEmbeddingModel}</span>
                </p>
              )}
              {selectedProvider && (
                <p className="text-sm text-slate-600 mt-1">
                  Provider: <span className="font-medium text-slate-900">{selectedProvider === "auto" ? "Otomatik" : selectedProvider}</span>
                </p>
              )}
              {selectedModel && (
                <p className="text-sm text-slate-600 mt-1">
                  Model: <span className="font-medium text-slate-900">{selectedModel}</span>
                </p>
              )}
            </div>
          )}

          {ragasSettings && (
            <div className="bg-gradient-to-r from-purple-50 to-indigo-50 rounded-xl p-4 border border-purple-200">
              <p className="font-semibold text-purple-700 mb-2 flex items-center gap-2">
                <Activity className="w-4 h-4" />
                Mevcut Ayarlar
              </p>
              <div className="grid grid-cols-2 gap-3 text-sm">
                <div>
                  <p className="text-purple-600">Sağlayıcı</p>
                  <p className="font-medium text-purple-900">
                    {ragasSettings.current_provider || "Otomatik"}
                  </p>
                </div>
                <div>
                  <p className="text-purple-600">Model</p>
                  <p className="font-medium text-purple-900">
                    {ragasSettings.current_model || "Varsayılan"}
                  </p>
                </div>
              </div>
              {ragasSettings.is_free && (
                <div className="mt-2 flex items-center gap-1 text-xs text-green-700">
                  <Zap className="w-3 h-3" />
                  <span>Ücretsiz model kullanılıyor</span>
                </div>
              )}
            </div>
          )}
        </div>
        <DialogFooter>
          <Button variant="outline" onClick={() => setIsOpen(false)}>
            İptal
          </Button>
          <Button onClick={handleSave} disabled={isSaving} className="bg-purple-600 hover:bg-purple-700">
            {isSaving ? (
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
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
