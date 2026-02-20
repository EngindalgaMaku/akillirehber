"use client";

import { useState } from "react";
import { api, ChunkResponse, streamChunking, ChunkingProgressEvent } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Input } from "@/components/ui/input";
import { Switch } from "@/components/ui/switch";
import { Progress } from "@/components/ui/progress";
import { toast } from "sonner";
import { Play, Copy, Check, ChevronDown, ChevronUp, Scissors, Brain, Zap, Globe, AlertCircle, Loader2 } from "lucide-react";
import { PageHeader } from "@/components/ui/page-header";

type ChunkStrategy = "fixed_size" | "recursive" | "sentence" | "semantic";

const COLORS = [
  "bg-blue-100 border-blue-300",
  "bg-green-100 border-green-300",
  "bg-yellow-100 border-yellow-300",
  "bg-purple-100 border-purple-300",
  "bg-pink-100 border-pink-300",
  "bg-orange-100 border-orange-300",
  "bg-teal-100 border-teal-300",
  "bg-red-100 border-red-300",
];

export default function ChunkingPage() {
  const [text, setText] = useState("");
  const [strategy, setStrategy] = useState<ChunkStrategy>("recursive");
  const [chunkSize, setChunkSize] = useState(500);
  const [overlap, setOverlap] = useState(50);
  const [similarityThreshold, setSimilarityThreshold] = useState(0.5);
  const [minChunkSize, setMinChunkSize] = useState(150);
  const [maxChunkSize, setMaxChunkSize] = useState(2000);
  const [bufferSize, setBufferSize] = useState(1);
  const [enableQaDetection, setEnableQaDetection] = useState(true);
  const [enableAdaptiveThreshold, setEnableAdaptiveThreshold] = useState(true);
  const [enableCache, setEnableCache] = useState(true);
  const [includeQualityMetrics, setIncludeQualityMetrics] = useState(true);
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState<ChunkResponse | null>(null);
  const [copiedIndex, setCopiedIndex] = useState<number | null>(null);
  const [expandedChunks, setExpandedChunks] = useState<Set<number>>(new Set());
  const [viewMode, setViewMode] = useState<"list" | "visual">("visual");
  
  // Progress tracking state
  const [progress, setProgress] = useState(0);
  const [progressStage, setProgressStage] = useState("");
  const [progressMessage, setProgressMessage] = useState("");
  const [progressDetails, setProgressDetails] = useState<Record<string, unknown> | null>(null);

  const handleChunk = async () => {
    if (!text.trim()) { toast.error("Metin girin"); return; }
    setIsLoading(true);
    setProgress(0);
    setProgressStage("");
    setProgressMessage("");
    setProgressDetails(null);
    setResult(null);
    
    try {
      const requestData: Record<string, unknown> = {
        text,
        strategy,
        chunk_size: chunkSize,
        overlap
      };
      
      if (strategy === "semantic") {
        requestData.similarity_threshold = similarityThreshold;
        requestData.min_chunk_size = minChunkSize;
        requestData.max_chunk_size = maxChunkSize;
        requestData.buffer_size = bufferSize;
        requestData.enable_qa_detection = enableQaDetection;
        requestData.enable_adaptive_threshold = enableAdaptiveThreshold;
        requestData.enable_cache = enableCache;
        requestData.include_quality_metrics = includeQualityMetrics;
        
        // Use streaming for semantic chunking
        const response = await streamChunking(
          requestData as Parameters<typeof streamChunking>[0],
          (event: ChunkingProgressEvent) => {
            setProgress(event.progress);
            setProgressStage(event.stage);
            setProgressMessage(event.message);
            if (event.details) {
              setProgressDetails(event.details);
            }
          }
        );
        setResult(response);
        setExpandedChunks(new Set());
        toast.success(`${response.chunks.length} parça oluşturuldu`);
      } else {
        // Use regular endpoint for non-semantic strategies
        const response = await api.chunk(requestData as Parameters<typeof api.chunk>[0]);
        setResult(response);
        setExpandedChunks(new Set());
        toast.success(`${response.chunks.length} parça oluşturuldu`);
      }
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "Hata oluştu");
    } finally { 
      setIsLoading(false);
      setProgress(0);
      setProgressStage("");
      setProgressMessage("");
    }
  };

  const copyChunk = (content: string, index: number) => {
    navigator.clipboard.writeText(content);
    setCopiedIndex(index);
    setTimeout(() => setCopiedIndex(null), 2000);
  };

  const toggleExpand = (index: number) => {
    const newSet = new Set(expandedChunks);
    if (newSet.has(index)) newSet.delete(index);
    else newSet.add(index);
    setExpandedChunks(newSet);
  };

  const expandAll = () => {
    if (result) setExpandedChunks(new Set(result.chunks.map((_, i) => i)));
  };

  const collapseAll = () => setExpandedChunks(new Set());

  const getLanguageLabel = (lang: string) => {
    const labels: Record<string, string> = {
      'tr': 'Türkçe',
      'en': 'İngilizce',
      'mixed': 'Karışık',
      'unknown': 'Bilinmiyor'
    };
    return labels[lang] || lang;
  };

  return (
    <div>
      <PageHeader
        icon={Scissors}
        title="Chunking Test"
        description="Metin parçalama stratejilerini test edin ve görselleştirin"
        iconColor="text-amber-600"
        iconBg="bg-amber-100"
      />

      <div className="grid gap-6 lg:grid-cols-5">
        {/* Input Panel */}
        <div className="lg:col-span-2 bg-white rounded-lg border border-slate-200 p-6">
          <h2 className="font-medium text-slate-900 mb-4">Ayarlar</h2>
          <div className="space-y-4">
            <div>
              <Label className="text-slate-700">Metin</Label>
              <Textarea
                placeholder="Metni buraya yapıştırın..."
                value={text}
                onChange={(e) => setText(e.target.value)}
                className="mt-1.5 min-h-[150px] resize-none text-sm"
              />
              <p className="text-xs text-slate-400 mt-1">{text.length.toLocaleString()} karakter</p>
            </div>
            <div>
              <Label className="text-slate-700">Strateji</Label>
              <Select value={strategy} onValueChange={(v) => setStrategy(v as ChunkStrategy)}>
                <SelectTrigger className="mt-1.5"><SelectValue /></SelectTrigger>
                <SelectContent>
                  <SelectItem value="fixed_size">Sabit Boyut</SelectItem>
                  <SelectItem value="recursive">Recursive</SelectItem>
                  <SelectItem value="sentence">Cümle Bazlı</SelectItem>
                  <SelectItem value="semantic">🧠 Semantic (Gelişmiş)</SelectItem>
                </SelectContent>
              </Select>
            </div>
            <div className="grid grid-cols-2 gap-3">
              <div>
                <Label className="text-slate-700 text-sm">Parça Boyutu</Label>
                <Input type="number" value={chunkSize} onChange={(e) => setChunkSize(Number(e.target.value))} className="mt-1" />
              </div>
              <div>
                <Label className="text-slate-700 text-sm">Overlap</Label>
                <Input type="number" value={overlap} onChange={(e) => setOverlap(Number(e.target.value))} className="mt-1" />
              </div>
            </div>
            
            {/* Semantic Chunking Options */}
            {strategy === "semantic" && (
              <div className="space-y-4 pt-4 border-t border-slate-200">
                <div className="flex items-center gap-2 text-indigo-600">
                  <Brain className="w-4 h-4" />
                  <span className="text-sm font-medium">Semantic Chunking Ayarları</span>
                </div>
                
                <div>
                  <Label className="text-slate-700 text-sm">Benzerlik Eşiği (0-1)</Label>
                  <Input 
                    type="number" 
                    step="0.05" 
                    min="0" 
                    max="1" 
                    value={similarityThreshold} 
                    onChange={(e) => setSimilarityThreshold(Number(e.target.value))} 
                    className="mt-1" 
                  />
                  <p className="text-xs text-slate-400 mt-1">Düşük = daha fazla parça, Yüksek = daha az parça</p>
                </div>
                
                <div className="grid grid-cols-2 gap-3">
                  <div>
                    <Label className="text-slate-700 text-sm">Min Boyut</Label>
                    <Input type="number" value={minChunkSize} onChange={(e) => setMinChunkSize(Number(e.target.value))} className="mt-1" />
                  </div>
                  <div>
                    <Label className="text-slate-700 text-sm">Max Boyut</Label>
                    <Input type="number" value={maxChunkSize} onChange={(e) => setMaxChunkSize(Number(e.target.value))} className="mt-1" />
                  </div>
                </div>
                
                <div>
                  <Label className="text-slate-700 text-sm">Buffer Size</Label>
                  <Input type="number" min="0" max="3" value={bufferSize} onChange={(e) => setBufferSize(Number(e.target.value))} className="mt-1" />
                  <p className="text-xs text-slate-400 mt-1">Cümle bağlamı için buffer (0-3)</p>
                </div>
                
                <div className="space-y-3">
                  <div className="flex items-center justify-between">
                    <Label className="text-slate-700 text-sm">Q&A Tespiti</Label>
                    <Switch checked={enableQaDetection} onCheckedChange={setEnableQaDetection} />
                  </div>
                  <div className="flex items-center justify-between">
                    <Label className="text-slate-700 text-sm">Adaptif Threshold</Label>
                    <Switch checked={enableAdaptiveThreshold} onCheckedChange={setEnableAdaptiveThreshold} />
                  </div>
                  <div className="flex items-center justify-between">
                    <Label className="text-slate-700 text-sm">Embedding Cache</Label>
                    <Switch checked={enableCache} onCheckedChange={setEnableCache} />
                  </div>
                  <div className="flex items-center justify-between">
                    <Label className="text-slate-700 text-sm">Kalite Metrikleri</Label>
                    <Switch checked={includeQualityMetrics} onCheckedChange={setIncludeQualityMetrics} />
                  </div>
                </div>
              </div>
            )}
            
            <Button onClick={handleChunk} disabled={isLoading} className="w-full bg-indigo-600 hover:bg-indigo-700">
              {isLoading ? (
                <>
                  <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                  İşleniyor...
                </>
              ) : (
                <>
                  <Play className="w-4 h-4 mr-2" />
                  Parçala
                </>
              )}
            </Button>
            
            {/* Progress Bar for Semantic Chunking */}
            {isLoading && strategy === "semantic" && (
              <div className="space-y-2 mt-4 p-4 bg-slate-50 rounded-lg border border-slate-200">
                <div className="flex items-center justify-between text-sm">
                  <span className="text-slate-600 font-medium">{progressMessage || "Başlatılıyor..."}</span>
                  <span className="text-indigo-600 font-semibold">{Math.round(progress)}%</span>
                </div>
                <Progress value={progress} className="h-2" />
                {progressStage && (
                  <div className="flex items-center gap-2 text-xs text-slate-500">
                    <span className="px-2 py-0.5 bg-indigo-100 text-indigo-700 rounded">
                      {progressStage.replaceAll('_', ' ')}
                    </span>
                    {progressDetails && (
                      <span>
                        {typeof progressDetails.chunk_count === 'number' && `${progressDetails.chunk_count} parça`}
                        {typeof progressDetails.analyzed === 'number' && typeof progressDetails.total === 'number' && 
                          `${progressDetails.analyzed}/${progressDetails.total} analiz edildi`}
                        {typeof progressDetails.estimated_sentences === 'number' && 
                          `~${progressDetails.estimated_sentences} cümle`}
                      </span>
                    )}
                  </div>
                )}
              </div>
            )}
          </div>
        </div>

        {/* Results Panel */}
        <div className="lg:col-span-3 bg-white rounded-lg border border-slate-200 p-6">
          <div className="flex items-center justify-between mb-4">
            <h2 className="font-medium text-slate-900">Sonuçlar</h2>
            {result && (
              <div className="flex items-center gap-2">
                <Button variant="outline" size="sm" onClick={() => setViewMode(viewMode === "list" ? "visual" : "list")}>
                  {viewMode === "list" ? "Görsel" : "Liste"}
                </Button>
                <Button variant="outline" size="sm" onClick={expandAll}>Tümünü Aç</Button>
                <Button variant="outline" size="sm" onClick={collapseAll}>Tümünü Kapat</Button>
              </div>
            )}
          </div>

          {result ? (
            <div className="space-y-4">
              {/* Basic Stats */}
              <div className="grid grid-cols-4 gap-3">
                <div className="bg-slate-50 rounded p-3 text-center">
                  <p className="text-xs text-slate-500">Parça</p>
                  <p className="text-lg font-semibold text-slate-900">{result.stats.total_chunks}</p>
                </div>
                <div className="bg-slate-50 rounded p-3 text-center">
                  <p className="text-xs text-slate-500">Toplam</p>
                  <p className="text-lg font-semibold text-slate-900">{result.stats.total_characters}</p>
                </div>
                <div className="bg-slate-50 rounded p-3 text-center">
                  <p className="text-xs text-slate-500">Ortalama</p>
                  <p className="text-lg font-semibold text-slate-900">{Math.round(result.stats.avg_chunk_size)}</p>
                </div>
                <div className="bg-slate-50 rounded p-3 text-center">
                  <p className="text-xs text-slate-500">Min/Max</p>
                  <p className="text-lg font-semibold text-slate-900">{result.stats.min_chunk_size}/{result.stats.max_chunk_size}</p>
                </div>
              </div>

              {/* Semantic Chunking Metrics */}
              {strategy === "semantic" && (result.detected_language || result.adaptive_threshold_used || result.processing_time_ms || result.quality_report) && (
                <div className="bg-gradient-to-r from-indigo-50 to-purple-50 rounded-lg p-4 border border-indigo-100">
                  <div className="flex items-center gap-2 mb-3">
                    <Brain className="w-4 h-4 text-indigo-600" />
                    <span className="text-sm font-medium text-indigo-900">Semantic Chunking Metrikleri</span>
                  </div>
                  
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                    {result.detected_language && (
                      <div className="bg-white rounded p-2 text-center">
                        <div className="flex items-center justify-center gap-1 mb-1">
                          <Globe className="w-3 h-3 text-slate-400" />
                          <p className="text-xs text-slate-500">Dil</p>
                        </div>
                        <p className="text-sm font-semibold text-slate-900">{getLanguageLabel(result.detected_language)}</p>
                      </div>
                    )}
                    
                    {result.adaptive_threshold_used != null && (
                      <div className="bg-white rounded p-2 text-center">
                        <div className="flex items-center justify-center gap-1 mb-1">
                          <Zap className="w-3 h-3 text-slate-400" />
                          <p className="text-xs text-slate-500">Adaptif Threshold</p>
                        </div>
                        <p className="text-sm font-semibold text-slate-900">{result.adaptive_threshold_used.toFixed(2)}</p>
                      </div>
                    )}
                    
                    {result.processing_time_ms != null && (
                      <div className="bg-white rounded p-2 text-center">
                        <p className="text-xs text-slate-500 mb-1">İşlem Süresi</p>
                        <p className="text-sm font-semibold text-slate-900">{(result.processing_time_ms / 1000).toFixed(2)}s</p>
                      </div>
                    )}
                    
                    {result.quality_report && (
                      <div className="bg-white rounded p-2 text-center">
                        <p className="text-xs text-slate-500 mb-1">Kalite Skoru</p>
                        <p className={`text-sm font-semibold ${
                          result.quality_report.overall_quality_score >= 0.8 ? 'text-green-600' :
                          result.quality_report.overall_quality_score >= 0.6 ? 'text-yellow-600' : 'text-red-600'
                        }`}>
                          {(result.quality_report.overall_quality_score * 100).toFixed(0)}%
                        </p>
                      </div>
                    )}
                  </div>
                  
                  {/* Quality Report Details */}
                  {result.quality_report && (
                    <div className="mt-3 pt-3 border-t border-indigo-100">
                      <div className="grid grid-cols-3 gap-3 text-center">
                        <div>
                          <p className="text-xs text-slate-500">Ort. Tutarlılık</p>
                          <p className="text-sm font-medium text-slate-700">{(result.quality_report.avg_coherence * 100).toFixed(0)}%</p>
                        </div>
                        <div>
                          <p className="text-xs text-slate-500">Min Tutarlılık</p>
                          <p className="text-sm font-medium text-slate-700">{(result.quality_report.min_coherence * 100).toFixed(0)}%</p>
                        </div>
                        <div>
                          <p className="text-xs text-slate-500">Max Tutarlılık</p>
                          <p className="text-sm font-medium text-slate-700">{(result.quality_report.max_coherence * 100).toFixed(0)}%</p>
                        </div>
                      </div>
                      
                      {result.quality_report.recommendations && result.quality_report.recommendations.length > 0 && (
                        <div className="mt-3">
                          <p className="text-xs text-slate-500 mb-1">Öneriler:</p>
                          <ul className="text-xs text-slate-600 space-y-1">
                            {result.quality_report.recommendations.slice(0, 3).map((rec, i) => (
                              <li key={i} className="flex items-start gap-1">
                                <span className="text-indigo-500">•</span>
                                {rec}
                              </li>
                            ))}
                          </ul>
                        </div>
                      )}
                    </div>
                  )}
                  
                  {/* Warning Message */}
                  {result.warning_message && (
                    <div className="mt-3 flex items-start gap-2 text-amber-700 bg-amber-50 rounded p-2">
                      <AlertCircle className="w-4 h-4 mt-0.5 flex-shrink-0" />
                      <p className="text-xs">{result.warning_message}</p>
                    </div>
                  )}
                  
                  {/* Fallback Used */}
                  {result.fallback_used && (
                    <div className="mt-3 flex items-start gap-2 text-orange-700 bg-orange-50 rounded p-2">
                      <AlertCircle className="w-4 h-4 mt-0.5 flex-shrink-0" />
                      <p className="text-xs">Fallback strateji kullanıldı: {result.fallback_used}</p>
                    </div>
                  )}
                </div>
              )}

              {/* Chunks Display */}
              {viewMode === "visual" ? (
                <div>
                  <Label className="text-slate-700 mb-2 block">Görsel Önizleme</Label>
                  <div className="bg-slate-50 rounded-lg p-4 max-h-[400px] overflow-y-auto">
                    <div className="space-y-1">
                      {result.chunks.map((chunk, i) => {
                        const qualityMetric = result.quality_metrics?.[i];
                        return (
                          <div key={i} className={`${COLORS[i % COLORS.length]} border-l-4 p-3 rounded-r`}>
                            <div className="flex items-center justify-between mb-2">
                              <div className="flex items-center gap-2">
                                <span className="text-xs font-bold text-slate-600">Chunk #{i + 1} ({chunk.content.length} karakter)</span>
                                {qualityMetric && (
                                  <span className={`text-xs px-1.5 py-0.5 rounded ${
                                    qualityMetric.semantic_coherence >= 0.8 ? 'bg-green-100 text-green-700' :
                                    qualityMetric.semantic_coherence >= 0.6 ? 'bg-yellow-100 text-yellow-700' : 'bg-red-100 text-red-700'
                                  }`}>
                                    {(qualityMetric.semantic_coherence * 100).toFixed(0)}%
                                  </span>
                                )}
                                {qualityMetric?.has_qa_pairs && (
                                  <span className="text-xs px-1.5 py-0.5 rounded bg-purple-100 text-purple-700">Q&A</span>
                                )}
                              </div>
                              <button onClick={() => copyChunk(chunk.content, i)} className="text-slate-400 hover:text-slate-600">
                                {copiedIndex === i ? <Check className="w-4 h-4 text-green-500" /> : <Copy className="w-4 h-4" />}
                              </button>
                            </div>
                            <p className="text-sm text-slate-800 whitespace-pre-wrap break-words">{chunk.content}</p>
                          </div>
                        );
                      })}
                    </div>
                  </div>
                </div>
              ) : (
                <div>
                  <Label className="text-slate-700 mb-2 block">Parça Listesi</Label>
                  <div className="space-y-2 max-h-[400px] overflow-y-auto">
                    {result.chunks.map((chunk, i) => {
                      const qualityMetric = result.quality_metrics?.[i];
                      return (
                        <div key={i} className={`border rounded-lg overflow-hidden ${COLORS[i % COLORS.length]}`}>
                          <button
                            onClick={() => toggleExpand(i)}
                            className="w-full flex items-center justify-between p-3 text-left hover:bg-white/50"
                          >
                            <div className="flex items-center gap-2">
                              <span className="font-medium text-sm text-slate-700">
                                Chunk #{i + 1} <span className="font-normal text-slate-500">({chunk.content.length} karakter)</span>
                              </span>
                              {qualityMetric && (
                                <span className={`text-xs px-1.5 py-0.5 rounded ${
                                  qualityMetric.semantic_coherence >= 0.8 ? 'bg-green-100 text-green-700' :
                                  qualityMetric.semantic_coherence >= 0.6 ? 'bg-yellow-100 text-yellow-700' : 'bg-red-100 text-red-700'
                                }`}>
                                  Tutarlılık: {(qualityMetric.semantic_coherence * 100).toFixed(0)}%
                                </span>
                              )}
                            </div>
                            <div className="flex items-center gap-2">
                              <button onClick={(e) => { e.stopPropagation(); copyChunk(chunk.content, i); }} className="text-slate-400 hover:text-slate-600">
                                {copiedIndex === i ? <Check className="w-4 h-4 text-green-500" /> : <Copy className="w-4 h-4" />}
                              </button>
                              {expandedChunks.has(i) ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
                            </div>
                          </button>
                          {expandedChunks.has(i) && (
                            <div className="p-3 pt-0 border-t border-slate-200 bg-white/70">
                              {qualityMetric && (
                                <div className="flex gap-3 mb-2 text-xs text-slate-500">
                                  <span>Cümle: {qualityMetric.sentence_count}</span>
                                  <span>Konu Tutarlılığı: {(qualityMetric.topic_consistency * 100).toFixed(0)}%</span>
                                  {qualityMetric.has_questions && <span className="text-purple-600">Soru içeriyor</span>}
                                  {qualityMetric.has_qa_pairs && <span className="text-purple-600">Q&A çifti</span>}
                                </div>
                              )}
                              <p className="text-sm text-slate-800 whitespace-pre-wrap break-words">{chunk.content}</p>
                            </div>
                          )}
                        </div>
                      );
                    })}
                  </div>
                </div>
              )}
            </div>
          ) : (
            <div className="text-center py-16 text-slate-400">
              <p>Sonuç görmek için metin girin ve parçalayın</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
