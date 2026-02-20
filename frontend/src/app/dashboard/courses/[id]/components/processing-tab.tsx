"use client";

import { useEffect, useState, useCallback } from "react";
import { api, Document, Chunk } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Checkbox } from "@/components/ui/checkbox";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { toast } from "sonner";
import {
  Loader2,
  FileText,
  Zap,
  Trash2,
  Database,
  CheckCircle2,
  AlertCircle,
  X,
} from "lucide-react";

interface ProcessingTabProps {
  readonly courseId: number;
  readonly isOwner: boolean;
}

type ChunkStrategy = "recursive" | "semantic";

interface ProcessingOptions {
  strategy: ChunkStrategy;
  chunk_size: number;
  overlap: number;
  similarity_threshold: number;
  embedding_model: string;
  min_chunk_size: number;
  max_chunk_size: number;
}

interface ProcessingProgress {
  docId: number;
  docName: string;
  status: "processing" | "success" | "error";
  operation: "chunking" | "embedding";
  error?: string;
}

const EMBEDDING_MODELS = [
  { value: "openai/text-embedding-3-small", label: "OpenAI text-embedding-3-small (1536 dim)" },
  { value: "openai/text-embedding-3-large", label: "OpenAI text-embedding-3-large (3072 dim)" },
  { value: "alibaba/text-embedding-v4", label: "Alibaba text-embedding-v4 (1024 dim)" },
  { value: "cohere/embed-multilingual-v3.0", label: "Cohere embed-multilingual-v3.0 (1024 dim)" },
  { value: "cohere/embed-multilingual-light-v3.0", label: "Cohere embed-multilingual-light-v3.0 (384 dim)" },
  { value: "jina/jina-embeddings-v3", label: "Jina jina-embeddings-v3 (1024 dim)" },
  { value: "voyage/voyage-4-large", label: "Voyage voyage-4-large (1536 dim)" },
  { value: "voyage/voyage-3-large", label: "Voyage voyage-3-large (1024 dim)" },
  { value: "voyage/voyage-3-lite", label: "Voyage voyage-3-lite (512 dim)" },
  { value: "qwen/qwen3-embedding-8b", label: "Qwen qwen3-embedding-8b (1024 dim)" },
];

const CHUNKS_PER_PAGE = 10;

export function ProcessingTab({ courseId, isOwner }: ProcessingTabProps) {
  const [documents, setDocuments] = useState<Document[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [selectedDocId, setSelectedDocId] = useState<number | null>(null);
  const [selectedDocIds, setSelectedDocIds] = useState<number[]>([]);
  const [docChunks, setDocChunks] = useState<Chunk[]>([]);
  const [chunkPage, setChunkPage] = useState(1);

  const [isProcessing, setIsProcessing] = useState(false);
  const [isDeletingChunks, setIsDeletingChunks] = useState(false);
  const [isDeletingChunkId, setIsDeletingChunkId] = useState<number | null>(null);
  const [isEmbedding, setIsEmbedding] = useState(false);
  const [isDeletingVectors, setIsDeletingVectors] = useState(false);
  const [isDeletingCollection, setIsDeletingCollection] = useState(false);
  const [processingProgress, setProcessingProgress] = useState<ProcessingProgress[]>([]);

  const [options, setOptions] = useState<ProcessingOptions>({
    strategy: "recursive",
    chunk_size: 500,
    overlap: 50,
    similarity_threshold: 0.5,
    embedding_model: "openai/text-embedding-3-small",
    min_chunk_size: 150,
    max_chunk_size: 2000,
  });

  // Load course settings and apply to options
  const loadCourseSettings = useCallback(async () => {
    try {
      const settings = await api.getCourseSettings(courseId);
      setOptions(prev => ({
        ...prev,
        strategy: settings.default_chunk_strategy as ChunkStrategy,
        chunk_size: settings.default_chunk_size,
        overlap: settings.default_overlap,
        embedding_model: settings.default_embedding_model,
      }));
    } catch {
      // Use defaults if settings fail to load
    }
  }, [courseId]);

  const loadDocuments = useCallback(async () => {
    try {
      const data = await api.getUserDocuments();
      const filtered = data.filter((d) => d.course_id === courseId);
      setDocuments(filtered);
      if (selectedDocId && !filtered.some((d) => d.id === selectedDocId)) {
        setSelectedDocId(null);
      }
    } catch {
      toast.error("Dokümanlar yüklenirken hata oluştu");
    } finally {
      setIsLoading(false);
    }
  }, [courseId, selectedDocId]);

  const loadChunks = useCallback(async (docId: number) => {
    try {
      const result = await api.getDocumentChunks(docId);
      setDocChunks(result.chunks);
      setChunkPage(1);
    } catch {
      toast.error("Chunk'lar yüklenirken hata oluştu");
    }
  }, []);

  useEffect(() => {
    loadCourseSettings();
    loadDocuments();
  }, [loadCourseSettings, loadDocuments]);

  useEffect(() => {
    if (selectedDocId) {
      loadChunks(selectedDocId);
    } else {
      setDocChunks([]);
    }
  }, [selectedDocId, loadChunks]);

  const handleProcess = async () => {
    if (!selectedDocId) return;
    if (!selectedDoc || !selectedDoc.char_count || selectedDoc.char_count <= 0) {
      toast.error("Bu dokümandan metin çıkarılamadı (içerik boş). Lütfen dosyayı yeniden yükleyin.");
      return;
    }
    setIsProcessing(true);
    try {
      const result = await api.processDocument(selectedDocId, {
        strategy: options.strategy,
        chunk_size: options.chunk_size,
        overlap: options.overlap,
        similarity_threshold: options.similarity_threshold,
        embedding_model: options.embedding_model,
        min_chunk_size: options.min_chunk_size,
        max_chunk_size: options.max_chunk_size,
      });
      setDocChunks(result.chunks);
      setChunkPage(1);
      toast.success(`${result.total} chunk oluşturuldu`);
      loadDocuments();
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "İşleme hatası");
    } finally {
      setIsProcessing(false);
    }
  };

  const handleDeleteChunks = async () => {
    if (!selectedDocId) return;
    if (!confirm("Tüm chunk'ları silmek istediğinizden emin misiniz?")) return;
    setIsDeletingChunks(true);
    try {
      await api.deleteDocumentChunks(selectedDocId);
      setDocChunks([]);
      toast.success("Chunk'lar silindi");
      loadDocuments();
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "Silme hatası");
    } finally {
      setIsDeletingChunks(false);
    }
  };

  const handleDeleteSingleChunk = async (chunkId: number) => {
    if (!selectedDocId || !chunkId) return;
    if (!confirm("Bu chunk'ı silmek istediğinizden emin misiniz?")) return;
    setIsDeletingChunkId(chunkId);
    try {
      await api.deleteChunk(selectedDocId, chunkId);
      setDocChunks((prev) => prev.filter((c) => c.id !== chunkId));
      toast.success("Chunk silindi");
      loadDocuments();
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "Silme hatası");
    } finally {
      setIsDeletingChunkId(null);
    }
  };

  const handleEmbed = async () => {
    if (!selectedDocId) return;
    if (!selectedDoc || !selectedDoc.char_count || selectedDoc.char_count <= 0) {
      toast.error("Bu dokümandan metin çıkarılamadı (içerik boş). Lütfen dosyayı yeniden yükleyin.");
      return;
    }
    setIsEmbedding(true);
    try {
      const result = await api.embedDocument(selectedDocId, options.embedding_model);
      toast.success(`${result.vector_count} vektör oluşturuldu`);
      loadDocuments();
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "Embedding hatası");
    } finally {
      setIsEmbedding(false);
    }
  };

  const handleDeleteVectors = async () => {
    if (!selectedDocId) return;
    if (!confirm("Tüm vektörleri silmek istediğinizden emin misiniz?")) return;
    setIsDeletingVectors(true);
    try {
      await api.deleteDocumentVectors(selectedDocId);
      toast.success("Vektörler silindi");
      loadDocuments();
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "Silme hatası");
    } finally {
      setIsDeletingVectors(false);
    }
  };

  const handleDeleteCollection = async () => {
    if (!confirm("Vektör koleksiyonunu silmek istediğinizden emin misiniz? Tüm dokümanların embedding durumu sıfırlanacak.")) return;
    setIsDeletingCollection(true);
    try {
      await api.deleteCourseCollection(courseId);
      toast.success("Koleksiyon silindi");
      loadDocuments();
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "Silme hatası");
    } finally {
      setIsDeletingCollection(false);
    }
  };

  const handleBulkProcess = async () => {
    if (selectedDocIds.length === 0) {
      toast.error("En az bir doküman seçin");
      return;
    }

    setIsProcessing(true);
    
    // Initialize processing progress for all selected documents
    const initialProgress: ProcessingProgress[] = selectedDocIds.map((docId) => {
      const doc = documents.find((d) => d.id === docId);
      return {
        docId,
        docName: doc?.original_filename || `Doküman ${docId}`,
        status: "processing",
        operation: "chunking",
      };
    });
    setProcessingProgress(initialProgress);

    let successCount = 0;
    let errorCount = 0;

    // Process documents sequentially
    for (const docId of selectedDocIds) {
      const doc = documents.find((d) => d.id === docId);
      if (!doc || !doc.char_count || doc.char_count <= 0) {
        errorCount++;
        setProcessingProgress((prev) =>
          prev.map((p) =>
            p.docId === docId
              ? { ...p, status: "error", error: "İçerik boş" }
              : p
          )
        );
        continue;
      }

      try {
        await api.processDocument(docId, {
          strategy: options.strategy,
          chunk_size: options.chunk_size,
          overlap: options.overlap,
          similarity_threshold: options.similarity_threshold,
          embedding_model: options.embedding_model,
          min_chunk_size: options.min_chunk_size,
          max_chunk_size: options.max_chunk_size,
        });
        successCount++;
        
        // Update progress for this document
        setProcessingProgress((prev) =>
          prev.map((p) =>
            p.docId === docId ? { ...p, status: "success" } : p
          )
        );
      } catch (error) {
        errorCount++;
        const errorMessage = error instanceof Error ? error.message : "İşleme hatası";
        
        // Update progress for this document with error
        setProcessingProgress((prev) =>
          prev.map((p) =>
            p.docId === docId ? { ...p, status: "error", error: errorMessage } : p
          )
        );
      }
    }

    // Show summary toast
    if (successCount > 0 && errorCount === 0) {
      toast.success(`${successCount} doküman başarıyla chunklandı`);
    } else if (successCount > 0 && errorCount > 0) {
      toast.warning(`${successCount} doküman chunklandı, ${errorCount} doküman işlenemedi`);
    } else if (errorCount > 0) {
      toast.error(`${errorCount} doküman işlenemedi`);
    }

    setIsProcessing(false);
    setSelectedDocIds([]);
    
    // Reload documents after a short delay
    setTimeout(() => {
      loadDocuments();
    }, 1000);
  };

  const handleBulkEmbed = async () => {
    if (selectedDocIds.length === 0) {
      toast.error("En az bir doküman seçin");
      return;
    }

    setIsEmbedding(true);
    
    // Initialize processing progress for all selected documents
    const initialProgress: ProcessingProgress[] = selectedDocIds.map((docId) => {
      const doc = documents.find((d) => d.id === docId);
      return {
        docId,
        docName: doc?.original_filename || `Doküman ${docId}`,
        status: "processing",
        operation: "embedding",
      };
    });
    setProcessingProgress(initialProgress);

    let successCount = 0;
    let errorCount = 0;

    // Embed documents sequentially
    for (const docId of selectedDocIds) {
      const doc = documents.find((d) => d.id === docId);
      if (!doc || !doc.char_count || doc.char_count <= 0) {
        errorCount++;
        setProcessingProgress((prev) =>
          prev.map((p) =>
            p.docId === docId
              ? { ...p, status: "error", error: "İçerik boş" }
              : p
          )
        );
        continue;
      }

      try {
        await api.embedDocument(docId, options.embedding_model);
        successCount++;
        
        // Update progress for this document
        setProcessingProgress((prev) =>
          prev.map((p) =>
            p.docId === docId ? { ...p, status: "success" } : p
          )
        );
      } catch (error) {
        errorCount++;
        const errorMessage = error instanceof Error ? error.message : "Embedding hatası";
        
        // Update progress for this document with error
        setProcessingProgress((prev) =>
          prev.map((p) =>
            p.docId === docId ? { ...p, status: "error", error: errorMessage } : p
          )
        );
      }
    }

    // Show summary toast
    if (successCount > 0 && errorCount === 0) {
      toast.success(`${successCount} doküman başarıyla embed edildi`);
    } else if (successCount > 0 && errorCount > 0) {
      toast.warning(`${successCount} doküman embed edildi, ${errorCount} doküman işlenemedi`);
    } else if (errorCount > 0) {
      toast.error(`${errorCount} doküman işlenemedi`);
    }

    setIsEmbedding(false);
    setSelectedDocIds([]);
    
    // Reload documents after a short delay
    setTimeout(() => {
      loadDocuments();
    }, 1000);
  };

  const clearProcessingProgress = () => {
    setProcessingProgress([]);
  };

  const handleSelectAll = (checked: boolean | undefined) => {
    if (checked) {
      setSelectedDocIds(documents.map((d) => d.id));
    } else {
      setSelectedDocIds([]);
    }
  };

  const handleSelectDoc = (docId: number, checked: boolean | undefined) => {
    if (checked) {
      setSelectedDocIds([...selectedDocIds, docId]);
    } else {
      setSelectedDocIds(selectedDocIds.filter((id) => id !== docId));
    }
  };

  const selectedDoc = documents.find((d) => d.id === selectedDocId);
  const totalPages = Math.ceil(docChunks.length / CHUNKS_PER_PAGE);
  const pagedChunks = docChunks.slice(
    (chunkPage - 1) * CHUNKS_PER_PAGE,
    chunkPage * CHUNKS_PER_PAGE
  );

  if (isLoading) {
    return (
      <div className="flex justify-center py-12">
        <Loader2 className="w-6 h-6 text-slate-400 animate-spin" />
      </div>
    );
  }

  if (!isOwner) {
    return (
      <div className="bg-white rounded-lg border border-slate-200 p-12 text-center">
        <p className="text-slate-500">Sadece ders sahibi işleme yapabilir</p>
      </div>
    );
  }

  return (
    <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
      {/* Left: Document Selection & Chunks */}
      <div className="lg:col-span-2 space-y-6">
        {/* Bulk Document Selection */}
        <div className="bg-white rounded-lg border border-slate-200 p-4">
          <div className="flex items-center justify-between mb-3">
            <Label className="text-sm font-medium">Toplu İşleme</Label>
            <div className="flex items-center gap-2">
              <span className="text-xs text-slate-500">
                {selectedDocIds.length} seçili
              </span>
              <Checkbox
                checked={selectedDocIds.length > 0 && selectedDocIds.length === documents.length}
                onCheckedChange={handleSelectAll}
                disabled={isProcessing || isEmbedding}
              />
            </div>
          </div>
          <div className="flex items-center gap-2 mb-3">
            <Button
              size="sm"
              onClick={handleBulkProcess}
              disabled={isProcessing || selectedDocIds.length === 0}
            >
              {isProcessing ? (
                <Loader2 className="w-4 h-4 mr-2 animate-spin" />
              ) : (
                <Zap className="w-4 h-4 mr-2" />
              )}
              Toplu Chunkla
            </Button>
            <Button
              size="sm"
              onClick={handleBulkEmbed}
              disabled={isEmbedding || selectedDocIds.length === 0}
            >
              {isEmbedding ? (
                <Loader2 className="w-4 h-4 mr-2 animate-spin" />
              ) : (
                <Database className="w-4 h-4 mr-2" />
              )}
              Toplu Embed
            </Button>
          </div>
          <div className="max-h-48 overflow-y-auto border border-slate-200 rounded">
            {documents.length === 0 ? (
              <div className="p-4 text-center text-sm text-slate-500">
                Henüz doküman yok
              </div>
            ) : (
              <div className="divide-y divide-slate-200">
                {documents.map((doc) => (
                  <div
                    key={doc.id}
                    className="flex items-center gap-3 p-3 hover:bg-slate-50"
                  >
                    <Checkbox
                      checked={selectedDocIds.includes(doc.id)}
                      onCheckedChange={(checked) => handleSelectDoc(doc.id, checked)}
                      disabled={isProcessing || isEmbedding}
                    />
                    <FileText className="w-4 h-4 text-slate-400 flex-shrink-0" />
                    <div className="flex-1 min-w-0">
                      <p className="text-sm font-medium text-slate-900 truncate">
                        {doc.original_filename}
                      </p>
                      <p className="text-xs text-slate-500">
                        {doc.chunk_count > 0
                          ? `${doc.chunk_count} chunk`
                          : "Chunklanmamış"}
                        {doc.embedding_status === "completed" && (
                          <span className="ml-2 text-green-600">
                            • {doc.vector_count} vektör
                          </span>
                        )}
                      </p>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>

        {/* Processing Progress */}
        {processingProgress.length > 0 && (
          <div className="bg-white rounded-lg border border-slate-200 p-4">
            <div className="flex items-center justify-between mb-3">
              <Label className="text-sm font-medium">
                İşleme Durumu ({processingProgress.length} doküman)
              </Label>
              <Button
                variant="ghost"
                size="sm"
                onClick={clearProcessingProgress}
                className="text-slate-500 hover:text-slate-700"
              >
                <X className="w-4 h-4" />
              </Button>
            </div>
            <div className="space-y-2">
              {processingProgress.map((progress) => (
                <div
                  key={progress.docId}
                  className="flex items-center gap-3 p-2 bg-slate-50 rounded border border-slate-200"
                >
                  {progress.status === "processing" && (
                    <Loader2 className="w-4 h-4 text-blue-500 animate-spin" />
                  )}
                  {progress.status === "success" && (
                    <CheckCircle2 className="w-4 h-4 text-green-500" />
                  )}
                  {progress.status === "error" && (
                    <AlertCircle className="w-4 h-4 text-red-500" />
                  )}
                  <div className="flex-1 min-w-0">
                    <p className="text-sm font-medium text-slate-900 truncate">
                      {progress.docName}
                    </p>
                    {progress.error && (
                      <p className="text-xs text-red-600 truncate">{progress.error}</p>
                    )}
                  </div>
                  <span className="text-xs text-slate-500">
                    {progress.operation === "chunking" ? "Chunklanıyor" : "Embed ediliyor"}
                  </span>
                  {progress.status === "success" && (
                    <span className="text-xs text-green-600">Başarılı</span>
                  )}
                  {progress.status === "error" && (
                    <span className="text-xs text-red-600">Hata</span>
                  )}
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Single Document Selection */}
        <div className="bg-white rounded-lg border border-slate-200 p-4">
          <Label className="text-sm font-medium">Tekil Doküman Seçimi</Label>
          <Select
            value={selectedDocId?.toString() || ""}
            onValueChange={(v) => setSelectedDocId(v ? Number(v) : null)}
          >
            <SelectTrigger className="mt-2">
              <SelectValue placeholder="Bir doküman seçin..." />
            </SelectTrigger>
            <SelectContent>
              {documents.map((doc) => (
                <SelectItem key={doc.id} value={doc.id.toString()}>
                  <div className="flex items-center gap-2">
                    <FileText className="w-4 h-4" />
                    {doc.original_filename}
                    {doc.chunk_count > 0 && (
                      <span className="text-xs text-slate-400">
                        ({doc.chunk_count} chunk)
                      </span>
                    )}
                  </div>
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>

        {/* Chunks Section */}
        {selectedDocId && (
          <div className="bg-white rounded-lg border border-slate-200">
            <div className="flex items-center justify-between px-4 py-3 border-b border-slate-200">
              <h3 className="font-medium text-slate-900 text-sm sm:text-base">
                Chunklar ({docChunks.length})
              </h3>
              <div className="flex items-center gap-2">
                <Button
                  size="sm"
                  onClick={handleProcess}
                  disabled={isProcessing}
                >
                  {isProcessing ? (
                    <Loader2 className="w-4 h-4 sm:mr-2 animate-spin" />
                  ) : (
                    <Zap className="w-4 h-4 sm:mr-2" />
                  )}
                  <span className="hidden sm:inline">{docChunks.length > 0 ? "Yeniden Oluştur" : "Chunkla"}</span>
                </Button>
                {docChunks.length > 0 && (
                  <Button
                    size="sm"
                    variant="outline"
                    onClick={handleDeleteChunks}
                    disabled={isDeletingChunks}
                    className="text-orange-600"
                  >
                    {isDeletingChunks ? (
                      <Loader2 className="w-4 h-4 sm:mr-2 animate-spin" />
                    ) : (
                      <Trash2 className="w-4 h-4 sm:mr-2" />
                    )}
                    <span className="hidden sm:inline">Temizle</span>
                  </Button>
                )}
              </div>
            </div>

            {docChunks.length === 0 ? (
              <div className="p-8 text-center">
                <p className="text-slate-500 text-sm">
                  Henüz chunk oluşturulmamış. Sağdaki ayarları kullanarak chunklayın.
                </p>
              </div>
            ) : (
              <div className="p-4">
                {/* Pagination */}
                {totalPages > 1 && (
                  <div className="flex items-center justify-between mb-4">
                    <span className="text-sm text-slate-500">
                      Sayfa {chunkPage} / {totalPages}
                    </span>
                    <div className="flex gap-2">
                      <Button
                        size="sm"
                        variant="outline"
                        disabled={chunkPage === 1}
                        onClick={() => setChunkPage(chunkPage - 1)}
                      >
                        Önceki
                      </Button>
                      <Button
                        size="sm"
                        variant="outline"
                        disabled={chunkPage === totalPages}
                        onClick={() => setChunkPage(chunkPage + 1)}
                      >
                        Sonraki
                      </Button>
                    </div>
                  </div>
                )}

                {/* Chunk List */}
                <div className="space-y-2">
                  {pagedChunks.map((chunk, idx) => {
                    const globalIdx = (chunkPage - 1) * CHUNKS_PER_PAGE + idx;
                    return (
                      <div
                        key={chunk.id ?? globalIdx}
                        className="bg-slate-50 rounded border border-slate-200 p-3"
                      >
                        <div className="flex items-center justify-between mb-2">
                          <span className="text-xs font-medium text-slate-600">
                            Chunk {globalIdx + 1}
                          </span>
                          <div className="flex items-center gap-2">
                            <span className="text-xs text-slate-400">
                              {chunk.content.length} karakter
                            </span>
                            {chunk.id && (
                              <Button
                                size="sm"
                                variant="ghost"
                                onClick={() => handleDeleteSingleChunk(chunk.id!)}
                                disabled={isDeletingChunkId === chunk.id}
                                className="h-6 w-6 p-0 text-red-400 hover:text-red-600 hover:bg-red-50"
                                title="Bu chunk'ı sil"
                              >
                                {isDeletingChunkId === chunk.id ? (
                                  <Loader2 className="w-3.5 h-3.5 animate-spin" />
                                ) : (
                                  <Trash2 className="w-3.5 h-3.5" />
                                )}
                              </Button>
                            )}
                          </div>
                        </div>
                        <p className="text-sm text-slate-700 whitespace-pre-wrap">
                          {chunk.content}
                        </p>
                      </div>
                    );
                  })}
                </div>
              </div>
            )}
          </div>
        )}

        {/* Embedding Section */}
        {selectedDocId && docChunks.length > 0 && (
          <div className="bg-white rounded-lg border border-slate-200">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 px-4 py-3 border-b border-slate-200">
              <div className="min-w-0">
                <h3 className="font-medium text-slate-900 text-sm sm:text-base">Embedding</h3>
                {selectedDoc?.embedding_status === "completed" && (
                  <p className="text-xs text-green-600 mt-0.5 truncate">
                    {selectedDoc.vector_count} vektör • {selectedDoc.embedding_model}
                  </p>
                )}
              </div>
              <div className="flex items-center gap-2 shrink-0">
                <Button
                  size="sm"
                  onClick={handleEmbed}
                  disabled={isEmbedding}
                >
                  {isEmbedding ? (
                    <Loader2 className="w-4 h-4 sm:mr-2 animate-spin" />
                  ) : (
                    <Database className="w-4 h-4 sm:mr-2" />
                  )}
                  <span className="hidden sm:inline">{selectedDoc?.embedding_status === "completed" ? "Yeniden Embed" : "Embed Et"}</span>
                </Button>
                {selectedDoc?.embedding_status === "completed" && (
                  <Button
                    size="sm"
                    variant="outline"
                    onClick={handleDeleteVectors}
                    disabled={isDeletingVectors}
                    className="text-orange-600"
                  >
                    {isDeletingVectors ? (
                      <Loader2 className="w-4 h-4 sm:mr-2 animate-spin" />
                    ) : (
                      <Trash2 className="w-4 h-4 sm:mr-2" />
                    )}
                    <span className="hidden sm:inline">Vektörleri Sil</span>
                  </Button>
                )}
              </div>
            </div>
            <div className="p-4">
              <p className="text-sm text-slate-500">
                Chunkları vektörlere dönüştürüp Weaviate&apos;e kaydedin. Bu işlem sohbet için gereklidir.
              </p>
              {documents.length > 0 && documents.every(d => !d.vector_count || d.vector_count === 0) && (
                <Button
                  size="sm"
                  variant="outline"
                  onClick={handleDeleteCollection}
                  disabled={isDeletingCollection}
                  className="mt-3 text-red-600 border-red-200 hover:bg-red-50"
                >
                  {isDeletingCollection ? (
                    <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                  ) : (
                    <Trash2 className="w-4 h-4 mr-2" />
                  )}
                  Koleksiyonu Sıfırla
                </Button>
              )}
            </div>
          </div>
        )}
      </div>

      {/* Right: Options */}
      <div className="lg:col-span-1">
        <div className="bg-gradient-to-br from-indigo-50 via-white to-purple-50 rounded-xl border-2 border-indigo-200 shadow-lg sticky top-6 overflow-hidden">
          {/* Header */}
          <div className="bg-gradient-to-r from-indigo-600 to-purple-600 px-4 py-3">
            <div className="flex items-center gap-2">
              <div className="w-8 h-8 bg-white/20 rounded-lg flex items-center justify-center">
                <Zap className="w-4 h-4 text-white" />
              </div>
              <h3 className="font-semibold text-white">İşleme Ayarları</h3>
            </div>
          </div>

          <div className="p-4 space-y-4">
            {/* Strategy */}
            <div className="space-y-1.5">
              <Label className="text-xs font-medium text-slate-700">Chunking Stratejisi</Label>
              <Select
                value={options.strategy}
                onValueChange={(v) => setOptions({ ...options, strategy: v as ChunkStrategy })}
              >
                <SelectTrigger className="bg-white border-slate-200 focus:border-indigo-300 focus:ring-indigo-200">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="recursive">Recursive</SelectItem>
                  <SelectItem value="semantic">Semantic</SelectItem>
                </SelectContent>
              </Select>
            </div>

            {/* Recursive options */}
            {options.strategy === "recursive" && (
              <div className="space-y-1.5">
                <Label className="text-xs font-medium text-slate-700">Chunk Boyutu</Label>
                <Input
                  type="number"
                  value={options.chunk_size}
                  onChange={(e) => setOptions({ ...options, chunk_size: Number(e.target.value) })}
                  min={100}
                  max={5000}
                  className="bg-white border-slate-200 focus:border-indigo-300 focus:ring-indigo-200"
                />
              </div>
            )}

            {/* Overlap */}
            <div className="space-y-1.5">
              <Label className="text-xs font-medium text-slate-700">Overlap</Label>
              <Input
                type="number"
                value={options.overlap}
                onChange={(e) => setOptions({ ...options, overlap: Number(e.target.value) })}
                min={0}
                max={500}
                className="bg-white border-slate-200 focus:border-indigo-300 focus:ring-indigo-200"
              />
            </div>

            {/* Semantic options */}
            {options.strategy === "semantic" && (
              <>
                <div className="space-y-1.5">
                  <Label className="text-xs font-medium text-slate-700">Benzerlik Eşiği (0-1)</Label>
                  <Input
                    type="number"
                    value={options.similarity_threshold}
                    onChange={(e) => setOptions({ ...options, similarity_threshold: Number(e.target.value) })}
                    min={0}
                    max={1}
                    step={0.05}
                    className="bg-white border-slate-200 focus:border-indigo-300 focus:ring-indigo-200"
                  />
                </div>

                <div className="space-y-1.5">
                  <Label className="text-xs font-medium text-slate-700">Min Chunk Boyutu</Label>
                  <Input
                    type="number"
                    value={options.min_chunk_size}
                    onChange={(e) => setOptions({ ...options, min_chunk_size: Number(e.target.value) })}
                    min={50}
                    max={1000}
                    className="bg-white border-slate-200 focus:border-indigo-300 focus:ring-indigo-200"
                  />
                </div>

                <div className="space-y-1.5">
                  <Label className="text-xs font-medium text-slate-700">Max Chunk Boyutu</Label>
                  <Input
                    type="number"
                    value={options.max_chunk_size}
                    onChange={(e) => setOptions({ ...options, max_chunk_size: Number(e.target.value) })}
                    min={500}
                    max={5000}
                    className="bg-white border-slate-200 focus:border-indigo-300 focus:ring-indigo-200"
                  />
                </div>
              </>
            )}

            {/* Embedding Model */}
            <div className="space-y-1.5">
              <Label className="text-xs font-medium text-slate-700">Embedding Model</Label>
              <Select
                value={options.embedding_model}
                onValueChange={(v) => setOptions({ ...options, embedding_model: v })}
              >
                <SelectTrigger className="bg-white border-slate-200 focus:border-indigo-300 focus:ring-indigo-200">
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

            {/* Info */}
            <div className="mt-4 pt-4 border-t border-indigo-100">
              <p className="text-xs text-slate-500">
                💡 Bu ayarlar ders varsayılanlarından yüklenir. Değişiklikler sadece bu işlem için geçerlidir.
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
