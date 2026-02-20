"use client";

import { useEffect, useState, useCallback } from "react";
import { api, Document } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { toast } from "sonner";
import { Upload, FileText, Trash2, Loader2, ChevronLeft, ChevronRight, X, CheckCircle2, AlertCircle } from "lucide-react";

interface DocumentsTabProps {
  courseId: number;
  isOwner: boolean;
}

const ITEMS_PER_PAGE = 10;

interface UploadProgress {
  fileName: string;
  status: "uploading" | "success" | "error";
  error?: string;
}

export function DocumentsTab({ courseId, isOwner }: DocumentsTabProps) {
  const [documents, setDocuments] = useState<Document[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [isUploading, setIsUploading] = useState(false);
  const [deletingId, setDeletingId] = useState<number | null>(null);
  const [currentPage, setCurrentPage] = useState(1);
  const [uploadProgress, setUploadProgress] = useState<UploadProgress[]>([]);

  const loadDocuments = useCallback(async () => {
    try {
      // Load user documents instead of course-specific documents
      const data = await api.getUserDocuments();
      setDocuments(data.filter((d) => d.course_id === courseId));
      setCurrentPage(1); // Reset to first page when loading documents
    } catch {
      toast.error("Dokümanlar yüklenirken hata oluştu");
    } finally {
      setIsLoading(false);
    }
  }, [courseId]);

  useEffect(() => {
    loadDocuments();
  }, [loadDocuments]);

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (!files || files.length === 0) return;

    setIsUploading(true);
    
    // Initialize upload progress for all files
    const initialProgress: UploadProgress[] = Array.from(files).map((file) => ({
      fileName: file.name,
      status: "uploading",
    }));
    setUploadProgress(initialProgress);

    const newDocuments: Document[] = [];
    let successCount = 0;
    let errorCount = 0;

    // Upload files sequentially
    for (let i = 0; i < files.length; i++) {
      const file = files[i];
      try {
        const doc = await api.uploadUserDocument(file, courseId);
        newDocuments.push(doc);
        successCount++;
        
        // Update progress for this file
        setUploadProgress((prev) =>
          prev.map((p) =>
            p.fileName === file.name
              ? { ...p, status: "success" }
              : p
          )
        );
      } catch (error) {
        errorCount++;
        const errorMessage = error instanceof Error ? error.message : "Yükleme hatası";
        
        // Update progress for this file with error
        setUploadProgress((prev) =>
          prev.map((p) =>
            p.fileName === file.name
              ? { ...p, status: "error", error: errorMessage }
              : p
          )
        );
      }
    }

    // Update documents list with successfully uploaded documents
    if (newDocuments.length > 0) {
      setDocuments([...newDocuments, ...documents]);
    }

    // Show summary toast
    if (successCount > 0 && errorCount === 0) {
      toast.success(`${successCount} doküman başarıyla yüklendi`);
    } else if (successCount > 0 && errorCount > 0) {
      toast.warning(`${successCount} doküman yüklendi, ${errorCount} doküman yüklenemedi`);
    } else if (errorCount > 0) {
      toast.error(`${errorCount} doküman yüklenemedi`);
    }

    setIsUploading(false);
    e.target.value = "";
    
    // Reload documents after a short delay to ensure all uploads are processed
    setTimeout(() => {
      loadDocuments();
    }, 1000);
  };

  const clearUploadProgress = () => {
    setUploadProgress([]);
  };

  const handleDelete = async (docId: number) => {
    if (!confirm("Bu dokümanı silmek istediğinizden emin misiniz?")) return;
    setDeletingId(docId);
    try {
      await api.deleteDocument(docId);
      setDocuments(documents.filter((d) => d.id !== docId));
      toast.success("Doküman silindi");
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "Silme hatası");
    } finally {
      setDeletingId(null);
    }
  };

  const getStatusBadge = (doc: Document) => {
    if (doc.embedding_status === "completed") {
      return (
        <span className="px-2 py-0.5 text-xs rounded-full bg-green-100 text-green-700">
          Hazır ({doc.vector_count} vektör)
        </span>
      );
    }
    if (doc.embedding_status === "processing") {
      return (
        <span className="px-2 py-0.5 text-xs rounded-full bg-yellow-100 text-yellow-700">
          İşleniyor...
        </span>
      );
    }
    if (doc.embedding_status === "error") {
      return (
        <span className="px-2 py-0.5 text-xs rounded-full bg-red-100 text-red-700">
          Hata
        </span>
      );
    }
    if (doc.chunk_count > 0) {
      return (
        <span className="px-2 py-0.5 text-xs rounded-full bg-blue-100 text-blue-700">
          {doc.chunk_count} chunk
        </span>
      );
    }
    return (
      <span className="px-2 py-0.5 text-xs rounded-full bg-slate-100 text-slate-600">
        İşlenmedi
      </span>
    );
  };

  if (isLoading) {
    return (
      <div className="flex justify-center py-12">
        <Loader2 className="w-6 h-6 text-slate-400 animate-spin" />
      </div>
    );
  }

  // Pagination calculations
  const totalPages = Math.ceil(documents.length / ITEMS_PER_PAGE);
  const startIndex = (currentPage - 1) * ITEMS_PER_PAGE;
  const endIndex = startIndex + ITEMS_PER_PAGE;
  const currentDocuments = documents.slice(startIndex, endIndex);
  const showingFrom = documents.length > 0 ? startIndex + 1 : 0;
  const showingTo = Math.min(endIndex, documents.length);

  return (
    <div className="bg-white rounded-lg border border-slate-200">
      <div className="flex items-center justify-between px-4 sm:px-6 py-3 sm:py-4 border-b border-slate-200">
        <h2 className="font-medium text-slate-900 text-sm sm:text-base">Dokümanlar ({documents.length})</h2>
        {isOwner && (
          <label className="cursor-pointer">
            <input
              type="file"
              className="hidden"
              accept=".pdf,.md,.docx,.txt"
              multiple
              onChange={handleFileUpload}
              disabled={isUploading}
            />
            <Button size="sm" disabled={isUploading} asChild>
              <span>
                {isUploading ? (
                  <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                ) : (
                  <Upload className="w-4 h-4 mr-2" />
                )}
                Toplu Yükle
              </span>
            </Button>
          </label>
        )}
      </div>

      {/* Upload Progress */}
      {uploadProgress.length > 0 && (
        <div className="px-4 sm:px-6 py-4 bg-slate-50 border-b border-slate-200">
          <div className="flex items-center justify-between mb-3">
            <h3 className="text-sm font-medium text-slate-700">
              Yükleme Durumu ({uploadProgress.length} dosya)
            </h3>
            <Button
              variant="ghost"
              size="sm"
              onClick={clearUploadProgress}
              className="text-slate-500 hover:text-slate-700"
            >
              <X className="w-4 h-4" />
            </Button>
          </div>
          <div className="space-y-2">
            {uploadProgress.map((progress, index) => (
              <div
                key={index}
                className="flex items-center gap-3 p-2 bg-white rounded border border-slate-200"
              >
                {progress.status === "uploading" && (
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
                    {progress.fileName}
                  </p>
                  {progress.error && (
                    <p className="text-xs text-red-600 truncate">{progress.error}</p>
                  )}
                </div>
                {progress.status === "uploading" && (
                  <span className="text-xs text-slate-500">Yükleniyor...</span>
                )}
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

      {documents.length === 0 ? (
        <div className="p-12 text-center">
          <FileText className="w-10 h-10 text-slate-300 mx-auto mb-3" />
          <p className="text-slate-500">Henüz doküman yok</p>
          {isOwner && (
            <p className="text-sm text-slate-400 mt-1">
              PDF, MD, DOCX veya TXT dosyası yükleyin
            </p>
          )}
        </div>
      ) : (
        <>
          <div className="divide-y divide-slate-200">
            {currentDocuments.map((doc) => (
              <div
                key={doc.id}
                className="px-4 sm:px-6 py-3 sm:py-4 flex flex-col sm:flex-row sm:items-center justify-between gap-2 sm:gap-0 hover:bg-slate-50"
              >
                <div className="flex items-center gap-3 min-w-0">
                  <FileText className="w-5 h-5 text-slate-400 shrink-0" />
                  <div className="min-w-0">
                    <p className="font-medium text-slate-900 text-sm truncate">{doc.original_filename}</p>
                    <p className="text-xs text-slate-500">
                      {(doc.file_size / 1024).toFixed(1)} KB •{" "}
                      {doc.char_count ? `${doc.char_count.toLocaleString()} karakter • ` : ""}
                      {new Date(doc.created_at).toLocaleDateString("tr-TR")}
                    </p>
                  </div>
                </div>
                <div className="flex items-center gap-2 sm:gap-3 self-end sm:self-auto shrink-0">
                  {getStatusBadge(doc)}
                  {isOwner && (
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => handleDelete(doc.id)}
                      disabled={deletingId === doc.id}
                      className="text-slate-400 hover:text-red-600"
                    >
                      {deletingId === doc.id ? (
                        <Loader2 className="w-4 h-4 animate-spin" />
                      ) : (
                        <Trash2 className="w-4 h-4" />
                      )}
                    </Button>
                  )}
                </div>
              </div>
            ))}
          </div>

          {/* Pagination */}
          {totalPages > 1 && (
            <div className="px-4 sm:px-6 py-3 sm:py-4 border-t border-slate-200 bg-slate-50">
              <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3">
                <p className="text-xs sm:text-sm text-slate-600">
                  {showingFrom}-{showingTo} / {documents.length}
                </p>
                <div className="flex items-center gap-2">
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => setCurrentPage((p) => Math.max(1, p - 1))}
                    disabled={currentPage === 1}
                  >
                    <ChevronLeft className="w-4 h-4" />
                  </Button>
                  <div className="flex items-center gap-1">
                    {Array.from({ length: Math.min(totalPages, 5) }, (_, i) => {
                      let pageNum;
                      if (totalPages <= 5) {
                        pageNum = i + 1;
                      } else if (currentPage <= 3) {
                        pageNum = i + 1;
                      } else if (currentPage >= totalPages - 2) {
                        pageNum = totalPages - 4 + i;
                      } else {
                        pageNum = currentPage - 2 + i;
                      }
                      return (
                        <Button
                          key={pageNum}
                          variant={currentPage === pageNum ? "default" : "outline"}
                          size="sm"
                          onClick={() => setCurrentPage(pageNum)}
                          className="w-8 h-8 p-0"
                        >
                          {pageNum}
                        </Button>
                      );
                    })}
                  </div>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => setCurrentPage((p) => Math.min(totalPages, p + 1))}
                    disabled={currentPage === totalPages}
                  >
                    <ChevronRight className="w-4 h-4" />
                  </Button>
                </div>
              </div>
            </div>
          )}
        </>
      )}
    </div>
  );
}
