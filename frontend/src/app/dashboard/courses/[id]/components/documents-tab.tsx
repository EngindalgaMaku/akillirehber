"use client";

import { useEffect, useState, useCallback } from "react";
import { api, Document } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { toast } from "sonner";
import { Upload, FileText, Trash2, Loader2 } from "lucide-react";

interface DocumentsTabProps {
  courseId: number;
  isOwner: boolean;
}

export function DocumentsTab({ courseId, isOwner }: DocumentsTabProps) {
  const [documents, setDocuments] = useState<Document[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [isUploading, setIsUploading] = useState(false);
  const [deletingId, setDeletingId] = useState<number | null>(null);

  const loadDocuments = useCallback(async () => {
    try {
      const data = await api.getCourseDocuments(courseId);
      setDocuments(data);
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
    const file = e.target.files?.[0];
    if (!file) return;

    setIsUploading(true);
    try {
      const doc = await api.uploadDocument(courseId, file);
      setDocuments([doc, ...documents]);
      toast.success("Doküman yüklendi");
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "Yükleme hatası");
    } finally {
      setIsUploading(false);
      e.target.value = "";
    }
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

  return (
    <div className="bg-white rounded-lg border border-slate-200">
      <div className="flex items-center justify-between px-6 py-4 border-b border-slate-200">
        <h2 className="font-medium text-slate-900">Dokümanlar ({documents.length})</h2>
        {isOwner && (
          <label className="cursor-pointer">
            <input
              type="file"
              className="hidden"
              accept=".pdf,.md,.docx,.txt"
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
                Yükle
              </span>
            </Button>
          </label>
        )}
      </div>

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
        <div className="divide-y divide-slate-200">
          {documents.map((doc) => (
            <div
              key={doc.id}
              className="px-6 py-4 flex items-center justify-between hover:bg-slate-50"
            >
              <div className="flex items-center gap-3">
                <FileText className="w-5 h-5 text-slate-400" />
                <div>
                  <p className="font-medium text-slate-900">{doc.original_filename}</p>
                  <p className="text-xs text-slate-500">
                    {(doc.file_size / 1024).toFixed(1)} KB •{" "}
                    {doc.char_count ? `${doc.char_count.toLocaleString()} karakter • ` : ""}
                    {new Date(doc.created_at).toLocaleDateString("tr-TR")}
                  </p>
                </div>
              </div>
              <div className="flex items-center gap-3">
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
      )}
    </div>
  );
}
