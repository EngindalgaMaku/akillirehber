"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { useAuth } from "@/lib/auth-context";
import { api, BackupInfo } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { toast } from "sonner";
import {
  Database,
  Download,
  Upload,
  Trash2,
  Loader2,
  HardDrive,
  FileArchive,
  AlertTriangle,
  CheckCircle2,
  RefreshCw,
} from "lucide-react";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";

export default function BackupPage() {
  const { user, isLoading: authLoading } = useAuth();
  const router = useRouter();

  const [backups, setBackups] = useState<BackupInfo[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [isCreating, setIsCreating] = useState(false);
  const [isRestoring, setIsRestoring] = useState(false);
  const [deleteDialogOpen, setDeleteDialogOpen] = useState(false);
  const [backupToDelete, setBackupToDelete] = useState<BackupInfo | null>(null);
  const [isDeleting, setIsDeleting] = useState(false);
  const [restoreDialogOpen, setRestoreDialogOpen] = useState(false);
  const [restoreType, setRestoreType] = useState<"postgres" | "weaviate">("postgres");
  const [restoreFile, setRestoreFile] = useState<File | null>(null);

  useEffect(() => {
    if (!authLoading && user?.role !== "admin") {
      toast.error("Bu sayfaya erişim yetkiniz yok");
      router.push("/dashboard");
    }
  }, [user, authLoading, router]);

  const fetchBackups = async () => {
    setIsLoading(true);
    try {
      const response = await api.listBackups();
      setBackups(response.backups);
    } catch (error) {
      toast.error("Yedekler yüklenirken hata oluştu");
      console.error("Failed to fetch backups:", error);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    if (user?.role === "admin") {
      fetchBackups();
    }
  }, [user]);

  const handleCreateBackup = async (type: "postgres" | "weaviate" | "full") => {
    setIsCreating(true);
    try {
      if (type === "postgres") {
        await api.createPostgresBackup();
        toast.success("PostgreSQL yedeği oluşturuldu");
      } else if (type === "weaviate") {
        await api.createWeaviateBackup();
        toast.success("Weaviate yedeği oluşturuldu");
      } else {
        await api.createFullBackup();
        toast.success("Tam yedek oluşturuldu");
      }
      fetchBackups();
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "Yedek oluşturulurken hata oluştu");
    } finally {
      setIsCreating(false);
    }
  };

  const handleDownloadBackup = async (filename: string) => {
    try {
      const blob = await api.downloadBackup(filename);
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = filename;
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);
      toast.success("Yedek indiriliyor");
    } catch {
      toast.error("Yedek indirilirken hata oluştu");
    }
  };

  const handleDeleteClick = (backup: BackupInfo) => {
    setBackupToDelete(backup);
    setDeleteDialogOpen(true);
  };

  const handleConfirmDelete = async () => {
    if (!backupToDelete) return;
    setIsDeleting(true);
    try {
      await api.deleteBackup(backupToDelete.filename);
      toast.success("Yedek silindi");
      setDeleteDialogOpen(false);
      setBackupToDelete(null);
      fetchBackups();
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "Silme sırasında hata oluştu");
    } finally {
      setIsDeleting(false);
    }
  };

  const handleRestoreClick = (type: "postgres" | "weaviate") => {
    setRestoreType(type);
    setRestoreDialogOpen(true);
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files?.[0]) {
      setRestoreFile(e.target.files[0]);
    }
  };

  const handleConfirmRestore = async () => {
    if (!restoreFile) {
      toast.error("Lütfen bir dosya seçin");
      return;
    }
    setIsRestoring(true);
    try {
      if (restoreType === "postgres") {
        await api.restorePostgresBackup(restoreFile);
        toast.success("PostgreSQL restore edildi");
      } else {
        await api.restoreWeaviateBackup(restoreFile);
        toast.success("Weaviate restore edildi");
      }
      setRestoreDialogOpen(false);
      setRestoreFile(null);
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "Restore sırasında hata oluştu");
    } finally {
      setIsRestoring(false);
    }
  };

  const formatBytes = (bytes: number) => {
    if (bytes === 0) return "0 Bytes";
    const k = 1024;
    const sizes = ["Bytes", "KB", "MB", "GB"];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return Math.round(bytes / Math.pow(k, i) * 100) / 100 + " " + sizes[i];
  };

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleString("tr-TR");
  };

  const getBackupIcon = (type: string) => {
    switch (type) {
      case "postgres": return <Database className="h-5 w-5" />;
      case "weaviate": return <HardDrive className="h-5 w-5" />;
      case "full": return <FileArchive className="h-5 w-5" />;
      default: return <FileArchive className="h-5 w-5" />;
    }
  };

  if (authLoading || user?.role !== "admin") {
    return null;
  }

  return (
    <div className="container mx-auto py-8 px-4">
      <div className="mb-8">
        <h1 className="text-3xl font-bold mb-2">Yedekleme ve Geri Yükleme</h1>
        <p className="text-muted-foreground">Veritabanı yedeklerini oluşturun, indirin ve geri yükleyin</p>
      </div>

      <div className="grid gap-6 md:grid-cols-3 mb-8">
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2"><Database className="h-5 w-5" />PostgreSQL</CardTitle>
            <CardDescription>Kullanıcılar, kurslar ve ayarlar</CardDescription>
          </CardHeader>
          <CardContent className="space-y-2">
            <Button onClick={() => handleCreateBackup("postgres")} disabled={isCreating} className="w-full">
              {isCreating ? <><Loader2 className="mr-2 h-4 w-4 animate-spin" />Oluşturuluyor...</> : <><CheckCircle2 className="mr-2 h-4 w-4" />Yedek Oluştur</>}
            </Button>
            <Button onClick={() => handleRestoreClick("postgres")} variant="outline" className="w-full">
              <Upload className="mr-2 h-4 w-4" />Geri Yükle
            </Button>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2"><HardDrive className="h-5 w-5" />Weaviate</CardTitle>
            <CardDescription>Vektör veritabanı ve embeddings</CardDescription>
          </CardHeader>
          <CardContent className="space-y-2">
            <Button onClick={() => handleCreateBackup("weaviate")} disabled={isCreating} className="w-full">
              {isCreating ? <><Loader2 className="mr-2 h-4 w-4 animate-spin" />Oluşturuluyor...</> : <><CheckCircle2 className="mr-2 h-4 w-4" />Yedek Oluştur</>}
            </Button>
            <Button onClick={() => handleRestoreClick("weaviate")} variant="outline" className="w-full">
              <Upload className="mr-2 h-4 w-4" />Geri Yükle
            </Button>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2"><FileArchive className="h-5 w-5" />Tam Yedek</CardTitle>
            <CardDescription>PostgreSQL + Weaviate birlikte</CardDescription>
          </CardHeader>
          <CardContent>
            <Button onClick={() => handleCreateBackup("full")} disabled={isCreating} className="w-full">
              {isCreating ? <><Loader2 className="mr-2 h-4 w-4 animate-spin" />Oluşturuluyor...</> : <><CheckCircle2 className="mr-2 h-4 w-4" />Tam Yedek Oluştur</>}
            </Button>
          </CardContent>
        </Card>
      </div>

      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <div>
              <CardTitle>Mevcut Yedekler</CardTitle>
              <CardDescription>Toplam {backups.length} yedek dosyası</CardDescription>
            </div>
            <Button onClick={fetchBackups} variant="outline" size="sm" disabled={isLoading}>
              <RefreshCw className={`h-4 w-4 mr-2 ${isLoading ? "animate-spin" : ""}`} />Yenile
            </Button>
          </div>
        </CardHeader>
        <CardContent>
          {isLoading ? (
            <div className="flex items-center justify-center py-8"><Loader2 className="h-8 w-8 animate-spin text-muted-foreground" /></div>
          ) : backups.length === 0 ? (
            <div className="text-center py-8 text-muted-foreground"><FileArchive className="h-12 w-12 mx-auto mb-4 opacity-50" /><p>Henüz yedek dosyası yok</p></div>
          ) : (
            <div className="space-y-2">
              {backups.map((backup) => (
                <div key={backup.filename} className="flex items-center justify-between p-4 border rounded-lg hover:bg-accent/50 transition-colors">
                  <div className="flex items-center gap-4">
                    <div className="p-2 bg-primary/10 rounded-lg">{getBackupIcon(backup.type)}</div>
                    <div>
                      <p className="font-medium">{backup.filename}</p>
                      <p className="text-sm text-muted-foreground">{formatBytes(backup.size)} • {formatDate(backup.created_at)}</p>
                    </div>
                  </div>
                  <div className="flex items-center gap-2">
                    <Button onClick={() => handleDownloadBackup(backup.filename)} variant="outline" size="sm"><Download className="h-4 w-4" /></Button>
                    <Button onClick={() => handleDeleteClick(backup)} variant="outline" size="sm" className="text-destructive hover:text-destructive"><Trash2 className="h-4 w-4" /></Button>
                  </div>
                </div>
              ))}
            </div>
          )}
        </CardContent>
      </Card>

      <Dialog open={deleteDialogOpen} onOpenChange={setDeleteDialogOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2"><AlertTriangle className="h-5 w-5 text-destructive" />Yedek Silinecek</DialogTitle>
            <DialogDescription><strong>{backupToDelete?.filename}</strong> dosyasını silmek istediğinizden emin misiniz? Bu işlem geri alınamaz.</DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <Button variant="outline" onClick={() => setDeleteDialogOpen(false)} disabled={isDeleting}>İptal</Button>
            <Button variant="destructive" onClick={handleConfirmDelete} disabled={isDeleting}>
              {isDeleting ? <><Loader2 className="mr-2 h-4 w-4 animate-spin" />Siliniyor...</> : "Sil"}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      <Dialog open={restoreDialogOpen} onOpenChange={setRestoreDialogOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2"><Upload className="h-5 w-5" />{restoreType === "postgres" ? "PostgreSQL" : "Weaviate"} Geri Yükle</DialogTitle>
            <DialogDescription>
              <div className="flex items-start gap-2 p-3 bg-destructive/10 border border-destructive/20 rounded-lg mb-4">
                <AlertTriangle className="h-5 w-5 text-destructive shrink-0 mt-0.5" />
                <div className="text-sm"><strong>Uyarı:</strong> Bu işlem mevcut verilerin üzerine yazacaktır. Devam etmeden önce mevcut verilerin yedeğini aldığınızdan emin olun.</div>
              </div>
              Yüklemek istediğiniz yedek dosyasını seçin.
            </DialogDescription>
          </DialogHeader>
          <div className="space-y-4">
            <div>
              <Label htmlFor="restore-file">Yedek Dosyası</Label>
              <Input id="restore-file" type="file" accept={restoreType === "postgres" ? ".sql" : ".json"} onChange={handleFileChange} disabled={isRestoring} />
            </div>
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => { setRestoreDialogOpen(false); setRestoreFile(null); }} disabled={isRestoring}>İptal</Button>
            <Button onClick={handleConfirmRestore} disabled={!restoreFile || isRestoring}>
              {isRestoring ? <><Loader2 className="mr-2 h-4 w-4 animate-spin" />Geri Yükleniyor...</> : "Geri Yükle"}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}
