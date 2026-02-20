'use client';

import { useState } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Upload, Database, CheckCircle, XCircle, Loader2, Terminal, Copy } from 'lucide-react';

interface UploadResult {
  success: boolean;
  message: string;
  filename?: string;
  path?: string;
}

export default function RestorePage() {
  const [postgresFile, setPostgresFile] = useState<File | null>(null);
  const [weaviateFile, setWeaviateFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [postgresResult, setPostgresResult] = useState<UploadResult | null>(null);
  const [weaviateResult, setWeaviateResult] = useState<UploadResult | null>(null);

  const handlePostgresChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files?.[0]) {
      setPostgresFile(e.target.files[0]);
      setPostgresResult(null);
    }
  };

  const handleWeaviateChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files?.[0]) {
      setWeaviateFile(e.target.files[0]);
      setWeaviateResult(null);
    }
  };

  const handleUpload = async () => {
    if (!postgresFile && !weaviateFile) {
      return;
    }

    setLoading(true);

    try {
      const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
      
      // Upload PostgreSQL
      if (postgresFile) {
        const formData = new FormData();
        formData.append('file', postgresFile);

        const pgResponse = await fetch(`${apiUrl}/api/restore/upload/postgres`, {
          method: 'POST',
          body: formData,
        });

        const pgData = await pgResponse.json();
        
        if (pgResponse.ok) {
          setPostgresResult({
            success: true,
            message: pgData.message,
            filename: pgData.details?.filename,
            path: pgData.details?.path,
          });
        } else {
          setPostgresResult({
            success: false,
            message: pgData.detail || 'Upload başarısız',
          });
        }
      }

      // Upload Weaviate
      if (weaviateFile) {
        const formData = new FormData();
        formData.append('file', weaviateFile);

        const wvResponse = await fetch(`${apiUrl}/api/restore/upload/weaviate`, {
          method: 'POST',
          body: formData,
        });

        const wvData = await wvResponse.json();
        
        if (wvResponse.ok) {
          setWeaviateResult({
            success: true,
            message: wvData.message,
            filename: wvData.details?.filename,
            path: wvData.details?.path,
          });
        } else {
          setWeaviateResult({
            success: false,
            message: wvData.detail || 'Upload başarısız',
          });
        }
      }
    } catch (error) {
      const errorMsg = error instanceof Error ? error.message : 'Upload işlemi başarısız';
      if (postgresFile && !postgresResult) {
        setPostgresResult({ success: false, message: errorMsg });
      }
      if (weaviateFile && !weaviateResult) {
        setWeaviateResult({ success: false, message: errorMsg });
      }
    } finally {
      setLoading(false);
    }
  };

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text);
  };

  const getRestoreCommands = () => {
    const commands: string[] = [];
    
    if (postgresResult?.success && postgresResult.filename) {
      const filename = postgresResult.filename;
      const isZip = filename.endsWith('.zip');
      
      if (isZip) {
        commands.push(
          '# PostgreSQL Restore (ZIP dosyası)',
          `cd /app/backups/uploads`,
          `unzip ${filename}`,
          `psql -U raguser -d ragchatbot -f *.sql`,
          ''
        );
      } else {
        commands.push(
          '# PostgreSQL Restore',
          `psql -U raguser -d ragchatbot -f /app/backups/uploads/${filename}`,
          ''
        );
      }
    }
    
    if (weaviateResult?.success && weaviateResult.filename) {
      commands.push(
        '# Weaviate Restore',
        `# Weaviate restore için backend API kullanılmalı`,
        `# Dosya: /app/backups/uploads/${weaviateResult.filename}`,
        ''
      );
    }
    
    return commands.join('\n');
  };

  const showCommands = (postgresResult?.success || weaviateResult?.success);

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 flex items-center justify-center p-4">
      <Card className="w-full max-w-3xl">
        <CardHeader>
          <CardTitle className="text-2xl flex items-center gap-2">
            <Database className="h-6 w-6" />
            Database Restore - File Upload
          </CardTitle>
          <CardDescription>
            Backup dosyalarını yükleyin, sonra terminalden restore edin
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          {/* PostgreSQL Backup */}
          <div className="space-y-2">
            <label className="text-sm font-medium">PostgreSQL Backup (.sql veya .zip)</label>
            <div className="flex items-center gap-2">
              <input
                type="file"
                accept=".sql,.zip"
                onChange={handlePostgresChange}
                className="flex-1 text-sm file:mr-4 file:py-2 file:px-4 file:rounded-md file:border-0 file:text-sm file:font-semibold file:bg-blue-50 file:text-blue-700 hover:file:bg-blue-100"
              />
              {postgresFile && (
                <CheckCircle className="h-5 w-5 text-green-500" />
              )}
            </div>
            {postgresFile && (
              <p className="text-xs text-gray-500">
                Seçilen: {postgresFile.name} ({(postgresFile.size / 1024 / 1024).toFixed(2)} MB)
              </p>
            )}
            {postgresResult && (
              <Alert variant={postgresResult.success ? 'default' : 'destructive'}>
                {postgresResult.success ? (
                  <CheckCircle className="h-4 w-4" />
                ) : (
                  <XCircle className="h-4 w-4" />
                )}
                <AlertDescription>{postgresResult.message}</AlertDescription>
              </Alert>
            )}
          </div>

          {/* Weaviate Backup */}
          <div className="space-y-2">
            <label className="text-sm font-medium">Weaviate Backup (.json veya .zip)</label>
            <div className="flex items-center gap-2">
              <input
                type="file"
                accept=".json,.zip"
                onChange={handleWeaviateChange}
                className="flex-1 text-sm file:mr-4 file:py-2 file:px-4 file:rounded-md file:border-0 file:text-sm file:font-semibold file:bg-blue-50 file:text-blue-700 hover:file:bg-blue-100"
              />
              {weaviateFile && (
                <CheckCircle className="h-5 w-5 text-green-500" />
              )}
            </div>
            {weaviateFile && (
              <p className="text-xs text-gray-500">
                Seçilen: {weaviateFile.name} ({(weaviateFile.size / 1024 / 1024).toFixed(2)} MB)
              </p>
            )}
            {weaviateResult && (
              <Alert variant={weaviateResult.success ? 'default' : 'destructive'}>
                {weaviateResult.success ? (
                  <CheckCircle className="h-4 w-4" />
                ) : (
                  <XCircle className="h-4 w-4" />
                )}
                <AlertDescription>{weaviateResult.message}</AlertDescription>
              </Alert>
            )}
          </div>

          {/* Upload Button */}
          <Button
            onClick={handleUpload}
            disabled={loading || (!postgresFile && !weaviateFile)}
            className="w-full"
            size="lg"
          >
            {loading ? (
              <>
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                Yükleniyor...
              </>
            ) : (
              <>
                <Upload className="mr-2 h-4 w-4" />
                Dosyaları Yükle
              </>
            )}
          </Button>

          {/* Restore Commands */}
          {showCommands && (
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <label className="text-sm font-medium flex items-center gap-2">
                  <Terminal className="h-4 w-4" />
                  Restore Komutları
                </label>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => copyToClipboard(getRestoreCommands())}
                >
                  <Copy className="h-3 w-3 mr-1" />
                  Kopyala
                </Button>
              </div>
              <pre className="bg-gray-900 text-green-400 p-4 rounded-md text-xs overflow-x-auto">
                {getRestoreCommands()}
              </pre>
              <p className="text-xs text-gray-500">
                Coolify terminalinde backend container&apos;a girin ve yukarıdaki komutları çalıştırın
              </p>
            </div>
          )}

          {/* Info */}
          <div className="text-xs text-gray-500 space-y-1 border-t pt-4">
            <p className="font-medium">Kullanım:</p>
            <p>1. Backup dosyalarını seçin ve yükleyin</p>
            <p>2. Coolify&apos;da backend container terminaline girin</p>
            <p>3. Yukarıdaki restore komutlarını çalıştırın</p>
            <p>4. Restore tamamlandıktan sonra /login sayfasından giriş yapın</p>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
