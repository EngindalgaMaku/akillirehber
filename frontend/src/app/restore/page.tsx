'use client';

import { useState } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Upload, Database, CheckCircle, XCircle, Loader2 } from 'lucide-react';

export default function RestorePage() {
  const [postgresFile, setPostgresFile] = useState<File | null>(null);
  const [weaviateFile, setWeaviateFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<{ success: boolean; message: string } | null>(null);

  const handlePostgresChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setPostgresFile(e.target.files[0]);
    }
  };

  const handleWeaviateChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setWeaviateFile(e.target.files[0]);
    }
  };

  const handleRestore = async () => {
    if (!postgresFile && !weaviateFile) {
      setResult({ success: false, message: 'En az bir backup dosyası seçmelisiniz' });
      return;
    }

    setLoading(true);
    setResult(null);

    try {
      const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
      
      // PostgreSQL restore
      if (postgresFile) {
        const formData = new FormData();
        formData.append('file', postgresFile);

        const pgResponse = await fetch(`${apiUrl}/api/restore/postgres`, {
          method: 'POST',
          body: formData,
        });

        if (!pgResponse.ok) {
          throw new Error('PostgreSQL restore başarısız');
        }
      }

      // Weaviate restore
      if (weaviateFile) {
        const formData = new FormData();
        formData.append('file', weaviateFile);

        const wvResponse = await fetch(`${apiUrl}/api/restore/weaviate`, {
          method: 'POST',
          body: formData,
        });

        if (!wvResponse.ok) {
          throw new Error('Weaviate restore başarısız');
        }
      }

      setResult({
        success: true,
        message: 'Backup başarıyla restore edildi! Artık giriş yapabilirsiniz.',
      });
    } catch (error) {
      setResult({
        success: false,
        message: error instanceof Error ? error.message : 'Restore işlemi başarısız',
      });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 flex items-center justify-center p-4">
      <Card className="w-full max-w-2xl">
        <CardHeader>
          <CardTitle className="text-2xl flex items-center gap-2">
            <Database className="h-6 w-6" />
            Database Restore
          </CardTitle>
          <CardDescription>
            İlk kurulum için backup dosyalarını yükleyin
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
          </div>

          {/* Result Message */}
          {result && (
            <Alert variant={result.success ? 'default' : 'destructive'}>
              {result.success ? (
                <CheckCircle className="h-4 w-4" />
              ) : (
                <XCircle className="h-4 w-4" />
              )}
              <AlertDescription>{result.message}</AlertDescription>
            </Alert>
          )}

          {/* Restore Button */}
          <Button
            onClick={handleRestore}
            disabled={loading || (!postgresFile && !weaviateFile)}
            className="w-full"
            size="lg"
          >
            {loading ? (
              <>
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                Restore ediliyor...
              </>
            ) : (
              <>
                <Upload className="mr-2 h-4 w-4" />
                Restore Et
              </>
            )}
          </Button>

          {/* Info */}
          <div className="text-xs text-gray-500 space-y-1">
            <p>• Bu sayfa sadece ilk kurulum için kullanılır</p>
            <p>• Dosya boyutu limiti yoktur</p>
            <p>• Hem .sql/.json hem de .zip formatları desteklenir</p>
            <p>• Restore sonrası /login sayfasından giriş yapabilirsiniz</p>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
