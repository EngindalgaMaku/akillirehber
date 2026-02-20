"use client";

import { useEffect, useState, useCallback } from "react";
import { useRouter } from "next/navigation";
import { useAuth } from "@/lib/auth-context";
import { api } from "@/lib/api";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { toast } from "sonner";
import { Settings, Shield, Key, Bot, Loader2, Save, Eye, EyeOff, RefreshCw, HardDrive, ArrowRight } from "lucide-react";

export default function SettingsPage() {
  const router = useRouter();
  const { user } = useAuth();
  const [isLoading, setIsLoading] = useState(true);
  const [isSaving, setIsSaving] = useState(false);
  const [showTeacherKey, setShowTeacherKey] = useState(false);
  const [showStudentKey, setShowStudentKey] = useState(false);
  const [showSecretKey, setShowSecretKey] = useState(false);

  const [formData, setFormData] = useState({
    teacher_registration_key: "",
    student_registration_key: "",
    hcaptcha_site_key: "",
    hcaptcha_secret_key: "",
    captcha_enabled: false,
  });

  const loadSettings = useCallback(async () => {
    try {
      const data = await api.getSystemSettings();
      setFormData({
        teacher_registration_key: data.teacher_registration_key || "",
        student_registration_key: data.student_registration_key || "",
        hcaptcha_site_key: data.hcaptcha_site_key || "",
        hcaptcha_secret_key: "",
        captcha_enabled: data.captcha_enabled,
      });
    } catch (error) {
      console.error("Settings load error:", error);
      toast.error("Ayarlar yüklenirken hata oluştu");
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    if (user && user.role !== "admin") {
      router.push("/dashboard");
      return;
    }
    if (user) {
      loadSettings();
    }
  }, [user, router, loadSettings]);

  const handleSave = async () => {
    setIsSaving(true);
    try {
      const updateData: Record<string, unknown> = {
        teacher_registration_key: formData.teacher_registration_key || null,
        student_registration_key: formData.student_registration_key || null,
        hcaptcha_site_key: formData.hcaptcha_site_key || null,
        captcha_enabled: formData.captcha_enabled,
      };
      
      if (formData.hcaptcha_secret_key) {
        updateData.hcaptcha_secret_key = formData.hcaptcha_secret_key;
      }
      
      await api.updateSystemSettings(updateData);
      toast.success("Ayarlar kaydedildi");
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "Kaydetme hatası");
    } finally {
      setIsSaving(false);
    }
  };

  const generateKey = (field: "teacher_registration_key" | "student_registration_key") => {
    const chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
    let key = "";
    for (let i = 0; i < 12; i++) {
      key += chars.charAt(Math.floor(Math.random() * chars.length));
    }
    setFormData({ ...formData, [field]: key });
  };

  if (!user || user.role !== "admin") {
    return null;
  }

  if (isLoading) {
    return (
      <div className="flex justify-center py-12">
        <Loader2 className="w-6 h-6 text-slate-400 animate-spin" />
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold text-slate-900">Sistem Ayarları</h1>
        <p className="text-slate-600 mt-2">Kayıt anahtarları ve güvenlik ayarlarını yönetin</p>
      </div>

      <div className="grid gap-6">
        {/* Admin Info */}
        <Card>
          <CardHeader>
            <div className="flex items-center gap-2">
              <Shield className="w-5 h-5 text-indigo-600" />
              <CardTitle>Yönetici Erişimi</CardTitle>
            </div>
            <CardDescription>Bu sayfa sadece sistem yöneticileri tarafından görüntülenebilir</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="bg-indigo-50 border border-indigo-200 rounded-lg p-4">
              <div className="flex items-start gap-3">
                <Settings className="w-5 h-5 text-indigo-600 mt-0.5" />
                <div>
                  <p className="text-sm font-medium text-indigo-900">Sistem Yöneticisi: {user.full_name}</p>
                  <p className="text-sm text-indigo-700 mt-1">
                    Kayıt anahtarları ve CAPTCHA ayarlarını buradan yönetebilirsiniz.
                  </p>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Registration Keys */}
        <Card>
          <CardHeader>
            <div className="flex items-center gap-2">
              <Key className="w-5 h-5 text-emerald-600" />
              <CardTitle>Kayıt Anahtarları</CardTitle>
            </div>
            <CardDescription>
              Öğretmen ve öğrenci kayıtları için gerekli anahtarları belirleyin. 
              Boş bırakılırsa anahtar istenmez.
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-6">
            {/* Teacher Key */}
            <div className="space-y-2">
              <Label className="text-sm font-medium text-slate-700">Öğretmen Kayıt Anahtarı</Label>
              <div className="flex gap-2">
                <div className="relative flex-1">
                  <Input
                    type={showTeacherKey ? "text" : "password"}
                    value={formData.teacher_registration_key}
                    onChange={(e) => setFormData({ ...formData, teacher_registration_key: e.target.value })}
                    placeholder="Öğretmen kayıt anahtarı"
                    className="pr-10"
                  />
                  <button
                    type="button"
                    onClick={() => setShowTeacherKey(!showTeacherKey)}
                    className="absolute right-3 top-1/2 -translate-y-1/2 text-slate-400 hover:text-slate-600"
                  >
                    {showTeacherKey ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                  </button>
                </div>
                <Button
                  type="button"
                  variant="outline"
                  size="icon"
                  onClick={() => generateKey("teacher_registration_key")}
                  title="Rastgele anahtar oluştur"
                >
                  <RefreshCw className="w-4 h-4" />
                </Button>
              </div>
              <p className="text-xs text-slate-500">Öğretmen olarak kayıt olmak isteyenler bu anahtarı girmelidir</p>
            </div>

            {/* Student Key */}
            <div className="space-y-2">
              <Label className="text-sm font-medium text-slate-700">Öğrenci Kayıt Anahtarı</Label>
              <div className="flex gap-2">
                <div className="relative flex-1">
                  <Input
                    type={showStudentKey ? "text" : "password"}
                    value={formData.student_registration_key}
                    onChange={(e) => setFormData({ ...formData, student_registration_key: e.target.value })}
                    placeholder="Öğrenci kayıt anahtarı"
                    className="pr-10"
                  />
                  <button
                    type="button"
                    onClick={() => setShowStudentKey(!showStudentKey)}
                    className="absolute right-3 top-1/2 -translate-y-1/2 text-slate-400 hover:text-slate-600"
                  >
                    {showStudentKey ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                  </button>
                </div>
                <Button
                  type="button"
                  variant="outline"
                  size="icon"
                  onClick={() => generateKey("student_registration_key")}
                  title="Rastgele anahtar oluştur"
                >
                  <RefreshCw className="w-4 h-4" />
                </Button>
              </div>
              <p className="text-xs text-slate-500">Öğrenci olarak kayıt olmak isteyenler bu anahtarı girmelidir</p>
            </div>
          </CardContent>
        </Card>

        {/* CAPTCHA Settings */}
        <Card>
          <CardHeader>
            <div className="flex items-center gap-2">
              <Bot className="w-5 h-5 text-purple-600" />
              <CardTitle>CAPTCHA Ayarları</CardTitle>
            </div>
            <CardDescription>
              hCaptcha ile bot koruması. Ücretsiz hesap için{" "}
              <a href="https://www.hcaptcha.com/" target="_blank" rel="noopener noreferrer" className="text-purple-600 hover:underline">
                hcaptcha.com
              </a>
              {" "}adresini ziyaret edin.
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-6">
            {/* Enable CAPTCHA */}
            <div className="flex items-center justify-between">
              <div className="space-y-0.5">
                <Label className="text-sm font-medium text-slate-700">CAPTCHA Aktif</Label>
                <p className="text-xs text-slate-500">Giriş ve kayıt sayfalarında CAPTCHA göster</p>
              </div>
              <Switch
                checked={formData.captcha_enabled}
                onCheckedChange={(checked) => setFormData({ ...formData, captcha_enabled: checked })}
              />
            </div>

            {formData.captcha_enabled && (
              <>
                {/* Site Key */}
                <div className="space-y-2">
                  <Label className="text-sm font-medium text-slate-700">hCaptcha Site Key</Label>
                  <Input
                    type="text"
                    value={formData.hcaptcha_site_key}
                    onChange={(e) => setFormData({ ...formData, hcaptcha_site_key: e.target.value })}
                    placeholder="10000000-ffff-ffff-ffff-000000000001"
                  />
                  <p className="text-xs text-slate-500">hCaptcha dashboard&apos;dan alınan site key</p>
                </div>

                {/* Secret Key */}
                <div className="space-y-2">
                  <Label className="text-sm font-medium text-slate-700">hCaptcha Secret Key</Label>
                  <div className="relative">
                    <Input
                      type={showSecretKey ? "text" : "password"}
                      value={formData.hcaptcha_secret_key}
                      onChange={(e) => setFormData({ ...formData, hcaptcha_secret_key: e.target.value })}
                      placeholder="Yeni secret key girin (mevcut değiştirilecek)"
                      className="pr-10"
                    />
                    <button
                      type="button"
                      onClick={() => setShowSecretKey(!showSecretKey)}
                      className="absolute right-3 top-1/2 -translate-y-1/2 text-slate-400 hover:text-slate-600"
                    >
                      {showSecretKey ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                    </button>
                  </div>
                  <p className="text-xs text-slate-500">hCaptcha dashboard&apos;dan alınan secret key (güvenlik için gösterilmez)</p>
                </div>
              </>
            )}
          </CardContent>
        </Card>

        {/* Save Button */}
        <Card className="bg-gradient-to-r from-slate-50 to-white">
          <CardContent className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <h4 className="font-semibold text-slate-900">Ayarları Kaydet</h4>
                <p className="text-sm text-slate-500 mt-1">
                  Değişiklikler hemen uygulanacak
                </p>
              </div>
              <Button 
                onClick={handleSave} 
                disabled={isSaving} 
                size="lg"
                className="bg-gradient-to-r from-indigo-600 to-indigo-700 hover:from-indigo-700 hover:to-indigo-800"
              >
                {isSaving ? (
                  <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                ) : (
                  <Save className="w-4 h-4 mr-2" />
                )}
                Kaydet
              </Button>
            </div>
          </CardContent>
        </Card>

        {/* Backup Link */}
        <Card className="border-blue-200 bg-blue-50/50">
          <CardContent className="p-6">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 rounded-lg bg-blue-600 flex items-center justify-center">
                  <HardDrive className="w-5 h-5 text-white" />
                </div>
                <div>
                  <h4 className="font-semibold text-slate-900">Veritabanı Yedekleme</h4>
                  <p className="text-sm text-slate-600 mt-1">
                    Yedekleri oluşturun, indirin ve geri yükleyin
                  </p>
                </div>
              </div>
              <Button 
                onClick={() => router.push("/dashboard/backup")}
                size="lg"
                className="bg-blue-600 hover:bg-blue-700"
              >
                Yedekleme Sayfası
                <ArrowRight className="w-4 h-4 ml-2" />
              </Button>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
