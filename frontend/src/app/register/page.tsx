"use client";

import { useState, useEffect, useRef } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import HCaptcha from "@hcaptcha/react-hcaptcha";
import { useAuth } from "@/lib/auth-context";
import { api, PublicSettings } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { toast } from "sonner";
import { Brain, GraduationCap, BookOpen } from "lucide-react";

export default function RegisterPage() {
  const router = useRouter();
  const { register } = useAuth();
  const captchaRef = useRef<HCaptcha>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [publicSettings, setPublicSettings] = useState<PublicSettings | null>(null);
  const [captchaToken, setCaptchaToken] = useState<string | null>(null);
  const [formData, setFormData] = useState({
    email: "",
    password: "",
    confirmPassword: "",
    full_name: "",
    role: "student" as "teacher" | "student",
    registration_key: "",
  });

  useEffect(() => {
    api.getPublicSettings()
      .then(s => setPublicSettings(s))
      .catch(() => setPublicSettings({ 
        captcha_enabled: false, 
        hcaptcha_site_key: null, 
        registration_key_required: true 
      }));
  }, []);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (formData.password !== formData.confirmPassword) {
      toast.error("Sifreler eslesmiyor");
      return;
    }
    if (formData.password.length < 6) {
      toast.error("Sifre en az 6 karakter olmali");
      return;
    }
    if (publicSettings?.captcha_enabled && !captchaToken) {
      toast.error("CAPTCHA gerekli");
      return;
    }
    if (publicSettings?.registration_key_required) {
      try {
        const r = await api.verifyRegistrationKey(formData.role, formData.registration_key);
        if (!r.valid) {
          toast.error("Gecersiz kayit anahtari");
          return;
        }
      } catch {
        toast.error("Anahtar dogrulanamadi");
        return;
      }
    }
    setIsLoading(true);
    try {
      await register({
        email: formData.email,
        password: formData.password,
        full_name: formData.full_name,
        role: formData.role,
      });
      router.push("/dashboard");
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Hata");
    } finally {
      setIsLoading(false);
    }
  };

  const subtitle = publicSettings 
    ? (publicSettings.registration_key_required ? "Kayit anahtari gerekli" : "") 
    : "Yukleniyor...";

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 flex items-center justify-center p-4">
      <div className="w-full max-w-md">
        <div className="text-center mb-8">
          <Brain className="w-16 h-16 text-purple-400 mx-auto mb-4" />
          <h1 className="text-3xl font-bold text-white">Kayit Ol</h1>
          <p className="text-gray-400 mt-2">{subtitle}</p>
        </div>

        <form onSubmit={handleSubmit} className="bg-white/10 backdrop-blur rounded-2xl p-6 space-y-4">
          <div>
            <Label className="text-white">Ad Soyad</Label>
            <Input
              value={formData.full_name}
              onChange={e => setFormData({...formData, full_name: e.target.value})}
              required
              className="bg-white/5 border-white/20 text-white"
            />
          </div>

          <div>
            <Label className="text-white">E-posta</Label>
            <Input
              type="email"
              value={formData.email}
              onChange={e => setFormData({...formData, email: e.target.value})}
              required
              className="bg-white/5 border-white/20 text-white"
            />
          </div>

          <div>
            <Label className="text-white">Sifre</Label>
            <Input
              type="password"
              value={formData.password}
              onChange={e => setFormData({...formData, password: e.target.value})}
              required
              className="bg-white/5 border-white/20 text-white"
            />
          </div>

          <div>
            <Label className="text-white">Sifre Tekrar</Label>
            <Input
              type="password"
              value={formData.confirmPassword}
              onChange={e => setFormData({...formData, confirmPassword: e.target.value})}
              required
              className="bg-white/5 border-white/20 text-white"
            />
          </div>

          <div className="grid grid-cols-2 gap-2">
            <button
              type="button"
              onClick={() => setFormData({...formData, role: "student", registration_key: ""})}
              className={`p-3 rounded-lg border ${formData.role === "student" ? "border-purple-500 bg-purple-500/20" : "border-white/20"}`}
            >
              <GraduationCap className="w-6 h-6 mx-auto text-purple-400" />
              <span className="text-white text-sm block mt-1">Ogrenci</span>
            </button>
            <button
              type="button"
              onClick={() => setFormData({...formData, role: "teacher", registration_key: ""})}
              className={`p-3 rounded-lg border ${formData.role === "teacher" ? "border-pink-500 bg-pink-500/20" : "border-white/20"}`}
            >
              <BookOpen className="w-6 h-6 mx-auto text-pink-400" />
              <span className="text-white text-sm block mt-1">Ogretmen</span>
            </button>
          </div>

          {publicSettings?.registration_key_required && (
            <div>
              <Label className="text-white">
                {formData.role === "teacher" ? "Ogretmen" : "Ogrenci"} Kayit Anahtari
              </Label>
              <Input
                value={formData.registration_key}
                onChange={e => setFormData({...formData, registration_key: e.target.value})}
                required
                placeholder="Anahtari girin"
                className="bg-white/5 border-white/20 text-white"
              />
            </div>
          )}

          {publicSettings?.captcha_enabled && publicSettings.hcaptcha_site_key && (
            <div className="flex justify-center">
              <HCaptcha
                ref={captchaRef}
                sitekey={publicSettings.hcaptcha_site_key}
                onVerify={t => setCaptchaToken(t)}
                theme="dark"
              />
            </div>
          )}

          <Button
            type="submit"
            disabled={isLoading}
            className="w-full bg-purple-600 hover:bg-purple-700"
          >
            {isLoading ? "Yukleniyor..." : "Kayit Ol"}
          </Button>
        </form>

        <p className="text-center text-gray-400 mt-4">
          Hesabiniz var mi?{" "}
          <Link href="/login" className="text-purple-400">Giris Yap</Link>
        </p>
      </div>
    </div>
  );
}
