"use client";

import { useState } from "react";
import { useAuth } from "@/lib/auth-context";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { User, Mail, Shield, Lock, Edit2, X, Check } from "lucide-react";
import { toast } from "sonner";
import { api } from "@/lib/api";

export default function ProfilePage() {
  const { user, refreshUser } = useAuth();
  const [isEditingProfile, setIsEditingProfile] = useState(false);
  const [isChangingPassword, setIsChangingPassword] = useState(false);
  const [loading, setLoading] = useState(false);

  // Profile form state
  const [profileData, setProfileData] = useState({
    full_name: user?.full_name || "",
    email: user?.email || "",
  });

  // Password form state
  const [passwordData, setPasswordData] = useState({
    current_password: "",
    new_password: "",
    confirm_password: "",
  });

  if (!user) return null;

  const handleProfileUpdate = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);

    try {
      await api.updateProfile({
        full_name: profileData.full_name,
        email: profileData.email,
      });

      await refreshUser();
      toast.success("Profil başarıyla güncellendi");
      setIsEditingProfile(false);
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : "Profil güncellenirken bir hata oluştu";
      toast.error(errorMessage);
    } finally {
      setLoading(false);
    }
  };

  const handlePasswordChange = async (e: React.FormEvent) => {
    e.preventDefault();

    if (passwordData.new_password !== passwordData.confirm_password) {
      toast.error("Yeni şifreler eşleşmiyor");
      return;
    }

    if (passwordData.new_password.length < 6) {
      toast.error("Yeni şifre en az 6 karakter olmalıdır");
      return;
    }

    setLoading(true);

    try {
      await api.changePassword({
        current_password: passwordData.current_password,
        new_password: passwordData.new_password,
      });

      toast.success("Şifre başarıyla değiştirildi");
      setIsChangingPassword(false);
      setPasswordData({
        current_password: "",
        new_password: "",
        confirm_password: "",
      });
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : "Şifre değiştirilirken bir hata oluştu";
      toast.error(errorMessage);
    } finally {
      setLoading(false);
    }
  };

  const cancelProfileEdit = () => {
    setProfileData({
      full_name: user.full_name,
      email: user.email,
    });
    setIsEditingProfile(false);
  };

  const cancelPasswordChange = () => {
    setPasswordData({
      current_password: "",
      new_password: "",
      confirm_password: "",
    });
    setIsChangingPassword(false);
  };

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold text-slate-900">Profil</h1>
        <p className="text-slate-600 mt-2">Hesap bilgilerinizi yönetin</p>
      </div>

      <div className="grid gap-6">
        {/* Profile Information Card */}
        <Card>
          <CardHeader>
            <div className="flex items-center justify-between">
              <div>
                <CardTitle>Kişisel Bilgiler</CardTitle>
                <CardDescription>Hesabınıza ait temel bilgiler</CardDescription>
              </div>
              {!isEditingProfile && (
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => setIsEditingProfile(true)}
                >
                  <Edit2 className="w-4 h-4 mr-2" />
                  Düzenle
                </Button>
              )}
            </div>
          </CardHeader>
          <CardContent>
            {isEditingProfile ? (
              <form onSubmit={handleProfileUpdate} className="space-y-4">
                <div className="space-y-2">
                  <Label htmlFor="full_name">Ad Soyad</Label>
                  <Input
                    id="full_name"
                    value={profileData.full_name}
                    onChange={(e) =>
                      setProfileData({ ...profileData, full_name: e.target.value })
                    }
                    required
                  />
                </div>

                <div className="space-y-2">
                  <Label htmlFor="email">E-posta</Label>
                  <Input
                    id="email"
                    type="email"
                    value={profileData.email}
                    onChange={(e) =>
                      setProfileData({ ...profileData, email: e.target.value })
                    }
                    required
                  />
                </div>

                <div className="flex gap-2">
                  <Button type="submit" disabled={loading}>
                    <Check className="w-4 h-4 mr-2" />
                    Kaydet
                  </Button>
                  <Button
                    type="button"
                    variant="outline"
                    onClick={cancelProfileEdit}
                    disabled={loading}
                  >
                    <X className="w-4 h-4 mr-2" />
                    İptal
                  </Button>
                </div>
              </form>
            ) : (
              <div className="space-y-6">
                <div className="flex items-start gap-4">
                  <div className="w-16 h-16 rounded-full bg-indigo-600 flex items-center justify-center text-white text-2xl font-semibold flex-shrink-0">
                    {user.full_name.charAt(0).toUpperCase()}
                  </div>
                  <div className="flex-1">
                    <div className="flex items-center gap-2 mb-1">
                      <User className="w-4 h-4 text-slate-500" />
                      <span className="text-sm font-medium text-slate-700">Ad Soyad</span>
                    </div>
                    <p className="text-lg font-semibold text-slate-900">{user.full_name}</p>
                  </div>
                </div>

                <div className="border-t pt-6">
                  <div className="flex items-center gap-2 mb-2">
                    <Mail className="w-4 h-4 text-slate-500" />
                    <span className="text-sm font-medium text-slate-700">E-posta</span>
                  </div>
                  <p className="text-slate-900">{user.email}</p>
                </div>

                <div className="border-t pt-6">
                  <div className="flex items-center gap-2 mb-2">
                    <Shield className="w-4 h-4 text-slate-500" />
                    <span className="text-sm font-medium text-slate-700">Rol</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <span
                      className={`inline-flex items-center px-3 py-1 rounded-full text-sm font-medium ${
                        user.role === "teacher"
                          ? "bg-indigo-100 text-indigo-800"
                          : "bg-green-100 text-green-800"
                      }`}
                    >
                      {user.role === "teacher" ? "Öğretmen" : "Öğrenci"}
                    </span>
                  </div>
                </div>
              </div>
            )}
          </CardContent>
        </Card>

        {/* Password Change Card */}
        <Card>
          <CardHeader>
            <div className="flex items-center justify-between">
              <div>
                <CardTitle>Şifre Değiştir</CardTitle>
                <CardDescription>Hesap güvenliğiniz için şifrenizi güncelleyin</CardDescription>
              </div>
              {!isChangingPassword && (
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => setIsChangingPassword(true)}
                >
                  <Lock className="w-4 h-4 mr-2" />
                  Şifre Değiştir
                </Button>
              )}
            </div>
          </CardHeader>
          <CardContent>
            {isChangingPassword ? (
              <form onSubmit={handlePasswordChange} className="space-y-4">
                <div className="space-y-2">
                  <Label htmlFor="current_password">Mevcut Şifre</Label>
                  <Input
                    id="current_password"
                    type="password"
                    value={passwordData.current_password}
                    onChange={(e) =>
                      setPasswordData({ ...passwordData, current_password: e.target.value })
                    }
                    required
                  />
                </div>

                <div className="space-y-2">
                  <Label htmlFor="new_password">Yeni Şifre</Label>
                  <Input
                    id="new_password"
                    type="password"
                    value={passwordData.new_password}
                    onChange={(e) =>
                      setPasswordData({ ...passwordData, new_password: e.target.value })
                    }
                    required
                    minLength={6}
                  />
                  <p className="text-xs text-slate-500">En az 6 karakter olmalıdır</p>
                </div>

                <div className="space-y-2">
                  <Label htmlFor="confirm_password">Yeni Şifre (Tekrar)</Label>
                  <Input
                    id="confirm_password"
                    type="password"
                    value={passwordData.confirm_password}
                    onChange={(e) =>
                      setPasswordData({ ...passwordData, confirm_password: e.target.value })
                    }
                    required
                    minLength={6}
                  />
                </div>

                <div className="flex gap-2">
                  <Button type="submit" disabled={loading}>
                    <Check className="w-4 h-4 mr-2" />
                    Şifreyi Değiştir
                  </Button>
                  <Button
                    type="button"
                    variant="outline"
                    onClick={cancelPasswordChange}
                    disabled={loading}
                  >
                    <X className="w-4 h-4 mr-2" />
                    İptal
                  </Button>
                </div>
              </form>
            ) : (
              <div className="bg-slate-50 rounded-lg p-4">
                <div className="flex items-start gap-3">
                  <Lock className="w-5 h-5 text-slate-600 mt-0.5" />
                  <div>
                    <p className="text-sm font-medium text-slate-900">Şifreniz güvende</p>
                    <p className="text-sm text-slate-600 mt-1">
                      Şifrenizi düzenli olarak değiştirmeniz hesap güvenliğiniz için önerilir.
                    </p>
                  </div>
                </div>
              </div>
            )}
          </CardContent>
        </Card>

        {/* Account Status Card */}
        <Card>
          <CardHeader>
            <CardTitle>Hesap Durumu</CardTitle>
            <CardDescription>Hesabınızın mevcut durumu</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="flex items-center gap-2">
              <div className="w-2 h-2 rounded-full bg-green-500"></div>
              <span className="text-sm text-slate-700">Hesap aktif ve kullanıma hazır</span>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}