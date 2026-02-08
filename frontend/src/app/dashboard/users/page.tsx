"use client";

import { useEffect, useState, useCallback } from "react";
import { useRouter } from "next/navigation";
import { useAuth } from "@/lib/auth-context";
import { api, AdminUser, AdminUserCreate, AdminStatistics, AdminUserUpdate } from "@/lib/api";
import { PageHeader } from "@/components/ui/page-header";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import { toast } from "sonner";
import {
  Users, Search, Loader2, UserCheck, UserX, GraduationCap, BookOpen,
  Shield, MoreVertical, ChevronLeft, ChevronRight, KeyRound, Trash2,
  Pencil, Power, PowerOff, X, Check, UserPlus,
} from "lucide-react";

const ROLE_MAP: Record<string, { label: string; color: string; icon: React.ElementType }> = {
  admin: { label: "Yönetici", color: "bg-amber-100 text-amber-700", icon: Shield },
  teacher: { label: "Öğretmen", color: "bg-blue-100 text-blue-700", icon: GraduationCap },
  student: { label: "Öğrenci", color: "bg-green-100 text-green-700", icon: BookOpen },
};

export default function UsersPage() {
  const router = useRouter();
  const { user } = useAuth();
  const [users, setUsers] = useState<AdminUser[]>([]);
  const [stats, setStats] = useState<AdminStatistics | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [search, setSearch] = useState("");
  const [roleFilter, setRoleFilter] = useState<string>("");
  const [page, setPage] = useState(1);
  const [totalPages, setTotalPages] = useState(1);
  const [total, setTotal] = useState(0);
  const [editingUser, setEditingUser] = useState<AdminUser | null>(null);
  const [editForm, setEditForm] = useState<AdminUserUpdate>({});
  const [actionLoading, setActionLoading] = useState<number | null>(null);
  const [tempPassword, setTempPassword] = useState<{ userId: number; password: string; expires: string } | null>(null);
  const [showCreateDialog, setShowCreateDialog] = useState(false);
  const [createForm, setCreateForm] = useState({ full_name: "", email: "", password: "", role: "student" as "teacher" | "student" });
  const [isCreating, setIsCreating] = useState(false);

  const loadUsers = useCallback(async () => {
    try {
      const res = await api.getAdminUsers({
        page,
        limit: 20,
        search: search || undefined,
        role: roleFilter || undefined,
        sort_by: "id",
        sort_order: "desc",
      });
      setUsers(res.users);
      setTotalPages(res.total_pages);
      setTotal(res.total);
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "Kullanıcılar yüklenemedi");
    }
  }, [page, search, roleFilter]);

  const loadStats = useCallback(async () => {
    try {
      const data = await api.getAdminStatistics();
      setStats(data);
    } catch {
      // silent
    }
  }, []);

  useEffect(() => {
    if (user && user.role !== "admin") {
      router.push("/dashboard");
      return;
    }
    if (user) {
      Promise.all([loadUsers(), loadStats()]).finally(() => setIsLoading(false));
    }
  }, [user, router, loadUsers, loadStats]);

  useEffect(() => {
    if (!isLoading) loadUsers();
  }, [page, search, roleFilter, loadUsers, isLoading]);

  const handleToggleActive = async (u: AdminUser) => {
    setActionLoading(u.id);
    try {
      if (u.is_active) {
        await api.deactivateUser(u.id);
        toast.success(`${u.full_name} devre dışı bırakıldı`);
      } else {
        await api.activateUser(u.id);
        toast.success(`${u.full_name} aktifleştirildi`);
      }
      await loadUsers();
      await loadStats();
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "İşlem başarısız");
    } finally {
      setActionLoading(null);
    }
  };

  const handleDelete = async (u: AdminUser) => {
    if (!confirm(`${u.full_name} (${u.email}) kullanıcısını silmek istediğinize emin misiniz? Bu işlem geri alınamaz.`)) return;
    setActionLoading(u.id);
    try {
      await api.deleteAdminUser(u.id);
      toast.success(`${u.full_name} silindi`);
      await loadUsers();
      await loadStats();
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "Silme başarısız");
    } finally {
      setActionLoading(null);
    }
  };

  const handleResetPassword = async (u: AdminUser) => {
    if (!confirm(`${u.full_name} için geçici şifre oluşturulsun mu?`)) return;
    setActionLoading(u.id);
    try {
      const res = await api.resetUserPassword(u.id);
      setTempPassword({ userId: u.id, password: res.temporary_password, expires: res.expires_at });
      toast.success("Geçici şifre oluşturuldu");
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "Şifre sıfırlama başarısız");
    } finally {
      setActionLoading(null);
    }
  };

  const handleEditSave = async () => {
    if (!editingUser) return;
    setActionLoading(editingUser.id);
    try {
      await api.updateAdminUser(editingUser.id, editForm);
      toast.success("Kullanıcı güncellendi");
      setEditingUser(null);
      setEditForm({});
      await loadUsers();
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "Güncelleme başarısız");
    } finally {
      setActionLoading(null);
    }
  };

  const startEdit = (u: AdminUser) => {
    setEditingUser(u);
    setEditForm({ full_name: u.full_name, email: u.email, role: u.role });
  };

  const handleCreate = async () => {
    if (!createForm.full_name || !createForm.email || !createForm.password) {
      toast.error("Tüm alanları doldurun");
      return;
    }
    setIsCreating(true);
    try {
      await api.createAdminUser(createForm);
      toast.success("Kullanıcı oluşturuldu");
      setShowCreateDialog(false);
      setCreateForm({ full_name: "", email: "", password: "", role: "student" });
      await loadUsers();
      await loadStats();
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "Oluşturma başarısız");
    } finally {
      setIsCreating(false);
    }
  };

  if (!user || user.role !== "admin") return null;

  if (isLoading) {
    return (
      <div className="flex justify-center py-12">
        <Loader2 className="w-8 h-8 text-slate-400 animate-spin" />
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <PageHeader icon={Users} title="Kullanıcı Yönetimi" description="Sistemdeki kullanıcıları yönetin">
        <Button onClick={() => setShowCreateDialog(true)} className="bg-indigo-600 hover:bg-indigo-700">
          <UserPlus className="w-4 h-4 mr-2" />
          Yeni Kullanıcı
        </Button>
      </PageHeader>

      {/* Stats */}
      {stats && (
        <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
          {[
            { label: "Toplam", value: stats.total_users, icon: Users, color: "text-slate-600 bg-slate-100" },
            { label: "Öğretmen", value: stats.active_teachers, icon: GraduationCap, color: "text-blue-600 bg-blue-100" },
            { label: "Öğrenci", value: stats.active_students, icon: BookOpen, color: "text-green-600 bg-green-100" },
            { label: "Pasif", value: stats.inactive_users, icon: UserX, color: "text-red-600 bg-red-100" },
            { label: "Bu Ay Yeni", value: stats.new_users_this_month, icon: UserPlus, color: "text-purple-600 bg-purple-100" },
          ].map((s) => (
            <Card key={s.label}>
              <CardContent className="p-4 flex items-center gap-3">
                <div className={`w-10 h-10 rounded-lg flex items-center justify-center ${s.color}`}>
                  <s.icon className="w-5 h-5" />
                </div>
                <div>
                  <p className="text-2xl font-bold text-slate-900">{s.value}</p>
                  <p className="text-xs text-slate-500">{s.label}</p>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      )}

      {/* Filters */}
      <Card>
        <CardContent className="p-4">
          <div className="flex flex-col sm:flex-row gap-3">
            <div className="relative flex-1">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-400" />
              <Input
                placeholder="İsim veya e-posta ara..."
                value={search}
                onChange={(e) => { setSearch(e.target.value); setPage(1); }}
                className="pl-9"
              />
            </div>
            <div className="flex gap-2">
              {["", "teacher", "student"].map((r) => (
                <Button
                  key={r}
                  variant={roleFilter === r ? "default" : "outline"}
                  size="sm"
                  onClick={() => { setRoleFilter(r); setPage(1); }}
                >
                  {r === "" ? "Tümü" : ROLE_MAP[r]?.label}
                </Button>
              ))}
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Temp Password Alert */}
      {tempPassword && (
        <Card className="border-amber-300 bg-amber-50">
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-amber-900">Geçici Şifre Oluşturuldu</p>
                <p className="text-lg font-mono font-bold text-amber-800 mt-1">{tempPassword.password}</p>
                <p className="text-xs text-amber-600 mt-1">
                  Son kullanma: {new Date(tempPassword.expires).toLocaleString("tr-TR")}
                </p>
              </div>
              <Button variant="ghost" size="icon" onClick={() => setTempPassword(null)}>
                <X className="w-4 h-4" />
              </Button>
            </div>
          </CardContent>
        </Card>
      )}

      {/* User Table */}
      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-base">Kullanıcılar ({total})</CardTitle>
        </CardHeader>
        <CardContent className="p-0">
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b bg-slate-50">
                  <th className="text-left p-3 font-medium text-slate-600">ID</th>
                  <th className="text-left p-3 font-medium text-slate-600">Ad Soyad</th>
                  <th className="text-left p-3 font-medium text-slate-600">E-posta</th>
                  <th className="text-left p-3 font-medium text-slate-600">Rol</th>
                  <th className="text-left p-3 font-medium text-slate-600">Durum</th>
                  <th className="text-left p-3 font-medium text-slate-600">Kayıt</th>
                  <th className="text-left p-3 font-medium text-slate-600">Son Giriş</th>
                  <th className="text-right p-3 font-medium text-slate-600">İşlemler</th>
                </tr>
              </thead>
              <tbody>
                {users.map((u) => {
                  const role = ROLE_MAP[u.role] || ROLE_MAP.student;
                  const RoleIcon = role.icon;
                  return (
                    <tr key={u.id} className="border-b hover:bg-slate-50 transition-colors">
                      <td className="p-3 text-slate-500">#{u.id}</td>
                      <td className="p-3 font-medium text-slate-900">{u.full_name}</td>
                      <td className="p-3 text-slate-600">{u.email}</td>
                      <td className="p-3">
                        <span className={`inline-flex items-center gap-1 px-2 py-1 rounded-full text-xs font-medium ${role.color}`}>
                          <RoleIcon className="w-3 h-3" />
                          {role.label}
                        </span>
                      </td>
                      <td className="p-3">
                        {u.is_active ? (
                          <span className="inline-flex items-center gap-1 text-xs text-green-700">
                            <UserCheck className="w-3 h-3" /> Aktif
                          </span>
                        ) : (
                          <span className="inline-flex items-center gap-1 text-xs text-red-600">
                            <UserX className="w-3 h-3" /> Pasif
                          </span>
                        )}
                      </td>
                      <td className="p-3 text-slate-500 text-xs">
                        {new Date(u.created_at).toLocaleDateString("tr-TR")}
                      </td>
                      <td className="p-3 text-slate-500 text-xs">
                        {u.last_login ? new Date(u.last_login).toLocaleDateString("tr-TR") : "-"}
                      </td>
                      <td className="p-3">
                        {u.role !== "admin" && (
                          <div className="flex items-center justify-end gap-1">
                            <Button
                              variant="ghost" size="icon" className="h-8 w-8"
                              onClick={() => startEdit(u)}
                              title="Düzenle"
                            >
                              <Pencil className="w-3.5 h-3.5" />
                            </Button>
                            <Button
                              variant="ghost" size="icon" className="h-8 w-8"
                              onClick={() => handleToggleActive(u)}
                              disabled={actionLoading === u.id}
                              title={u.is_active ? "Devre Dışı Bırak" : "Aktifleştir"}
                            >
                              {u.is_active ? <PowerOff className="w-3.5 h-3.5 text-orange-500" /> : <Power className="w-3.5 h-3.5 text-green-500" />}
                            </Button>
                            <Button
                              variant="ghost" size="icon" className="h-8 w-8"
                              onClick={() => handleResetPassword(u)}
                              disabled={actionLoading === u.id}
                              title="Şifre Sıfırla"
                            >
                              <KeyRound className="w-3.5 h-3.5 text-amber-500" />
                            </Button>
                            <Button
                              variant="ghost" size="icon" className="h-8 w-8"
                              onClick={() => handleDelete(u)}
                              disabled={actionLoading === u.id}
                              title="Sil"
                            >
                              <Trash2 className="w-3.5 h-3.5 text-red-500" />
                            </Button>
                          </div>
                        )}
                      </td>
                    </tr>
                  );
                })}
                {users.length === 0 && (
                  <tr>
                    <td colSpan={8} className="p-8 text-center text-slate-400">
                      Kullanıcı bulunamadı
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </CardContent>
      </Card>

      {/* Pagination */}
      {totalPages > 1 && (
        <div className="flex items-center justify-center gap-2">
          <Button
            variant="outline" size="sm"
            disabled={page <= 1}
            onClick={() => setPage(page - 1)}
          >
            <ChevronLeft className="w-4 h-4" />
          </Button>
          <span className="text-sm text-slate-600">
            {page} / {totalPages}
          </span>
          <Button
            variant="outline" size="sm"
            disabled={page >= totalPages}
            onClick={() => setPage(page + 1)}
          >
            <ChevronRight className="w-4 h-4" />
          </Button>
        </div>
      )}

      {/* Edit Dialog */}
      {editingUser && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
          <Card className="w-full max-w-md mx-4">
            <CardHeader>
              <div className="flex items-center justify-between">
                <CardTitle className="text-base">Kullanıcı Düzenle</CardTitle>
                <Button variant="ghost" size="icon" onClick={() => setEditingUser(null)}>
                  <X className="w-4 h-4" />
                </Button>
              </div>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-2">
                <Label>Ad Soyad</Label>
                <Input
                  value={editForm.full_name || ""}
                  onChange={(e) => setEditForm({ ...editForm, full_name: e.target.value })}
                />
              </div>
              <div className="space-y-2">
                <Label>E-posta</Label>
                <Input
                  type="email"
                  value={editForm.email || ""}
                  onChange={(e) => setEditForm({ ...editForm, email: e.target.value })}
                />
              </div>
              <div className="space-y-2">
                <Label>Rol</Label>
                <select
                  className="w-full rounded-md border border-slate-200 px-3 py-2 text-sm"
                  value={editForm.role || ""}
                  onChange={(e) => setEditForm({ ...editForm, role: e.target.value as AdminUserUpdate["role"] })}
                >
                  <option value="teacher">Öğretmen</option>
                  <option value="student">Öğrenci</option>
                </select>
              </div>
              <div className="flex justify-end gap-2 pt-2">
                <Button variant="outline" onClick={() => setEditingUser(null)}>İptal</Button>
                <Button onClick={handleEditSave} disabled={actionLoading === editingUser.id}>
                  {actionLoading === editingUser.id ? <Loader2 className="w-4 h-4 animate-spin mr-2" /> : <Check className="w-4 h-4 mr-2" />}
                  Kaydet
                </Button>
              </div>
            </CardContent>
          </Card>
        </div>
      )}

      {/* Create Dialog */}
      {showCreateDialog && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
          <Card className="w-full max-w-md mx-4">
            <CardHeader>
              <div className="flex items-center justify-between">
                <CardTitle className="text-base">Yeni Kullanıcı Oluştur</CardTitle>
                <Button variant="ghost" size="icon" onClick={() => setShowCreateDialog(false)}>
                  <X className="w-4 h-4" />
                </Button>
              </div>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-2">
                <Label>Ad Soyad</Label>
                <Input
                  value={createForm.full_name}
                  onChange={(e) => setCreateForm({ ...createForm, full_name: e.target.value })}
                  placeholder="Ad Soyad"
                />
              </div>
              <div className="space-y-2">
                <Label>E-posta</Label>
                <Input
                  type="email"
                  value={createForm.email}
                  onChange={(e) => setCreateForm({ ...createForm, email: e.target.value })}
                  placeholder="ornek@email.com"
                />
              </div>
              <div className="space-y-2">
                <Label>Şifre</Label>
                <Input
                  type="text"
                  value={createForm.password}
                  onChange={(e) => setCreateForm({ ...createForm, password: e.target.value })}
                  placeholder="En az 6 karakter"
                />
              </div>
              <div className="space-y-2">
                <Label>Rol</Label>
                <select
                  className="w-full rounded-md border border-slate-200 px-3 py-2 text-sm"
                  value={createForm.role}
                  onChange={(e) => setCreateForm({ ...createForm, role: e.target.value as "teacher" | "student" })}
                >
                  <option value="student">Öğrenci</option>
                  <option value="teacher">Öğretmen</option>
                </select>
              </div>
              <div className="flex justify-end gap-2 pt-2">
                <Button variant="outline" onClick={() => setShowCreateDialog(false)}>İptal</Button>
                <Button onClick={handleCreate} disabled={isCreating}>
                  {isCreating ? <Loader2 className="w-4 h-4 animate-spin mr-2" /> : <UserPlus className="w-4 h-4 mr-2" />}
                  Oluştur
                </Button>
              </div>
            </CardContent>
          </Card>
        </div>
      )}
    </div>
  );
}
