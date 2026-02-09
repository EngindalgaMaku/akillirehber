"use client";

import { useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import { api, type Course, type TestSet } from "@/lib/api";
import { Loader2, Plus, FileText, Edit, Trash2, BookOpen } from "lucide-react";
import Link from "next/link";
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog";
import { Textarea } from "@/components/ui/textarea";
import { toast } from "sonner";

export default function TestSetsPage() {
  const router = useRouter();
  const [courses, setCourses] = useState<Course[]>([]);
  const [selectedCourse, setSelectedCourse] = useState<number | null>(null);
  const [testSets, setTestSets] = useState<TestSet[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [isNewDialogOpen, setIsNewDialogOpen] = useState(false);
  const [newTestSetName, setNewTestSetName] = useState("");
  const [newTestSetDescription, setNewTestSetDescription] = useState("");
  const [isCreating, setIsCreating] = useState(false);

  useEffect(() => {
    loadCourses();
  }, []);

  useEffect(() => {
    if (selectedCourse) {
      loadTestSets(selectedCourse);
    }
  }, [selectedCourse]);

  const loadCourses = async () => {
    try {
      const data = await api.getCourses();
      setCourses(data);
      if (data.length > 0) {
        const savedCourseId = localStorage.getItem("ragas_selected_course_id");
        if (savedCourseId && data.find((c) => c.id === Number.parseInt(savedCourseId))) {
          setSelectedCourse(Number.parseInt(savedCourseId));
        } else {
          setSelectedCourse(data[0].id);
          localStorage.setItem("ragas_selected_course_id", data[0].id.toString());
        }
      }
    } catch (error) {
      console.error("Failed to load courses:", error);
      setCourses([]);
    } finally {
      setIsLoading(false);
    }
  };

  const loadTestSets = async (courseId: number) => {
    setIsLoading(true);
    try {
      const data = await api.getTestSets(courseId);
      setTestSets(data);
    } catch (error) {
      console.error("Failed to load test sets:", error);
      setTestSets([]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleCreateTestSet = async () => {
    if (!selectedCourse || !newTestSetName.trim()) return;
    setIsCreating(true);
    try {
      const newTestSet = await api.createTestSet({
        course_id: selectedCourse,
        name: newTestSetName.trim(),
        description: newTestSetDescription.trim() || undefined,
      });
      toast.success("Test seti oluşturuldu");
      await loadTestSets(selectedCourse);
      setIsNewDialogOpen(false);
      setNewTestSetName("");
      setNewTestSetDescription("");
      router.push(`/dashboard/ragas/test-sets/${newTestSet.id}`);
    } catch (error) {
      console.error("Failed to create test set:", error);
      toast.error("Test seti oluşturulamadı");
    } finally {
      setIsCreating(false);
    }
  };

  const handleDeleteTestSet = async (testSetId: number) => {
    if (!confirm("Bu test setini silmek istediğinizden emin misiniz?")) return;
    try {
      await api.deleteTestSet(testSetId);
      toast.success("Test seti silindi");
      if (selectedCourse) {
        await loadTestSets(selectedCourse);
      }
    } catch (error) {
      console.error("Failed to delete test set:", error);
      toast.error("Test seti silinemedi");
    }
  };

  return (
    <div className="container mx-auto py-8 px-4">
      <div className="mb-8 flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold">Test Setleri</h1>
          <p className="text-muted-foreground mt-2">Test sorularınızı yönetin ve düzenleyin</p>
        </div>
        <Dialog open={isNewDialogOpen} onOpenChange={setIsNewDialogOpen}>
          <DialogTrigger asChild>
            <Button disabled={!selectedCourse}>
              <Plus className="h-4 w-4 mr-2" />
              Yeni Test Seti
            </Button>
          </DialogTrigger>
          <DialogContent>
            <DialogHeader>
              <DialogTitle>Yeni Test Seti Oluştur</DialogTitle>
              <DialogDescription>Test sorularınız için yeni bir set oluşturun</DialogDescription>
            </DialogHeader>
            <div className="space-y-4 py-4">
              <div className="space-y-2">
                <Label htmlFor="name">Test Seti Adı *</Label>
                <Input id="name" placeholder="Örn: Bölüm 1 Test Soruları" value={newTestSetName} onChange={(e) => setNewTestSetName(e.target.value)} disabled={isCreating} />
              </div>
              <div className="space-y-2">
                <Label htmlFor="description">Açıklama (Opsiyonel)</Label>
                <Textarea id="description" placeholder="Test seti hakkında açıklama..." value={newTestSetDescription} onChange={(e) => setNewTestSetDescription(e.target.value)} disabled={isCreating} rows={3} />
              </div>
            </div>
            <DialogFooter>
              <Button variant="outline" onClick={() => { setIsNewDialogOpen(false); setNewTestSetName(""); setNewTestSetDescription(""); }} disabled={isCreating}>İptal</Button>
              <Button onClick={handleCreateTestSet} disabled={!newTestSetName.trim() || isCreating}>{isCreating ? (<><Loader2 className="mr-2 h-4 w-4 animate-spin" />Oluşturuluyor...</>) : ("Oluştur")}</Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>
      </div>

      <Card className="mb-6">
        <CardHeader>
          <CardTitle>Ders Seçin</CardTitle>
          <CardDescription>Test setlerini görüntülemek için bir ders seçin</CardDescription>
        </CardHeader>
        <CardContent>
          <select className="w-full px-3 py-2 border rounded-md" value={selectedCourse || ""} onChange={(e) => { const nextId = Number(e.target.value); setSelectedCourse(nextId); localStorage.setItem("ragas_selected_course_id", nextId.toString()); }} disabled={courses.length === 0}>
            <option value="">Seçin...</option>
            {courses.map((c) => (<option key={c.id} value={c.id}>{c.name}</option>))}
          </select>
        </CardContent>
      </Card>

      {isLoading && (
        <div className="flex items-center justify-center py-12">
          <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
        </div>
      )}

      {!isLoading && selectedCourse && testSets.length === 0 && (
        <Card>
          <CardContent className="py-12 text-center">
            <FileText className="h-12 w-12 mx-auto text-muted-foreground mb-4" />
            <h3 className="text-lg font-semibold mb-2">Henüz test seti yok</h3>
            <p className="text-muted-foreground mb-4">Bu ders için henüz test seti oluşturulmamış</p>
            <Button onClick={() => setIsNewDialogOpen(true)}>
              <Plus className="h-4 w-4 mr-2" />
              İlk Test Setini Oluştur
            </Button>
          </CardContent>
        </Card>
      )}

      {!isLoading && selectedCourse && testSets.length > 0 && (
        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
          {testSets.map((testSet) => (
            <Card key={testSet.id} className="hover:shadow-lg transition-shadow">
              <CardHeader>
                <div className="flex items-start justify-between">
                  <div className="flex-1">
                    <CardTitle className="text-lg">{testSet.name}</CardTitle>
                    {testSet.description && (
                      <CardDescription className="mt-1">{testSet.description}</CardDescription>
                    )}
                  </div>
                </div>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div className="flex items-center justify-between text-sm">
                    <span className="text-muted-foreground">Soru Sayısı</span>
                    <Badge variant="secondary">{testSet.question_count} soru</Badge>
                  </div>
                  <div className="flex items-center justify-between text-sm">
                    <span className="text-muted-foreground">Oluşturulma</span>
                    <span className="text-sm">{new Date(testSet.created_at).toLocaleDateString('tr-TR')}</span>
                  </div>
                  <div className="flex gap-2 pt-2">
                    <Button variant="default" size="sm" className="flex-1" onClick={() => router.push(`/dashboard/ragas/test-sets/${testSet.id}`)}>
                      <Edit className="h-4 w-4 mr-1" />
                      Düzenle
                    </Button>
                    <Button variant="outline" size="sm" onClick={() => handleDeleteTestSet(testSet.id)}>
                      <Trash2 className="h-4 w-4" />
                    </Button>
                  </div>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      )}

      {!isLoading && selectedCourse && testSets.length > 0 && (
        <div className="mt-6 flex justify-center">
          <Link href="/dashboard/ragas/test-sets/generate">
            <Button variant="outline">
              <BookOpen className="h-4 w-4 mr-2" />
              Yeni Soru Üret
            </Button>
          </Link>
        </div>
      )}
    </div>
  );
}
