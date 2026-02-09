"use client";

import { useAuth } from "@/lib/auth-context";
import { api } from "@/lib/api";
import { BookOpen, FileText, MessageSquare, Boxes, GraduationCap, Brain, Database, Search, Layers, Edit3, Save, X, Plus, Trash2, Home } from "lucide-react";
import { useState, useEffect, useRef } from "react";
import { PageHeader } from "@/components/ui/page-header";

interface Note { id: string; content: string; createdAt: string; }
interface DashboardStats { course_count: number; document_count: number; chunk_count: number; }

function getInitialNotes(): Note[] {
  if (typeof window === "undefined") return [];
  try {
    const saved = localStorage.getItem("dashboard_notes");
    return saved ? JSON.parse(saved) : [];
  } catch { return []; }
}

export default function DashboardPage() {
  const { user } = useAuth();
  const [notes, setNotes] = useState<Note[]>(getInitialNotes);
  const [newNote, setNewNote] = useState("");
  const [editingId, setEditingId] = useState<string | null>(null);
  const [editContent, setEditContent] = useState("");
  const [stats, setStats] = useState<DashboardStats | null>(null);
  const statsLoaded = useRef(false);

  useEffect(() => {
    if (statsLoaded.current) return;
    statsLoaded.current = true;
    api.getDashboardStats().then((data) => setStats(data)).catch(() => {});
  }, []);

  const saveNotes = (updatedNotes: Note[]) => { setNotes(updatedNotes); localStorage.setItem("dashboard_notes", JSON.stringify(updatedNotes)); };
  const addNote = () => { if (!newNote.trim()) return; saveNotes([{ id: Date.now().toString(), content: newNote, createdAt: new Date().toLocaleDateString("tr-TR") }, ...notes]); setNewNote(""); };
  const updateNote = (id: string) => { if (!editContent.trim()) return; saveNotes(notes.map((n) => n.id === id ? { ...n, content: editContent } : n)); setEditingId(null); setEditContent(""); };
  const deleteNote = (id: string) => saveNotes(notes.filter((n) => n.id !== id));

  if (!user) return null;
  const isTeacher = user.role === "teacher";

  return (
    <div>
      <PageHeader
        icon={Home}
        title={`Hos geldin, ${user.full_name.split(" ")[0]}`}
        description={isTeacher ? "Derslerinizi yonetin ve icerik ekleyin" : "Ogrenmeye devam edin"}
        iconColor="text-indigo-600"
        iconBg="bg-indigo-100"
      />

      <div className="grid grid-cols-2 sm:grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
        <div className="bg-indigo-50 rounded-lg border border-indigo-200 p-5">
          <div className="w-10 h-10 bg-indigo-100 rounded-lg flex items-center justify-center">
            <BookOpen className="w-5 h-5 text-indigo-600" />
          </div>
          <p className="text-2xl font-semibold text-slate-900 mt-3">{stats?.course_count ?? 0}</p>
          <p className="text-sm text-slate-500">{isTeacher ? "Dersler" : "Kayitli Dersler"}</p>
        </div>
        <div className="bg-emerald-50 rounded-lg border border-emerald-200 p-5">
          <div className="w-10 h-10 bg-emerald-100 rounded-lg flex items-center justify-center">
            <FileText className="w-5 h-5 text-emerald-600" />
          </div>
          <p className="text-2xl font-semibold text-slate-900 mt-3">{stats?.document_count ?? 0}</p>
          <p className="text-sm text-slate-500">Dokumanlar</p>
        </div>
        <div className="bg-amber-50 rounded-lg border border-amber-200 p-5">
          <div className="w-10 h-10 bg-amber-100 rounded-lg flex items-center justify-center">
            <Boxes className="w-5 h-5 text-amber-600" />
          </div>
          <p className="text-2xl font-semibold text-slate-900 mt-3">{stats?.chunk_count ?? 0}</p>
          <p className="text-sm text-slate-500">Chunk Sayisi</p>
        </div>
        <div className="bg-rose-50 rounded-lg border border-rose-200 p-5">
          <div className="w-10 h-10 bg-rose-100 rounded-lg flex items-center justify-center">
            <MessageSquare className="w-5 h-5 text-rose-600" />
          </div>
          <p className="text-2xl font-semibold text-slate-900 mt-3">-</p>
          <p className="text-sm text-slate-500">Sohbetler</p>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
        <div className="bg-gradient-to-br from-indigo-50 to-white rounded-lg border border-indigo-100 p-6">
          <div className="flex items-center gap-3 mb-4">
            <div className="p-2 bg-indigo-100 rounded-lg"><GraduationCap className="w-6 h-6 text-indigo-600" /></div>
            <h2 className="text-lg font-medium text-slate-900">Proje Hakkinda</h2>
          </div>
          <div className="space-y-3 text-sm">
            <div className="bg-white/60 rounded-lg p-3"><p className="text-slate-600 font-medium">Universite</p><p className="text-slate-900">Burdur Mehmet Akif Ersoy Universitesi</p></div>
            <div className="bg-white/60 rounded-lg p-3"><p className="text-slate-600 font-medium">Program</p><p className="text-slate-900">Yuksek Lisans Tezi</p></div>
            <div className="bg-white/60 rounded-lg p-3"><p className="text-slate-600 font-medium">Proje</p><p className="text-slate-900">RAG Temelli Egitim Chatbot Sistemi</p></div>
            <div className="bg-white/60 rounded-lg p-3"><p className="text-slate-600 font-medium">Gelistirici</p><p className="text-slate-900">Engin Dalga</p></div>
            <div className="bg-white/60 rounded-lg p-3"><p className="text-slate-600 font-medium">Danisman</p><p className="text-slate-900">Prof. Dr. Serkan Balli</p></div>
          </div>
        </div>

        <div className="bg-gradient-to-br from-emerald-50 to-white rounded-lg border border-emerald-100 p-6">
          <div className="flex items-center gap-3 mb-4">
            <div className="p-2 bg-emerald-100 rounded-lg"><Brain className="w-6 h-6 text-emerald-600" /></div>
            <h2 className="text-lg font-medium text-slate-900">RAG Yaklasimlari</h2>
          </div>
          <div className="space-y-3">
            <div className="flex items-start gap-3 bg-white/60 rounded-lg p-3"><Search className="w-5 h-5 text-emerald-600 mt-0.5" /><div><p className="text-slate-900 font-medium text-sm">Semantic Search</p><p className="text-slate-600 text-xs">Anlam tabanli arama</p></div></div>
            <div className="flex items-start gap-3 bg-white/60 rounded-lg p-3"><Database className="w-5 h-5 text-emerald-600 mt-0.5" /><div><p className="text-slate-900 font-medium text-sm">BM25</p><p className="text-slate-600 text-xs">Anahtar kelime tabanli arama</p></div></div>
            <div className="flex items-start gap-3 bg-white/60 rounded-lg p-3"><Layers className="w-5 h-5 text-emerald-600 mt-0.5" /><div><p className="text-slate-900 font-medium text-sm">Hybrid RAG</p><p className="text-slate-600 text-xs">Hibrit arama yaklasimi</p></div></div>
            <div className="flex items-start gap-3 bg-white/60 rounded-lg p-3"><Brain className="w-5 h-5 text-emerald-600 mt-0.5" /><div><p className="text-slate-900 font-medium text-sm">Reranking</p><p className="text-slate-600 text-xs">Sonuc optimizasyonu</p></div></div>
          </div>
          <div className="mt-4 p-3 bg-emerald-100/50 rounded-lg">
            <p className="text-xs text-emerald-800"><span className="font-medium">Hibrit Yaklasim:</span> Semantic + BM25 birlesimi ile daha dogru sonuclar.</p>
          </div>
        </div>
      </div>

      <div className="bg-white rounded-lg border border-slate-200 p-6">
        <div className="flex items-center gap-3 mb-4">
          <div className="p-2 bg-amber-100 rounded-lg"><Edit3 className="w-5 h-5 text-amber-600" /></div>
          <h2 className="text-lg font-medium text-slate-900">Gelistirme Notlari</h2>
        </div>
        <div className="flex gap-2 mb-4">
          <input type="text" value={newNote} onChange={(e) => setNewNote(e.target.value)} onKeyDown={(e) => e.key === "Enter" && addNote()} placeholder="Yeni not ekle..." className="flex-1 px-3 py-2 border border-slate-200 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500" />
          <button onClick={addNote} className="px-4 py-2 bg-indigo-600 text-white rounded-lg text-sm font-medium hover:bg-indigo-700 flex items-center gap-2"><Plus className="w-4 h-4" />Ekle</button>
        </div>
        <div className="space-y-2 max-h-64 overflow-y-auto">
          {notes.length === 0 ? <p className="text-slate-500 text-sm text-center py-4">Henuz not eklenmemis.</p> : notes.map((note) => (
            <div key={note.id} className="flex items-start gap-2 p-3 bg-slate-50 rounded-lg group">
              {editingId === note.id ? (
                <>
                  <input type="text" value={editContent} onChange={(e) => setEditContent(e.target.value)} onKeyDown={(e) => e.key === "Enter" && updateNote(note.id)} className="flex-1 px-2 py-1 border border-slate-200 rounded text-sm" autoFocus />
                  <button onClick={() => updateNote(note.id)} className="p-1 text-green-600 hover:bg-green-100 rounded"><Save className="w-4 h-4" /></button>
                  <button onClick={() => setEditingId(null)} className="p-1 text-slate-400 hover:bg-slate-200 rounded"><X className="w-4 h-4" /></button>
                </>
              ) : (
                <>
                  <div className="flex-1"><p className="text-sm text-slate-700">{note.content}</p><p className="text-xs text-slate-400 mt-1">{note.createdAt}</p></div>
                  <button onClick={() => { setEditingId(note.id); setEditContent(note.content); }} className="p-1 text-slate-400 hover:bg-slate-200 rounded opacity-0 group-hover:opacity-100"><Edit3 className="w-4 h-4" /></button>
                  <button onClick={() => deleteNote(note.id)} className="p-1 text-red-400 hover:bg-red-100 rounded opacity-0 group-hover:opacity-100"><Trash2 className="w-4 h-4" /></button>
                </>
              )}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
