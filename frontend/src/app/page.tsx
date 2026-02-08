"use client";

import { useEffect } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import { Button } from "@/components/ui/button";
import { BookOpen, Brain, GraduationCap, MessageSquare, Sparkles, Users } from "lucide-react";
import { useAuth } from "@/lib/auth-context";

export default function HomePage() {
  const { user, loading } = useAuth();
  const router = useRouter();

  useEffect(() => {
    if (!loading && user) {
      router.replace("/dashboard");
    }
  }, [user, loading, router]);

  // Show nothing while checking auth or redirecting
  if (loading || user) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 flex items-center justify-center">
        <div className="w-8 h-8 border-4 border-purple-500 border-t-transparent rounded-full animate-spin" />
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900">
      {/* Header */}
      <header className="container mx-auto px-4 py-6">
        <nav className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-purple-500 to-pink-500 flex items-center justify-center">
              <Brain className="w-6 h-6 text-white" />
            </div>
            <span className="text-2xl font-bold text-white">AkıllıRehber</span>
          </div>
          <div className="flex items-center gap-4">
            <Link href="/login">
              <Button variant="ghost" className="text-white hover:text-purple-300 hover:bg-white/10">
                Giriş Yap
              </Button>
            </Link>
            <Link href="/register">
              <Button className="bg-gradient-to-r from-purple-500 to-pink-500 hover:from-purple-600 hover:to-pink-600 text-white border-0">
                Kayıt Ol
              </Button>
            </Link>
          </div>
        </nav>
      </header>

      {/* Hero Section */}
      <main className="container mx-auto px-4 py-20">
        <div className="text-center max-w-4xl mx-auto">
          <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-white/10 text-purple-300 text-sm mb-8">
            <Sparkles className="w-4 h-4" />
            <span>Yapay Zeka Destekli Eğitim Asistanı</span>
          </div>
          
          <h1 className="text-5xl md:text-7xl font-bold text-white mb-6 leading-tight">
            Öğrenmenin{" "}
            <span className="text-transparent bg-clip-text bg-gradient-to-r from-purple-400 to-pink-400">
              Akıllı
            </span>{" "}
            Yolu
          </h1>
          
          <p className="text-xl text-gray-300 mb-12 max-w-2xl mx-auto">
            RAG teknolojisi ile güçlendirilmiş chatbot sayesinde ders materyallerinizi yükleyin, 
            sorularınızı sorun ve anında doğru cevaplar alın.
          </p>

          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <Link href="/register">
              <Button size="lg" className="bg-gradient-to-r from-purple-500 to-pink-500 hover:from-purple-600 hover:to-pink-600 text-white text-lg px-8 py-6 rounded-xl">
                <GraduationCap className="w-5 h-5 mr-2" />
                Hemen Başla
              </Button>
            </Link>
            <Link href="/login">
              <Button size="lg" variant="ghost" className="border-2 border-white/30 text-white bg-transparent hover:bg-white/10 hover:border-white/50 text-lg px-8 py-6 rounded-xl backdrop-blur-sm">
                Giriş Yap
              </Button>
            </Link>
          </div>
        </div>

        {/* Features */}
        <div className="grid md:grid-cols-3 gap-8 mt-32">
          <FeatureCard
            icon={<BookOpen className="w-8 h-8" />}
            title="Doküman Yönetimi"
            description="PDF, Word ve metin dosyalarınızı kolayca yükleyin. Sistem otomatik olarak içeriği analiz eder."
          />
          <FeatureCard
            icon={<MessageSquare className="w-8 h-8" />}
            title="Akıllı Sohbet"
            description="Yüklediğiniz materyaller hakkında sorular sorun, anında doğru ve kaynaklı cevaplar alın."
          />
          <FeatureCard
            icon={<Users className="w-8 h-8" />}
            title="Öğretmen & Öğrenci"
            description="Öğretmenler ders oluşturur, öğrenciler derslere katılır ve öğrenir."
          />
        </div>
      </main>

      {/* Footer */}
      <footer className="container mx-auto px-4 py-8 mt-20 border-t border-white/10">
        <div className="text-center text-gray-400">
          <p>© 2026 AkıllıRehber. Tüm hakları saklıdır.</p>
        </div>
      </footer>
    </div>
  );
}

function FeatureCard({ icon, title, description }: { icon: React.ReactNode; title: string; description: string }) {
  return (
    <div className="p-8 rounded-2xl bg-white/5 border border-white/10 hover:bg-white/10 transition-all duration-300 group">
      <div className="w-14 h-14 rounded-xl bg-gradient-to-br from-purple-500/20 to-pink-500/20 flex items-center justify-center text-purple-400 mb-6 group-hover:scale-110 transition-transform">
        {icon}
      </div>
      <h3 className="text-xl font-semibold text-white mb-3">{title}</h3>
      <p className="text-gray-400">{description}</p>
    </div>
  );
}
