"use client";

import Link from "next/link";
import { Card } from "@/components/ui/card";
import { 
  Scissors, 
  FlaskConical, 
  Target, 
  Shield, 
  Database, 
  BarChart3, 
  FileText,
  ArrowRight,
  Sparkles
} from "lucide-react";

const testCategories = [
  {
    title: "Chunking Testleri",
    description: "Metin bölümleme stratejilerini test edin ve karşılaştırın",
    icon: Scissors,
    href: "/dashboard/chunking",
    color: "from-blue-500 to-cyan-500",
    features: [
      "Fixed, Semantic ve Recursive stratejiler",
      "Chunk kalite metrikleri",
      "Performans karşılaştırması"
    ]
  },
  {
    title: "RAGAS Değerlendirme",
    description: "RAG sisteminin doğruluğunu ve kalitesini ölçün",
    icon: FlaskConical,
    href: "/dashboard/ragas",
    color: "from-purple-500 to-pink-500",
    features: [
      "Faithfulness, Answer Relevancy",
      "Context Precision & Recall",
      "Answer Correctness"
    ]
  },
  {
    title: "Test Setleri",
    description: "Değerlendirme için test soruları yönetin",
    icon: FileText,
    href: "/dashboard/ragas/test-sets",
    color: "from-indigo-500 to-purple-500",
    features: [
      "Test seti oluşturma ve yönetimi",
      "Soru kategorileri (Bloom seviyeleri)",
      "Toplu test çalıştırma"
    ]
  },
  {
    title: "Test Sorusu Üretimi",
    description: "Kalite filtreli otomatik soru üretimi",
    icon: Sparkles,
    href: "/dashboard/ragas/test-sets/generate",
    color: "from-amber-500 to-orange-500",
    features: [
      "Bloom taksonomisi bazlı üretim",
      "ROUGE-1 kalite filtresi",
      "Streaming ile anlık sonuçlar"
    ]
  },
  {
    title: "ROUGE & BERTScore",
    description: "Semantik benzerlik ve metin kalitesi testleri",
    icon: Target,
    href: "/dashboard/semantic-similarity",
    color: "from-teal-500 to-emerald-500",
    features: [
      "ROUGE-1, ROUGE-2, ROUGE-L",
      "BERTScore (Precision, Recall, F1)",
      "Batch test ve PDF export"
    ]
  },
  {
    title: "Giskard Testleri",
    description: "LLM güvenlik ve güvenilirlik testleri",
    icon: Shield,
    href: "/dashboard/giskard",
    color: "from-red-500 to-rose-500",
    features: [
      "Hallucination tespiti",
      "Bias ve toxicity kontrolü",
      "Robustness testleri"
    ]
  },
  {
    title: "W&B Runs",
    description: "Weights & Biases entegrasyonu ve run yönetimi",
    icon: Database,
    href: "/dashboard/wandb-runs",
    color: "from-violet-500 to-purple-500",
    features: [
      "Experiment tracking",
      "Metrik görselleştirme",
      "Run karşılaştırma"
    ]
  },
  {
    title: "MTEB Benchmark",
    description: "Embedding model performans karşılaştırması",
    icon: BarChart3,
    href: "/dashboard/mteb-benchmark",
    color: "from-green-500 to-teal-500",
    features: [
      "Çoklu model karşılaştırma",
      "Retrieval performansı",
      "Latency analizi"
    ]
  }
];

export default function SystemTestsPage() {
  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="bg-gradient-to-r from-indigo-600 via-purple-600 to-pink-600 rounded-2xl p-8 text-white shadow-xl">
        <div className="flex items-center gap-4 mb-4">
          <div className="w-16 h-16 bg-white/20 backdrop-blur-sm rounded-2xl flex items-center justify-center">
            <FlaskConical className="w-8 h-8" />
          </div>
          <div>
            <h1 className="text-3xl font-bold">Sistem Testleri</h1>
            <p className="text-indigo-100 mt-1">
              RAG sisteminizin performansını ölçün ve optimize edin
            </p>
          </div>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mt-6">
          <div className="bg-white/10 backdrop-blur-sm rounded-xl p-4">
            <p className="text-sm text-indigo-100 mb-1">Test Kategorileri</p>
            <p className="text-2xl font-bold">{testCategories.length}</p>
          </div>
          <div className="bg-white/10 backdrop-blur-sm rounded-xl p-4">
            <p className="text-sm text-indigo-100 mb-1">Metrik Türleri</p>
            <p className="text-2xl font-bold">15+</p>
          </div>
          <div className="bg-white/10 backdrop-blur-sm rounded-xl p-4">
            <p className="text-sm text-indigo-100 mb-1">Entegrasyonlar</p>
            <p className="text-2xl font-bold">W&B, Giskard</p>
          </div>
        </div>
      </div>

      {/* Test Categories Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {testCategories.map((category) => {
          const Icon = category.icon;
          return (
            <Link key={category.href} href={category.href}>
              <Card className="group h-full overflow-hidden border-0 shadow-lg hover:shadow-2xl transition-all duration-300 hover:-translate-y-1 cursor-pointer">
                {/* Gradient Header */}
                <div className={`bg-gradient-to-r ${category.color} p-6 text-white relative overflow-hidden`}>
                  <div className="absolute top-0 right-0 w-32 h-32 bg-white/10 rounded-full -mr-16 -mt-16" />
                  <div className="absolute bottom-0 left-0 w-24 h-24 bg-white/10 rounded-full -ml-12 -mb-12" />
                  <div className="relative">
                    <div className="w-12 h-12 bg-white/20 backdrop-blur-sm rounded-xl flex items-center justify-center mb-4">
                      <Icon className="w-6 h-6" />
                    </div>
                    <h3 className="text-xl font-bold mb-2">{category.title}</h3>
                    <p className="text-sm text-white/90">{category.description}</p>
                  </div>
                </div>

                {/* Features List */}
                <div className="p-6 bg-white">
                  <ul className="space-y-2 mb-4">
                    {category.features.map((feature, idx) => (
                      <li key={idx} className="flex items-start gap-2 text-sm text-slate-600">
                        <div className="w-1.5 h-1.5 rounded-full bg-indigo-500 mt-1.5 shrink-0" />
                        <span>{feature}</span>
                      </li>
                    ))}
                  </ul>
                  
                  {/* Action Button */}
                  <div className="flex items-center gap-2 text-indigo-600 font-medium text-sm group-hover:gap-3 transition-all">
                    <span>Teste Git</span>
                    <ArrowRight className="w-4 h-4" />
                  </div>
                </div>
              </Card>
            </Link>
          );
        })}
      </div>

      {/* Info Section */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <Card className="p-6 border-0 shadow-lg bg-gradient-to-br from-blue-50 to-indigo-50">
          <h3 className="text-lg font-bold text-slate-900 mb-3 flex items-center gap-2">
            <Target className="w-5 h-5 text-indigo-600" />
            Test Stratejisi
          </h3>
          <p className="text-sm text-slate-600 mb-4">
            Sisteminizi kapsamlı bir şekilde test etmek için önerilen sıra:
          </p>
          <ol className="space-y-2 text-sm text-slate-700">
            <li className="flex items-start gap-2">
              <span className="font-bold text-indigo-600">1.</span>
              <span><strong>Chunking:</strong> Optimal bölümleme stratejisini belirleyin</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="font-bold text-indigo-600">2.</span>
              <span><strong>Test Sorusu Üretimi:</strong> Kaliteli test setleri oluşturun</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="font-bold text-indigo-600">3.</span>
              <span><strong>RAGAS:</strong> RAG pipeline'ınızı değerlendirin</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="font-bold text-indigo-600">4.</span>
              <span><strong>ROUGE/BERTScore:</strong> Cevap kalitesini ölçün</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="font-bold text-indigo-600">5.</span>
              <span><strong>Giskard:</strong> Güvenlik ve güvenilirlik testleri</span>
            </li>
          </ol>
        </Card>

        <Card className="p-6 border-0 shadow-lg bg-gradient-to-br from-purple-50 to-pink-50">
          <h3 className="text-lg font-bold text-slate-900 mb-3 flex items-center gap-2">
            <Sparkles className="w-5 h-5 text-purple-600" />
            Öne Çıkan Özellikler
          </h3>
          <ul className="space-y-3 text-sm text-slate-700">
            <li className="flex items-start gap-3">
              <div className="w-8 h-8 rounded-lg bg-purple-100 flex items-center justify-center shrink-0">
                <FlaskConical className="w-4 h-4 text-purple-600" />
              </div>
              <div>
                <p className="font-medium">Streaming Test Sonuçları</p>
                <p className="text-xs text-slate-500">Anlık geri bildirim ile hızlı iterasyon</p>
              </div>
            </li>
            <li className="flex items-start gap-3">
              <div className="w-8 h-8 rounded-lg bg-indigo-100 flex items-center justify-center shrink-0">
                <Database className="w-4 h-4 text-indigo-600" />
              </div>
              <div>
                <p className="font-medium">W&B Entegrasyonu</p>
                <p className="text-xs text-slate-500">Tüm metriklerinizi merkezi bir yerde takip edin</p>
              </div>
            </li>
            <li className="flex items-start gap-3">
              <div className="w-8 h-8 rounded-lg bg-pink-100 flex items-center justify-center shrink-0">
                <BarChart3 className="w-4 h-4 text-pink-600" />
              </div>
              <div>
                <p className="font-medium">Detaylı Raporlama</p>
                <p className="text-xs text-slate-500">PDF export ve görselleştirme araçları</p>
              </div>
            </li>
          </ul>
        </Card>
      </div>
    </div>
  );
}
