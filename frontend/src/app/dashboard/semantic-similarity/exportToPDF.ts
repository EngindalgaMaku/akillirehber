// PDF Export utility for Semantic Similarity results
import {
  type SemanticSimilarityBatchTestResponse,
  type SemanticSimilarityResult,
} from "@/lib/api";

interface ExportPDFOptions {
  results: SemanticSimilarityResult[];
  aggregate: SemanticSimilarityBatchTestResponse["aggregate"];
  courseName: string;
  groupName: string;
}

// Bloom level display names
const BLOOM_LABELS: Record<string, string> = {
  remembering: "🧠 Hatırlama",
  understanding_applying: "🔧 Anlama/Uygulama",
  analyzing_evaluating: "⭐ Analiz/Değerlendirme",
  unknown: "❓ Bilinmeyen",
};

// Calculate bloom-level statistics
const calculateBloomStats = (results: SemanticSimilarityResult[]) => {
  const bloomGroups: Record<string, SemanticSimilarityResult[]> = {};
  
  results.forEach(r => {
    const level = r.bloom_level || 'unknown';
    if (!bloomGroups[level]) {
      bloomGroups[level] = [];
    }
    bloomGroups[level].push(r);
  });

  const bloomStats: Record<string, any> = {};
  
  Object.entries(bloomGroups).forEach(([level, items]) => {
    const rouge1Scores = items.filter(r => r.rouge1 != null).map(r => r.rouge1!);
    const rouge2Scores = items.filter(r => r.rouge2 != null).map(r => r.rouge2!);
    const rougelScores = items.filter(r => r.rougel != null).map(r => r.rougel!);
    const bertScores = items.filter(r => r.original_bertscore_f1 != null).map(r => r.original_bertscore_f1!);
    
    bloomStats[level] = {
      count: items.length,
      avg_rouge1: rouge1Scores.length > 0 ? rouge1Scores.reduce((a, b) => a + b, 0) / rouge1Scores.length : null,
      avg_rouge2: rouge2Scores.length > 0 ? rouge2Scores.reduce((a, b) => a + b, 0) / rouge2Scores.length : null,
      avg_rougel: rougelScores.length > 0 ? rougelScores.reduce((a, b) => a + b, 0) / rougelScores.length : null,
      avg_bertscore: bertScores.length > 0 ? bertScores.reduce((a, b) => a + b, 0) / bertScores.length : null,
    };
  });
  
  return bloomStats;
};

export const generateSemanticSimilarityPDF = (options: ExportPDFOptions) => {
  const { results, aggregate, courseName, groupName } = options;
  
  const bloomStats = calculateBloomStats(results);

  const getMetricClass = (val: number) => 
    val >= 0.8 ? 'metric-good' : val >= 0.6 ? 'metric-medium' : 'metric-bad';

  const htmlContent = `
    <!DOCTYPE html>
    <html>
    <head>
      <meta charset="utf-8">
      <title>RAG Cevap Kalitesi Test Raporu - ${groupName}</title>
      <style>
        body { font-family: Arial, sans-serif; margin: 40px; color: #333; }
        h1 { color: #0d9488; border-bottom: 3px solid #0d9488; padding-bottom: 10px; }
        h2 { color: #0891b2; margin-top: 30px; }
        .header { margin-bottom: 30px; }
        .stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 20px 0; }
        .stat-card { background: #f0fdfa; border: 1px solid #5eead4; border-radius: 8px; padding: 15px; }
        .stat-label { font-size: 12px; color: #0d9488; font-weight: bold; }
        .stat-value { font-size: 24px; font-weight: bold; color: #0f766e; margin-top: 5px; }
        table { width: 100%; border-collapse: collapse; margin: 20px 0; font-size: 11px; }
        th { background: #0d9488; color: white; padding: 12px; text-align: left; font-size: 11px; }
        td { padding: 10px; border-bottom: 1px solid #ddd; }
        tr:hover { background: #f0fdfa; }
        .metric-good { color: #059669; font-weight: bold; }
        .metric-medium { color: #d97706; font-weight: bold; }
        .metric-bad { color: #dc2626; font-weight: bold; }
        .footer { margin-top: 40px; padding-top: 20px; border-top: 2px solid #e5e7eb; font-size: 11px; color: #6b7280; }
        @media print {
          body { margin: 20px; }
          .no-print { display: none; }
        }
      </style>
    </head>
    <body>
      <div class="header">
        <h1>🎯 RAG Cevap Kalitesi Test Raporu</h1>
        <p><strong>Ders:</strong> ${courseName}</p>
        <p><strong>Grup:</strong> ${groupName}</p>
        <p><strong>Tarih:</strong> ${new Date().toLocaleString("tr-TR")}</p>
        <p><strong>Test Sayısı:</strong> ${results.length}</p>
      </div>

      <h2>📊 Özet İstatistikler</h2>
      <div class="stats">
        ${aggregate.avg_rouge1 != null ? `
          <div class="stat-card">
            <div class="stat-label">Ort. ROUGE-1</div>
            <div class="stat-value">${(aggregate.avg_rouge1 * 100).toFixed(1)}%</div>
          </div>
        ` : ''}
        ${aggregate.avg_rouge2 != null ? `
          <div class="stat-card">
            <div class="stat-label">Ort. ROUGE-2</div>
            <div class="stat-value">${(aggregate.avg_rouge2 * 100).toFixed(1)}%</div>
          </div>
        ` : ''}
        ${aggregate.avg_rougel != null ? `
          <div class="stat-card">
            <div class="stat-label">Ort. ROUGE-L</div>
            <div class="stat-value">${(aggregate.avg_rougel * 100).toFixed(1)}%</div>
          </div>
        ` : ''}
        ${aggregate.avg_original_bertscore_precision != null ? `
          <div class="stat-card">
            <div class="stat-label">Ort. BERTScore P</div>
            <div class="stat-value">${(aggregate.avg_original_bertscore_precision * 100).toFixed(1)}%</div>
          </div>
        ` : ''}
        ${aggregate.avg_original_bertscore_recall != null ? `
          <div class="stat-card">
            <div class="stat-label">Ort. BERTScore R</div>
            <div class="stat-value">${(aggregate.avg_original_bertscore_recall * 100).toFixed(1)}%</div>
          </div>
        ` : ''}
        ${aggregate.avg_original_bertscore_f1 != null ? `
          <div class="stat-card">
            <div class="stat-label">Ort. BERTScore F1</div>
            <div class="stat-value">${(aggregate.avg_original_bertscore_f1 * 100).toFixed(1)}%</div>
          </div>
        ` : ''}
      </div>

      ${Object.keys(bloomStats).length > 0 ? `
        <h2>🎯 Bloom Kategorilerine Göre İstatistikler</h2>
        ${Object.entries(bloomStats).map(([level, stats]: [string, any]) => `
          <div style="margin: 20px 0; padding: 15px; background: #f8fafc; border-left: 4px solid #0d9488; border-radius: 4px;">
            <h3 style="margin: 0 0 10px 0; color: #0f766e;">${BLOOM_LABELS[level] || level} (${stats.count} soru)</h3>
            <div class="stats">
              ${stats.avg_rouge1 != null ? `
                <div class="stat-card">
                  <div class="stat-label">Ort. ROUGE-1</div>
                  <div class="stat-value">${(stats.avg_rouge1 * 100).toFixed(1)}%</div>
                </div>
              ` : ''}
              ${stats.avg_rouge2 != null ? `
                <div class="stat-card">
                  <div class="stat-label">Ort. ROUGE-2</div>
                  <div class="stat-value">${(stats.avg_rouge2 * 100).toFixed(1)}%</div>
                </div>
              ` : ''}
              ${stats.avg_rougel != null ? `
                <div class="stat-card">
                  <div class="stat-label">Ort. ROUGE-L</div>
                  <div class="stat-value">${(stats.avg_rougel * 100).toFixed(1)}%</div>
                </div>
              ` : ''}
              ${stats.avg_bertscore != null ? `
                <div class="stat-card">
                  <div class="stat-label">Ort. BERTScore F1</div>
                  <div class="stat-value">${(stats.avg_bertscore * 100).toFixed(1)}%</div>
                </div>
              ` : ''}
            </div>
          </div>
        `).join('')}
      ` : ''}

      <h2>📋 Detaylı Sonuçlar</h2>
      <table>
        <thead>
          <tr>
            <th style="width: 40px;">No</th>
            <th>Soru</th>
            <th style="width: 100px;">Bloom Seviyesi</th>
            <th style="width: 70px;">ROUGE-1</th>
            <th style="width: 70px;">ROUGE-2</th>
            <th style="width: 70px;">ROUGE-L</th>
            <th style="width: 70px;">BERTScore P</th>
            <th style="width: 70px;">BERTScore R</th>
            <th style="width: 70px;">BERTScore F1</th>
            <th style="width: 60px;">Süre</th>
          </tr>
        </thead>
        <tbody>
          ${results.map((r, idx) => `
            <tr>
              <td>${idx + 1}</td>
              <td>
                <strong>Soru:</strong> ${r.question.substring(0, 100)}${r.question.length > 100 ? '...' : ''}<br/>
                <span style="color: #059669; font-size: 10px;"><strong>Doğru Cevap:</strong> ${r.ground_truth.substring(0, 100)}${r.ground_truth.length > 100 ? '...' : ''}</span><br/>
                <span style="color: #0891b2; font-size: 10px;"><strong>Üretilen:</strong> ${r.generated_answer.substring(0, 100)}${r.generated_answer.length > 100 ? '...' : ''}</span>
              </td>
              <td style="font-size: 10px; font-weight: bold; color: #0f766e;">${r.bloom_level ? BLOOM_LABELS[r.bloom_level] || r.bloom_level : '-'}</td>
              <td class="${r.rouge1 != null ? getMetricClass(r.rouge1) : ''}">${r.rouge1 != null ? (r.rouge1 * 100).toFixed(1) + '%' : '-'}</td>
              <td class="${r.rouge2 != null ? getMetricClass(r.rouge2) : ''}">${r.rouge2 != null ? (r.rouge2 * 100).toFixed(1) + '%' : '-'}</td>
              <td class="${r.rougel != null ? getMetricClass(r.rougel) : ''}">${r.rougel != null ? (r.rougel * 100).toFixed(1) + '%' : '-'}</td>
              <td class="${r.original_bertscore_precision != null ? getMetricClass(r.original_bertscore_precision) : ''}">${r.original_bertscore_precision != null ? (r.original_bertscore_precision * 100).toFixed(1) + '%' : '-'}</td>
              <td class="${r.original_bertscore_recall != null ? getMetricClass(r.original_bertscore_recall) : ''}">${r.original_bertscore_recall != null ? (r.original_bertscore_recall * 100).toFixed(1) + '%' : '-'}</td>
              <td class="${r.original_bertscore_f1 != null ? getMetricClass(r.original_bertscore_f1) : ''}">${r.original_bertscore_f1 != null ? (r.original_bertscore_f1 * 100).toFixed(1) + '%' : '-'}</td>
              <td>${r.latency_ms}ms</td>
            </tr>
          `).join('')}
        </tbody>
      </table>

      <div class="footer">
        <p><strong>Metrik Açıklamaları:</strong></p>
        <ul style="margin: 10px 0; padding-left: 20px;">
          <li><strong>ROUGE-1/2/L:</strong> N-gram overlap metrikleri (0-100%). Üretilen cevabın referans cevapla kelime bazında ne kadar örtüştüğünü ölçer.</li>
          <li><strong>BERTScore P/R/F1:</strong> BERT embedding tabanlı anlamsal benzerlik metrikleri (0-100%). <code>bert-score</code> Python kütüphanesi ile token-level hesaplanır.</li>
        </ul>
        <p style="margin-top: 20px;">Bu rapor Akıllı Rehber RAG Sistemi tarafından otomatik olarak oluşturulmuştur.</p>
      </div>
    </body>
    </html>
  `;

  return htmlContent;
};
