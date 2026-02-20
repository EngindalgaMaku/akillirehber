// PDF Export utility for Semantic Similarity Analysis
import { GroupComparison, QuestionComparison } from "./page";

interface ExperimentInfo {
  title: string;
  experimentName: string;
  date: string;
  experimenter: string;
  summary: string;
  objective: string;
  methodology: string;
  evaluation: string;
}

interface AnalysisPDFOptions {
  experimentInfo: ExperimentInfo;
  groupComparisons: GroupComparison[];
  questionComparisons: QuestionComparison[];
  courseName: string;
  courseSettings: { default_embedding_model?: string; llm_provider?: string; llm_model?: string } | null;
}

export const generateAnalysisPDF = (options: AnalysisPDFOptions) => {
  const { experimentInfo, groupComparisons, questionComparisons, courseName, courseSettings } = options;

  // Generate random test number
  const testNumber = `TEST-${Math.floor(Math.random() * 100000).toString().padStart(5, '0')}`;

  // Calculate overall averages across all groups
  const allRouge1Values = groupComparisons.flatMap(g => g.results.filter(r => r.rouge1 != null).map(r => r.rouge1!));
  const allRouge2Values = groupComparisons.flatMap(g => g.results.filter(r => r.rouge2 != null).map(r => r.rouge2!));
  const allRougelValues = groupComparisons.flatMap(g => g.results.filter(r => r.rougel != null).map(r => r.rougel!));
  const allBertF1Values = groupComparisons.flatMap(g => g.results.filter(r => r.original_bertscore_f1 != null).map(r => r.original_bertscore_f1!));
  const allLatencyValues = groupComparisons.flatMap(g => g.results.filter(r => r.latency_ms != null).map(r => r.latency_ms!));

  const overallAvgRouge1 = allRouge1Values.length > 0 ? allRouge1Values.reduce((a, b) => a + b, 0) / allRouge1Values.length : 0;
  const overallAvgRouge2 = allRouge2Values.length > 0 ? allRouge2Values.reduce((a, b) => a + b, 0) / allRouge2Values.length : 0;
  const overallAvgRougel = allRougelValues.length > 0 ? allRougelValues.reduce((a, b) => a + b, 0) / allRougelValues.length : 0;
  const overallAvgBertF1 = allBertF1Values.length > 0 ? allBertF1Values.reduce((a, b) => a + b, 0) / allBertF1Values.length : 0;
  const overallAvgLatency = allLatencyValues.length > 0 ? allLatencyValues.reduce((a, b) => a + b, 0) / allLatencyValues.length : 0;

  const getMetricClass = (val: number) =>
    val >= 0.8 ? 'metric-good' : val >= 0.6 ? 'metric-medium' : 'metric-bad';

  const htmlContent = `
    <!DOCTYPE html>
    <html>
    <head>
      <meta charset="utf-8">
      <title>${experimentInfo.title} - ${experimentInfo.experimentName}</title>
      <style>
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 40px; color: #1e293b; background: #f8fafc; }
        .container { max-width: 900px; margin: 0 auto; background: white; padding: 40px; border-radius: 12px; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1); }
        
        /* Header */
        .header { border-bottom: 3px solid #6366f1; padding-bottom: 20px; margin-bottom: 30px; }
        .header h1 { color: #4f46e5; margin: 0 0 10px 0; font-size: 28px; }
        .header .subtitle { color: #64748b; font-size: 14px; }
        
        /* Experiment Info */
        .experiment-info { background: linear-gradient(135deg, #eef2ff 0%, #e0e7ff 100%); border-radius: 10px; padding: 25px; margin-bottom: 30px; border-left: 4px solid #6366f1; }
        .experiment-info h2 { color: #4338ca; margin-top: 0; font-size: 18px; border-bottom: 1px solid #c7d2fe; padding-bottom: 10px; margin-bottom: 15px; }
        .info-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 15px; }
        .info-item label { display: block; font-size: 11px; color: #6366f1; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; }
        .info-item p { margin: 5px 0 0 0; font-size: 13px; color: #1e293b; }
        .info-item.full-width { grid-column: span 2; }
        .info-item.full-width p { line-height: 1.6; }
        
        /* Section Headers */
        h2 { color: #0891b2; margin-top: 35px; font-size: 20px; border-left: 4px solid #0891b2; padding-left: 12px; }
        
        /* Stats Grid */
        .stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 15px; margin: 20px 0; }
        .stat-card { background: #f0fdfa; border: 1px solid #5eead4; border-radius: 8px; padding: 18px; text-align: center; }
        .stat-card.purple { background: #f5f3ff; border-color: #c4b5fd; }
        .stat-card.amber { background: #fffbeb; border-color: #fcd34d; }
        .stat-label { font-size: 11px; color: #0d9488; font-weight: bold; text-transform: uppercase; }
        .stat-value { font-size: 28px; font-weight: bold; color: #0f766e; margin-top: 8px; }
        .stat-card.purple .stat-value { color: #7c3aed; }
        .stat-card.amber .stat-value { color: #d97706; }
        
        /* Tables */
        table { width: 100%; border-collapse: collapse; margin: 20px 0; font-size: 11px; background: white; border-radius: 8px; overflow: hidden; }
        th { background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%); color: white; padding: 14px 12px; text-align: left; font-size: 11px; font-weight: 600; }
        td { padding: 12px; border-bottom: 1px solid #e2e8f0; }
        tr:nth-child(even) { background: #f8fafc; }
        tr:hover { background: #eef2ff; }
        
        /* Metric Classes */
        .metric-good { color: #059669; font-weight: bold; }
        .metric-medium { color: #d97706; font-weight: bold; }
        .metric-bad { color: #dc2626; font-weight: bold; }
        .variance-low { color: #059669; font-weight: bold; }
        .variance-medium { color: #d97706; font-weight: bold; }
        .variance-high { color: #dc2626; font-weight: bold; }
        
        /* Variance Badge */
        .variance-badge { display: inline-block; padding: 4px 10px; border-radius: 20px; font-size: 10px; font-weight: 600; }
        .variance-badge.low { background: #dcfce7; color: #166534; }
        .variance-badge.medium { background: #fef3c7; color: #92400e; }
        .variance-badge.high { background: #fee2e2; color: #991b1b; }
        
        /* Group Badge */
        .group-badge { display: inline-block; padding: 4px 12px; background: #e0e7ff; color: #4338ca; border-radius: 20px; font-size: 11px; font-weight: 600; }
        
        /* Warning Box */
        .warning-box { background: linear-gradient(135deg, #fff7ed 0%, #ffedd5 100%); border: 1px solid #fdba74; border-radius: 8px; padding: 20px; margin-top: 20px; }
        .warning-box h3 { color: #c2410c; margin: 0 0 15px 0; font-size: 14px; }
        .warning-item { background: white; border-radius: 6px; padding: 12px; margin-bottom: 10px; }
        .warning-item p { margin: 0; font-size: 11px; color: #9a3412; }

        /* Charts */
        .chart-container { background: white; border-radius: 8px; padding: 20px; margin: 20px 0; border: 1px solid #e2e8f0; }
        .chart-title { font-size: 14px; font-weight: bold; color: #475569; margin-bottom: 15px; text-align: center; }
        .bar-chart { display: flex; flex-direction: column; gap: 12px; }
        .bar-row { display: flex; align-items: center; gap: 10px; }
        .bar-label { width: 120px; font-size: 11px; color: #64748b; text-align: right; }
        .bar-track { flex: 1; background: #f1f5f9; border-radius: 4px; height: 24px; position: relative; overflow: hidden; }
        .bar-fill { height: 100%; border-radius: 4px; transition: width 0.3s ease; display: flex; align-items: center; padding-left: 8px; font-size: 11px; font-weight: bold; color: white; }
        .bar-fill.rouge1 { background: linear-gradient(90deg, #a855f7 0%, #7c3aed 100%); }
        .bar-fill.rouge2 { background: linear-gradient(90deg, #ec4899 0%, #db2777 100%); }
        .bar-fill.rougel { background: linear-gradient(90deg, #f472b6 0%, #e11d48 100%); }
        .bar-fill.bert { background: linear-gradient(90deg, #3b82f6 0%, #2563eb 100%); }
        .bar-value { width: 50px; text-align: right; font-size: 12px; font-weight: bold; color: #1e293b; }
        .chart-legend { display: flex; justify-content: center; gap: 20px; margin-top: 15px; font-size: 10px; }
        .legend-item { display: flex; align-items: center; gap: 5px; }
        .legend-color { width: 12px; height: 12px; border-radius: 2px; }
        .legend-color.rouge1 { background: linear-gradient(90deg, #a855f7 0%, #7c3aed 100%); }
        .legend-color.rouge2 { background: linear-gradient(90deg, #ec4899 0%, #db2777 100%); }
        .legend-color.rougel { background: linear-gradient(90deg, #f472b6 0%, #e11d48 100%); }
        .legend-color.bert { background: linear-gradient(90deg, #3b82f6 0%, #2563eb 100%); }

        /* Evaluation Section */
        .evaluation-section { background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%); border-radius: 10px; padding: 25px; margin-bottom: 30px; border-left: 4px solid #f59e0b; }
        .evaluation-section h2 { color: #92400e; margin-top: 0; font-size: 18px; border-bottom: 1px solid #fcd34d; padding-bottom: 10px; margin-bottom: 15px; }
        .evaluation-content { font-size: 13px; color: #1e293b; line-height: 1.8; white-space: pre-wrap; }

        /* Footer */
        .footer { margin-top: 50px; padding-top: 25px; border-top: 2px solid #e5e7eb; font-size: 11px; color: #64748b; }
        .footer h4 { color: #475569; margin: 0 0 10px 0; font-size: 12px; }
        .footer ul { margin: 10px 0; padding-left: 20px; }
        .footer li { margin-bottom: 5px; line-height: 1.6; }
        
        /* Page Break */
        .page-break { page-break-before: always; }
        
        @media print {
          body { margin: 0; padding: 20px; background: white; }
          .container { box-shadow: none; padding: 20px; }
          .no-print { display: none; }
        }
      </style>
    </head>
    <body>
      <div class="container">
        <!-- Header -->
        <div class="header">
          <h1>🔬 Semantic Similarity Analiz Raporu</h1>
          <p class="subtitle">${experimentInfo.title} - ${testNumber}</p>
        </div>

        <!-- Experiment Information -->
        <div class="experiment-info">
          <h2>📋 Deney Bilgileri</h2>
          <div class="info-grid">
            <div class="info-item">
              <label>Deney Adı</label>
              <p>${experimentInfo.experimentName}</p>
            </div>
            <div class="info-item">
              <label>Deneyi Yapan</label>
              <p>${experimentInfo.experimenter}</p>
            </div>
            <div class="info-item">
              <label>Tarih</label>
              <p>${experimentInfo.date}</p>
            </div>
            <div class="info-item">
              <label>Ders</label>
              <p>${courseName}</p>
            </div>
            <div class="info-item full-width">
              <label>Özet</label>
              <p>${experimentInfo.summary}</p>
            </div>
            <div class="info-item full-width">
              <label>Amaç</label>
              <p>${experimentInfo.objective}</p>
            </div>
            <div class="info-item full-width">
              <label>Metodoloji</label>
              <p>${experimentInfo.methodology}</p>
            </div>
          </div>
        </div>

        <!-- Overall Summary -->
        <div class="experiment-info" style="background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%); border-left-color: #f59e0b;">
          <h2 style="color: #92400e; border-color: #fcd34d;">📊 Genel Özet</h2>
          <div class="stats">
            <div class="stat-card">
              <div class="stat-label">Toplam Test</div>
              <div class="stat-value">${groupComparisons.length}</div>
            </div>
            <div class="stat-card purple">
              <div class="stat-label">Ort. ROUGE-1</div>
              <div class="stat-value">${(overallAvgRouge1 * 100).toFixed(2)}%</div>
            </div>
            <div class="stat-card purple">
              <div class="stat-label">Ort. ROUGE-2</div>
              <div class="stat-value">${(overallAvgRouge2 * 100).toFixed(2)}%</div>
            </div>
            <div class="stat-card purple">
              <div class="stat-label">Ort. ROUGE-L</div>
              <div class="stat-value">${(overallAvgRougel * 100).toFixed(2)}%</div>
            </div>
            <div class="stat-card">
              <div class="stat-label">Ort. BERTScore F1</div>
              <div class="stat-value">${(overallAvgBertF1 * 100).toFixed(2)}%</div>
            </div>
            <div class="stat-card amber">
              <div class="stat-label">Ort. Yanıt Süresi</div>
              <div class="stat-value">${overallAvgLatency.toFixed(0)} ms</div>
            </div>
          </div>
        </div>

        <!-- Technical Settings -->
        <div class="experiment-info" style="background: linear-gradient(135deg, #f0fdfa 0%, #ccfbf1 100%); border-left-color: #0891b2;">
          <h2 style="color: #0e7490; border-color: #99f6e4;">⚙️ Teknik Ayarlar</h2>
          <div class="info-grid">
            <div class="info-item">
              <label>Embedding Model</label>
              <p>${courseSettings?.default_embedding_model || '-'}</p>
            </div>
            <div class="info-item">
              <label>LLM Model</label>
              <p>${courseSettings?.llm_provider && courseSettings?.llm_model ? `${courseSettings.llm_provider}/${courseSettings.llm_model}` : '-'}</p>
            </div>
          </div>
        </div>

        <!-- Metrics Explanation -->
        <div class="experiment-info" style="background: linear-gradient(135deg, #e0f2fe 0%, #fef3c7 100%); border-left-color: #ef4444;">
          <h2 style="color: #991b1b; border-color: #fbbf24;">📖 Kullanılan Metrikler</h2>
          <div class="info-grid">
            <div class="info-item full-width">
              <label>ROUGE-1</label>
              <p>1-gram örtüşme oranı. Üretilen cevabın referans cevapla ne kadar kelime bazında örtüştüğünü ölçer. 0-100% aralığında değer alır.</p>
            </div>
            <div class="info-item full-width">
              <label>ROUGE-2</label>
              <p>2-gram örtüşme oranı. Üretilen cevabın referans cevapla ne kadar 2'li kelime kombinasyonları örtüştüğünü ölçer. 0-100% aralığında değer alır.</p>
            </div>
            <div class="info-item full-width">
              <label>ROUGE-L</label>
              <p>En uzun ortak alt dizi oranı. Üretilen cevabın referans cevapla ne kadar uzunlukta ortak alt dizi içerdiğini ölçer. 0-100% aralığında değer alır.</p>
            </div>
            <div class="info-item full-width">
              <label>BERTScore</label>
              <p>Anlamsal benzerlik skoru. Deep learning tabanlı bir model (BERT) kullanılarak üretilen cevabın referans cevapla ne kadar anlamsal olarak benzer olduğunu ölçer. Cosine similarity kullanılarak hesaplanır. 0-100% aralığında değer alır.</p>
            </div>
          </div>
        </div>

        <!-- Group Comparison -->
        <h2>📊 Grup Karşılaştırması</h2>
        <table>
          <thead>
            <tr>
              <th>Test</th>
              <th style="text-align: center;">Test Sayısı</th>
              <th style="text-align: center;">ROUGE-1</th>
              <th style="text-align: center;">ROUGE-2</th>
              <th style="text-align: center;">ROUGE-L</th>
              <th style="text-align: center;">BERTScore F1</th>
              <th style="text-align: center;">Ort. Yanıt Süresi</th>
            </tr>
          </thead>
          <tbody>
            ${groupComparisons.map((comp, idx) => `
              <tr>
                <td><span class="group-badge">Test ${idx + 1}</span></td>
                <td style="text-align: center; font-weight: bold;">${comp.aggregate.test_count}</td>
                <td style="text-align: center;" class="${comp.aggregate.avg_rouge1 != null ? getMetricClass(comp.aggregate.avg_rouge1) : ''}">
                  ${comp.aggregate.avg_rouge1 != null ? (comp.aggregate.avg_rouge1 * 100).toFixed(2) + '%' : '-'}
                </td>
                <td style="text-align: center;" class="${comp.aggregate.avg_rouge2 != null ? getMetricClass(comp.aggregate.avg_rouge2) : ''}">
                  ${comp.aggregate.avg_rouge2 != null ? (comp.aggregate.avg_rouge2 * 100).toFixed(2) + '%' : '-'}
                </td>
                <td style="text-align: center;" class="${comp.aggregate.avg_rougel != null ? getMetricClass(comp.aggregate.avg_rougel) : ''}">
                  ${comp.aggregate.avg_rougel != null ? (comp.aggregate.avg_rougel * 100).toFixed(2) + '%' : '-'}
                </td>
                <td style="text-align: center;" class="${comp.aggregate.avg_bertscore_f1 != null ? getMetricClass(comp.aggregate.avg_bertscore_f1) : ''}">
                  ${comp.aggregate.avg_bertscore_f1 != null ? (comp.aggregate.avg_bertscore_f1 * 100).toFixed(2) + '%' : '-'}
                </td>
                <td style="text-align: center; font-weight: bold;">
                  ${comp.aggregate.avg_latency_ms != null ? comp.aggregate.avg_latency_ms.toFixed(0) + ' ms' : '-'}
                </td>
              </tr>
            `).join('')}
          </tbody>
        </table>

        <!-- Charts Section -->
        <div class="chart-container">
          <div class="chart-title">📈 Metrik Karşılaştırma Grafikleri</div>
          ${groupComparisons.map((comp, idx) => `
            <div style="margin-bottom: 20px; ${idx > 0 ? 'padding-top: 20px; border-top: 1px solid #e2e8f0;' : ''}">
              <div style="font-size: 13px; font-weight: bold; color: #475569; margin-bottom: 12px;">
                Test ${idx + 1}
              </div>
              <div class="bar-chart">
                ${comp.aggregate.avg_rouge1 != null ? `
                  <div class="bar-row">
                    <div class="bar-label">ROUGE-1</div>
                    <div class="bar-track">
                      <div class="bar-fill rouge1" style="width: ${comp.aggregate.avg_rouge1 * 100}%">
                        ${(comp.aggregate.avg_rouge1 * 100).toFixed(1)}%
                      </div>
                    </div>
                    <div class="bar-value">${(comp.aggregate.avg_rouge1 * 100).toFixed(1)}%</div>
                  </div>
                ` : ''}
                ${comp.aggregate.avg_rouge2 != null ? `
                  <div class="bar-row">
                    <div class="bar-label">ROUGE-2</div>
                    <div class="bar-track">
                      <div class="bar-fill rouge2" style="width: ${comp.aggregate.avg_rouge2 * 100}%">
                        ${(comp.aggregate.avg_rouge2 * 100).toFixed(1)}%
                      </div>
                    </div>
                    <div class="bar-value">${(comp.aggregate.avg_rouge2 * 100).toFixed(1)}%</div>
                  </div>
                ` : ''}
                ${comp.aggregate.avg_rougel != null ? `
                  <div class="bar-row">
                    <div class="bar-label">ROUGE-L</div>
                    <div class="bar-track">
                      <div class="bar-fill rougel" style="width: ${comp.aggregate.avg_rougel * 100}%">
                        ${(comp.aggregate.avg_rougel * 100).toFixed(1)}%
                      </div>
                    </div>
                    <div class="bar-value">${(comp.aggregate.avg_rougel * 100).toFixed(1)}%</div>
                  </div>
                ` : ''}
                ${comp.aggregate.avg_bertscore_f1 != null ? `
                  <div class="bar-row">
                    <div class="bar-label">BERTScore F1</div>
                    <div class="bar-track">
                      <div class="bar-fill bert" style="width: ${comp.aggregate.avg_bertscore_f1 * 100}%">
                        ${(comp.aggregate.avg_bertscore_f1 * 100).toFixed(1)}%
                      </div>
                    </div>
                    <div class="bar-value">${(comp.aggregate.avg_bertscore_f1 * 100).toFixed(1)}%</div>
                  </div>
                ` : ''}
              </div>
            </div>
          `).join('')}
          <div class="chart-legend">
            <div class="legend-item"><div class="legend-color rouge1"></div>ROUGE-1</div>
            <div class="legend-item"><div class="legend-color rouge2"></div>ROUGE-2</div>
            <div class="legend-item"><div class="legend-color rougel"></div>ROUGE-L</div>
            <div class="legend-item"><div class="legend-color bert"></div>BERTScore F1</div>
          </div>
        </div>

        <!-- Evaluation Section -->
        <div class="evaluation-section">
          <h2>💬 Değerlendirme</h2>
          <div class="evaluation-content">${experimentInfo.evaluation}</div>
        </div>

        <!-- Group Differences -->
        ${groupComparisons.length === 2 ? `
          <h2>🔄 Gruplar Arası Farklar</h2>
          <div class="stats">
            ${groupComparisons[0].aggregate.avg_rouge1 != null && groupComparisons[1].aggregate.avg_rouge1 != null ? `
              <div class="stat-card amber">
                <div class="stat-label">ROUGE-1 Farkı</div>
                <div class="stat-value" style="color: ${groupComparisons[0].aggregate.avg_rouge1! > groupComparisons[1].aggregate.avg_rouge1! ? '#059669' : '#dc2626'};">
                  ${Math.abs(groupComparisons[0].aggregate.avg_rouge1! - groupComparisons[1].aggregate.avg_rouge1!) * 100 > 0 ? '+' : ''}${((groupComparisons[0].aggregate.avg_rouge1! - groupComparisons[1].aggregate.avg_rouge1!) * 100).toFixed(2)}%
                </div>
                <div style="font-size: 10px; color: #64748b; margin-top: 5px;">Test 1 vs Test 2</div>
              </div>
            ` : ''}
            ${groupComparisons[0].aggregate.avg_rougel != null && groupComparisons[1].aggregate.avg_rougel != null ? `
              <div class="stat-card amber">
                <div class="stat-label">ROUGE-L Farkı</div>
                <div class="stat-value" style="color: ${groupComparisons[0].aggregate.avg_rougel! > groupComparisons[1].aggregate.avg_rougel! ? '#059669' : '#dc2626'};">
                  ${Math.abs(groupComparisons[0].aggregate.avg_rougel! - groupComparisons[1].aggregate.avg_rougel!) * 100 > 0 ? '+' : ''}${((groupComparisons[0].aggregate.avg_rougel! - groupComparisons[1].aggregate.avg_rougel!) * 100).toFixed(2)}%
                </div>
                <div style="font-size: 10px; color: #64748b; margin-top: 5px;">Test 1 vs Test 2</div>
              </div>
            ` : ''}
            ${groupComparisons[0].aggregate.avg_bertscore_f1 != null && groupComparisons[1].aggregate.avg_bertscore_f1 != null ? `
              <div class="stat-card amber">
                <div class="stat-label">BERTScore F1 Farkı</div>
                <div class="stat-value" style="color: ${groupComparisons[0].aggregate.avg_bertscore_f1! > groupComparisons[1].aggregate.avg_bertscore_f1! ? '#059669' : '#dc2626'};">
                  ${Math.abs(groupComparisons[0].aggregate.avg_bertscore_f1! - groupComparisons[1].aggregate.avg_bertscore_f1!) * 100 > 0 ? '+' : ''}${((groupComparisons[0].aggregate.avg_bertscore_f1! - groupComparisons[1].aggregate.avg_bertscore_f1!) * 100).toFixed(2)}%
                </div>
                <div style="font-size: 10px; color: #64748b; margin-top: 5px;">Test 1 vs Test 2</div>
              </div>
            ` : ''}
          </div>
        ` : ''}

        <!-- Question Analysis -->
        <div class="page-break"></div>
        <h2>📝 Soru Bazlı Analiz</h2>
        <table>
          <thead>
            <tr>
              <th style="width: 35%;">Soru</th>
              <th style="text-align: center; width: 10%;">Grup Sayısı</th>
              <th style="text-align: center; width: 18%;">Ort. ROUGE-1</th>
              <th style="text-align: center; width: 18%;">Ort. ROUGE-2</th>
              <th style="text-align: center; width: 18%;">Ort. BERTScore F1</th>
            </tr>
          </thead>
          <tbody>
            ${questionComparisons.map((comp) => {
              const avgRouge1 = comp.groupResults.reduce((sum, r) => sum + (r.rouge1 || 0), 0) / comp.groupResults.length;
              const avgRouge2 = comp.groupResults.reduce((sum, r) => sum + (r.rouge2 || 0), 0) / comp.groupResults.length;
              const avgBertF1 = comp.groupResults.reduce((sum, r) => sum + (r.bertscore_f1 || 0), 0) / comp.groupResults.length;
              return `
              <tr>
                <td>${comp.question.substring(0, 150)}${comp.question.length > 150 ? '...' : ''}</td>
                <td style="text-align: center;"><span class="group-badge">${comp.groupResults.length}</span></td>
                <td style="text-align: center;" class="${getMetricClass(avgRouge1)}">
                  ${(avgRouge1 * 100).toFixed(2)}%
                </td>
                <td style="text-align: center;" class="${getMetricClass(avgRouge2)}">
                  ${(avgRouge2 * 100).toFixed(2)}%
                </td>
                <td style="text-align: center;" class="${getMetricClass(avgBertF1)}">
                  ${(avgBertF1 * 100).toFixed(2)}%
                </td>
              </tr>
              `;
            }).join('')}
          </tbody>
        </table>

        <!-- Answer vs Ground Truth Comparison -->
        <div class="page-break"></div>
        <h2>📋 Cevap ve Ground Truth Karşılaştırması</h2>
        ${groupComparisons.map((comp, groupIdx) => `
          <div style="margin-bottom: 30px;">
            <div style="background: linear-gradient(135deg, #eef2ff 0%, #e0e7ff 100%); border-left: 4px solid #6366f1; padding: 15px; border-radius: 8px; margin-bottom: 15px;">
              <h3 style="color: #4338ca; margin: 0; font-size: 16px;">Test ${groupIdx + 1}</h3>
            </div>
            <table>
              <thead>
                <tr>
                  <th style="width: 18%;">Soru</th>
                  <th style="width: 22%;">Üretilen Cevap</th>
                  <th style="width: 22%;">Ground Truth</th>
                  <th style="text-align: center; width: 6%;">ROUGE-1</th>
                  <th style="text-align: center; width: 6%;">ROUGE-2</th>
                  <th style="text-align: center; width: 6%;">ROUGE-L</th>
                  <th style="text-align: center; width: 6%;">BERTScore F1</th>
                  <th style="text-align: center; width: 8%;">Yanıt Süresi</th>
                </tr>
              </thead>
              <tbody>
                ${comp.results.map((result, resultIdx) => `
                  <tr>
                    <td style="vertical-align: top; font-size: 10px;">${result.question.substring(0, 200)}${result.question.length > 200 ? '...' : ''}</td>
                    <td style="vertical-align: top; font-size: 10px;">${result.generated_answer.substring(0, 300)}${result.generated_answer.length > 300 ? '...' : ''}</td>
                    <td style="vertical-align: top; font-size: 10px;">${result.ground_truth.substring(0, 300)}${result.ground_truth.length > 300 ? '...' : ''}</td>
                    <td style="text-align: center; vertical-align: top;" class="${result.rouge1 != null ? getMetricClass(result.rouge1) : ''}">
                      ${result.rouge1 != null ? (result.rouge1 * 100).toFixed(2) + '%' : '-'}
                    </td>
                    <td style="text-align: center; vertical-align: top;" class="${result.rouge2 != null ? getMetricClass(result.rouge2) : ''}">
                      ${result.rouge2 != null ? (result.rouge2 * 100).toFixed(2) + '%' : '-'}
                    </td>
                    <td style="text-align: center; vertical-align: top;" class="${result.rougel != null ? getMetricClass(result.rougel) : ''}">
                      ${result.rougel != null ? (result.rougel * 100).toFixed(2) + '%' : '-'}
                    </td>
                    <td style="text-align: center; vertical-align: top;" class="${result.original_bertscore_f1 != null ? getMetricClass(result.original_bertscore_f1) : ''}">
                      ${result.original_bertscore_f1 != null ? (result.original_bertscore_f1 * 100).toFixed(2) + '%' : '-'}
                    </td>
                    <td style="text-align: center; vertical-align: top; font-weight: bold;">
                      ${result.latency_ms != null ? result.latency_ms.toFixed(0) + ' ms' : '-'}
                    </td>
                  </tr>
                `).join('')}
              </tbody>
            </table>
          </div>
        `).join('')}

        <!-- Footer -->
        <div class="footer">
          <h4>📖 Metrik Açıklamaları</h4>
          <ul>
            <li><strong>ROUGE-1/2/L:</strong> N-gram overlap metrikleri (0-100%). Üretilen cevabın referans cevapla kelime bazında ne kadar örtüştüğünü ölçer.</li>
            <li><strong>BERTScore:</strong> Anlamsal benzerlik metrikleri (0-100%). Embedding tabanlı semantik benzerlik.</li>
          </ul>
          <p style="margin-top: 25px; text-align: center; font-style: italic;">
            Bu rapor Akıllı Rehber RAG Sistemi tarafından otomatik olarak oluşturulmuştur.<br/>
            Rapor Tarihi: ${new Date().toLocaleString('tr-TR')}
          </p>
        </div>
      </div>
    </body>
    </html>
  `;

  return htmlContent;
};
