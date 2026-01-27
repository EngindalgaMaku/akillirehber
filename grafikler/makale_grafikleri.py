import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load Data
# Config 1 (Jina) - Jina v3 Embed + v2 Rerank + GLM-4.7
c1_data = {
    'Metrik': ['ROUGE-1', 'ROUGE-2', 'ROUGE-L', 'BERTScore', 'Gecikme'],
    'Baseline': [0.712, 0.625, 0.682, 0.822, 4907],
    'Rerank': [0.773, 0.693, 0.743, 0.855, 4571]
}
df1 = pd.DataFrame(c1_data)
df1['Konfigürasyon'] = 'Konfigürasyon 1'

# Config 2 (OpenAI/Alibaba) - OpenAI v3 Embed + GTE Rerank + Qwen-Flash
c2_data = {
    'Metrik': ['ROUGE-1', 'ROUGE-2', 'ROUGE-L', 'BERTScore', 'Gecikme'],
    'Baseline': [0.758, 0.675, 0.720, 0.852, 4580],
    'Rerank': [0.779, 0.697, 0.742, 0.864, 4422]
}
df2 = pd.DataFrame(c2_data)
df2['Konfigürasyon'] = 'Konfigürasyon 2'

# Config 3 (Cohere) - Cohere v3 Embed + v3 Rerank + GLM-4.7
c3_data = {
    'Metrik': ['ROUGE-1', 'ROUGE-2', 'ROUGE-L', 'BERTScore', 'Gecikme'],
    'Baseline': [0.735, 0.649, 0.710, 0.835, 4374],
    'Rerank': [0.776, 0.695, 0.751, 0.855, 4201]
}
df3 = pd.DataFrame(c3_data)
df3['Konfigürasyon'] = 'Konfigürasyon 3'

# Config 4 (BAAI) - BGE M3 Embed + BGE Reranker v2-m3
c4_data = {
    'Metrik': ['ROUGE-1', 'ROUGE-2', 'ROUGE-L', 'BERTScore', 'Gecikme'],
    'Baseline': [0.788, 0.711, 0.757, 0.863, 5227],
    'Rerank': [0.7963, 0.725, 0.769, 0.862, 5872]
}
df4 = pd.DataFrame(c4_data)
df4['Konfigürasyon'] = 'Konfigürasyon 4'

# Combine for plotting
all_configs = pd.concat([df1, df2, df3, df4])
melted = all_configs.melt(id_vars=['Metrik', 'Konfigürasyon'], value_vars=['Baseline', 'Rerank'], var_name='Senaryo', value_name='Deger')

# Q1 Journal Quality Professional Plots

# Figure 1: ROUGE-1 Performance
plt.figure(figsize=(6, 4))
rouge1_data = melted[melted['Metrik'] == 'ROUGE-1']
ax = sns.barplot(data=rouge1_data, x='Konfigürasyon', y='Deger', hue='Senaryo', 
                 palette=['#2E86AB', '#A23B72'], width=0.8)
plt.ylim(0.65, 0.80)
plt.xlabel('')
plt.ylabel('ROUGE-1 Skoru', fontsize=11)
plt.legend(title='Senaryo', title_fontsize=10, fontsize=9, bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(rotation=45, ha='right', fontsize=9)
plt.yticks(fontsize=9)
sns.despine()
plt.tight_layout()
plt.savefig('figure1_rouge1.png', dpi=300, bbox_inches='tight')

# Figure 2: ROUGE-2 Performance  
plt.figure(figsize=(6, 4))
rouge2_data = melted[melted['Metrik'] == 'ROUGE-2']
ax = sns.barplot(data=rouge2_data, x='Konfigürasyon', y='Deger', hue='Senaryo',
                 palette=['#2E86AB', '#A23B72'], width=0.8)
plt.ylim(0.55, 0.75)
plt.xlabel('')
plt.ylabel('ROUGE-2 Skoru', fontsize=11)
plt.legend(title='Senaryo', title_fontsize=10, fontsize=9, bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(rotation=45, ha='right', fontsize=9)
plt.yticks(fontsize=9)
sns.despine()
plt.tight_layout()
plt.savefig('figure2_rouge2.png', dpi=300, bbox_inches='tight')

# Figure 3: ROUGE-L Performance
plt.figure(figsize=(6, 4))
rougel_data = melted[melted['Metrik'] == 'ROUGE-L']
ax = sns.barplot(data=rougel_data, x='Konfigürasyon', y='Deger', hue='Senaryo',
                 palette=['#2E86AB', '#A23B72'], width=0.8)
plt.ylim(0.60, 0.80)
plt.xlabel('')
plt.ylabel('ROUGE-L Skoru', fontsize=11)
plt.legend(title='Senaryo', title_fontsize=10, fontsize=9, bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(rotation=45, ha='right', fontsize=9)
plt.yticks(fontsize=9)
sns.despine()
plt.tight_layout()
plt.savefig('figure3_rougel.png', dpi=300, bbox_inches='tight')

# Figure 4: BERTScore Performance
plt.figure(figsize=(6, 4))
bert_data = melted[melted['Metrik'] == 'BERTScore']
ax = sns.barplot(data=bert_data, x='Konfigürasyon', y='Deger', hue='Senaryo',
                 palette=['#2E86AB', '#A23B72'], width=0.8)
plt.ylim(0.75, 0.90)
plt.xlabel('')
plt.ylabel('BERTScore F1', fontsize=11)
plt.legend(title='Senaryo', title_fontsize=10, fontsize=9, bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(rotation=45, ha='right', fontsize=9)
plt.yticks(fontsize=9)
sns.despine()
plt.tight_layout()
plt.savefig('figure4_bertscore.png', dpi=300, bbox_inches='tight')

# Figure 5: Latency Comparison
plt.figure(figsize=(6, 4))
latency_data = melted[melted['Metrik'] == 'Gecikme']
ax = sns.barplot(data=latency_data, x='Konfigürasyon', y='Deger', hue='Senaryo',
                 palette=['#2E86AB', '#A23B72'], width=0.8)
plt.ylim(4000, 5500)
plt.xlabel('')
plt.ylabel('Gecikme Süresi (ms)', fontsize=11)
plt.legend(title='Senaryo', title_fontsize=10, fontsize=9, bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(rotation=45, ha='right', fontsize=9)
plt.yticks(fontsize=9)
sns.despine()
plt.tight_layout()
plt.savefig('figure5_latency.png', dpi=300, bbox_inches='tight')

# Visualization 2: ROUGE-2 Detailed Uplift (The most significant metric)
plt.figure(figsize=(10, 6))
r2_data = melted[melted['Metrik'] == 'ROUGE-2']
ax = sns.barplot(data=r2_data, x='Konfigürasyon', y='Deger', hue='Senaryo', palette='flare')
plt.title('ROUGE-2 (Öbek Bazlı Sadakat) Karşılaştırması', fontsize=14)
plt.ylim(0.5, 0.8)
for p in ax.patches:
    ax.annotate(format(p.get_height(), '.3f'), (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', xytext=(0, 9), textcoords='offset points')
plt.savefig('rouge2_karsilastirma.png')

# Visualization 3: Latency Comparison
plt.figure(figsize=(10, 6))
latency_data = melted[melted['Metrik'] == 'Gecikme']
sns.barplot(data=latency_data, x='Konfigürasyon', y='Deger', hue='Senaryo', palette='magma')
plt.title('Sistem Gecikme Süresi (ms) Karşılaştırması', fontsize=14)
plt.ylabel('Milisaniye (ms)')
plt.savefig('gecikme_karsilastirma.png')

print("Final charts created successfully.")
