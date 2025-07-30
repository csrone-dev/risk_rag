import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
plt.rcParams['font.family'] = ['Times New Roman', 'AppleGothic']  # 先用 Times New Roman，找不到中文字才 fallback 用 AppleGothic


# ---------- 1. 讀入資料 ----------
df = pd.read_csv('risk_prompt_metrics.csv')

# ---------- 2. 派別標籤 ----------
df['family'] = df['prompt_type'].apply(lambda x: 'Few-shot' if 'FEW' in x else 'Zero-shot')
df['is_cot'] = df['prompt_type'].str.contains('_COT_')

# ---------- 3. 計算 Δ 值 (CoT − Non‑CoT) ----------
records = []
for (risk, fam), grp in df.groupby(['risk_name', 'family']):
    cot = grp[grp.is_cot]
    non = grp[~grp.is_cot]
    if cot.empty or non.empty:
        continue
    records.append({
        'risk_name': risk,
        'family': fam,
        'Δprecision': cot['precision'].iloc[0]   - non['precision'].iloc[0],
        'Δrecall':    cot['rel_recall'].iloc[0]  - non['rel_recall'].iloc[0],
        'Δf1':        cot['f1_rel'].iloc[0]      - non['f1_rel'].iloc[0]
    })

delta = pd.DataFrame(records)

# ---------- 4. 繪製箱型圖 ----------
fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
metrics = ['Δprecision', 'Δrecall', 'Δf1']
titles  = ['ΔPrecision', 'ΔRecall', 'ΔF1']

for ax, metric, title in zip(axes, metrics, titles):
    zs = delta.loc[delta.family == 'Zero-shot', metric]
    fs = delta.loc[delta.family == 'Few-shot', metric]
    ax.boxplot([zs, fs],
               labels=['Zero-shot', 'Few-shot'],
               meanline=True, showmeans=True)
    ax.axhline(0, color='gray', linewidth=0.8, linestyle='--')
    ax.set_title(title)
    ax.set_ylabel('Δ (CoT - Non-CoT)')

fig.suptitle('Distribution of ΔMetrics by Prompt Variants (CoT - Non-CoT)', fontsize=14)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
# plt.savefig('risk_metrics_boxplots.png', dpi=300)
# plt.show()


import pandas as pd


# Map helper columns
df['family'] = df['prompt_type'].apply(lambda x: 'Few-shot' if 'FEW' in x else 'Zero-shot')
df['is_cot'] = df['prompt_type'].str.contains('_COT_')

# Compute deltas
records = []
for (risk, fam), grp in df.groupby(['risk_name', 'family']):
    cot = grp[grp.is_cot]
    non = grp[~grp.is_cot]
    if cot.empty or non.empty:
        continue
    records.append({
        'risk_name': risk,
        'family': fam,
        'Δprecision': cot['precision'].iloc[0] - non['precision'].iloc[0],
        'Δrecall':    cot['rel_recall'].iloc[0] - non['rel_recall'].iloc[0],
        'Δf1':        cot['f1_rel'].iloc[0] - non['f1_rel'].iloc[0],
    })

delta = pd.DataFrame(records)

# Summary stats
summary = (delta
           .groupby('family')[['Δprecision','Δrecall','Δf1']]
           .agg(['median','mean','min','max']))
print(summary)

