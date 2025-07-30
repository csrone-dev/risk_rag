import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
plt.rcParams['font.family'] = ['Times New Roman', 'AppleGothic']  # 先用 Times New Roman，找不到中文字才 fallback 用 AppleGothic


df = pd.read_csv('risk_prompt_metrics.csv')

# ---------------- 前置欄位 ----------------
df['family'] = df['prompt_type'].apply(lambda x: 'Few-shot' if 'FEW' in x else 'Zero-shot')
df['is_cot'] = df['prompt_type'].str.contains('_COT_')

# ---------------- 計算 ΔF1 (CoT − Non-CoT) ----------------
records = []
for (risk, fam), grp in df.groupby(['risk_name', 'family']):
    cot_row  = grp[grp.is_cot]
    non_row  = grp[~grp.is_cot]
    if cot_row.empty or non_row.empty:
        continue
    delta_f1 = cot_row['f1_rel'].iloc[0] - non_row['f1_rel'].iloc[0]
    records.append({'risk_name': risk, 'family': fam, 'Δf1': delta_f1})

delta_df = pd.DataFrame(records)

# ---------------- 繪圖函式 ----------------
def plot_delta(family, color):
    sub = delta_df[delta_df.family == family].sort_values('Δf1')
    fig, ax = plt.subplots(figsize=(8, 0.3 * len(sub) + 2))
    ax.barh(sub['risk_name'], sub['Δf1'], color=color, alpha=0.85)
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_xlabel('ΔF1 (CoT − Non-CoT)', fontsize=12)
    ax.set_title(f"{family}'s ΔF1 per Risk", fontsize=14)
    plt.tight_layout()
    plt.savefig(f'{family} ΔF1 per Risk.png', dpi=300)
    # plt.show()
    

# ---------------- 繪製兩張圖 ----------------
plot_delta('Few-shot',  'indianred')

plot_delta('Zero-shot', 'steelblue')


