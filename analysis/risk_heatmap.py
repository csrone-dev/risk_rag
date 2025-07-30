import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
plt.rcParams['font.family'] = ['Times New Roman', 'AppleGothic']  # 先用 Times New Roman，找不到中文字才 fallback 用 AppleGothic


# 1. 載入資料
df_prompt = pd.read_csv('risk_prompt_metrics.csv')           # 欄位: risk_name, prompt_type, f1_rel
df_ens    = pd.read_csv('ensemble_stats_by_risk.csv')    # 欄位: 風險名稱, F1_rel

# 2. Pivot 並合併 Ensemble
df_heat = df_prompt.pivot(index='risk_name', columns='prompt_type', values='rel_recall')
rename_map = {
    'ZERO_SHOT_PROMPT'       : 'Zero-shot',
    'ZERO_SHOT_COT_PROMPT'   : 'Zero-shot CoT',
    'FEW_SHOT_PROMPT'        : 'Few-shot',
    'FEW_SHOT_COT_PROMPT'    : 'Few-shot CoT',
}
df_heat.rename(columns=rename_map, inplace=True)

df_heat['Ensemble'] = df_ens.set_index('風險名稱')['Recall']

# 3. 依 Ensemble F1 值由大到小排序
df_heat = df_heat.sort_values('Ensemble', ascending=False)

# 4. 指定 x 軸順序
ordered_cols = ['Zero-shot', 'Zero-shot CoT', 'Few-shot', 'Few-shot CoT', 'Ensemble']
df_heat = df_heat[ordered_cols]

# 5. 繪製 Heatmap
fig, ax = plt.subplots(figsize=(10, 10))
im = ax.imshow(df_heat.values, aspect='auto', cmap='viridis_r', vmin=0.57, vmax=1)

# 設定刻度與標籤
ax.set_xticks(range(len(df_heat.columns)))
ax.set_xticklabels(df_heat.columns, rotation=45, ha='right')
ax.set_yticks(range(len(df_heat.index)))
ax.set_yticklabels(df_heat.index)

# 在每個格子寫入數值
for i in range(df_heat.shape[0]):
    for j in range(df_heat.shape[1]):
        val = df_heat.iloc[i, j]
        ax.text(j, i, f"{val:.2f}", ha='center', va='center',
                color='black' if val < 0.8 else 'white', fontsize=8)

ax.set_xlabel('Method')
ax.set_ylabel('Risk name')
# ax.set_title('各 Prompt 及 Ensemble F1 Heatmap (依 Ensemble 排序)')
fig.colorbar(im, ax=ax, label='Recall score')

plt.tight_layout()
plt.savefig(f'prompt_and_ensemble_heatmap_recall.png', dpi=300)
plt.show()