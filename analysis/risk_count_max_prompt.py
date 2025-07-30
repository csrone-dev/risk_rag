import pandas as pd

# 讀檔
df = pd.read_csv('risk_prompt_metrics.csv') 

# 1. 取各風險 f1 最高那一列
best = df.loc[df.groupby('risk_name')['f1_rel'].idxmax()]

# 2. 統計不同 prompt_type 拿到「最高 f1」的次數
win_counts = best['prompt_type'].value_counts().reindex(
    ['FEW_SHOT_PROMPT','FEW_SHOT_COT_PROMPT','ZERO_SHOT_COT_PROMPT','ZERO_SHOT_PROMPT']
).fillna(0)

print(win_counts.to_string())
