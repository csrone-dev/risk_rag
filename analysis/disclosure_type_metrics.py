import pandas as pd

# 1. 讀取 CSV
path = 'disclosure_type_annotation.csv'
df = pd.read_csv(path, encoding='utf-8-sig')

# 2. Ground Truth 二值化（0.5 視為 0）
df['truth_keyword'] = df['類別_關鍵字'].apply(lambda x: 1 if x == 1 else 0)
df['truth_desc'] = df['類別_風險描述']
df['truth_resp'] = df['類別_風險因應']

# 3. 預測結果二值化（從「揭露類別」欄位判斷）
df['pred_keyword'] = df['揭露類別'].str.contains('關鍵字').astype(int)
df['pred_desc'] = df['揭露類別'].str.contains('風險描述').astype(int)
df['pred_resp'] = df['揭露類別'].str.contains('風險因應').astype(int)

# 4. 計算 TP/FP/FN 及指標
results = []
for name, tcol, pcol in [
    ('關鍵字', 'truth_keyword', 'pred_keyword'),
    ('風險描述', 'truth_desc', 'pred_desc'),
    ('風險因應', 'truth_resp', 'pred_resp'),
]:
    TP = int(((df[tcol] == 1) & (df[pcol] == 1)).sum())
    FP = int(((df[tcol] == 0) & (df[pcol] == 1)).sum())
    FN = int(((df[tcol] == 1) & (df[pcol] == 0)).sum())
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0
    results.append({
        '類別': name,
        'TP': TP,
        'FP': FP,
        'FN': FN,
        'Precision': round(precision, 3),
        'Recall': round(recall, 3),
        'F1': round(f1, 3)
    })

metrics_df = pd.DataFrame(results)

print(metrics_df)
