import pandas as pd
import numpy as np

# 0. 參數設定
SRC = "full_annotated_table.csv"
OUT_RISK = "ensemble_stats_by_risk.csv"
THRESHOLD = 2  # ≥2 票即預測為 1

# 1. 讀入原始資料
df = pd.read_csv(SRC)

# 2. 建立 ground truth：對 (company, 風險代碼, chunk_id) 做 OR 合併
chunk_label = (
    df
    .groupby(["company", "風險代碼", "chunk_id"])["是否真的有揭露"]
    .max()
    .astype(int)
)

# 3. 建立 is_pred 並 pivot 出投票矩陣
df["is_pred"] = 1  # 因原始檔每列都是模型預測為 1

preds = (
    df
    .pivot_table(
        index=["company", "風險代碼", "chunk_id"],
        columns="prompt_type",
        values="is_pred",
        aggfunc="max",
        fill_value=0
    )
)

# 4. 計算 ensemble y_pred
vote_sum = preds.sum(axis=1)
y_pred   = (vote_sum >= THRESHOLD).astype(int)

# 5. 對齊 y_true 與 y_pred
y_true = chunk_label.loc[preds.index]  # 索引相同：(company, 風險代碼, chunk_id)

# 6. 準備風險對應表
risk_names = (
    df[["風險代碼", "風險名稱"]]
    .drop_duplicates()
    .set_index("風險代碼")["風險名稱"]
)

# 7. 計算每個風險的指標，並檢查 FN 與之前的 FN_partial 是否一致
records = []
for risk_code in y_true.index.get_level_values("風險代碼").unique():
    # 篩出這個風險的所有 (company, risk_code, chunk_id)
    idx = [idx for idx in y_true.index if idx[1] == risk_code]
    yt = y_true.loc[idx]
    yp = y_pred.loc[idx]
    
    # 傳統指標
    tp = int(((yt==1) & (yp==1)).sum())
    fp = int(((yt==0) & (yp==1)).sum())
    fn = int(((yt==1) & (yp==0)).sum())
    
    # 先前 FN_partial 定義：金標為 1 但不在 pred_keys (即 yp==0)
    FN_partial = int(sum((yt == 1) & (yp == 0)))
    assert fn == FN_partial, f"Risk {risk_code} 的 FN 不一致！"
    
    # 計算 precision, recall, f1_rel
    precision = tp / (tp + fp) if tp + fp > 0 else np.nan
    recall    = tp / (tp + fn) if tp + fn > 0 else np.nan
    f1_rel    = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0 else np.nan
    )
    
    records.append({
        "風險代碼": risk_code,
        "風險名稱": risk_names[risk_code],
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "Precision": precision,
        "Recall": recall,
        "F1_rel": f1_rel
    })

# 8. 輸出結果
res_df = pd.DataFrame(records).sort_values("風險代碼")
res_df.to_csv(OUT_RISK, index=False)
print(f"Saved overall per-risk stats → {OUT_RISK}")
