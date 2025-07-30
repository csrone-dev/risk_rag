import pandas as pd
import numpy as np
import glob

# === 1. 讀進所有公司檔 ===
files = glob.glob("../output_annotated/phase3/*_dev_result_with_label.csv")
if not files:
    raise FileNotFoundError("找不到 output_annotated/phase3 資料夾下的 *_dev_result_with_label.csv 檔案")

df = pd.concat([pd.read_csv(f, encoding="utf-8") for f in files], ignore_index=True)

breakpoint()

# === 2. 保留必要欄位（含風險代碼） ===
df = (
    df[["prompt_type", "風險代碼", "風險名稱", "chunk_id", "是否真的有揭露"]]
    .drop_duplicates()
)

# === 3. 建真實標籤：key = (風險代碼, chunk_id) ===
key_cols    = ["風險代碼", "chunk_id"]
chunk_label = df.groupby(key_cols)["是否真的有揭露"].max().to_dict()
all_keys      = set(chunk_label.keys())
prompt_types  = df["prompt_type"].unique()
risk_codes    = df["風險代碼"].unique()

records = []

# === 4. 逐風險 × prompt_type 計 TP / FP / FN_partial / Precision / Recall / F1 ===
for rc in risk_codes:
    risk_keys = {k for k in all_keys if k[0] == rc}
    risk_name = df.loc[df["風險代碼"] == rc, "風險名稱"].iloc[0]

    for p in prompt_types:
        pred_keys = set(
            df.loc[
                (df["prompt_type"] == p) & (df["風險代碼"] == rc),
                ["風險代碼", "chunk_id"]
            ].itertuples(index=False, name=None)
        )

        TP = sum(chunk_label[k] == 1 for k in pred_keys)
        FP = sum(chunk_label[k] == 0 for k in pred_keys)
        FN = sum((chunk_label[k] == 1) and (k not in pred_keys) for k in risk_keys)

        precision  = TP / (TP + FP) if TP + FP else np.nan
        rel_recall = TP / (TP + FN)    if TP + FN else np.nan
        f1_rel     = (2 * precision * rel_recall / (precision + rel_recall)
                      if (precision + rel_recall) else np.nan)

        records.append([
            rc, risk_name, p,
            TP, FP, FN,
            precision, rel_recall, f1_rel
        ])

# === 5. 整理每風險 × prompt 詳細指標 ===
cols = [
    "risk_code","risk_name","prompt_type",
    "TP","FP","FN_partial",
    "precision","rel_recall","f1_rel"
]
metrics = (
    pd.DataFrame(records, columns=cols)
      .sort_values(["risk_code","f1_rel"], ascending=[True, False])
)
metrics.to_csv("risk_prompt_metrics.csv", index=False)

# === 6. 針對 每個風險 彙總 precision/recall/f1 的平均與標準差 ===
risk_stats = (
    metrics
      .groupby(["risk_code","risk_name"])
      [["precision","rel_recall","f1_rel"]]
      .agg(["mean","std"])
)

# flatten MultiIndex
risk_stats.columns = [
    f"{metric}_{stat}"
    for metric, stat in risk_stats.columns
]
risk_stats = risk_stats.reset_index()

# 6-1 數值版輸出
risk_stats.to_csv("risk_stats_numeric.csv", index=False)

# 6-3 同時保留 avg F1 的舊檔名
risk_stats[["risk_code","risk_name","f1_rel_mean"]].to_csv(
    "risk_f1_average.csv", index=False
)

print("✅ 已輸出：")
print("  • risk_prompt_metrics.csv    (風險 × prompt 詳細指標)")
print("  • risk_stats_numeric.csv     (precision/recall/f1 mean & std 數值版)")
print("  • risk_f1_average.csv        (各風險平均 F1_rel)")
