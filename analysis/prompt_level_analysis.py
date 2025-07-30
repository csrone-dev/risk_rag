import pandas as pd

FILE = "full_annotated_table.csv"          # chunk-level 原始檔
KEYS = ["company", "風險名稱", "chunk_id"]   # ← 三欄合起來唯一

# 讀檔並先去掉同 prompt 重複列
cols = ["prompt_type", *KEYS, "是否真的有揭露"]
raw  = (pd.read_csv(FILE, usecols=cols)
          .drop_duplicates())              # (prompt, company, risk, chunk) 唯一

# ① 正例全集（被任一 prompt 抓到的人標=1）
is_pos = (raw.groupby(KEYS)["是否真的有揭露"].max() == 1)
all_pos = {k for k, v in is_pos.items() if v}

# ② 逐 prompt 計 TP, FP, FN (相對 recall)
records = []
for p in sorted(raw.prompt_type.unique()):
    pred = {tuple(x) for x in raw.loc[raw.prompt_type==p, KEYS].to_numpy()}
    TP = len(pred & all_pos)
    FP = len(pred - all_pos)
    FN = len(all_pos - pred)
    prec = TP/(TP+FP)
    rec  = TP/(TP+FN)
    f1   = 2*prec*rec/(prec+rec)
    records.append([p, TP, FP, FN, prec, rec, f1])

out = (pd.DataFrame(records,
        columns=["prompt", "TP", "FP", "FN",
                 "precision", "recall", "f1"])
         .sort_values("f1", ascending=False))
print(out.to_string(index=False))
