import pandas as pd
import numpy as np

company_name = "微星2377"
df = pd.read_csv(f"output_annotated/phase3/{company_name}_dev_result_with_label.csv")
df = df[["prompt_type", "風險名稱", "chunk_id", "是否真的有揭露"]].drop_duplicates()

# 人標好的真實答案
# 同風險 chunk 的人工標註應該一致 → 取 max (但若真的同時出現 0 與 1，取 max 當正例)
chunk_label = df.groupby(["風險名稱", "chunk_id"])["是否真的有揭露"].max().to_dict()
print(len(chunk_label))

# 跟上面印出來應該要一樣
n_unique_pairs = df[["風險名稱", "chunk_id"]].drop_duplicates().shape[0]
print(n_unique_pairs)
all_chunks = set(chunk_label.keys())
print(len(all_chunks))

# === 各個 prompt_type 計算 TP / FP / FN（沒有 TN 所以不算 accuracy）===
prompt_types = df["prompt_type"].unique()
results = []

for p in prompt_types:
    # 此 prompt 認為有揭露的 (風險名稱, chunk_id)，轉成 tuple 集合
    pred_keys = set(
        df.loc[df["prompt_type"] == p, ["風險名稱", "chunk_id"]].itertuples(
            index=False, name=None
        )  # → (risk, chunk) tuple
    )

    # print(len(pred_keys))

    TP = sum(chunk_label[k] == 1 for k in pred_keys)
    FP = sum(chunk_label[k] == 0 for k in pred_keys)

    # FN 可能會有在所有 prompt 沒抓的 chunk 中，出現一些漏抓的
    # k not in pred_keys => 也就是此 prompt「沒抓到，視為 0」# 但實際上人標為 1
    FN_partial = sum(chunk_label[k] == 1 and k not in pred_keys for k in all_chunks)

    # TN 最不準，因為在所有 prompt 沒抓的 chunk 中，多了很多 TN
    TN_partial = len(all_chunks) - (TP + FP + FN_partial)

    # print(len(all_chunks) - len(pred_keys))
    # print(TP, FP, FN)

    precision = TP / (TP + FP)
    rel_recall = TP / (TP + FN_partial)
    rel_f1 = 2 * precision * rel_recall / (precision + rel_recall)

    results.append([p, TP, FP, FN_partial, TN_partial, precision, rel_recall, rel_f1])


# === 輸出結果 ===
metrics = pd.DataFrame(
    results,
    columns=[
        "prompt_type",
        "TP",
        "FP",
        "FN_partial",
        "TN_partial",
        "precision",
        "rel_recall",
        "rel_f1",
    ],
).sort_values("rel_f1", ascending=False)

metrics[["precision", "rel_recall", "rel_f1"]] = metrics[
    ["precision", "rel_recall", "rel_f1"]
].applymap(lambda x: f"{x:.2%}")

name_map = {
    "ZERO_SHOT_PROMPT":        "Zero-shot",
    "ZERO_SHOT_COT_PROMPT":    "Zero-shot CoT",
    "FEW_SHOT_PROMPT":         "Few-shot",
    "FEW_SHOT_COT_PROMPT":     "Few-shot CoT"
}

metrics["prompt_type"] = metrics["prompt_type"].map(name_map)
metrics = metrics.rename(columns={"prompt_type": "Prompt type"})

print(metrics.to_string(index=False))
