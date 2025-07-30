import pandas as pd

company_name = "微星2377"
answer = pd.read_csv(f"output_annotated/phase3/{company_name}_sinyi_result_with_label.csv")
result = pd.read_csv(
    f"output_generation/phase3/{company_name}_gen_result_for_dev.csv"
)

answer['是否真的有揭露'] = answer['是否真的有揭露'].map({
    '正確': 1,
    '錯誤': 0
})

merged = result.merge(
    # 從 answer 選出三欄合併用：風險名稱、揭露句子、標籤
    answer[["風險名稱", "chunk_id", "是否真的有揭露"]],
    how="left",
    left_on=["風險名稱", "chunk_id"],  # result 裡的 key
    right_on=["風險名稱", "chunk_id"],  # answer 裡的 key
)

merged.to_csv(
    f"output_annotated/phase3/{company_name}_dev_result_with_label.csv", index=False, encoding="utf-8"
)
