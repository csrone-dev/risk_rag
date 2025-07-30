import pandas as pd
import numpy as np
import glob
import os

files = glob.glob("../output_annotated/phase3/*_dev_result_with_label.csv")
if not files:
    raise FileNotFoundError("找不到 output_annotated/phase3 資料夾下的 *_dev_result_with_label.csv 檔案")

dfs = []
for f in files:
    df_temp = pd.read_csv(f, encoding="utf-8")
    # 從檔名抽出公司名稱 (如 "永豐金2890_dev_result_with_label.csv" -> "永豐金2890")
    company = os.path.basename(f).replace("_dev_result_with_label.csv", "")
    df_temp["company"] = company
    dfs.append(df_temp)

df = pd.concat(dfs, ignore_index=True)

# === 2. 輸出含 company 的完整表 ===
df.to_csv("full_annotated_table.csv", index=False, encoding="utf-8")
