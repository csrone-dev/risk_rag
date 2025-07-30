import pandas as pd
import numpy as np

# 僅保留兩種 CoT 且有 reasoning 的列 
df = pd.read_csv("full_annotated_table.csv")

cot_prompts = ["ZERO_SHOT_COT_PROMPT", "FEW_SHOT_COT_PROMPT"] 
df = df[
    df["prompt_type"].isin(cot_prompts)
    & df["模型推論過程"].notna()          # reasoning 不為空
]

# 隨機抽樣：每種 CoT 15 TP + 15 FP 
rng = np.random.default_rng(0)
samples = []

for ptype in cot_prompts:
    g = df[df["prompt_type"] == ptype]

    tp = g[g["是否真的有揭露"] == 1].sample(15, random_state=rng)
    # fp = g[g["是否真的有揭露"] == 0].sample(15, random_state=rng)
    samples.append(pd.concat([tp]))

qualitative_set = pd.concat(samples).reset_index(drop=True)  


qualitative_set.to_csv("cot_sample_2.csv", index=False)
print("抽樣完成！總筆數：", len(qualitative_set))
