import pandas as pd

df_all = pd.read_csv("output_acc/phase2_all_acc.csv")

micro = (
    df_all.groupby("prompt_type")[["TP", "FP", "FN_partial"]]
          .sum()
          .assign(
              precision   = lambda x: x.TP / (x.TP + x.FP),
              rel_recall  = lambda x: x.TP / (x.TP + x.FN_partial),
              f1_rel      = lambda x: 2 * x.precision * x.rel_recall /
                                      (x.precision + x.rel_recall)
          ).sort_values("f1_rel", ascending=False)  
)

print("=== Micro (加總) ===")
print(micro, end="\n\n")
