import pandas as pd
from itertools import combinations
from sklearn.metrics import cohen_kappa_score
from statsmodels.stats.inter_rater import fleiss_kappa

df = pd.read_csv('full_annotated_table.csv', dtype={'chunk_id': str, 'prompt_type': str})

# 建立 chunk × prompt 的二元存在矩陣
# 有抓到的標 1，沒抓到的標 0
presence = (
    df.assign(present=1)
      .pivot_table(index='chunk_id', columns='prompt_type', values='present', fill_value=0)
      .astype(int)  
)

print(" ---- presence matrix ----")
print(presence.head())

print("\n---- Pairwise metrics ----")
for p1, p2 in combinations(presence.columns, 2):
    v1 = presence[p1]
    v2 = presence[p2]
    # Jaccard = |A∩B| / |A∪B|
    inter = (v1 & v2).sum()
    union = (v1 | v2).sum()
    jaccard = inter / union if union > 0 else float('nan')
    print(f"{p1} vs {p2}: Jaccard = {jaccard:.3f}")

