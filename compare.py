# Compare AUROC / AUPRC of main.py results: cluster vs smooth preprocessing
import os
import pandas as pd
from Evaluate import concat_ref, cal_auc_aupr

rows = []
for method in ["cluster", "smooth"]:
    path = f"rankedEdges_{method}.csv"
    if not os.path.exists(path):
        print(f"Missing {path}, run main.py with --preprocess {method} first")
        continue
    df = pd.read_csv(path)
    df_ref = pd.read_csv("input/refNetwork.csv", usecols=["Gene1", "Gene2"])
    res = cal_auc_aupr(concat_ref(df, df_ref))
    rows.append({"preprocess": method, "edges": len(df), **res})

out = pd.DataFrame(rows)
print(out.to_string(index=False))
out.to_csv("compare_results.csv", index=False)
