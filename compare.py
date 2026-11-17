"""
compare.py - score main.py's two output files against the reference network.

SUMMARY
    Reads rankedEdges_cluster.csv and rankedEdges_smooth.csv (made by main.py),
    computes AUROC and AUPRC for each with Evaluate.cal_auc_aupr, prints a table
    and saves it to compare_results.csv.

WARNING (old scoring)
    cal_auc_aupr grades each method only on the edges it output. Edges mRMR
    deleted never count as misses, so two methods with different edge lists get
    graded on different tests. This script produced last week's
    "cluster 0.62 vs smooth 0.558" result. Use eval_full.eval_full for a fair
    comparison on all gene pairs.

HOW TO RUN
    python main.py ... --preprocess cluster
    python main.py ... --preprocess smooth
    python compare.py
"""
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
    # concat_ref labels each output edge real (1) or fake (0); only output edges get scored
    res = cal_auc_aupr(concat_ref(df, df_ref))
    rows.append({"preprocess": method, "edges": len(df), **res})

out = pd.DataFrame(rows)
print(out.to_string(index=False))
out.to_csv("compare_results.csv", index=False)
