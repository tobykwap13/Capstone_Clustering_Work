# -*- coding:utf-8 -*-
"""
ensemble_best.py - AUROC booster check.

SUMMARY
    Builds three networks per run (smooth lasso, cluster + distance correlation,
    cluster + backward KL), tries each alone and in rank-averaged combinations,
    and grades all with fair scoring. Output: three_network_results/ensemble_best.csv.

CAUTION
    The two divergence networks and their lambda values came from
    divergence_sweep.py results on this same reference network. Treat the
    combined scores as optimistic until tested on a second dataset.

DETAILS
Rank-average the smooth lasso network with the best
divergence networks from divergence_sweep.py (cluster + dcor, cluster + backward KL).
Same clean + 5 noisy runs and seeds as three_networks.py.

  python ensemble_best.py
"""
import os
os.environ.setdefault("TQDM_DISABLE", "1")
import warnings
import numpy as np
import pandas as pd

from order_cluster import order_cluster
from mRMR import MRMR2_divergence
from divergence_scores import score_pairs
from smooth_lasso import build_stages, _pairs, fit_dynamic_network, edges_from_A, cv_select
from eval_full import eval_full
from three_networks import rank_ensemble
from sweep import add_noise

warnings.filterwarnings("ignore")


def div_net(proc, count, scorer, lam):
    # Score all pairs, then re-rank with divergence mRMR (lambda 0 = no mRMR)
    e = score_pairs(proc, count, scorer)
    return MRMR2_divergence(e, lambda_val=lam) if lam > 0 else e


df_exp = pd.read_csv("input/ExpressionData.csv", index_col=0)
df_pse = pd.read_csv("input/PseudoTime.csv", index_col=0)
df_ref = pd.read_csv("input/refNetwork.csv")
genes = list(df_exp.columns)
rows = []
for run in range(0, 6):
    data = df_exp if run == 0 else add_noise(df_exp, np.random.default_rng(42 + run), 0.05)
    c, proc = order_cluster(df_pse, data, cells_per_cluster=5)
    nets = {"cluster_dcor": div_net(proc, c, "dcor", 1.0),
            "cluster_KLbwd": div_net(proc, c, "KL_bwd", 0.5)}
    a, b, _ = cv_select(df_pse, data, 5, 30, standardize=False)
    Y, seq = build_stages(df_pse, data, 5, 30, standardize=False)
    nets["smooth_lasso"] = edges_from_A(fit_dynamic_network(_pairs(Y, seq, "scpadgrn"), a, b), genes)
    combos = [("cluster_dcor",), ("cluster_KLbwd",), ("smooth_lasso",),
              ("smooth_lasso", "cluster_dcor"), ("smooth_lasso", "cluster_KLbwd"),
              ("smooth_lasso", "cluster_dcor", "cluster_KLbwd")]
    for combo in combos:
        e = nets[combo[0]] if len(combo) == 1 else rank_ensemble([nets[k] for k in combo], genes)
        fa, fp = eval_full(e, df_ref, genes)
        rows.append(dict(run=run, network=" + ".join(combo), AUROC_full=fa, AUPRC_full=fp))
    print(f"run {run} done", flush=True)

d = pd.DataFrame(rows)
os.makedirs("three_network_results", exist_ok=True)
d.to_csv("three_network_results/ensemble_best.csv", index=False)
s = d[d.run > 0].groupby("network")[["AUROC_full", "AUPRC_full"]].agg(["mean", "std"]).round(4)
s[("AUROC_full", "clean")] = d[d.run == 0].set_index("network").AUROC_full.round(4)
print(s.sort_values(("AUROC_full", "mean"), ascending=False).to_string())
