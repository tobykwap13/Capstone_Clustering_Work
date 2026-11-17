# -*- coding:utf-8 -*-
"""
divergence_sweep.py - which divergence gives the best AUROC, for cluster vs smooth preprocessing?

SUMMARY
    For each run (clean + 5 noisy) and each preprocessing (cluster, smooth):
    score all gene pairs with each of the 14 scores in divergence_scores.py,
    apply mRMR at lambda 0 / 0.5 / 1 / 1.5, and grade with both old and fair
    scoring. Results live in divergence_results/.

FUNCTIONS
    prune(df, scorer, lam, n_jobs)   mRMR step (MRMR2 for MI, MRMR2_divergence otherwise)
    main()                           the sweep loop and summary tables

DETAILS

Runs: run 0 = clean data, runs 1..R = Gaussian noise (same noise model and
seeds as sweep.py), so every scorer and preprocess sees the same data in a run.
Each scorer is pruned with mRMR at several lambda values (lambda 0 = no pruning,
plain ranking).

Outputs in --out:
  div_raw.csv       one row per run x preprocess x scorer x lambda
  div_summary.csv   mean / SD over noisy runs, plus the clean-data value
  div_best.csv      best lambda per scorer x preprocess (by full AUROC)

  python divergence_sweep.py               (5 noisy runs + clean)
  python divergence_sweep.py --runs 2      (quick)
"""
import os
os.environ.setdefault("TQDM_DISABLE", "1")
import argparse
import time
import numpy as np
import pandas as pd

from PreprocessData import smooth
from order_cluster import order_cluster
from mRMR import MRMR2, MRMR2_divergence
from divergence_scores import SCORERS, score_pairs
from eval_full import eval_pipeline, eval_full
from sweep import add_noise


def prune(df, scorer, lam, n_jobs):
    # lambda 0 = skip mRMR and keep the raw ranking
    if lam == 0:
        return df
    # MI path deletes pairs with adjusted score <= 0 (as in main.py).
    # Divergence path keeps all pairs and only re-ranks them.
    if scorer == "MI":
        out = MRMR2(df.loc[df.score > 0], n_jobs, lambda_val=lam)
        out = out.loc[out.score > 0]
    else:
        out = MRMR2_divergence(df, n_jobs=n_jobs, lambda_val=lam)
    out["score"] = pd.to_numeric(out["score"], errors="coerce")
    return out[np.isfinite(out.score)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--size", type=int, default=5)
    ap.add_argument("--lambdas", type=float, nargs="+", default=[0, 0.5, 1, 1.5])
    ap.add_argument("--scorers", nargs="+", default=list(SCORERS))
    ap.add_argument("--noise", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n_jobs", type=int, default=2)
    ap.add_argument("--out", default="divergence_results")
    opt = ap.parse_args()
    os.makedirs(opt.out, exist_ok=True)

    df_exp = pd.read_csv("input/ExpressionData.csv", index_col=0)
    df_pse = pd.read_csv("input/PseudoTime.csv", index_col=0)
    df_ref = pd.read_csv("input/refNetwork.csv")
    genes = list(df_exp.columns)

    rows, t0 = [], time.time()
    for run in range(0, opt.runs + 1):
        data = df_exp if run == 0 else add_noise(df_exp, np.random.default_rng(opt.seed + run), opt.noise)
        for method in ["cluster", "smooth"]:
            if method == "cluster":
                count, proc = order_cluster(df_pse, data, cells_per_cluster=opt.size)
            else:
                count, proc = smooth(df_pse, data, 1, opt.size)
            for sc in opt.scorers:
                raw = score_pairs(proc, count, sc)
                for lam in opt.lambdas:
                    ed = prune(raw.copy(), sc, lam, opt.n_jobs)
                    pa, pp = eval_pipeline(ed, df_ref)
                    fa, fp = eval_full(ed, df_ref, genes)
                    rows.append(dict(run=run, preprocess=method, scorer=sc, lam=lam, edges=len(ed),
                                     AUROC_full=fa, AUPRC_full=fp, AUROC_pipe=pa, AUPRC_pipe=pp))
                print(f"run {run} {method:<7} {sc:<13} full AUROC (lam 0/1) "
                      f"{rows[-len(opt.lambdas)]['AUROC_full']:.3f} / "
                      f"{[r for r in rows[-len(opt.lambdas):] if r['lam']==1][0]['AUROC_full'] if 1 in opt.lambdas else float('nan'):.3f}"
                      f"  [{(time.time()-t0)/60:.1f} min]", flush=True)
            pd.DataFrame(rows).to_csv(os.path.join(opt.out, "div_raw.csv"), index=False)

    raw = pd.DataFrame(rows)
    noisy = raw[raw.run > 0]
    keys = ["preprocess", "scorer", "lam"]
    summ = noisy.groupby(keys)[["AUROC_full", "AUPRC_full", "AUROC_pipe", "AUPRC_pipe", "edges"]].agg(["mean", "std"]).round(4)
    summ.columns = ["_".join(c) for c in summ.columns]
    clean = raw[raw.run == 0].set_index(keys)[["AUROC_full", "AUROC_pipe"]].add_suffix("_clean").round(4)
    summ = summ.join(clean).reset_index()
    summ.to_csv(os.path.join(opt.out, "div_summary.csv"), index=False)
    best = summ.loc[summ.groupby(["preprocess", "scorer"]).AUROC_full_mean.idxmax()]
    best = best.sort_values("AUROC_full_mean", ascending=False)
    best.to_csv(os.path.join(opt.out, "div_best.csv"), index=False)
    pd.set_option("display.width", 220)
    print(best[["preprocess", "scorer", "lam", "AUROC_full_mean", "AUROC_full_std", "AUROC_full_clean",
                "AUROC_pipe_mean", "AUPRC_full_mean"]].to_string(index=False))


if __name__ == "__main__":
    main()
