# -*- coding:utf-8 -*-
"""
Robustness sweep: scPADGRN clustering (order_cluster) vs original sliding windows (smooth).

For each run, Gaussian noise is added to the expression data (same setup as
run_slingshot_distance_example.py). Every method / size / scorer combination in
the same run sees the same noisy data, so cluster vs smooth comparisons are paired.

Outputs (in --out folder):
  sweep_raw.csv      one row per run x scorer x size x preprocess
  sweep_summary.csv  mean and SD of AUROC / AUPRC
  sweep_paired.csv   cluster minus smooth, paired t-test p-values, win counts
  sweep_auroc.pdf    AUROC vs size, one panel per scorer

Quick test (about 2 min):
  python sweep.py --quick
Full sweep:
  python sweep.py
"""
import os
os.environ.setdefault("TQDM_DISABLE", "1")   # silence pqdm progress bars

import argparse
import time
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel

from PreprocessData import smooth
from order_cluster import order_cluster
from psedoScore import cal_mi2, cal_kl2, cal_js2, cal_wasserstein2, cal_energy2, cal_cramer2
from mRMR import MRMR2, MRMR2_divergence
from Evaluate import concat_ref, cal_auc_aupr

DIVERGENCES = {
    "KL": cal_kl2,
    "JS": cal_js2,
    "wasserstein": cal_wasserstein2,
    "energy": cal_energy2,
    "cramer": cal_cramer2,
}
ALL_SCORERS = ["MI"] + list(DIVERGENCES)

# These four measure distance: a high score means the two genes behave DIFFERENTLY,
# the opposite of what an edge score should mean. Multiply by -1 so high = likely edge.
# MI and cramer (distance correlation) measure dependence and keep their sign.
DISTANCE_SCORERS = {"KL", "JS", "wasserstein", "energy"}


def add_noise(df_expr, rng, noise_level=0.05):
    """Gaussian noise scaled by each gene's range, negatives clipped to 0."""
    ranges = (df_expr.max() - df_expr.min()).values
    noise = rng.standard_normal(df_expr.shape) * ranges
    df_noisy = df_expr + noise * noise_level
    return df_noisy.clip(lower=0)


def preprocess(method, df_pse, df_exp, size):
    if method == "cluster":
        return order_cluster(df_pse, df_exp, cells_per_cluster=size)
    return smooth(df_pse, df_exp, 1, size)


def infer_edges(scorer, df_proc, count, n_jobs, lambda_val, flip=True):
    """Score gene pairs, then prune with mRMR. Returns Gene1, Gene2, score."""
    if scorer == "MI":
        df = cal_mi2(df_proc, count, n_jobs)
        df = df.loc[df.score > 0]
        df = MRMR2(df, n_jobs)
        df = df.loc[df.score > 0]
        return df[["Gene1", "Gene2", "score"]]

    df = DIVERGENCES[scorer](df_proc, n_jobs=n_jobs)
    if set(df.columns) == set(df_proc.columns):          # matrix output -> long form
        df = df.stack().reset_index()
        df.columns = ["Gene1", "Gene2", "score"]
    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    df = df[np.isfinite(df["score"])]
    if flip and scorer in DISTANCE_SCORERS:
        df["score"] = -df["score"]
    df = MRMR2_divergence(df, n_jobs=n_jobs, lambda_val=lambda_val)
    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    return df[np.isfinite(df["score"])][["Gene1", "Gene2", "score"]]


def evaluate(df_edges, ref_path):
    df_ref = pd.read_csv(ref_path, usecols=["Gene1", "Gene2"])
    df_eval = concat_ref(df_edges.copy(), df_ref)
    if df_eval.real_score.nunique() < 2:
        return np.nan, np.nan
    res = cal_auc_aupr(df_eval)
    return res["AUROC"], res["AUPRC"]


def summarize(df_raw, out_dir):
    summary = (df_raw.groupby(["scorer", "size", "preprocess"])[["AUROC", "AUPRC"]]
               .agg(["mean", "std"]).round(4))
    summary.columns = ["_".join(c) for c in summary.columns]
    summary = summary.reset_index()
    summary.to_csv(os.path.join(out_dir, "sweep_summary.csv"), index=False)

    rows = []
    for (scorer, size), g in df_raw.groupby(["scorer", "size"]):
        wide = g.pivot(index="run", columns="preprocess", values=["AUROC", "AUPRC"]).dropna()
        if wide.empty or "cluster" not in wide["AUROC"] or "smooth" not in wide["AUROC"]:
            continue
        row = {"scorer": scorer, "size": size, "n_runs": len(wide)}
        for metric in ["AUROC", "AUPRC"]:
            c, s = wide[metric]["cluster"], wide[metric]["smooth"]
            diff = c - s
            p = ttest_rel(c, s).pvalue if len(wide) > 1 and diff.std() > 0 else np.nan
            row[f"{metric}_cluster"] = round(c.mean(), 4)
            row[f"{metric}_smooth"] = round(s.mean(), 4)
            row[f"{metric}_diff"] = round(diff.mean(), 4)
            row[f"{metric}_p"] = round(p, 4) if pd.notna(p) else np.nan
            row[f"{metric}_cluster_wins"] = f"{int((diff > 0).sum())}/{len(wide)}"
        rows.append(row)
    paired = pd.DataFrame(rows)
    paired.to_csv(os.path.join(out_dir, "sweep_paired.csv"), index=False)
    return summary, paired


def plot(summary, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scorers = list(summary.scorer.unique())
    fig, axes = plt.subplots(1, len(scorers), figsize=(3.2 * len(scorers), 3.2), sharey=True, squeeze=False)
    for ax, scorer in zip(axes[0], scorers):
        g = summary[summary.scorer == scorer]
        for method, color in [("cluster", "tab:blue"), ("smooth", "tab:gray")]:
            m = g[g.preprocess == method].sort_values("size")
            ax.errorbar(m["size"], m["AUROC_mean"], yerr=m["AUROC_std"], marker="o",
                        capsize=3, color=color, label=method)
        ax.set_title(scorer)
        ax.set_xlabel("cells per window / cluster")
        ax.grid(alpha=0.3)
    axes[0][0].set_ylabel("AUROC (mean ± SD)")
    axes[0][0].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "sweep_auroc.pdf"))
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_file", default="input/ExpressionData.csv")
    p.add_argument("--time_file", default="input/PseudoTime.csv")
    p.add_argument("--ref_file", default="input/refNetwork.csv")
    p.add_argument("--runs", type=int, default=5)
    p.add_argument("--sizes", type=int, nargs="+", default=[3, 5, 10, 20])
    p.add_argument("--scorers", nargs="+", default=ALL_SCORERS, choices=ALL_SCORERS)
    p.add_argument("--noise", type=float, default=0.05)
    p.add_argument("--lambda_val", type=float, default=1.0, help="mRMR redundancy penalty for divergence scorers")
    p.add_argument("--n_jobs", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="sweep_results")
    p.add_argument("--quick", action="store_true", help="2 runs, size 5, MI and cramer only")
    p.add_argument("--no_flip", action="store_true",
                   help="keep the original sign on KL, JS, wasserstein, energy (reproduces the first sweep)")
    opt = p.parse_args()

    if opt.quick:
        opt.runs, opt.sizes, opt.scorers = 2, [5], ["MI", "cramer"]
        opt.out = opt.out + "_quick"

    os.makedirs(opt.out, exist_ok=True)
    raw_path = os.path.join(opt.out, "sweep_raw.csv")

    df_exp_orig = pd.read_csv(opt.data_file, index_col=0)
    df_pse = pd.read_csv(opt.time_file, index_col=0)

    total = opt.runs * len(opt.scorers) * len(opt.sizes) * 2
    done, t0, rows = 0, time.time(), []

    for run in range(1, opt.runs + 1):
        rng = np.random.default_rng(opt.seed + run)
        df_noisy = add_noise(df_exp_orig, rng, opt.noise)

        for size in opt.sizes:
            for method in ["cluster", "smooth"]:
                count, df_proc = preprocess(method, df_pse, df_noisy, size)
                for scorer in opt.scorers:
                    try:
                        edges = infer_edges(scorer, df_proc, count, opt.n_jobs, opt.lambda_val,
                                            flip=not opt.no_flip)
                        auroc, auprc = evaluate(edges, opt.ref_file)
                        n_edges = len(edges)
                    except Exception as e:
                        print(f"  !! {scorer} {method} size={size} run={run} failed: {e}")
                        auroc, auprc, n_edges = np.nan, np.nan, 0
                    rows.append({"run": run, "scorer": scorer, "size": size, "preprocess": method,
                                 "rows": len(df_proc), "edges": n_edges, "AUROC": auroc, "AUPRC": auprc})
                    pd.DataFrame(rows).to_csv(raw_path, index=False)   # save as we go
                    done += 1
                    elapsed = time.time() - t0
                    eta = elapsed / done * (total - done)
                    print(f"[{done}/{total}] run {run} | {scorer:<11} | {method:<7} size {size:<2} "
                          f"| AUROC {auroc:.3f} AUPRC {auprc:.3f} | ETA {eta/60:.1f} min")

    df_raw = pd.DataFrame(rows)
    summary, paired = summarize(df_raw, opt.out)
    try:
        plot(summary, opt.out)
    except Exception as e:
        print(f"Plot failed: {e}")

    pd.set_option("display.width", 200)
    print("\n=== Cluster vs smooth (paired across runs) ===")
    cols = ["scorer", "size", "AUROC_cluster", "AUROC_smooth", "AUROC_diff", "AUROC_p",
            "AUROC_cluster_wins", "AUPRC_diff", "AUPRC_p"]
    print(paired[[c for c in cols if c in paired]].to_string(index=False))
    print(f"\nSaved results to {opt.out}/")


if __name__ == "__main__":
    main()
