# -*- coding:utf-8 -*-
"""
Compare the three networks on the same data.

  original      sliding windows (PreprocessData.smooth)   -> MI -> mRMR   (main.py --preprocess smooth)
  cluster       scPADGRN order_cluster                    -> MI -> mRMR   (main.py --preprocess cluster)
  smooth_lasso  pseudotime windows + order_cluster inside each window
                -> scPADGRN dynamic network (L1 sparsity + L1 smoothness between windows)

Run 0 is the clean data. Runs 1..R add Gaussian noise (same model and seeds as
sweep.py), so the three networks are compared pairwise inside each run.

Smooth lasso alpha and beta are picked per run by cross-validated prediction
error (smooth_lasso.cv_select), never by AUROC, so the reference network does
not leak into the fit. beta = 0 (plain lasso, same alpha) is also fit to show
what the smoothness penalty does.

Two AUROC versions are reported (see eval_full.py):
  AUROC_full  all 342 gene pairs, fair across methods (main number)
  AUROC_pipe  only the edges each method outputs, what compare.py reports

Outputs in --out:
  three_raw.csv, three_summary.csv, three_paired.csv, three_overlap.csv,
  smoothness.csv, ensemble.csv, three_networks.pdf / .png

  python three_networks.py
"""
import os
os.environ.setdefault("TQDM_DISABLE", "1")
import argparse
import warnings
import numpy as np
import pandas as pd
from itertools import combinations
from scipy.stats import ttest_rel

from PreprocessData import smooth
from order_cluster import order_cluster
from psedoScore import cal_mi2
from mRMR import MRMR2
from smooth_lasso import build_stages, _pairs, fit_dynamic_network, edges_from_A, cv_select, network_change
from eval_full import eval_pipeline, eval_full, jaccard_top
from sweep import add_noise

warnings.filterwarnings("ignore")
NETS = ["original", "cluster", "smooth_lasso"]
COLORS = {"original": "#2a78d6", "cluster": "#eb6834", "smooth_lasso": "#1baf7a"}


def mi_mrmr(df_proc, count, n_jobs):
    df = cal_mi2(df_proc, count, n_jobs)
    df = df.loc[df.score > 0].sort_values("score", ascending=False)
    df = MRMR2(df, n_jobs)
    df["score"] = pd.to_numeric(df["score"])
    return df.loc[df.score > 0][["Gene1", "Gene2", "score"]]


def rank_ensemble(nets, genes):
    """Average the rank of every gene pair across networks (missing pairs rank last)."""
    from itertools import permutations
    allp = pd.DataFrame(list(permutations(genes, 2)), columns=["Gene1", "Gene2"])
    ranks = []
    for e in nets:
        d = allp.merge(e[["Gene1", "Gene2", "score"]].drop_duplicates(["Gene1", "Gene2"]),
                       how="left", on=["Gene1", "Gene2"])
        ranks.append(d.score.rank(method="average", na_option="top", ascending=True).to_numpy())
    allp["score"] = np.mean(ranks, axis=0)
    return allp


def plot(raw, smooth_tab, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ink, ink2, grid = "#0b0b0b", "#52514e", "#e4e3df"
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": ink2, "axes.labelcolor": ink,
                         "xtick.color": ink2, "ytick.color": ink2})
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    noisy = raw[raw.run > 0]
    for ax, metric, title in [(axes[0], "AUROC_full", "AUROC, all 342 pairs"),
                              (axes[1], "AUROC_pipe", "AUROC, output edges only")]:
        for i, n in enumerate(NETS):
            v = noisy[noisy.net == n][metric].to_numpy()
            c = raw[(raw.run == 0) & (raw.net == n)][metric].iloc[0]
            ax.bar(i, v.mean(), width=0.6, color=COLORS[n], edgecolor="white", linewidth=2)
            ax.errorbar(i, v.mean(), yerr=v.std(ddof=1), color=ink, capsize=4, lw=1.2)
            ax.scatter(np.full(len(v), i) + np.linspace(-0.12, 0.12, len(v)), v, s=14,
                       color="white", edgecolor=ink, lw=0.8, zorder=3)
            ax.scatter(i + 0.36, c, marker="D", s=30, color=ink, zorder=3)
            ax.text(i, 0.415, f"{v.mean():.3f}", ha="center", color="white", fontsize=10, fontweight="bold")
        ax.axhline(0.5, color=ink2, lw=1, ls=(0, (4, 3)))
        ax.set_xticks(range(3), ["Original", "Cluster", "Smooth lasso"])
        ax.set_ylim(0.4, 0.8)
        ax.set_title(title, color=ink, fontsize=11)
        ax.grid(axis="y", color=grid, lw=0.8)
        ax.set_axisbelow(True)
        for s in ["top", "right"]:
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel("AUROC (bars: mean ± SD, 5 noisy runs)")
    ax = axes[2]
    st = smooth_tab[smooth_tab.run > 0].groupby(["model", "step"]).jaccard.mean().reset_index()
    for model, color, label in [("lasso_beta0", "#eda100", "β = 0 (no smoothness)"),
                                ("smooth_lasso", "#1baf7a", "β by CV (smooth lasso)")]:
        m = st[st.model == model]
        ax.plot(range(len(m)), m.jaccard, marker="o", ms=7, lw=2, color=color, label=label)
    ax.set_xticks(range(len(m)), list(m.step), rotation=0, fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Edge-set Jaccard, window t vs t+1")
    ax.set_title("Network change between windows", color=ink, fontsize=11)
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    ax.grid(axis="y", color=grid, lw=0.8)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    fig.text(0.01, 0.01, "◆ = clean data.  Dashed line = random (0.5).", fontsize=8, color=ink2)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(os.path.join(out, "three_networks.pdf"))
    fig.savefig(os.path.join(out, "three_networks.png"), dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--size", type=int, default=5, help="window size (original) / cells per cluster (cluster)")
    ap.add_argument("--n_windows", type=int, default=5, help="smooth lasso pseudotime windows (scPADGRN time points)")
    ap.add_argument("--r", type=int, default=30, help="smooth lasso clusters per window")
    ap.add_argument("--noise", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n_jobs", type=int, default=2)
    ap.add_argument("--topk", type=int, default=50)
    ap.add_argument("--out", default="three_network_results")
    opt = ap.parse_args()
    os.makedirs(opt.out, exist_ok=True)

    df_exp = pd.read_csv("input/ExpressionData.csv", index_col=0)
    df_pse = pd.read_csv("input/PseudoTime.csv", index_col=0)
    df_ref = pd.read_csv("input/refNetwork.csv")
    genes = list(df_exp.columns)

    raw, overlap, smooth_rows, ens_rows = [], [], [], []
    for run in range(0, opt.runs + 1):
        data = df_exp if run == 0 else add_noise(df_exp, np.random.default_rng(opt.seed + run), opt.noise)
        nets, info = {}, {}

        c, d = smooth(df_pse, data, 1, opt.size)
        nets["original"] = mi_mrmr(d, c, opt.n_jobs)
        c, d = order_cluster(df_pse, data, cells_per_cluster=opt.size)
        nets["cluster"] = mi_mrmr(d, c, opt.n_jobs)

        a, b, _ = cv_select(df_pse, data, opt.n_windows, opt.r, standardize=False)
        Y, seq = build_stages(df_pse, data, opt.n_windows, opt.r, standardize=False)
        pr = _pairs(Y, seq, "scpadgrn")
        A = fit_dynamic_network(pr, a, b)
        A0 = fit_dynamic_network(pr, a, 0.0)
        nets["smooth_lasso"] = edges_from_A(A, genes)
        lasso0 = edges_from_A(A0, genes)
        info["smooth_lasso"] = dict(alpha=a, beta=b)

        for n in NETS:
            e = nets[n]
            fa, fp = eval_full(e, df_ref, genes)
            pa, pp = eval_pipeline(e[e.score > 1e-8] if n == "smooth_lasso" else e, df_ref)
            raw.append(dict(run=run, net=n, edges=int((e.score > 1e-8).sum()) if n == "smooth_lasso" else len(e),
                            AUROC_full=fa, AUPRC_full=fp, AUROC_pipe=pa, AUPRC_pipe=pp, **info.get(n, {})))
        fa0, fp0 = eval_full(lasso0, df_ref, genes)
        raw.append(dict(run=run, net="lasso_beta0", edges=int((lasso0.score > 1e-8).sum()),
                        AUROC_full=fa0, AUPRC_full=fp0, alpha=a, beta=0.0))

        for n1, n2 in combinations(NETS, 2):
            overlap.append(dict(run=run, pair=f"{n1} vs {n2}", jaccard_topk=jaccard_top(nets[n1], nets[n2], opt.topk)))
        for model, AA in [("smooth_lasso", A), ("lasso_beta0", A0)]:
            ch = network_change(AA)
            ch.insert(0, "model", model)
            ch.insert(0, "run", run)
            smooth_rows.append(ch)

        for combo in [("original", "cluster"), ("original", "smooth_lasso"), ("cluster", "smooth_lasso"), tuple(NETS)]:
            fa, fp = eval_full(rank_ensemble([nets[k] for k in combo], genes), df_ref, genes)
            ens_rows.append(dict(run=run, ensemble=" + ".join(combo), AUROC_full=fa, AUPRC_full=fp))

        r = pd.DataFrame(raw)
        print(f"run {run}: " + "  ".join(f"{n} {r[(r.run==run)&(r.net==n)].AUROC_full.iloc[0]:.3f}"
                                         for n in NETS + ["lasso_beta0"]) + f"   (alpha {a}, beta {b})", flush=True)

    raw = pd.DataFrame(raw)
    raw.to_csv(os.path.join(opt.out, "three_raw.csv"), index=False)
    noisy = raw[raw.run > 0]
    summ = noisy.groupby("net")[["AUROC_full", "AUPRC_full", "AUROC_pipe", "AUPRC_pipe", "edges"]].agg(["mean", "std"]).round(4)
    summ.columns = ["_".join(x) for x in summ.columns]
    summ = summ.join(raw[raw.run == 0].set_index("net")[["AUROC_full", "AUROC_pipe"]].add_suffix("_clean").round(4))
    summ = summ.reindex(NETS + ["lasso_beta0"]).reset_index()
    summ.to_csv(os.path.join(opt.out, "three_summary.csv"), index=False)

    paired = []
    for n1, n2 in [("cluster", "original"), ("smooth_lasso", "original"), ("smooth_lasso", "cluster"),
                   ("smooth_lasso", "lasso_beta0")]:
        for m in ["AUROC_full", "AUPRC_full", "AUROC_pipe"]:
            x = noisy[noisy.net == n1].set_index("run")[m]
            y = noisy[noisy.net == n2].set_index("run")[m]
            if x.isna().all() or y.isna().all():
                continue
            diff = x - y
            paired.append(dict(comparison=f"{n1} - {n2}", metric=m, mean_diff=round(diff.mean(), 4),
                               p_paired_t=round(ttest_rel(x, y).pvalue, 4), wins=f"{int((diff > 0).sum())}/{len(diff)}"))
    pd.DataFrame(paired).to_csv(os.path.join(opt.out, "three_paired.csv"), index=False)

    ov = pd.DataFrame(overlap)
    ov.to_csv(os.path.join(opt.out, "three_overlap.csv"), index=False)
    sm = pd.concat(smooth_rows, ignore_index=True)
    sm.to_csv(os.path.join(opt.out, "smoothness.csv"), index=False)
    ens = pd.DataFrame(ens_rows)
    ens.to_csv(os.path.join(opt.out, "ensemble.csv"), index=False)

    pd.set_option("display.width", 220)
    print("\n=== Summary (noisy runs) ===")
    print(summ.to_string(index=False))
    print("\n=== Paired differences ===")
    print(pd.DataFrame(paired).to_string(index=False))
    print("\n=== Top-%d edge overlap (Jaccard) ===" % opt.topk)
    print(ov[ov.run > 0].groupby("pair").jaccard_topk.agg(["mean", "std"]).round(3))
    print("\n=== Network change between windows (noisy runs) ===")
    print(sm[sm.run > 0].groupby("model")[["L1_change", "jaccard", "edges_t"]].mean().round(3))
    print("\n=== Rank ensembles (noisy runs) ===")
    print(ens[ens.run > 0].groupby("ensemble")[["AUROC_full", "AUPRC_full"]].agg(["mean", "std"]).round(4))
    plot(raw, sm, opt.out)
    print(f"\nSaved to {opt.out}/")


if __name__ == "__main__":
    main()
