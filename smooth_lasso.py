# -*- coding:utf-8 -*-
"""
Network 3: "smooth lasso". scPADGRN's dynamic network model run on
pseudotime instead of real time points.

Steps for each pseudotime branch:
  1. Sort cells by pseudotime and cut them into n_windows equal-count windows.
     These windows play the role of scPADGRN's time points t = 1..N.
  2. Inside each window, run the scPADGRN clustering (order_cluster): cut the
     ordered cells at the largest adjacent-cell gaps into r clusters and average.
     Y(t) is genes x r.
  3. Branches are stacked side by side, so Y(t) is genes x (r * n_branches).

Then fit the scPADGRN objective (Zheng, Huang, Zou 2020, Eq 6):

  min  1/2 sum_t || [Y(t+1) - Y(t)] - A(t) Y(t) ||_F^2
       + alpha * sum_t ||A(t)||_1                  (sparsity)
       + beta  * sum_t ||A(t+1) - A(t)||_1         (smoothness: neighboring
                                                    networks change little)

A(t)[i, j] is the effect of gene j on gene i in window t. The edge score for
j -> i is mean_t |A(t)[i, j]|. The diagonal (self decay) is fit but not
penalized and not scored.

pairing="scpadgrn": cluster k of window t is paired with cluster k of window
                    t+1, as in the paper.
pairing="consecutive": each cluster is paired with the next cluster along the
                    branch, and A(t) is chosen by the window the first cluster
                    sits in. This is a shorter time step.
"""
import numpy as np
import pandas as pd
import cvxpy as cp

from order_cluster import order_cluster


def _branch_windows(pt, n_windows):
    """Split an ordered pseudotime Series into n_windows equal-count windows."""
    return np.array_split(pt.index.to_numpy(), n_windows)


def build_stages(df_pse, df_exp, n_windows=5, r=30, standardize=True):
    """
    Returns
      Y: list over windows of arrays genes x (r * n_branches)
      seq: list over branches of (clusters x genes array, window id per cluster)
    """
    X = df_exp.astype(float)
    if standardize:
        X = (X - X.mean()) / X.std(ddof=0).replace(0, 1)

    Y = [[] for _ in range(n_windows)]
    seq = []
    for col in df_pse.columns:
        pt = df_pse[col].dropna().sort_values()
        rows, wid = [], []
        for t, cells in enumerate(_branch_windows(pt, n_windows)):
            sub_p = pd.DataFrame({col: pt.loc[cells]})
            _, clus = order_cluster(sub_p, X.loc[cells], k=min(r, len(cells)))
            M = clus.to_numpy()
            Y[t].append(M.T)                    # genes x r
            rows.append(M)
            wid += [t] * len(M)
        seq.append((np.vstack(rows), np.array(wid)))
    Y = [np.hstack(y) for y in Y]
    return Y, seq


def _pairs(Y, seq, pairing):
    """Build (X_t, dX_t) for each transition t: dX = A(t) X."""
    if pairing == "scpadgrn":
        return [(Y[t], Y[t + 1] - Y[t]) for t in range(len(Y) - 1)]
    T = len(Y)
    src = [[] for _ in range(T)]
    dif = [[] for _ in range(T)]
    for M, wid in seq:
        for c in range(len(M) - 1):
            src[wid[c]].append(M[c])
            dif[wid[c]].append(M[c + 1] - M[c])
    return [(np.array(s).T, np.array(d).T) for s, d in zip(src, dif) if len(s)]


def fit_dynamic_network(pairs, alpha=0.05, beta=0.05, solver="CLARABEL"):
    """Solve the scPADGRN objective. Returns list of A(t), genes x genes."""
    m = pairs[0][0].shape[0]
    n_tot = sum(Xt.shape[1] for Xt, _ in pairs)
    off = 1.0 - np.eye(m)
    A = [cp.Variable((m, m)) for _ in pairs]
    fit = sum(cp.sum_squares(dX - A[t] @ Xt) for t, (Xt, dX) in enumerate(pairs)) / (2 * n_tot)
    sparse = sum(cp.sum(cp.abs(cp.multiply(off, a))) for a in A)
    smooth = sum(cp.sum(cp.abs(cp.multiply(off, A[t + 1] - A[t]))) for t in range(len(A) - 1))
    prob = cp.Problem(cp.Minimize(fit + alpha * sparse + beta * smooth))
    prob.solve(solver=solver)
    return [a.value for a in A]


def edges_from_A(A_list, genes):
    """Edge j -> i scored by mean |A(t)[i, j]| over windows. Also returns the sign."""
    W = np.mean([np.abs(a) for a in A_list], axis=0)
    S = np.sign(np.mean(A_list, axis=0))
    rows = []
    for i, gi in enumerate(genes):          # target
        for j, gj in enumerate(genes):      # regulator
            if i != j:
                rows.append({"Gene1": gj, "Gene2": gi, "score": W[i, j], "sign": S[i, j]})
    return pd.DataFrame(rows).sort_values("score", ascending=False, ignore_index=True)


def smooth_lasso_network(df_pse, df_exp, n_windows=5, r=30, alpha=0.05, beta=0.05,
                         pairing="scpadgrn", standardize=True, return_A=False):
    Y, seq = build_stages(df_pse, df_exp, n_windows, r, standardize)
    A = fit_dynamic_network(_pairs(Y, seq, pairing), alpha, beta)
    edges = edges_from_A(A, list(df_exp.columns))
    return (edges, A) if return_A else edges


def network_change(A_list, tol=1e-4):
    """How much the network moves between neighboring windows.
    Returns mean L1 change per step and the Jaccard overlap of edge sets."""
    rows = []
    for t in range(len(A_list) - 1):
        a, b = A_list[t].copy(), A_list[t + 1].copy()
        np.fill_diagonal(a, 0)
        np.fill_diagonal(b, 0)
        ea, eb = np.abs(a) > tol, np.abs(b) > tol
        union = (ea | eb).sum()
        rows.append({"step": f"W{t+1}->W{t+2}",
                     "L1_change": np.abs(b - a).sum(),
                     "edges_t": int(ea.sum()), "edges_t1": int(eb.sum()),
                     "jaccard": (ea & eb).sum() / union if union else 1.0})
    return pd.DataFrame(rows)


def cv_select(df_pse, df_exp, n_windows=5, r=30, standardize=True,
              alphas=(0.001, 0.003, 0.01, 0.03), betas=(0, 0.003, 0.01, 0.03, 0.1),
              folds=5, seed=0):
    """
    Pick alpha and beta without looking at the reference network.
    Cluster columns are split into folds (the same columns in every window, so
    the scPADGRN pairing stays intact). Fit on the training columns, score the
    squared error of predicting Y(t+1) - Y(t) on the held-out columns.
    Returns (best_alpha, best_beta, table of CV errors).
    """
    Y, _ = build_stages(df_pse, df_exp, n_windows, r, standardize)
    n_col = Y[0].shape[1]
    fold = np.random.default_rng(seed).permutation(n_col) % folds
    rows = []
    for a in alphas:
        for b in betas:
            err = 0.0
            for f in range(folds):
                tr, te = fold != f, fold == f
                pr = [(Y[t][:, tr], (Y[t + 1] - Y[t])[:, tr]) for t in range(len(Y) - 1)]
                A = fit_dynamic_network(pr, a, b)
                err += sum(np.sum(((Y[t + 1] - Y[t])[:, te] - A[t] @ Y[t][:, te]) ** 2)
                           for t in range(len(Y) - 1))
            rows.append({"alpha": a, "beta": b, "cv_error": err})
    tab = pd.DataFrame(rows)
    best = tab.loc[tab.cv_error.idxmin()]
    return float(best.alpha), float(best.beta), tab
