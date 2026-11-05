# -*- coding:utf-8 -*-
"""
Step 1: verify order_cluster.py does what scPADGRN's orderCluster.R does.

Run from the Pseudo-network folder:
    python verify_clustering.py

Each check prints PASS or FAIL. A final summary shows the totals.
"""
import numpy as np
import pandas as pd
from scipy.stats import rankdata

from order_cluster import order_cluster

results = []


def check(name, ok, detail=""):
    results.append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))


# ---------------------------------------------------------------------------
# Line-by-line Python copy of R/orderCluster.R (1-based R indexing kept in comments).
# fixed=False keeps the R code exactly as written, including its off-by-one bug.
# fixed=True changes the one buggy line: N[, j] <- M[, i]  ->  N[, j] <- M[, i+1]
# ---------------------------------------------------------------------------
def r_order_cluster(M, k, fixed=True):
    """M: genes x ordered cells (R layout). Returns genes x k."""
    n = M.shape[1]
    N = np.zeros((M.shape[0], k))
    # distance <- diag(as.matrix(dist(t(M)))[2:n, 1:(n-1)])  -> dist(cell i+1, cell i)
    distance = np.linalg.norm(M[:, 1:] - M[:, :-1], axis=0)
    link = rankdata(distance) <= n - k          # rank(distance) <= n - k
    j = 0
    N[:, 0] = M[:, 0]
    for i in range(n - 1):                      # for (i in 1:(n-1))
        if link[i]:
            N[:, j] += M[:, i + 1]
        else:
            j += 1
            N[:, j] = M[:, i + 1] if fixed else M[:, i]
    shre = np.where(~link)[0] + 1               # which(!link)
    shre = np.concatenate([[0], shre, [n]])
    clu_number = np.diff(shre)
    return N / clu_number


def as_inputs(M_cells_by_genes, pseudotime=None):
    n = M_cells_by_genes.shape[0]
    cells = [f"c{i}" for i in range(n)]
    pt = np.arange(n, dtype=float) if pseudotime is None else pseudotime
    df_e = pd.DataFrame(M_cells_by_genes, index=cells)
    df_p = pd.DataFrame({"t": pt}, index=cells)
    return df_p, df_e


print("\n=== A. Python port matches the R algorithm ===")
rng = np.random.default_rng(0)
max_err, n_cases = 0.0, 0
for trial in range(200):
    n = int(rng.integers(10, 300))
    g = int(rng.integers(2, 25))
    k = int(rng.integers(2, n))
    M = rng.random((n, g))
    df_p, df_e = as_inputs(M)
    _, py = order_cluster(df_p, df_e, k=k)
    r = r_order_cluster(M.T, k, fixed=True).T
    max_err = max(max_err, np.abs(py.to_numpy() - r).max())
    n_cases += 1
check("matches fixed R orderCluster on 200 random datasets", max_err < 1e-10,
      f"max abs difference = {max_err:.2e}")

print("\n=== B. The R original has an off-by-one bug ===")
M = np.array([[0.0], [0.1], [10.0], [10.1], [20.0], [20.1]])   # 3 obvious pairs
r_buggy = r_order_cluster(M.T, 3, fixed=False).T.ravel()
r_fixed = r_order_cluster(M.T, 3, fixed=True).T.ravel()
print(f"   true cluster means : [0.05, 10.05, 20.05]")
print(f"   R as written       : {np.round(r_buggy, 3).tolist()}")
print(f"   R with fix / Python: {np.round(r_fixed, 3).tolist()}")
check("R original gives wrong means (bug confirmed)", not np.allclose(r_buggy, [0.05, 10.05, 20.05]))
check("fixed version gives correct means", np.allclose(r_fixed, [0.05, 10.05, 20.05]))

print("\n=== C. Known-answer toy example ===")
groups_true = [np.full((4, 2), 0.0), np.full((6, 2), 5.0), np.full((3, 2), 9.0)]
M = np.vstack(groups_true) + rng.normal(0, 0.01, (13, 2))
df_p, df_e = as_inputs(M)
count, df, lab = order_cluster(df_p, df_e, k=3, return_labels=True)
sizes = lab.groupby("cluster").size().tolist()
check("recovers 3 planted groups with sizes 4, 6, 3", sizes == [4, 6, 3], f"got {sizes}")

print("\n=== D. Real data: input/ExpressionData.csv + input/PseudoTime.csv ===")
df_e = pd.read_csv("input/ExpressionData.csv", index_col=0)
df_p = pd.read_csv("input/PseudoTime.csv", index_col=0)
count, df, lab = order_cluster(df_p, df_e, cells_per_cluster=5, return_labels=True)

n_branch = df_p.notna().sum().tolist()
expected_k = [n // 5 for n in n_branch]
check("cluster count per branch = n_cells // 5", count == expected_k, f"{count} from {n_branch} cells")
check("output rows = sum of clusters", len(df) == sum(count), f"{len(df)} rows")
check("no missing values in output", int(df.isna().sum().sum()) == 0)

used = lab.groupby("branch").size().tolist()
check("every cell in each branch used exactly once", used == n_branch)

contiguous = all(
    (g.sort_values("pseudotime").cluster.diff().dropna().isin([0, 1])).all()
    for _, g in lab.groupby("branch"))
check("clusters are contiguous along pseudotime (no cell skips ahead)", contiguous)

# Pseudotime has many ties in this dataset, so neighboring clusters inside a tie
# share the same mean. The check is "never goes backwards", not "always strictly up".
mean_pt = lab.groupby("cluster").pseudotime.mean()
n_ties = {b: int(df_p[b].dropna().duplicated().sum()) for b in df_p.columns}
ordered = all(np.all(np.diff(mean_pt[lab[lab.branch == b].cluster.unique()].values) >= 0)
              for b in df_p.columns)
check("cluster mean pseudotime never decreases within a branch", ordered,
      f"tied pseudotime values per branch: {n_ties}")

recomputed = lab.merge(df_e, left_on="cell", right_index=True).groupby("cluster")[df_e.columns].mean()
err = np.abs(recomputed.to_numpy() - df.loc[recomputed.index].to_numpy()).max()
check("each output row = mean of its cells", err < 1e-10, f"max abs difference = {err:.2e}")

# breaks land on the k-1 largest gaps
ok_breaks = True
for b, kk in zip(df_p.columns, count):
    g = lab[lab.branch == b].sort_values("pseudotime")
    X = df_e.loc[g.cell].to_numpy()
    gaps = np.linalg.norm(np.diff(X, axis=0), axis=1)
    is_break = np.diff(g.cluster.to_numpy()) == 1
    ok_breaks &= gaps[is_break].min() >= gaps[~is_break].max()
check("every break gap is larger than every within-cluster gap", bool(ok_breaks))

print("\n=== E. What the clusters look like (report, no pass/fail) ===")
for cpc in [3, 5, 10, 20]:
    _, _, lab_c = order_cluster(df_p, df_e, cells_per_cluster=cpc, return_labels=True)
    s = lab_c.groupby("cluster").size()
    print(f"   cells_per_cluster={cpc:<2}: {len(s)} clusters | median size {s.median():.0f} | "
          f"singletons {(s == 1).mean() * 100:.0f}% | largest {s.max()} cells")

print("\n   Clusters with more than 20 cells at cells_per_cluster=5:")
s = lab.groupby(["branch", "cluster"]).agg(size=("cell", "size"),
                                           pt_min=("pseudotime", "min"),
                                           pt_max=("pseudotime", "max"))
for b in df_p.columns:
    big = s.loc[b][s.loc[b]["size"] > 20]
    print(f"   {b}: {int(big['size'].sum())} of {int(s.loc[b]['size'].sum())} cells sit in "
          f"{len(big)} big clusters at pseudotime "
          + ", ".join(f"{r.pt_min:.2f}-{r.pt_max:.2f}" for r in big.itertuples()))

print(f"\n=== {sum(results)}/{len(results)} checks passed ===")
