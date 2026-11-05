import numpy as np
import pandas as pd


def order_cluster(df_p, df_e, k=None, cells_per_cluster=5, return_labels=False):
    """
    Python port of scPADGRN's orderCluster(), built as a drop-in
    replacement for PreprocessData.smooth().

    For each pseudotime branch:
      1. Sort cells by pseudotime.
      2. Compute the Euclidean distance between each pair of adjacent cells.
      3. Cut the trajectory at the k-1 largest gaps, giving k clusters
         of adjacent, similar cells (sizes vary).
      4. Average expression within each cluster.

    Input:
      df_p: pseudotime dataframe, one column per branch, NaN for cells
            outside the branch (same format as input/PseudoTime.csv)
      df_e: expression dataframe, cells as rows, genes as columns
      k: number of clusters per branch. If None, k = n_cells // cells_per_cluster
      cells_per_cluster: average cluster size when k is None. Default 5
            matches smooth()'s default window size.

    Output: (count, df) in the same format smooth() returns.
      count: number of clusters in each branch
      df: cluster-averaged expression, branches stacked in order
      labels (only if return_labels=True): one row per cell per branch with
            columns cell, branch, pseudotime, cluster (row number in df)
    """
    count = []
    out = []
    labels = []
    offset = 0
    for col in df_p.columns:
        pt = df_p[col].dropna().sort_values()
        M = df_e.loc[pt.index].to_numpy(dtype=float)
        n = M.shape[0]

        kk = k if k is not None else max(2, n // cells_per_cluster)
        kk = min(kk, n)

        # distance between cell i and cell i+1 along pseudotime
        gaps = np.linalg.norm(np.diff(M, axis=0), axis=1)

        # break at the kk-1 largest gaps (R version links the n-k smallest)
        if kk > 1:
            breaks = np.sort(np.argsort(gaps, kind="stable")[-(kk - 1):]) + 1
        else:
            breaks = np.array([], dtype=int)

        groups = np.split(M, breaks)
        clustered = np.vstack([g.mean(axis=0) for g in groups])

        out.append(pd.DataFrame(clustered, columns=df_e.columns))
        count.append(len(groups))

        cluster_id = np.repeat(np.arange(len(groups)), [len(g) for g in groups]) + offset + 1
        labels.append(pd.DataFrame({"cell": pt.index, "branch": col,
                                    "pseudotime": pt.values, "cluster": cluster_id}))
        offset += len(groups)

    df_all = pd.concat(out, ignore_index=True)
    df_all.index = range(1, len(df_all) + 1)
    if return_labels:
        return count, df_all, pd.concat(labels, ignore_index=True)
    return count, df_all
