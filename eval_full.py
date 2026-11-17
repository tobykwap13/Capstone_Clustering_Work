# -*- coding:utf-8 -*-
"""
eval_full.py - grade a ranked edge list against refNetwork.csv, two ways.

FUNCTIONS
    eval_pipeline(df_edges, df_ref)      old scoring, output edges only
    eval_full(df_edges, df_ref, genes)   fair scoring, all gene pairs (use this)
    jaccard_top(e1, e2, k)               overlap of two networks' top-k edges

DETAILS

pipeline: Evaluate.cal_auc_aupr on the edges the method outputs. This is what
          compare.py and sweep.py report. Methods that output different edge
          sets are scored on different pair sets.
full:     every ordered gene pair (19 x 18 = 342) is scored. Pairs a method
          does not output tie at the bottom. Same pair set for every method,
          so this is the fair comparison (BEELINE style).
"""
import numpy as np
import pandas as pd
from itertools import permutations
from sklearn import metrics

from Evaluate import concat_ref, cal_auc_aupr


def eval_pipeline(df_edges, df_ref):
    # Same as Evaluate.concat_ref + cal_auc_aupr, with non-finite scores removed
    d = concat_ref(df_edges[["Gene1", "Gene2", "score"]].copy(), df_ref[["Gene1", "Gene2"]].copy())
    d = d[np.isfinite(d.score.astype(float))]
    if d.real_score.nunique() < 2:
        return np.nan, np.nan
    r = cal_auc_aupr(d)
    return r["AUROC"], r["AUPRC"]


def eval_full(df_edges, df_ref, genes):
    # Every ordered gene pair, no self pairs: 19 genes -> 342 pairs
    allp = pd.DataFrame(list(permutations(genes, 2)), columns=["Gene1", "Gene2"])
    e = df_edges[["Gene1", "Gene2", "score"]].copy()
    e["score"] = pd.to_numeric(e["score"], errors="coerce")
    e = e[np.isfinite(e.score)].drop_duplicates(["Gene1", "Gene2"])
    d = allp.merge(e, how="left", on=["Gene1", "Gene2"])
    # Pairs the method did not output get one shared score below all others,
    # so they rank last and tie with each other
    floor = (d.score.min() - 1) if d.score.notna().any() else 0
    d["score"] = d.score.fillna(floor)
    # Label: 1 if the pair sits in the reference network (self loops dropped), else 0
    ref = df_ref[["Gene1", "Gene2"]].drop_duplicates()
    ref = ref[ref.Gene1 != ref.Gene2].assign(y=1)
    d = d.merge(ref, how="left", on=["Gene1", "Gene2"]).fillna({"y": 0})
    fpr, tpr, _ = metrics.roc_curve(d.y, d.score)
    prec, rec, _ = metrics.precision_recall_curve(d.y, d.score)
    return metrics.auc(fpr, tpr), metrics.auc(rec, prec)


def jaccard_top(e1, e2, k):
    """Overlap of the top-k edges of two networks."""
    a = set(map(tuple, e1.nlargest(k, "score")[["Gene1", "Gene2"]].values))
    b = set(map(tuple, e2.nlargest(k, "score")[["Gene1", "Gene2"]].values))
    return len(a & b) / len(a | b)
