# -*- coding:utf-8 -*-
"""
divergence_scores.py - 14 pairwise edge scores for the divergence comparison.

FUNCTIONS
    lag_pairs      regulator at step t, target at step t+1, inside each branch
    _prob          turn a profile into a probability vector (clip at 0, add tiny constant, sum to 1)
    kl_fwd ... ks  one function per score (see SCORERS)
    score_pairs    run one score over every ordered gene pair

DETAILS
Pairwise edge scores for the divergence comparison. Every scorer returns a
long table Gene1 (regulator), Gene2 (target), score, with HIGH score = more
likely edge.

Lag: the regulator at step t is compared with the target at step t+1, inside
each branch only (psedoScore.py's cal_kl2 etc. lag across the branch boundary
once; this version does not).

Two families:
  profile divergences  (KL forward/backward/symmetric, JS, Hellinger, total
      variation, Bhattacharyya): each gene's lagged time profile is turned into
      a probability vector over time (same as psedoScore.kl_divergence), then
      compared. Low divergence = similar shape = likely edge, so the sign is
      flipped.
  sample distances  (Wasserstein, energy, KS): compare the two genes' value
      distributions, as psedoScore.py does. Sign flipped.
  dependence scores (MI, distance correlation, |Pearson|, |Spearman|): high =
      edge, no flip.
"""
import numpy as np
import pandas as pd
from itertools import permutations
from scipy.stats import entropy, wasserstein_distance, energy_distance, ks_2samp, pearsonr, spearmanr
import dcor

from psedoScore import MI_Gao

EPS = 1e-10


def lag_pairs(x, y, count, lag=1):
    # count = rows per branch. Shift inside each branch, then join the branches.
    xs, ys, s = [], [], 0
    for c in count:
        xs.append(x[s:s + c - lag])
        ys.append(y[s + lag:s + c])
        s += c
    return np.concatenate(xs), np.concatenate(ys)


def _prob(v):
    v = np.clip(v, 0, None) + EPS
    return v / v.sum()


# Kullback-Leibler: forward KL(regulator || target), backward KL(target || regulator)
def kl_fwd(x, y):  return entropy(_prob(x), _prob(y))
def kl_bwd(x, y):  return entropy(_prob(y), _prob(x))
def kl_sym(x, y):  return 0.5 * (kl_fwd(x, y) + kl_bwd(x, y))


def js(x, y):
    p, q = _prob(x), _prob(y)
    m = 0.5 * (p + q)
    return 0.5 * entropy(p, m) + 0.5 * entropy(q, m)


# Hellinger, total variation and Bhattacharyya all compare the two probability
# vectors point by point. Hellinger and Bhattacharyya rank pairs almost the same.
def hellinger(x, y):
    p, q = _prob(x), _prob(y)
    return np.sqrt(0.5 * np.sum((np.sqrt(p) - np.sqrt(q)) ** 2))


def total_variation(x, y):
    return 0.5 * np.abs(_prob(x) - _prob(y)).sum()


def bhattacharyya(x, y):
    return -np.log(np.sum(np.sqrt(_prob(x) * _prob(y))))


def mi(x, y):              return max(MI_Gao(x.reshape(-1, 1), y.reshape(-1, 1)), 0)
def dist_corr(x, y):       return dcor.distance_correlation(x, y)
def abs_pearson(x, y):     return abs(pearsonr(x, y)[0]) if x.std() > 0 and y.std() > 0 else 0.0
def abs_spearman(x, y):    return abs(spearmanr(x, y)[0]) if x.std() > 0 and y.std() > 0 else 0.0
def wasserstein(x, y):     return wasserstein_distance(x, y)
def energy(x, y):          return energy_distance(x, y)
def ks(x, y):              return ks_2samp(x, y).statistic


# name: (function, flip sign?)
# flip = True for distances (low = similar = likely edge), so high always means likely edge
SCORERS = {
    "MI": (mi, False),
    "dcor": (dist_corr, False),
    "pearson": (abs_pearson, False),
    "spearman": (abs_spearman, False),
    "KL_fwd": (kl_fwd, True),
    "KL_bwd": (kl_bwd, True),
    "KL_sym": (kl_sym, True),
    "JS": (js, True),
    "hellinger": (hellinger, True),
    "TV": (total_variation, True),
    "bhattacharyya": (bhattacharyya, True),
    "wasserstein": (wasserstein, True),
    "energy": (energy, True),
    "KS": (ks, True),
}


def score_pairs(df, count, name, lag=1):
    f, flip = SCORERS[name]
    rows = []
    for g1, g2 in permutations(df.columns, 2):
        x, y = lag_pairs(df[g1].to_numpy(float), df[g2].to_numpy(float), count, lag)
        try:
            s = float(f(x, y))
        except Exception:
            s = np.nan
        rows.append((g1, g2, -s if flip else s))
    out = pd.DataFrame(rows, columns=["Gene1", "Gene2", "score"])
    return out[np.isfinite(out.score)]
