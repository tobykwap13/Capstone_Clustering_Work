# Capstone results: three networks, divergences, AUROC boosters

Data: BEELINE GSD synthetic set. 19 genes, 2000 cells, 2 Slingshot branches, 79 reference edges (76 after self loops drop out).
Runs: clean data plus 5 noisy runs (5% Gaussian noise, seeds 43 to 47, same as sweep.py). Every method sees the same data inside a run, so differences are paired.

## The three networks

| Network | How it is built |
|---|---|
| Original | sliding windows (k = 5) → MI → mRMR. Same as `main.py --preprocess smooth`. |
| Cluster | scPADGRN `order_cluster` (5 cells per cluster) → MI → mRMR. Same as `main.py --preprocess cluster`. |
| Smooth lasso | 5 equal-count pseudotime windows per branch, `order_cluster` into 30 clusters inside each window, then the scPADGRN objective: squared fit error + α·L1 sparsity + β·L1 change between neighboring windows. α and β come from 5-fold cross-validated prediction error, never from the reference network. Edge score = mean \|A(t)\| over windows. |

## Headline numbers (AUROC over all 342 gene pairs)

| Network | AUROC mean ± SD (noisy) | AUROC clean | AUPRC mean (noisy) |
|---|---|---|---|
| Original | 0.611 ± 0.017 | 0.621 | 0.293 |
| Cluster | 0.582 ± 0.027 | 0.566 | 0.270 |
| Smooth lasso | **0.639 ± 0.021** | **0.673** | **0.315** |
| Lasso, β = 0 (no smoothness) | 0.624 ± 0.021 | 0.625 | 0.307 |

Paired tests over the 5 noisy runs:

- Smooth lasso beats Original by +0.028 AUROC (p = 0.02, 5 of 5 runs).
- Smooth lasso beats Cluster by +0.057 (p = 0.01, 5 of 5).
- Cluster trails Original by 0.029 (p = 0.04, 1 of 5) when both use MI.
- Smoothness adds +0.015 over plain lasso (p = 0.20, 4 of 5). Cross-validation picked β > 0 in all 6 runs, so the data itself prefers smooth networks.

## Smoothness: networks change little between windows

| Model | Edge Jaccard, window t vs t+1 | L1 change per step |
|---|---|---|
| Smooth lasso (β by CV) | 0.88 | 4.8 |
| Lasso, β = 0 | 0.39 | 37.0 |

The β penalty keeps neighboring networks close, the scPADGRN continuity assumption, and it costs no AUROC.

## Network overlap (top 50 edges, Jaccard)

Original vs Cluster 0.20. Original vs Smooth lasso 0.11. Cluster vs Smooth lasso 0.12. The three methods find different edges, which explains why the ensemble below helps.

## Evaluation warning

`Evaluate.cal_auc_aupr` scores only the edges a method outputs. Each method outputs a different edge set, so those AUROCs compare different questions. On clean data this metric ranks Cluster (0.610) above Original (0.547). The fair metric, all 342 pairs with missing pairs tied at the bottom, flips the order (0.566 vs 0.621). The earlier compare_results.csv result (cluster 0.620 vs smooth 0.558) comes from this artifact. Report the all-pairs AUROC in the paper. `eval_full.py` implements it.

How many real edges mRMR deletes on clean data (76 real edges in total):

| | Original | Cluster |
|---|---|---|
| Edges kept | 178 | 164 |
| Real edges kept | 52 | 41 |
| Real edges deleted | 24 | 35 |

The old grader never counted those deleted real edges as misses. The function comes from the pipeline code (`Evaluate.py`, July 2025). It works for one method's own ranking and breaks for comparisons between methods.

mRMR on vs off is a separate question from the scoring fix. With mutual information, under fair scoring:

| | mRMR off (λ = 0) | mRMR on (λ = 1) |
|---|---|---|
| Original | 0.585 | 0.611 |
| Cluster | 0.579 | 0.582 |

mRMR still helps mutual information on balance, so the original pipeline keeps it.

## Divergences: cluster vs smooth

All 14 scorers, run on both preprocessings, at mRMR λ = 0, 0.5, 1, 1.5. Full table in `divergence_results/div_summary.csv`. Best λ per scorer:

| Rank | Preprocess | Scorer | λ | AUROC mean ± SD |
|---|---|---|---|---|
| 1 | cluster | backward KL | 0.5 | 0.655 ± 0.003 |
| 2 | cluster | distance correlation ("cramer" in sweep.py) | 1.0 | 0.654 ± 0.013 |
| 3 | smooth | \|Pearson\| | 1.5 | 0.649 ± 0.014 |
| 4 | cluster | total variation | 0 | 0.641 ± 0.005 |
| 5 | cluster | JS | 0 | 0.637 ± 0.004 |
| . | smooth | MI (the original pipeline) | 1.0 | 0.611 ± 0.017 |

Findings:

1. Clustering helps once you drop MI. At λ = 1, cluster beats smooth for energy (+0.057), distance correlation (+0.049), KS (+0.040) and backward KL (+0.036). At λ = 0, cluster wins for JS, TV, Hellinger, Bhattacharyya and symmetric KL, 5 of 5 runs each. MI is the one scorer where cluster loses.
2. Profile divergences (KL, JS, TV, Hellinger, Bhattacharyya) need λ = 0. mRMR at λ = 1 drops them to 0.50 to 0.54. Their redundancy term does not fit the mRMR formula.
3. Wasserstein, energy and KS rank last (0.54 to 0.62). They compare value distributions and ignore time order.

## Ways to raise AUROC

Rank-average ensembles (`ensemble_best.py`, `three_networks.py`):

| Network | AUROC mean ± SD | AUROC clean | AUPRC mean |
|---|---|---|---|
| Smooth lasso + cluster dcor + cluster backward KL | **0.702 ± 0.012** | **0.725** | **0.439** |
| Smooth lasso + cluster dcor | 0.690 ± 0.024 | 0.711 | 0.391 |
| Original + Smooth lasso | 0.668 ± 0.021 | 0.694 | 0.373 |
| Original (baseline) | 0.611 ± 0.017 | 0.621 | 0.293 |

Caveat: I chose dcor, backward KL and their λ by AUROC on this same reference network. Treat 0.70 as optimistic until a second dataset confirms the choice. The smooth lasso number (0.639) has no such leak.

## Code bugs and notes

- `run_slingshot_distance_example.py` imports `cal_pearson2`, `cal_symmetric_pearson2`, `cal_js_pearson2`, `cal_neyman2`. None exist in `psedoScore.py`, so the script fails on import.
- `cal_kl2`, `cal_js2` and the other psedoScore scorers lag across the branch boundary once. `divergence_scores.py` lags inside each branch.
- `cal_cramer` computes distance correlation, not the Cramér distance. Name it distance correlation in the paper.
- Rerunning `main.py` here gives 178 and 164 edges vs 169 and 167 in the committed rankedEdges files. Package versions likely differ. `three_networks.py` matches a fresh `main.py` run exactly.

## Files

- `smooth_lasso.py`: smooth lasso network, CV selection, window-to-window change.
- `eval_full.py`: all-pairs AUROC/AUPRC, top-k Jaccard.
- `divergence_scores.py`: 14 branch-aware scorers.
- `three_networks.py` → `three_network_results/`: three-way comparison, smoothness, overlap, ensembles, figure.
- `divergence_sweep.py` → `divergence_results/`: divergence × preprocess × λ sweep.
- `ensemble_best.py` → `three_network_results/ensemble_best.csv`.

Reproduce: `python three_networks.py`, `python divergence_sweep.py`, `python ensemble_best.py`. About 15 minutes total on 2 cores.
