# -*- coding:utf-8 -*-
"""
main.py - run the original Pseudo-network pipeline once and save the edge list.

SUMMARY
    Builds one gene regulatory network from scRNA-seq data ordered by pseudotime.
    Three steps:
      1. Preprocess: turn single cells into smoother "samples" along pseudotime.
           --preprocess smooth   sliding-window averages (the original pipeline)
           --preprocess cluster  scPADGRN clustering (order_cluster.py)
      2. Score every gene pair with time-lagged mutual information (psedoScore.cal_mi2).
      3. Prune redundant regulators with mRMR (mRMR.MRMR2).
    Output: rankedEdges_<preprocess>.csv with columns Gene1 (regulator),
    Gene2 (target), score. Higher score = more likely edge.

HOW TO RUN
    python main.py --data_file input/ExpressionData.csv \
                   --time_file input/PseudoTime.csv --preprocess smooth

NOTE
    This script does not score the network. Use compare.py (old scoring) or
    eval_full.py (fair all-pairs scoring) for AUROC.
"""
import pandas as pd
import argparse
from PreprocessData import *
from psedoScore import *          # was: from EstimateMI import *  (EstimateMI.py is missing from the repo)
from mRMR import *
from Evaluate import *
from order_cluster import order_cluster   # scPADGRN clustering, ported from R/orderCluster.R


def main():
    # ---- Command-line options ----
    parser = argparse.ArgumentParser()
    parser.add_argument('--window_size', type=int, default=5, help='The size of sliding window to preprocess expressioon data. Also the average cluster size when --preprocess cluster.')
    parser.add_argument('--slide', type=int, default=1, help='Sliding length when preprocessing expression data.')
    parser.add_argument('--n_jobs', type=int, default=1, help='The num of available CPU to run this program.')
    parser.add_argument('--data_file', type=str, help='The input scRNA-seq gene expression file.')
    parser.add_argument('--time_file', type=str, help='The input time file.')
    parser.add_argument('--tf_file', type=str, default='', help='TF list.')
    parser.add_argument('--save_dir', type=str, default='./')
    parser.add_argument('--preprocess', type=str, default='cluster', choices=['cluster', 'smooth'],
                        help='cluster = scPADGRN orderCluster, smooth = original sliding windows.')

    opt = parser.parse_args()

    k = opt.window_size          # window size (smooth) or cells per cluster (cluster)
    slide = opt.slide            # step between windows (smooth only)
    n_jobs = opt.n_jobs          # CPU cores for the MI and mRMR steps
    save_dir = opt.save_dir

    # --------------- STEP1 preprocess the dataset ---------------
    data_file = opt.data_file
    time_file = opt.time_file
    tf_file = opt.tf_file

    # Expression: cells as rows, genes as columns.
    # Pseudotime: cells as rows, one column per branch (NaN = cell not in branch).
    df_exp = pd.read_csv(data_file, header=0, index_col=0)
    df_pse = pd.read_csv(time_file, header=0, index_col=0)

    # Optional list of transcription factors. If given, only these genes act as regulators.
    if tf_file != '':
        TF_set = set(pd.read_csv(tf_file, header=None, index_col=None).values.reshape(-1))
    else:
        TF_set = []

    # Both options return (count, df):
    #   count = number of rows per branch, df = processed expression, branches stacked
    if opt.preprocess == 'cluster':
        count, df_exp = order_cluster(df_pse, df_exp, cells_per_cluster=k)
    else:
        count, df_exp = smooth(df_pse, df_exp, slide, k)

    # --------------- STEP2 estimate the MI ---------------
    # Score every regulator -> target pair with lag-1 mutual information.
    # Keep only positive scores and sort strongest first.
    df_mi = cal_mi2(df_exp, count, n_jobs, TF_set)
    df_mi = df_mi.loc[df_mi.score>0]
    df_mi.sort_values(by='score', ascending=False, inplace=True)

    # --------------- STEP3 filter out redundant edges using mRMR ---------------
    # mRMR lowers the score of regulators carrying the same information as ones
    # already picked for a target. Pairs whose score drops to 0 or below get removed.
    df_mrmr = MRMR2(df_mi, n_jobs)
    df_mrmr = df_mrmr.sort_values(by='score', ascending=False)
    df_mrmr = df_mrmr.loc[df_mrmr.score>0]
    df_mrmr.to_csv(save_dir + "/rankedEdges_" + opt.preprocess + ".csv", header=True, index=False)



if __name__ == '__main__':
    main()
