import os

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.stats import spearmanr

from load_measures import DATASETS, DEFAULT_MEASURE_PAIR, KEY_COLUMNS, load_results


RANKING_COLUMNS = [
    "average_rank",
    "dpfr_rank",
    "topsis_dpfr_rank",
    "promethee_rank",
]
RANKING_LABELS = {
    "average_rank": "Average",
    "dpfr_rank": "DPFR",
    "topsis_dpfr_rank": "TOPSIS-DPFR",
    "promethee_rank": "PROMETHEE",
}
OUTPUT_DIR = "corr_analysis/ranking_plots"


def load_ranking_rows(pair: str) -> pd.DataFrame:
    columns = [*KEY_COLUMNS, "measure_pair", *RANKING_COLUMNS]
    rows = load_results(columns)
    rows = rows[rows["measure_pair"] == pair].copy()
    return rows


def compute_correlation(df: pd.DataFrame) -> pd.DataFrame:
    n = len(RANKING_COLUMNS)
    rho_matrix = np.ones((n, n))
    values = df[RANKING_COLUMNS].to_numpy(dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            rho, _ = spearmanr(values[:, i], values[:, j], nan_policy="omit")
            rho_matrix[i, j] = rho
            rho_matrix[j, i] = rho

    labels = [RANKING_LABELS[column] for column in RANKING_COLUMNS]
    return pd.DataFrame(rho_matrix, index=labels, columns=labels)


def plot_heatmap(corr_matrix: pd.DataFrame, output_dir: str, dataset: str):
    annot = np.empty_like(corr_matrix, dtype=object)
    for i in range(len(corr_matrix)):
        for j in range(len(corr_matrix)):
            annot[i, j] = f"{corr_matrix.iloc[i, j]:.2f}"

    fig, ax = plt.subplots(figsize=(5, 5))
    sns.heatmap(
        corr_matrix,
        annot=annot,
        fmt="",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        linewidths=0.5,
        square=True,
        ax=ax,
        annot_kws={"fontsize": 10},
    )
    ax.set_title(f"{dataset}: Ranking Spearman Correlation", fontsize=10)
    fig.tight_layout()

    safe_name = dataset.replace("/", "-").replace(" ", "_")
    fig.savefig(
        os.path.join(output_dir, f"ranking_corr_heatmap_{safe_name}.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def build_heatmaps(pair: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    ranking_rows = load_ranking_rows(pair)
    print(f"Loaded {len(ranking_rows)} alternatives for {pair}")

    for dataset in DATASETS:
        data = ranking_rows[ranking_rows["dataset"] == dataset]
        corr = compute_correlation(data)
        print(f"\n{dataset} ranking correlations (n={len(data)}):")
        print(corr.to_string(float_format="{:.3f}".format))
        plot_heatmap(corr, output_dir, dataset)

    print(f"\nRanking plots saved to {output_dir}")


def main():
    build_heatmaps(DEFAULT_MEASURE_PAIR, OUTPUT_DIR)


if __name__ == "__main__":
    main()
