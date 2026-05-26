import numpy as np
import pandas as pd
from pymcdm.helpers import normalize_matrix
from pymcdm.methods import PROMETHEE_II
from pymcdm.normalizations import sum_normalization

from critic_weights import critic_weights
from dpfr import standard_dpfr_scores
from load_measures import (
    DEFAULT_MEASURE_PAIR,
    KEY_COLUMNS,
    RESULTS_PATH,
    dataset_order_map,
    load_measures,
    load_midpoints,
    resolve_measure_pair,
    select_midpoints,
)
from topsis_dpfr import topsis_dpfr_scores


PROFIT_TYPES = np.array([1, 1], dtype=int)

def promethee_scores(alts: np.ndarray):
    method = PROMETHEE_II("usual")
    types = PROFIT_TYPES
    alts = normalize_matrix(alts, sum_normalization, types)
    weights = critic_weights(alts)
    pref = method(alts, weights, types)
    return pref, weights, alts

def rank_scores(scores: pd.Series | np.ndarray, ascending: bool) -> pd.Series:
    return pd.Series(scores).rank(ascending=ascending, method="min").astype(int)

def build_dataset_results(
    dataset_rows: pd.DataFrame,
    midpoint_row: pd.Series,
) -> pd.DataFrame:
    relevance_metric = midpoint_row["relevance_metric"]
    fairness_metric = midpoint_row["fairness_metric"]
    matrix = dataset_rows[[relevance_metric, fairness_metric]].to_numpy(dtype=float)
    midpoint = np.array(
        [midpoint_row["midpoint_relevance"], midpoint_row["midpoint_fairness"]],
        dtype=float,
    )
    result = dataset_rows[KEY_COLUMNS].copy().reset_index(drop=True)
    result["measure_pair"] = midpoint_row["measure_pair"]
    result["relevance_measure"] = midpoint_row["relevance_measure"]
    result["fairness_measure"] = midpoint_row["fairness_measure"]
    result["relevance_metric"] = relevance_metric
    result["fairness_metric"] = fairness_metric
    result["relevance_value"] = matrix[:, 0]
    result["fairness_value"] = matrix[:, 1]
    result["midpoint_relevance"] = midpoint[0]
    result["midpoint_fairness"] = midpoint[1]

    average_scores = matrix.mean(axis=1)
    dpfr_scores = standard_dpfr_scores(matrix, midpoint)
    topsis_scores, topsis_weights, sums, normalized_negative, normalized_positive = (
        topsis_dpfr_scores(matrix, midpoint)
    )
    promethee_values, promethee_weights, normalized_matrix = promethee_scores(matrix)

    result["average_score"] = average_scores
    result["average_rank"] = rank_scores(average_scores, ascending=False)
    result["dpfr_score"] = dpfr_scores
    result["dpfr_rank"] = rank_scores(dpfr_scores, ascending=True)
    result["topsis_dpfr_score"] = topsis_scores
    result["topsis_dpfr_rank"] = rank_scores(topsis_scores, ascending=False)
    result["promethee_score"] = promethee_values
    result["promethee_rank"] = rank_scores(promethee_values, ascending=False)

    result["relevance_weight"] = topsis_weights[0]
    result["fairness_weight"] = topsis_weights[1]
    result["promethee_relevance_weight"] = promethee_weights[0]
    result["promethee_fairness_weight"] = promethee_weights[1]
    result["relevance_sum"] = sums[0]
    result["fairness_sum"] = sums[1]
    result["normalized_relevance_value"] = normalized_matrix[:, 0]
    result["normalized_fairness_value"] = normalized_matrix[:, 1]
    result["normalized_midpoint_relevance"] = normalized_positive[0]
    result["normalized_midpoint_fairness"] = normalized_positive[1]
    result["normalized_relevance_lower_bound"] = normalized_negative[0]
    result["normalized_fairness_lower_bound"] = normalized_negative[1]
    result["normalization_method"] = "sum"
    return result


def build_results(pair: str = DEFAULT_MEASURE_PAIR) -> pd.DataFrame:
    canonical_pair, _, _, relevance_metric, fairness_metric = resolve_measure_pair(pair)
    measures = load_measures([*KEY_COLUMNS, relevance_metric, fairness_metric])
    midpoints = select_midpoints(load_midpoints(), canonical_pair)

    result_frames = []
    for _, midpoint_row in midpoints.iterrows():
        dataset_rows = measures[measures["dataset"] == midpoint_row["dataset"]].copy()
        result_frames.append(build_dataset_results(dataset_rows, midpoint_row))

    results = pd.concat(result_frames, ignore_index=True)
    results["_dataset_order"] = results["dataset"].map(dataset_order_map())
    results["_source_order"] = results.groupby("dataset").cumcount()
    return results.sort_values(["_dataset_order", "_source_order"]).drop(
        columns=["_dataset_order", "_source_order"]
    )


def main():
    results = build_results(DEFAULT_MEASURE_PAIR)
    results.to_csv(RESULTS_PATH, index=False, float_format="%.12g")
    print(f"Wrote {len(results)} rows to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
