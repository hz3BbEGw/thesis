import os
import numpy as np
import pandas as pd

from load_measures import (
    DATASETS,
    DEFAULT_MEASURE_PAIR,
    INPUT_DIR,
    MIDPOINT_PATH,
    resolve_measure_pair,
)
ALPHA = 0.5

def pareto_path(dataset: str):
    return os.path.join(
        INPUT_DIR,
        "pareto",
        "result",
        f"pareto_new_{dataset}_oraclefair_at10.pickle",
    )


def find_midpoint_on_pareto(
    pareto_df: pd.DataFrame,
    relevance_metric: str,
    fairness_metric: str,
):
    pair_df = (
        pareto_df[[relevance_metric, fairness_metric]]
        .dropna()
        .sort_values(relevance_metric, ascending=False, kind="stable")
        .drop_duplicates(relevance_metric, keep="last")
    )

    pair_values = pair_df.to_numpy(dtype=float)
    if len(pair_values) == 1:
        selected_position = 0
        total_arc_length = 0.0
        target_arc_length = 0.0
        selected_arc_length = 0.0
    else:
        segment_lengths = np.linalg.norm(np.diff(pair_values, axis=0), axis=1)
        cumulative_lengths = np.cumsum(segment_lengths)
        total_arc_length = float(cumulative_lengths[-1])
        target_arc_length = ALPHA * total_arc_length
        selected_segment = int(np.abs(cumulative_lengths - target_arc_length).argmin())
        selected_position = selected_segment + 1
        selected_arc_length = float(cumulative_lengths[selected_segment])

    return {
        "midpoint_relevance": float(pair_values[selected_position, 0]),
        "midpoint_fairness": float(pair_values[selected_position, 1]),
        "pareto_index": int(pair_df.index[selected_position]),
        "n_pair_pareto_points": int(len(pair_df)),
        "total_arc_length": total_arc_length,
        "target_arc_length": target_arc_length,
        "selected_arc_length": selected_arc_length,
    }


def build_midpoints() -> pd.DataFrame:
    midpoint_rows = []
    (
        measure_pair,
        relevance_measure,
        fairness_measure,
        relevance_metric,
        fairness_metric,
    ) = resolve_measure_pair(DEFAULT_MEASURE_PAIR)

    for dataset in DATASETS:
        pareto_df = pd.read_pickle(pareto_path(dataset))
        midpoint = find_midpoint_on_pareto(
            pareto_df,
            relevance_metric,
            fairness_metric,
        )
        midpoint_rows.append(
            {
                "dataset": dataset,
                "measure_pair": measure_pair,
                "relevance_measure": relevance_measure,
                "fairness_measure": fairness_measure,
                "relevance_metric": relevance_metric,
                "fairness_metric": fairness_metric,
                "alpha": ALPHA,
                "n_full_pareto_points": int(len(pareto_df)),
                **midpoint,
            }
        )

    return pd.DataFrame(midpoint_rows)


def main():
    midpoints = build_midpoints()
    midpoints.to_csv(MIDPOINT_PATH, index=False, float_format="%.12g")
    print(f"Wrote {len(midpoints)} rows to {MIDPOINT_PATH}")


if __name__ == "__main__":
    main()
