import pandas as pd

INPUT_DIR = "DPFR-recsys-evaluation"
MEASURES_PATH = "measures.csv"
MIDPOINT_PATH = "midpoint.csv"
RESULTS_PATH = "results.csv"

DATASETS = ["Amazon-lb", "Jester", "Lastfm", "ML-10M", "ML-20M", "QK-video"]
MODELS = ["BPR", "EASE", "ItemKNN", "MultiVAE"]
RERANKERS = ["borda", "combmnz", "GS-subset-0.05"]
KEY_COLUMNS = ["dataset", "model", "variant", "reranker", "alternative"]
MIDPOINT_COLUMNS = [
    "dataset",
    "measure_pair",
    "relevance_measure",
    "fairness_measure",
    "relevance_metric",
    "fairness_metric",
    "midpoint_relevance",
    "midpoint_fairness",
]
RESULT_COLUMNS = [*KEY_COLUMNS, "measure_pair"]

RELEVANCE_METRICS = {
    "P": "P@10",
    "MAP": "MAP@10",
    "R": "R@10",
    "NDCG": "NDCG@10",
}
FAIRNESS_METRICS = {
    "Jain": "Jain_our@10",
    "Ent": "Ent_our@10",
    "Gini": "Gini_our@10",
}

DEFAULT_MEASURE_PAIR = "NDCG-Ent"


def load_csv(path: str, columns: list[str]) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df[list(columns)].copy()


def load_measures(
    columns: list[str] = KEY_COLUMNS,
    path: str = MEASURES_PATH,
) -> pd.DataFrame:
    return load_csv(path, columns)


def load_midpoints(
    columns: list[str] = MIDPOINT_COLUMNS,
    path: str = MIDPOINT_PATH,
) -> pd.DataFrame:
    return load_csv(path, columns)


def load_results(
    columns: list[str] = RESULT_COLUMNS,
    path: str = RESULTS_PATH,
) -> pd.DataFrame:
    return load_csv(path, columns)


def resolve_measure_pair(pair: str) -> tuple[str, str, str, str, str]:
    relevance_measure, fairness_measure = pair.split("-", maxsplit=1)
    canonical_pair = f"{relevance_measure}-{fairness_measure}"
    return (
        canonical_pair,
        relevance_measure,
        fairness_measure,
        RELEVANCE_METRICS[relevance_measure],
        FAIRNESS_METRICS[fairness_measure],
    )


def select_midpoints(midpoints: pd.DataFrame, pair: str) -> pd.DataFrame:
    canonical_pair, _, _, _, _ = resolve_measure_pair(pair)
    return midpoints[midpoints["measure_pair"] == canonical_pair].copy()


def dataset_order_map():
    return {dataset: idx for idx, dataset in enumerate(DATASETS)}
