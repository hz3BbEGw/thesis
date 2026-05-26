import os
import pickle
import pandas as pd
from load_measures import DATASETS, INPUT_DIR, MEASURES_PATH, MODELS, RERANKERS


METRICS = [
    "HR@10",
    "MRR@10",
    "P@10",
    "MAP@10",
    "R@10",
    "NDCG@10",
    "Jain_ori@10",
    "Jain_our@10",
    "QF_ori@10",
    "QF_our@10",
    "Ent_ori@10",
    "Ent_our@10",
    "Gini_ori@10",
    "Gini_our@10",
    "FSat_ori@10",
    "FSat_our@10",
    "IAA_true_ori@10",
    "II-F_ori@10",
    "AI-F_ori@10",
    "IBO_ori@10",
    "IBO_our@10",
    "MME_ori@10",
]


def alternative_specs():
    specs: list[dict[str, str]] = []
    for dataset in DATASETS:
        for model in MODELS:
            specs.append(
                {
                    "dataset": dataset,
                    "model": model,
                    "variant": "base",
                    "reranker": "none",
                    "alternative": model,
                    "path": os.path.join(
                        INPUT_DIR,
                        "eval",
                        "base",
                        f"base_{dataset}_{model}.pickle",
                    ),
                }
            )
            for reranker in RERANKERS:
                specs.append(
                    {
                        "dataset": dataset,
                        "model": model,
                        "variant": "reranked",
                        "reranker": reranker,
                        "alternative": f"{model}+{reranker}",
                        "path": os.path.join(
                            INPUT_DIR,
                            "reranking",
                            "result",
                            f"{dataset}_{model}_at10_rerank25_{reranker}.pickle",
                        ),
                    }
                )
    return specs


def read_metric_pickle(path: str):
    with open(path, "rb") as file:
        values = dict(pickle.load(file))

    return {metric: float(values[metric]) for metric in METRICS}


def build_measures() -> pd.DataFrame:
    rows = []
    for spec in alternative_specs():
        rows.append(
            {
                "dataset": spec["dataset"],
                "model": spec["model"],
                "variant": spec["variant"],
                "reranker": spec["reranker"],
                "alternative": spec["alternative"],
                **read_metric_pickle(spec["path"]),
            }
        )
    return pd.DataFrame(rows)


def main():
    measures = build_measures()
    measures.to_csv(MEASURES_PATH, index=False, float_format="%.12g", na_rep="nan")
    print(f"Wrote {len(measures)} rows to {MEASURES_PATH}")


if __name__ == "__main__":
    main()
