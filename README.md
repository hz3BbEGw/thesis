# Joint Fairness and Relevance Evaluation

## Contents

- `DPFR-recsys-evaluation/` - source evaluation data and RecBole-based experiment code.
- `measures.py` - extracts metric values from evaluation pickles into `measures.csv`.
- `midpoints.py` - computes dataset-specific Pareto midpoints into `midpoint.csv`.
- `results.py` - builds MCDA rankings using average score, DPFR, TOPSIS-DPFR, and PROMETHEE into `results.csv`.
- `heatmaps.py` - generates ranking-correlation heatmaps under `corr_analysis/ranking_plots/`.
- `changes` - git diff of `DPFR-recsys-evaluation/`.

## Requirements

```bash
pip install numpy pandas scipy matplotlib seaborn pymcdm
```

## Usage

```bash
python measures.py
python midpoints.py
python results.py
python heatmaps.py
```

The scripts expect the DPFR evaluation pickles to be available under `DPFR-recsys-evaluation/`.
