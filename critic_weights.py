import numpy as np

def critic_weights(normalized_matrix: np.ndarray):
    std = np.std(normalized_matrix, axis=0, ddof=1)
    corr = np.corrcoef(normalized_matrix, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0)
    weight = std * np.sum(1 - corr, axis=0)

    total = np.sum(weight)
    return weight / total
