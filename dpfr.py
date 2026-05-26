import numpy as np

def standard_dpfr_scores(matrix: np.ndarray, midpoint: np.ndarray):
    return np.linalg.norm(matrix - midpoint, axis=1)
