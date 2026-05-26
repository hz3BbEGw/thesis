import numpy as np
from pymcdm.helpers import normalize_matrix
from pymcdm.normalizations import sum_normalization
from critic_weights import critic_weights

PROFIT_TYPES = np.array([1, 1], dtype=int)

def topsis_dpfr_method(
    alts: np.ndarray,
    weights: np.ndarray,
    normalized_positive_ideal: np.ndarray,
):
    negative_ideal = np.min(alts, axis=0)
    positive_ideal = normalized_positive_ideal

    weighted_alts = alts * weights
    weighted_negative_ideal = negative_ideal * weights
    weighted_positive_ideal = positive_ideal * weights

    d_minus = np.linalg.norm(
        weighted_alts - weighted_negative_ideal,
        axis=1,
    )
    d_plus = np.linalg.norm(
        weighted_alts - weighted_positive_ideal,
        axis=1,
    )
    denominator = d_minus + d_plus
    pref = np.divide(
        d_minus,
        denominator,
        out=np.full_like(d_minus, 0.5, dtype=float),
        where=~np.isclose(denominator, 0.0),
    )
    return pref, negative_ideal, positive_ideal

def topsis_dpfr_scores(
    alts: np.ndarray,
    midpoint: np.ndarray,
):
    method = topsis_dpfr_method
    types = PROFIT_TYPES
    sums = alts.sum(axis=0)
    alts = normalize_matrix(alts, sum_normalization, types)
    normalized_midpoint = midpoint / sums
    weights = critic_weights(alts)
    pref, normalized_negative_ideal, normalized_positive_ideal = method(
        alts,
        weights,
        normalized_midpoint,
    )
    return pref, weights, sums, normalized_negative_ideal, normalized_positive_ideal
