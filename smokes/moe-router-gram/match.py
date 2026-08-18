from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

from gram import cosine_gram, row_normalize, upper_tri_cosine


def hungarian_perm(Wa: np.ndarray, Wb: np.ndarray) -> np.ndarray:
    a = row_normalize(Wa)
    b = row_normalize(Wb)
    cost = 1.0 - (a @ b.T)
    _r, c = linear_sum_assignment(cost)
    return c


def align_gram(g: np.ndarray, perm: np.ndarray) -> np.ndarray:
    return g[np.ix_(perm, perm)]


def layer_scores(wa: np.ndarray, wb: np.ndarray) -> dict[str, float]:
    ga = cosine_gram(wa)
    gb = cosine_gram(wb)
    raw = upper_tri_cosine(ga, gb)
    perm = hungarian_perm(wa, wb)
    aligned = upper_tri_cosine(ga, align_gram(gb, perm))
    return {"raw": raw, "aligned": aligned}
