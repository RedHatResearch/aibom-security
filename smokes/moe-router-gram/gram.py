from __future__ import annotations

import numpy as np


def row_normalize(W: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(W, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise RuntimeError("zero_row_in_normalize")
    return W / norms


def cosine_gram(W: np.ndarray) -> np.ndarray:
    return row_normalize(W) @ row_normalize(W).T


def upper_tri_vec(G: np.ndarray) -> np.ndarray:
    idx = np.triu_indices(G.shape[0], k=1)
    return G[idx]


def upper_tri_cosine(Ga: np.ndarray, Gb: np.ndarray) -> float:
    a = upper_tri_vec(Ga)
    b = upper_tri_vec(Gb)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        raise RuntimeError("zero_gram_vec")
    return float(np.dot(a, b) / (na * nb))
