"""Dimensionality-reduction wrappers (PCA / ICA / UMAP), optional standardization."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import PCA, FastICA
from sklearn.preprocessing import StandardScaler


@dataclass
class Embedding:
    scores: np.ndarray            # (N, k)
    method: str
    explained_variance: np.ndarray | None = None
    model: object | None = None


def _maybe_scale(X: np.ndarray, standardize: bool) -> np.ndarray:
    if standardize:
        return StandardScaler().fit_transform(X)
    return X


def pca(X: np.ndarray, n_components: int = 6, standardize: bool = False) -> Embedding:
    Xs = _maybe_scale(X, standardize)
    n_components = min(n_components, *Xs.shape)
    model = PCA(n_components=n_components, random_state=42)
    scores = model.fit_transform(Xs)
    return Embedding(scores, "PCA", model.explained_variance_ratio_, model)


def ica(X: np.ndarray, n_components: int = 6, standardize: bool = False) -> Embedding:
    Xs = _maybe_scale(X, standardize)
    n_components = min(n_components, *Xs.shape)
    model = FastICA(n_components=n_components, random_state=42, max_iter=1000)
    scores = model.fit_transform(Xs)
    return Embedding(scores, "ICA", None, model)


def umap_embed(
    X: np.ndarray, n_components: int = 2, n_neighbors: int = 15, min_dist: float = 0.1,
    standardize: bool = False, metric: str = "euclidean",
) -> Embedding:
    import umap

    Xs = _maybe_scale(X, standardize)
    model = umap.UMAP(
        n_components=n_components,
        n_neighbors=min(n_neighbors, len(Xs) - 1),
        min_dist=min_dist,
        metric=metric,
        random_state=42,
    )
    scores = model.fit_transform(Xs)
    return Embedding(scores, "UMAP", None, model)
