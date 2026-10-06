"""Objective metrics for *cluster recoverability* from an embedding.

Because the fishy dataset has known ground-truth factors, we can measure — for any
embedding (PCA/ICA/UMAP scores, or even raw features) — how recoverable each planted
factor is. The headline metric is **balanced classification accuracy** of each factor
from a small number of embedding dimensions (so it is fair to the 5% minority class),
plus a silhouette and the single best-axis separation.
"""
from __future__ import annotations

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    adjusted_rand_score,
    balanced_accuracy_score,
    roc_auc_score,
    silhouette_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def _cv_balanced_accuracy(X: np.ndarray, y: np.ndarray, clf="logistic") -> float:
    """Stratified 5-fold balanced accuracy predicting y from X."""
    classes, counts = np.unique(y, return_counts=True)
    if len(classes) < 2:
        return float("nan")
    n_splits = int(min(5, counts.min()))
    if n_splits < 2:
        return float("nan")
    if clf == "logistic":
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced"),
        )
    elif clf == "knn":
        model = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5))
    else:  # lda
        model = make_pipeline(StandardScaler(), LinearDiscriminantAnalysis())
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
    pred = cross_val_predict(model, X, y, cv=cv)
    return float(balanced_accuracy_score(y, pred))


def _best_axis_auc(X: np.ndarray, y: np.ndarray) -> tuple[float, int]:
    """For a binary factor, the best single-dimension separation (AUC)."""
    if len(np.unique(y)) != 2:
        return float("nan"), -1
    best, best_dim = 0.5, -1
    for d in range(X.shape[1]):
        col = X[:, d]
        auc = roc_auc_score(y, col)
        auc = max(auc, 1 - auc)  # axis sign-agnostic
        if auc > best:
            best, best_dim = auc, d
    return float(best), int(best_dim)


def factor_recoverability(X: np.ndarray, y: np.ndarray) -> dict:
    """Recoverability of one factor from embedding X."""
    out = {
        "balanced_acc": _cv_balanced_accuracy(X, y, "logistic"),
        "knn_acc": _cv_balanced_accuracy(X, y, "knn"),
        "prevalence": float(np.mean(y == y.max())),
    }
    auc, dim = _best_axis_auc(X, y)
    out["best_axis_auc"] = auc
    out["best_axis"] = dim
    try:
        if len(np.unique(y)) > 1:
            out["silhouette"] = float(silhouette_score(X, y))
        else:
            out["silhouette"] = float("nan")
    except Exception:
        out["silhouette"] = float("nan")
    return out


def scorecard(X: np.ndarray, labels: dict[str, np.ndarray]) -> dict[str, dict]:
    """Recoverability scorecard over all factors for one embedding."""
    return {name: factor_recoverability(X, y) for name, y in labels.items()}


def ari_for_joint(X: np.ndarray, joint: np.ndarray, n_clusters: int | None = None) -> float:
    """Unsupervised: KMeans on X vs joint ground-truth labels (Adjusted Rand)."""
    from sklearn.cluster import KMeans

    k = n_clusters or len(np.unique(joint))
    km = KMeans(n_clusters=k, n_init=10, random_state=0)
    pred = km.fit_predict(StandardScaler().fit_transform(X))
    return float(adjusted_rand_score(joint, pred))


def format_scorecard(card: dict[str, dict], title: str = "") -> str:
    lines = []
    if title:
        lines.append(title)
    header = f"{'factor':<8} {'prev':>5} {'bal_acc':>8} {'knn_acc':>8} {'best_auc':>9} {'axis':>5} {'silh':>7}"
    lines.append(header)
    lines.append("-" * len(header))
    for name, m in card.items():
        lines.append(
            f"{name:<8} {m['prevalence']:>5.2f} {m['balanced_acc']:>8.3f} "
            f"{m['knn_acc']:>8.3f} {m['best_axis_auc']:>9.3f} {str(m['best_axis']):>5} "
            f"{m['silhouette']:>7.3f}"
        )
    return "\n".join(lines)
