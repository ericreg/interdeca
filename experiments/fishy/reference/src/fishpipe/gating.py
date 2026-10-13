"""Trustworthy small-n cluster gating: HDLSS-aware significance + resampling stability.

The team's existing gate is the 1-D Hartigan dip test + a column-shuffled null. That null
destroys feature covariance (so it is too easy to beat) and the dip is blind to structure
that lives in a tilted multivariate subspace or in a rare minority. This module adds two
calibrated, CPU-only, n<<p-aware gates from the HDLSS literature:

  * SigClust (Liu, Hayes, Nobel & Marron 2008 JASA; soft-threshold null per Huang et al. 2015):
    "is this a single Gaussian blob, or two clusters?" The null is a single Gaussian whose
    eigen-SPECTRUM is estimated (not the full p x p covariance), with small eigenvalues
    soft-thresholded up to an estimated background noise level so noise dimensions don't
    inflate the false-positive rate. Multivariate and direction-agnostic.

  * Consensus PAC + covariance-preserving null (Monti 2003; Senbabaoglu PAC 2014; M3C-style
    null, John 2020): resample specimens, recluster, score the Proportion of Ambiguous
    Clustering, and compare to a PCA-preserving null so the gate emits a calibrated p-value
    for EACH k INCLUDING k=1 (abstain). Plus per-cluster bootstrap Jaccard (Hennig 2007).

Together: a discovery that finds real separated structure and *refuses to invent it* — the
property that makes an unsupervised morphospace safe to publish from at n=25-40.
"""
from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# SigClust
# ---------------------------------------------------------------------------
def cluster_index(X: np.ndarray, labels: np.ndarray) -> float:
    """2-means cluster index: within-cluster sum-of-squares / total sum-of-squares.
    Lower = more clustered. The SigClust test statistic."""
    tot = ((X - X.mean(0)) ** 2).sum()
    wss = 0.0
    for c in np.unique(labels):
        Xi = X[labels == c]
        wss += ((Xi - Xi.mean(0)) ** 2).sum()
    return wss / tot if tot > 0 else 1.0


def _soft_threshold_spectrum(X: np.ndarray):
    """Estimate the null Gaussian's eigen-spectrum for n<<p: sample eigenvalues for the top
    n-1 directions, a robust background-noise level sigma^2 for the rest, with small
    eigenvalues soft-thresholded up to sigma^2 (Huang et al. 2015 in spirit)."""
    n, p = X.shape
    Xc = X - X.mean(0)
    # sample eigenvalues via SVD (top min(n-1,p) directions)
    s = np.linalg.svd(Xc, full_matrices=False, compute_uv=False)
    lam = (s ** 2) / (n - 1)                          # (min(n-1,p),)
    # robust background noise: median per-feature variance is a stable sigma^2 estimate
    sigma2 = np.median(Xc.var(0, ddof=1))
    sigma2 = max(sigma2, 1e-12)
    lam_th = np.maximum(lam, sigma2)                 # soft-threshold up to noise floor
    return lam_th, sigma2, p


def sigclust(X: np.ndarray, n_sim: int = 1000, seed: int = 0) -> dict:
    """Test H0: data are one Gaussian blob vs H1: two 2-means clusters.
    Returns {ci, pvalue, n_sim}. Small p-value => a real (not single-Gaussian) split.

    Null samples are drawn with covariance = U diag(lam_th) U^T along the top directions
    plus sigma^2 in every remaining dimension — i.e. a single anisotropic Gaussian matched
    to the data's spectrum, the correct null for 'is there more than one blob'."""
    rng = np.random.default_rng(seed)
    # mean-center only — do NOT unit-scale: cluster separation lives in the high-variance
    # directions, and column-standardizing would inflate noise dims to equal the signal,
    # whitening the very structure SigClust must detect. (Input is already PCA morphospace.)
    Xs = X - X.mean(0)
    n, p = Xs.shape
    km = KMeans(2, n_init=10, random_state=seed).fit(Xs)
    ci_real = cluster_index(Xs, km.labels_)

    lam_th, sigma2, _ = _soft_threshold_spectrum(Xs)
    k = len(lam_th)
    # extra std along top-k eigvecs beyond the isotropic sigma; isotropic part added separately
    extra = np.sqrt(np.maximum(lam_th - sigma2, 0.0))      # (k,)
    # top eigvecs
    Xc = Xs - Xs.mean(0)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    V = Vt.T[:, :k]                                        # (p, k)

    ci_null = np.empty(n_sim)
    for b in range(n_sim):
        g1 = rng.standard_normal((n, k)) * extra
        Z = g1 @ V.T + rng.standard_normal((n, p)) * np.sqrt(sigma2)
        lab = KMeans(2, n_init=3, random_state=b).fit_predict(Z)
        ci_null[b] = cluster_index(Z, lab)
    # p = P(null CI <= real CI): a more-clustered (lower CI) real split than the null
    pval = (1.0 + np.sum(ci_null <= ci_real)) / (n_sim + 1.0)
    return {"ci": float(ci_real), "pvalue": float(pval), "ci_null_mean": float(ci_null.mean())}


# ---------------------------------------------------------------------------
# Consensus clustering + PAC + covariance-preserving null + per-cluster Jaccard
# ---------------------------------------------------------------------------
def _consensus_matrix(X, k, n_resample=100, frac=0.8, seed=0):
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    co = np.zeros((n, n)); cnt = np.zeros((n, n))
    for b in range(n_resample):
        idx = rng.choice(n, size=int(frac * n), replace=False)
        lab = KMeans(k, n_init=3, random_state=b).fit_predict(X[idx])
        for c in np.unique(lab):
            members = idx[lab == c]
            co[np.ix_(members, members)] += 1
        cnt[np.ix_(idx, idx)] += 1
    with np.errstate(invalid="ignore"):
        M = np.where(cnt > 0, co / cnt, 0.0)
    return M


def _pac(M, lo=0.1, hi=0.9):
    iu = np.triu_indices_from(M, k=1)
    v = M[iu]
    return float(np.mean((v > lo) & (v < hi)))


def consensus_gate(X: np.ndarray, ks=(2, 3, 4, 5, 6), n_resample=100, n_null=40, seed=0) -> dict:
    """For each k, PAC of the consensus matrix vs a PCA-preserving null (M3C-style).
    Returns per-k {pac, null_pac_mean, pvalue} and the chosen k (lowest PAC passing p<0.05,
    else k=1 = ABSTAIN)."""
    Xs = X - X.mean(0)                                # mean-center only (see sigclust note)
    rng = np.random.default_rng(seed)
    # PCA-preserving null generator: sample PC scores ~ N(0, sd of real PCs), rotate back
    Xc = Xs - Xs.mean(0)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    scores = Xc @ Vt.T
    sd = scores.std(0)

    out = {}
    for k in ks:
        pac_real = _pac(_consensus_matrix(Xs, k, n_resample, seed=seed))
        null_pacs = []
        for b in range(n_null):
            Znull = (rng.standard_normal(scores.shape) * sd) @ Vt
            null_pacs.append(_pac(_consensus_matrix(Znull, k, max(30, n_resample // 2), seed=b)))
        null_pacs = np.array(null_pacs)
        # real is "more clustered" if its PAC is LOWER than the null PAC
        pval = (1.0 + np.sum(null_pacs <= pac_real)) / (len(null_pacs) + 1.0)
        # RCSI (M3C): how much MORE stable the real clustering is than the null at this k
        rcsi = float(np.log10(max(null_pacs.mean(), 1e-3)) - np.log10(max(pac_real, 1e-3)))
        out[k] = {"pac": pac_real, "null_pac_mean": float(null_pacs.mean()),
                  "pvalue": float(pval), "rcsi": rcsi}
    # choose k with the strongest real-vs-null stability (M3C RCSI), among significant; else abstain
    sig = [k for k in ks if out[k]["pvalue"] < 0.05]
    chosen = max(sig, key=lambda k: out[k]["rcsi"]) if sig else 1
    return {"per_k": out, "chosen_k": chosen}


def cluster_jaccard(X: np.ndarray, k: int, n_boot=100, seed=0) -> np.ndarray:
    """Per-cluster bootstrap Jaccard stability (Hennig 2007). Bands: <=0.5 dissolved,
    0.6-0.75 questionable, >=0.75 valid, >=0.85 highly stable."""
    Xs = X - X.mean(0)                                # mean-center only (see sigclust note)
    n = Xs.shape[0]
    base = KMeans(k, n_init=10, random_state=seed).fit(Xs)
    base_lab = base.labels_
    rng = np.random.default_rng(seed)
    jac = np.zeros((n_boot, k))
    for b in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        lab_b = KMeans(k, n_init=3, random_state=b).fit_predict(Xs[idx])
        for c in range(k):
            orig = set(np.where(base_lab == c)[0])
            best = 0.0
            for cb in np.unique(lab_b):
                boot = set(idx[lab_b == cb])
                inter = len(orig & boot)
                union = len(orig | boot)
                if union:
                    best = max(best, inter / union)
            jac[b, c] = best
    return jac.mean(0)
