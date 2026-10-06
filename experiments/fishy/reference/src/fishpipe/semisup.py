"""Minimal-supervision escape: constraint-driven metric learning + active query selection.

The team proved subtle/overlapping structure (stripe 4-vs-5) is supervised-recoverable (~0.93)
but has NO unsupervised density gap. The HDLSS literature's highest-leverage fix for n<<p is
NOT a better unsupervised method — it is to let a handful of expert pairwise judgments WARP the
feature space so the previously-invisible axis becomes a real density direction the team's own
unsupervised gates can then fire on (Davis et al. 2007 ITML; Bilenko et al. 2004 MPCK-Means;
Xiong et al. 2014 active constraints).

Elicitation a biologist can do: shown two rendered specimens, answer "same pattern type?"
-> must-link (ML) or cannot-link (CL). This module:
  * learns a Mahalanobis warp from ML/CL pairs (a regularized RCA/DCA-style generalized
    eigenproblem: directions where CL pairs spread and ML pairs are tight), then
  * runs ORDINARY unsupervised clustering in the warped space, and
  * selects WHICH pairs to query actively (uncertainty sampling) so the expert budget is small.
CPU-trivial; the learned projection is inspectable (which descriptor coords were up-weighted).
"""
from __future__ import annotations

import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


def learn_warp(X: np.ndarray, ml_pairs, cl_pairs, n_dim: int = 4, reg: float = 1.0) -> np.ndarray:
    """Learn a linear warp L (p x r) from must-link / cannot-link pairs.

    Build constraint scatter matrices S_ml, S_cl from pair-difference outer products, then take
    the top generalized eigenvectors of (S_cl, S_ml + reg*I): directions that PULL APART
    cannot-link pairs while keeping must-link pairs together. `reg` is the ITML-style prior
    toward the Euclidean metric (stabilizes when constraints are few — essential at small n).
    Returns L so that X @ L is the warped representation.
    """
    p = X.shape[1]
    S_ml = np.zeros((p, p)); S_cl = np.zeros((p, p))
    for i, j in ml_pairs:
        d = (X[i] - X[j])[:, None]
        S_ml += d @ d.T
    for i, j in cl_pairs:
        d = (X[i] - X[j])[:, None]
        S_cl += d @ d.T
    S_ml = S_ml / max(len(ml_pairs), 1) + reg * np.eye(p)
    if len(cl_pairs) == 0:
        return np.eye(p)[:, :n_dim]
    S_cl = S_cl / max(len(cl_pairs), 1)
    # generalized eigenproblem S_cl v = lam S_ml v  ->  whiten by S_ml, eig of whitened S_cl
    from scipy.linalg import eigh
    w, V = eigh(S_cl, S_ml)
    order = np.argsort(w)[::-1]
    return V[:, order[:n_dim]]


def triplet_warp(X, triplets, n_dim=10, margin=1.0, lr=0.05, n_epochs=40,
                 reg=1e-2, L0=None, seed=0):
    """Learn a linear warp L (p x n_dim) from RANKING triplets (a, p, q) meaning
    'a is more similar to p than to q' (p ranked nearer the anchor than q).

    SGD on the hinge loss  max(0, margin + ||L^T(a-p)||^2 - ||L^T(a-q)||^2)  with an
    ITML-style pull toward the prior L0 (the round-0 PCA basis) so few triplets at small n
    don't overfit. This is the 'morph the space' step driven by relative judgments; iterate
    it across feedback rounds. Returns L so X @ L is the morphed embedding."""
    rng = np.random.default_rng(seed)
    p = X.shape[1]
    if L0 is None:
        # init = top-n_dim PCA basis -> round-0 space is the ordinary unsupervised morphospace
        Xc = X - X.mean(0)
        _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
        L0 = Vt[:n_dim].T.copy()
    L = L0.copy()
    trips = list(triplets)
    if not trips:
        return L

    # Pin the Frobenius norm of L to its round-0 value at EVERY step: this removes the
    # overall-scale degree of freedom the hinge would otherwise exploit (shrink -> trivially
    # satisfied = collapse) AND caps gradient blow-up. With unit-norm pair-differences the
    # update is about DIRECTION (which axes to up/down-weight), not scale.
    fro0 = np.linalg.norm(L0)
    d0 = np.median([((X[a] - X[q]) @ L0) @ ((X[a] - X[q]) @ L0) for (a, _, q) in trips[:300]])
    m = margin * max(d0, 1e-9)

    for _ in range(n_epochs):
        rng.shuffle(trips)
        for a, pp, qq in trips:
            dp = X[a] - X[pp]; dq = X[a] - X[qq]
            nd = np.sqrt(max(dp @ dp, 1e-12)) + np.sqrt(max(dq @ dq, 1e-12))
            dp = dp / nd; dq = dq / nd                     # unit-scale the difference vectors
            ep = dp @ L; eq = dq @ L
            if m / (nd * nd) + ep @ ep - eq @ eq > 0:      # violated triplet (scaled margin)
                grad = 2.0 * (np.outer(dp, ep) - np.outer(dq, eq))
                L -= lr * (grad + reg * (L - L0))
        f = np.linalg.norm(L)
        if not np.isfinite(f) or f < 1e-9:                 # NaN/collapse guard
            return L0.copy()
        L *= fro0 / f                                      # pin Frobenius norm
    return L


def triplets_from_ranking(anchor, ranked, remote=None):
    """A ranking [near ... far] of candidates around `anchor` -> ordered triplets
    (anchor, closer, farther). Optionally append a `remote` item as farther than all."""
    items = list(ranked) + ([remote] if remote is not None else [])
    out = []
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            out.append((anchor, items[i], items[j]))
    return out


def spring_embed(Y0, triplets, n_iter=400, lr=0.01, margin=0.5, lam=0.05, seed=0):
    """SPRING / ENERGY ordinal embedding (move the POINTS, not a linear transform).

    Minimise  E = sum_triplets max(0, m + ||Ya-Yp||^2 - ||Ya-Yq||^2)  +  lam * sum ||Yi - Y0i||^2
    where each triplet (a, p, q) means 'a more similar to p than to q'. Mechanically: a violated
    triplet acts as a SPRING that pulls the anchor toward the closer point p and pushes it from
    the farther point q (p attracted to a, q repelled). The lam term is a spring tethering every
    point to its initial morphospace position Y0 — it stabilises the layout and keeps UNRANKED
    points in place (the ordinal-embedding analogue of out-of-sample placement). Returns Y.

    This is the soft-ordinal-embedding / t-STE family (Agarwal 2007; van der Maaten-Weinberger
    2012), i.e. the force-directed alternative to the linear `triplet_warp`."""
    rng = np.random.default_rng(seed)
    Y = Y0.copy().astype(float)
    if not triplets:
        return Y
    scale = np.median(np.sum((Y0 - Y0.mean(0)) ** 2, axis=1)) + 1e-9
    m = margin * scale
    trips = list(triplets)
    for _ in range(n_iter):
        rng.shuffle(trips)
        for a, p, q in trips:
            dap = Y[a] - Y[p]; daq = Y[a] - Y[q]
            if m + dap @ dap - daq @ daq > 0:             # violated -> apply spring forces
                Y[a] -= lr * 2.0 * (Y[q] - Y[p])          # anchor toward p, away from q
                Y[p] -= lr * (-2.0 * dap)                 # p attracted to a
                Y[q] -= lr * (2.0 * daq)                  # q repelled from a
        Y -= lr * 2.0 * lam * (Y - Y0)                    # tether spring to initial layout
    return Y


def cluster_in_warp(X, L, k=2, seed=0):
    """GMM clustering in the warped space (unsupervised given the warp)."""
    Z = X @ L
    Z = StandardScaler().fit_transform(Z)
    return GaussianMixture(k, covariance_type="full", random_state=seed).fit_predict(Z)


# ---------------------------------------------------------------------------
# Pair sampling: random vs active (uncertainty)
# ---------------------------------------------------------------------------
def _pairs_from_labels(idx_pairs, y):
    ml = [(i, j) for i, j in idx_pairs if y[i] == y[j]]
    cl = [(i, j) for i, j in idx_pairs if y[i] != y[j]]
    return ml, cl


def random_pairs(n, m, rng):
    out = set()
    while len(out) < m:
        i, j = rng.integers(0, n, 2)
        if i != j:
            out.add((int(min(i, j)), int(max(i, j))))
    return list(out)


def active_pairs(X, L, m, queried, rng, n_ensemble=12, k=2):
    """Uncertainty sampling (NPU-style): co-cluster each pair across a bootstrap GMM ensemble
    in the current warped space; query the pairs whose co-association is most ambiguous
    (closest to 0.5) and not yet queried."""
    Z = StandardScaler().fit_transform(X @ L)
    n = Z.shape[0]
    co = np.zeros((n, n))
    for b in range(n_ensemble):
        bidx = rng.choice(n, n, replace=True)
        gm = GaussianMixture(k, covariance_type="full", random_state=b).fit(Z[bidx])
        lab = gm.predict(Z)
        for c in np.unique(lab):
            mem = np.where(lab == c)[0]
            co[np.ix_(mem, mem)] += 1
    co /= n_ensemble
    iu = np.triu_indices(n, k=1)
    amb = -np.abs(co[iu] - 0.5)                       # higher = more ambiguous
    order = np.argsort(amb)[::-1]
    out = []
    qset = set(queried)
    for t in order:
        pair = (int(iu[0][t]), int(iu[1][t]))
        if pair not in qset:
            out.append(pair); qset.add(pair)
        if len(out) >= m:
            break
    return out
