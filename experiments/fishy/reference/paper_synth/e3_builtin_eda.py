"""E3 — what a user gets from ColorAtlas's OWN exploratory-analysis panel.

Faithful reproduction of the module's population analysis on the atlas output, scored
against the generator's ground truth. NO custom representation learning: only the two
descriptors the module ships (area-weighted colour composition `area_hist`, and the
"subsample & average only" raw flattening `spatial_flatten`) and the three reducers the
panel offers (PCA, FastICA, UMAP).

Everything ground-truth is used for VALIDATION ONLY; nothing is tuned against it.

Outputs
  results/e3_builtin_eda.json
  results/e3_builtin_eda_arrays.npz
"""
from __future__ import annotations

import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C                                                        # noqa: E402
from fishpipe import features, gating                                     # noqa: E402
from fishpipe.data import FaceColorData                                   # noqa: E402

from sklearn.decomposition import PCA, FastICA                            # noqa: E402
from sklearn.metrics import adjusted_rand_score, silhouette_score         # noqa: E402
from sklearn.mixture import GaussianMixture                               # noqa: E402
from threadpoolctl import threadpool_limits                               # noqa: E402

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

T0 = time.time()
# E3_QUICK=1 is a smoke-test switch only (shrinks every resampling budget); all reported
# numbers come from the default full-budget run.
QUICK = bool(int(os.environ.get("E3_QUICK", "0")))
OUT_JSON = C.RESULTS / ("e3_builtin_eda_QUICK.json" if QUICK else "e3_builtin_eda.json")
OUT_NPZ = C.RESULTS / ("e3_builtin_eda_QUICK_arrays.npz" if QUICK else "e3_builtin_eda_arrays.npz")

N_DIP_SHUFFLE = 4 if QUICK else 200
GMM_KS = list(range(1, 9))
SMALL_N = (25, 40)
SMALL_DRAWS = 2 if QUICK else 30
SIG_NSIM = 40 if QUICK else 1000
CONS_KW = dict(n_resample=10, n_null=3) if QUICK else {}
# E3_ADDENDUM_ONLY=1 recomputes only the `reference` block (a deterministic function of the
# ground-truth label arrays, no descriptor and no model fitting on the data) and merges it
# into an existing results JSON, so the 72-minute main run does not have to be repeated.
ADDENDUM_ONLY = bool(int(os.environ.get("E3_ADDENDUM_ONLY", "0")))


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


def bcv(X, y):
    """C.balanced_cv, run with BLAS pinned to one thread.

    Pure performance guard, no change to the protocol: with p >> n the logistic solve is a
    sequence of tiny matvecs and OpenMP thread spin-up on a 32-core box dominates the run
    time (measured 137 s -> 1.9 s per fold on the 12k-dim descriptor). Identical numbers.
    """
    with threadpool_limits(limits=1):
        return C.balanced_cv(X, y)


def py(o):
    """Recursively convert numpy scalars/arrays to plain python for json."""
    if isinstance(o, dict):
        return {str(k): py(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [py(v) for v in o]
    if isinstance(o, np.ndarray):
        return [py(v) for v in o.tolist()]
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return None if not np.isfinite(f) else f
    if isinstance(o, (np.integer, int)):
        return int(o)
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    return o


# --------------------------------------------------------------------------- data
log("loading atlas data ...")
fcd, art, mesh = C.atlas_data()
names = list(fcd.names)
df, labels = C.load_gt(names)
N = len(names)
log(f"N={N} specimens, {fcd.colors.shape[1]} atlas faces")

y_bt = C.joint_label(labels, ("belly", "tail"))            # 4 groups
y_bts = C.joint_label(labels, ("belly", "tail", "stripe"))  # 8 groups
donor = df["specimen_index"].to_numpy().astype(int)

# generating parameters (drop constants + identifiers)
skip = {"sample_tag", "sample_index", "mesh_file", "landmark_file"}
PARAMS = [c for c in df.columns
          if c not in skip and pd.api.types.is_numeric_dtype(df[c]) and df[c].nunique() > 1]
P = df[PARAMS].to_numpy(float)
# `specimen_index` / `tps_rmse` / `n_landmarks_used` are NOT colour knobs: they are exactly
# constant within each of the 7 donor scans (verified), so a high |r| against them means the
# component is tracking DONOR SHAPE, not pigmentation. Kept in the table on purpose.
DONOR_PROXY = [c for c in ("specimen_index", "tps_rmse", "n_landmarks_used") if c in PARAMS]
GEN_PARAMS = [c for c in PARAMS if c not in DONOR_PROXY]
log(f"{len(PARAMS)} non-constant columns: {len(GEN_PARAMS)} generator knobs + "
    f"{len(DONOR_PROXY)} donor proxies {DONOR_PROXY}")


def reference_ceilings():
    """How much of the belly x tail target is recoverable AT ALL — the honest yardstick.

    ARI against the 4-group belly x tail partition is only interpretable next to the ARI a
    *perfect* reader of the generator's own parameters would score. Two things limit it:
      1. belly_hue is genuinely bimodal, tail_hue is not (Hartigan dip on the parameter
         itself), so the 'tail' 2-group label is a k-means cut through a continuum;
      2. therefore a method that perfectly recovers only the separable factor cannot exceed
         a specific ARI, no matter how good the representation is.
    Uses ground truth for VALIDATION CONTEXT only; nothing here touches a representation.
    """
    import diptest as _dt
    from sklearn.cluster import KMeans as _KM
    out = {"gt_marginal_dip": {}}
    for c in ("belly_hue", "tail_hue", "base_color_hue", "base_color_sat",
              "belly_strength", "tail_strength"):
        if c in df.columns:
            d, p = _dt.diptest(np.ascontiguousarray(df[c].to_numpy(float)))
            out["gt_marginal_dip"][c] = {"dip": float(d), "p": float(p)}
    out["ari_vs_belly_x_tail"] = {
        "perfect_belly_only": float(adjusted_rand_score(y_bt, labels["belly"])),
        "perfect_tail_only": float(adjusted_rand_score(y_bt, labels["tail"])),
        "perfect_belly_x_tail": float(adjusted_rand_score(y_bt, y_bt)),
        "perfect_stripe_only": float(adjusted_rand_score(y_bt, labels["stripe"])),
        "random_4way": float(np.mean([
            adjusted_rand_score(y_bt, np.random.default_rng(s).integers(0, 4, len(y_bt)))
            for s in range(50)])),
    }
    # oracle: cluster the GENERATOR'S OWN belly/tail hue values (the best possible colour
    # reading) and score it the same way the module's morphospace is scored
    G = df[["belly_hue", "tail_hue"]].to_numpy(float)
    G = (G - G.mean(0)) / G.std(0)
    with threadpool_limits(limits=1):
        k_o, lab_o, bic_o = gmm_bic_select(G)
        km4 = _KM(4, n_init=10, random_state=0).fit_predict(G)
    out["oracle_on_true_hues"] = {
        "gmm_bic_chosen_k": int(k_o),
        "ari_gmm": float(adjusted_rand_score(y_bt, lab_o)),
        "ari_kmeans_k4": float(adjusted_rand_score(y_bt, km4)),
        "bic": [float(b) for b in bic_o],
    }
    return out


def corr_ratio(x, g):
    """Correlation ratio eta (sqrt of eta^2) of continuous x against categorical g."""
    x = np.asarray(x, float)
    tot = ((x - x.mean()) ** 2).sum()
    if tot <= 0:
        return 0.0
    between = 0.0
    for u in np.unique(g):
        xi = x[g == u]
        between += len(xi) * (xi.mean() - x.mean()) ** 2
    return float(np.sqrt(between / tot))


def pearson_table(Z, Pm):
    """|r| between every column of Z (n, k) and every column of Pm (n, p) -> (k, p)."""
    Zc = (Z - Z.mean(0)) / (Z.std(0) + 1e-12)
    Pc = (Pm - Pm.mean(0)) / (Pm.std(0) + 1e-12)
    return np.abs(Zc.T @ Pc) / len(Z)


# --------------------------------------------------------------------------- features
FEATS: dict[str, np.ndarray] = {}
if not ADDENDUM_ONLY:
    log("building the module's two descriptors ...")
    for K in (16, 24, 30):
        t = time.time()
        FEATS[f"area_hist_K{K}"] = features.area_hist(fcd, n_clusters=K)
        log(f"  area_hist K={K}  {FEATS[f'area_hist_K{K}'].shape}  ({time.time() - t:.1f}s)")
    for S in (20000, 4000):
        FEATS[f"spatial_flatten_{S}"] = features.spatial_flatten(fcd, color_space="lab",
                                                                 subsample=S)
        log(f"  spatial_flatten sub={S}  {FEATS[f'spatial_flatten_{S}'].shape}")

FEAT_ORDER = ["area_hist_K16", "area_hist_K24", "area_hist_K30",
              "spatial_flatten_20000", "spatial_flatten_4000"]
HEADLINE_FEAT = "area_hist_K24"
if QUICK:
    FEAT_ORDER = ["area_hist_K24", "spatial_flatten_4000"]

# --------------------------------------------------------------------------- reducers
def reduce_pca(X, k=10):
    p = PCA(n_components=k, random_state=0).fit(X)
    return p.transform(X), {"evr": p.explained_variance_ratio_.copy()}


def reduce_ica(X, k=10):
    ica = FastICA(n_components=k, random_state=0, max_iter=1000, whiten="unit-variance")
    S = ica.fit_transform(X)
    # FastICA components are unordered; order by the data variance each explains
    # (||mixing column||^2 * var(source)) so "top-2 / top-4" is a defensible reading.
    w = (ica.mixing_ ** 2).sum(0) * S.var(0)
    order = np.argsort(w)[::-1]
    return S[:, order], {"weight": w[order] / (w.sum() + 1e-12), "n_iter": getattr(ica, "n_iter_", -1)}


def reduce_umap(X, k=2):
    import umap
    U = umap.UMAP(n_components=k, random_state=42).fit_transform(X)
    return np.asarray(U, float), {}


REDUCERS = {"PCA10": (reduce_pca, 10), "ICA10": (reduce_ica, 10), "UMAP2": (reduce_umap, 2)}


# --------------------------------------------------------------------------- analysis units
def gmm_bic_select(X2, ks=GMM_KS, seed=0):
    bics, models = [], {}
    with threadpool_limits(limits=1):      # see bcv(): tiny-array KMeans/EM, 4.7 s -> 0.4 s
        for k in ks:
            gm = GaussianMixture(n_components=k, covariance_type="full", n_init=5,
                                 random_state=seed).fit(X2)
            bics.append(float(gm.bic(X2)))
            models[k] = gm
        kstar = ks[int(np.argmin(bics))]
        lab = models[kstar].predict(X2) if kstar > 1 else np.zeros(len(X2), dtype=int)
    return kstar, lab, np.array(bics)


def fit_gmm(X2, k, seed=0):
    with threadpool_limits(limits=1):
        return GaussianMixture(k, covariance_type="full", n_init=5, random_state=seed).fit(X2)


def cluster_scores(X2, lab, kstar):
    out = {
        "chosen_k": int(kstar),
        "n_effective_clusters": int(len(np.unique(lab))),
        "ari_belly_x_tail": float(adjusted_rand_score(y_bt, lab)),
        "ari_belly_x_tail_x_stripe": float(adjusted_rand_score(y_bts, lab)),
    }
    for f in C.FACTORS:
        out[f"ari_{f}"] = float(adjusted_rand_score(labels[f], lab))
    out["ari_donor"] = float(adjusted_rand_score(donor, lab))
    out["silhouette"] = (float(silhouette_score(X2, lab))
                         if len(np.unique(lab)) > 1 else float("nan"))
    return out


def gate_report(X2, kstar):
    kj = int(max(kstar, 2))
    # every gate is thousands of KMeans fits on a 250x2 array: OpenMP spin-up dominates,
    # so pin to one thread (measured 168 s -> 4.5 s on the consensus gate). Same numbers.
    with threadpool_limits(limits=1):
        sc = gating.sigclust(X2, n_sim=SIG_NSIM, seed=0)
        cg = gating.consensus_gate(X2, seed=0, **CONS_KW)
        jac = gating.cluster_jaccard(X2, k=kj, seed=0)
    return {
        "sigclust_ci": float(sc["ci"]), "sigclust_p": float(sc["pvalue"]),
        "sigclust_ci_null_mean": float(sc["ci_null_mean"]),
        "consensus_chosen_k": int(cg["chosen_k"]),
        "consensus_per_k": {int(k): {kk: float(vv) for kk, vv in v.items()}
                            for k, v in cg["per_k"].items()},
        "jaccard_k": kj,
        "jaccard": [float(v) for v in jac],
        "jaccard_min": float(np.min(jac)),
        "abstains": bool(cg["chosen_k"] == 1 or sc["pvalue"] >= 0.05),
    }


def cv_block(Z, tag_dims=(2, 4, 10), do_null=True):
    """balanced_cv per factor on the top-d components, plus the shuffled-label null."""
    res, nul = {}, {}
    rng = np.random.default_rng(0)
    for d in tag_dims:
        if Z.shape[1] < d:
            continue
        Xd = Z[:, :d]
        res[f"top{d}"] = {}
        nul[f"top{d}"] = {}
        for f in C.FACTORS:
            a, s = bcv(Xd, labels[f])
            res[f"top{d}"][f] = [float(a), float(s)]
            if do_null:
                a0, s0 = bcv(Xd, rng.permutation(labels[f]))
                nul[f"top{d}"][f] = [float(a0), float(s0)]
    return res, nul


def dip_ranked_view(S10, n_rank=10):
    """The same PCA morphospace, but with the 2 axes chosen by Hartigan dip instead of variance.

    This is still FULLY UNSUPERVISED — the dip statistic of a component is a property of that
    component alone and never sees a label. It tests a concrete claim about the module's
    panel: that plotting PC1 vs PC2 (the highest-variance pair) is the wrong default when the
    biology of interest is a *split*, because variance ranks axes by spread, not by gap.
    """
    import diptest as _dt
    k = min(n_rank, S10.shape[1])
    dips = np.array([_dt.dipstat(np.ascontiguousarray(S10[:, j])) for j in range(k)])
    pick = np.sort(np.argsort(dips)[::-1][:2])
    X2 = S10[:, pick]
    kstar, lab, bic = gmm_bic_select(X2)
    out = {"picked_components": [int(v + 1) for v in pick],
           "dip_per_component": [float(v) for v in dips]}
    out.update(cluster_scores(X2, lab, kstar))
    out["ari_belly_x_tail_at_k4"] = float(adjusted_rand_score(y_bt, fit_gmm(X2, 4).predict(X2)))
    out["cv_top2"] = {f: [float(a) for a in bcv(X2, labels[f])] for f in C.FACTORS}
    out["gate"] = gate_report(X2, kstar)
    return out, X2, lab


if ADDENDUM_ONLY:
    # Post-hoc blocks that need no descriptor rebuild: `reference` is a pure function of the
    # ground-truth labels, `dip_ranked` reuses the PCA scores already saved to the npz.
    ref = reference_ceilings()
    Zarr = dict(np.load(OUT_NPZ, allow_pickle=True))
    dipr = {}
    for fname in [f for f in FEAT_ORDER if f"scores10__{f}" in Zarr]:
        d, X2, lab = dip_ranked_view(Zarr[f"scores10__{fname}"])
        dipr[fname] = d
        Zarr[f"X2dip__{fname}"] = X2
        Zarr[f"gmmdip__{fname}"] = lab
        log(f"  dip-ranked {fname}: PCs {d['picked_components']} k*={d['chosen_k']} "
            f"ARI4={d['ari_belly_x_tail']:.3f} (default PC1-2 gave "
            f"{json.load(open(OUT_JSON))['combos'][fname + '|PCA10']['gmm']['ari_belly_x_tail']:.3f})")
    with open(OUT_JSON) as fh:
        prev = json.load(fh)
    prev["reference"] = py(ref)
    prev["dip_ranked"] = py(dipr)
    prev.setdefault("meta", {})["addendum_added_post_hoc"] = [
        "reference", "dip_ranked",
        "computed from the saved PCA scores / ground-truth labels; no descriptor was rebuilt"]
    with open(OUT_JSON, "w") as fh:
        json.dump(prev, fh, indent=2)
    np.savez_compressed(OUT_NPZ, **Zarr)
    print("\n--- G. dip-ranked vs variance-ranked 2-D morphospace (unsupervised axis choice) ---")
    print(f"{'feature':22s} {'PCs':>8s} {'k*':>3s} {'ARI4':>6s} {'ARI_belly':>9s} "
          f"{'ARI_tail':>8s} {'sil':>6s} | default PC1-2 ARI4")
    for fname, d in dipr.items():
        base = prev["combos"][f"{fname}|PCA10"]["gmm"]
        print(f"{fname:22s} {str(d['picked_components']):>8s} {d['chosen_k']:3d} "
              f"{d['ari_belly_x_tail']:6.3f} {d['ari_belly']:9.3f} {d['ari_tail']:8.3f} "
              f"{d['silhouette']:6.3f} | {base['ari_belly_x_tail']:.3f}")
    log(f"merged `reference` + `dip_ranked` into {OUT_JSON}")
    raise SystemExit(0)


# --------------------------------------------------------------------------- run combos
results = {"meta": {}, "features": {}, "combos": {}, "dip": {}, "small_n": {}}
npz: dict[str, np.ndarray] = {
    "param_names": np.array(PARAMS),
    "factor_names": np.array(list(C.FACTORS)),
    "y_belly_x_tail": y_bt, "y_belly_x_tail_x_stripe": y_bts, "donor": donor,
    **{f"y_{f}": labels[f] for f in C.FACTORS},
    "specimen_names": np.array(names),
}

# ---- full-descriptor supervised ceiling (per feature, reducer-independent)
log("full-descriptor balanced_cv (this is the slow part) ...")
rng_null = np.random.default_rng(1)
for fname in FEAT_ORDER:
    X = FEATS[fname]
    t = time.time()
    blk, nblk = {}, {}
    for f in (C.FACTORS[:1] if QUICK else C.FACTORS):
        a, s = bcv(X, labels[f])
        blk[f] = [float(a), float(s)]
        a0, s0 = bcv(X, rng_null.permutation(labels[f]))
        nblk[f] = [float(a0), float(s0)]
    results["features"][fname] = {"dim": int(X.shape[1]), "cv_full": blk, "cv_full_null": nblk}
    log(f"  {fname}: dim={X.shape[1]} " +
        " ".join(f"{f}={blk[f][0]:.3f}/null{nblk[f][0]:.3f}" for f in blk) +
        f"  ({time.time() - t:.0f}s)")

# ---- per (feature, reducer)
for fname in FEAT_ORDER:
    X = FEATS[fname]
    for rname, (fn, k) in REDUCERS.items():
        tag = f"{fname}|{rname}"
        t = time.time()
        Z, info = fn(X, k)
        entry: dict = {"feature": fname, "reducer": rname, "n_components": int(Z.shape[1])}
        if rname == "PCA10":
            evr = info["evr"]
            entry["evr_pc1"] = float(evr[0])
            entry["evr_pc2"] = float(evr[1])
            entry["evr_pc1_pc2"] = float(evr[:2].sum())
            entry["evr_cum10"] = float(evr.sum())
            entry["evr"] = [float(v) for v in evr]
        if rname == "ICA10":
            entry["ica_variance_weight"] = [float(v) for v in info["weight"]]

        # PC x parameter |Pearson r|
        ncomp = min(6, Z.shape[1])
        Rtab = pearson_table(Z[:, :ncomp], P)
        npz[f"corr__{fname}__{rname}"] = Rtab
        eta = np.array([corr_ratio(Z[:, j], donor) for j in range(ncomp)])
        npz[f"donor_eta__{fname}__{rname}"] = eta
        entry["donor_eta_per_comp"] = [float(v) for v in eta]
        entry["top_param_per_comp"] = [
            [PARAMS[int(np.argmax(Rtab[j]))], float(Rtab[j].max())] for j in range(ncomp)]
        gcols = [PARAMS.index(c) for c in GEN_PARAMS]
        entry["top_generator_param_per_comp"] = [
            [GEN_PARAMS[int(np.argmax(Rtab[j, gcols]))], float(Rtab[j, gcols].max())]
            for j in range(ncomp)]
        entry["max_abs_r_per_param"] = {PARAMS[i]: float(Rtab[:, i].max()) for i in range(len(PARAMS))}
        # how much of the leading 2-D view is donor shape rather than pigment
        entry["donor_eta_pc1_pc2_max"] = float(eta[:2].max())

        # supervised readout on the leading components
        cv, cvn = cv_block(Z)
        entry["cv"] = cv
        entry["cv_null"] = cvn

        # auto-clustering on the 2-D morphospace exactly as the panel shows it
        X2 = Z[:, :2]
        kstar, glab, bics = gmm_bic_select(X2)
        entry["gmm"] = cluster_scores(X2, glab, kstar)
        entry["gmm"]["bic"] = [float(b) for b in bics]
        # what a user who FORCES the true number of groups would get
        gm4 = fit_gmm(X2, 4)
        entry["gmm"]["ari_belly_x_tail_at_k4"] = float(adjusted_rand_score(y_bt, gm4.predict(X2)))
        npz[f"X2__{fname}__{rname}"] = X2
        npz[f"gmm__{fname}__{rname}"] = glab
        npz[f"bic__{fname}__{rname}"] = bics
        if rname == "PCA10":
            npz[f"scores10__{fname}"] = Z

        entry["gate"] = gate_report(X2, kstar)
        entry["seconds"] = float(time.time() - t)
        results["combos"][tag] = entry
        log(f"  {tag}: k*={kstar} ARI(b x t)={entry['gmm']['ari_belly_x_tail']:.3f} "
            f"sil={entry['gmm']['silhouette']:.3f} sigclust_p={entry['gate']['sigclust_p']:.4f} "
            f"cons_k={entry['gate']['consensus_chosen_k']} ({entry['seconds']:.0f}s)")

# --------------------------------------------------------------------------- dip test
# Hartigan dip on PC1..PC6 vs a COLUMN-SHUFFLED null (destroys inter-feature covariance,
# keeps every marginal). Done for the PCA reducer only — refitting UMAP/FastICA 200x is
# not affordable and the spec asks about "PC1..PC6".
import diptest                                                            # noqa: E402

log("Hartigan dip test vs column-shuffled null (PCA only) ...")
for fname in FEAT_ORDER:
    X = FEATS[fname]
    t = time.time()
    Z, _ = reduce_pca(X, 10)
    dips = np.array([diptest.dipstat(np.ascontiguousarray(Z[:, j])) for j in range(6)])
    pana = np.array([diptest.diptest(np.ascontiguousarray(Z[:, j]))[1] for j in range(6)])
    rng = np.random.default_rng(7)
    null = np.zeros((N_DIP_SHUFFLE, 6))
    for b in range(N_DIP_SHUFFLE):
        Xs = rng.permuted(X, axis=0)
        Zs, _ = reduce_pca(Xs, 10)
        for j in range(6):
            null[b, j] = diptest.dipstat(np.ascontiguousarray(Zs[:, j]))
    p95 = np.percentile(null, 95, axis=0)
    results["dip"][fname] = {
        "dip": [float(v) for v in dips],
        "p_analytic": [float(v) for v in pana],
        "shuffle_null_p95": [float(v) for v in p95],
        "exceeds_null_p95": [bool(v) for v in (dips > p95)],
        "empirical_p": [float((1 + (null[:, j] >= dips[j]).sum()) / (N_DIP_SHUFFLE + 1))
                        for j in range(6)],
        "n_shuffles": N_DIP_SHUFFLE,
    }
    npz[f"dip_null__{fname}"] = null
    log(f"  {fname}: dip={np.round(dips, 4)} p95={np.round(p95, 4)} "
        f"exceeds={list(dips > p95)} ({time.time() - t:.0f}s)")

# --------------------------------------------------------------------------- small-n
# A real study runs the module on the specimens it HAS. So for each draw we recompute the
# descriptor from scratch on the n sampled specimens (not a slice of the 250-specimen
# palette) — that is what a user with n=25 would actually obtain.
log("small-n regime (recomputing the descriptor inside every draw) ...")
small_arrays = {}
for n in SMALL_N:
    rows = []
    for d in range(SMALL_DRAWS):
        rng = np.random.default_rng(1000 + 7 * d + n)
        idx = rng.choice(N, size=n, replace=False)
        sub = FaceColorData(colors=fcd.colors[idx], areas=fcd.areas,
                            centroids=fcd.centroids, names=[names[i] for i in idx])
        Xs = features.area_hist(sub, n_clusters=24)
        Zs, info = reduce_pca(Xs, min(10, n - 1))
        X2 = Zs[:, :2]
        kstar, glab, _ = gmm_bic_select(X2)
        gm4 = fit_gmm(X2, 4)
        with threadpool_limits(limits=1):
            sc = gating.sigclust(X2, n_sim=SIG_NSIM, seed=0)
            cg = gating.consensus_gate(X2, seed=0, **CONS_KW)
        rows.append({
            "n": n, "draw": d,
            "chosen_k": int(kstar),
            "ari_bt": float(adjusted_rand_score(y_bt[idx], glab)),
            "ari_bt_k4": float(adjusted_rand_score(y_bt[idx], gm4.predict(X2))),
            "ari_bts": float(adjusted_rand_score(y_bts[idx], glab)),
            "evr12": float(info["evr"][:2].sum()),
            "sigclust_p": float(sc["pvalue"]),
            "consensus_k": int(cg["chosen_k"]),
            "abstain_consensus": bool(cg["chosen_k"] == 1),
            "abstain_sigclust": bool(sc["pvalue"] >= 0.05),
            "abstain_either": bool(cg["chosen_k"] == 1 or sc["pvalue"] >= 0.05),
        })
        if d % 10 == 0:
            log(f"  n={n} draw {d}: k*={kstar} ari={rows[-1]['ari_bt']:.3f} "
                f"cons_k={cg['chosen_k']} sig_p={sc['pvalue']:.3f}")
    R = pd.DataFrame(rows)
    small_arrays[f"smalln_{n}_ari_bt"] = R["ari_bt"].to_numpy()
    small_arrays[f"smalln_{n}_ari_bt_k4"] = R["ari_bt_k4"].to_numpy()
    small_arrays[f"smalln_{n}_chosen_k"] = R["chosen_k"].to_numpy()
    small_arrays[f"smalln_{n}_sigclust_p"] = R["sigclust_p"].to_numpy()
    small_arrays[f"smalln_{n}_consensus_k"] = R["consensus_k"].to_numpy()
    results["small_n"][str(n)] = {
        "n_draws": SMALL_DRAWS,
        "ari_bt_mean": float(R["ari_bt"].mean()), "ari_bt_sd": float(R["ari_bt"].std(ddof=1)),
        "ari_bt_min": float(R["ari_bt"].min()), "ari_bt_max": float(R["ari_bt"].max()),
        "ari_bt_k4_mean": float(R["ari_bt_k4"].mean()),
        "ari_bt_k4_sd": float(R["ari_bt_k4"].std(ddof=1)),
        "ari_bts_mean": float(R["ari_bts"].mean()),
        "chosen_k_mean": float(R["chosen_k"].mean()),
        "chosen_k_hist": {int(k): int(v) for k, v in R["chosen_k"].value_counts().items()},
        "evr12_mean": float(R["evr12"].mean()),
        "abstain_rate_consensus": float(R["abstain_consensus"].mean()),
        "abstain_rate_sigclust": float(R["abstain_sigclust"].mean()),
        "abstain_rate_either": float(R["abstain_either"].mean()),
        "consensus_k_hist": {int(k): int(v) for k, v in R["consensus_k"].value_counts().items()},
    }
    log(f"  n={n}: ARI(b x t)={R['ari_bt'].mean():.3f}+-{R['ari_bt'].std(ddof=1):.3f} "
        f"abstain(either)={R['abstain_either'].mean():.2f}")
npz.update(small_arrays)

# reference: same headline analysis at full n, for the small-n comparison
ref = results["combos"][f"{HEADLINE_FEAT}|PCA10"]
results["small_n"]["full_250_reference"] = {
    "ari_bt": ref["gmm"]["ari_belly_x_tail"],
    "ari_bt_k4": ref["gmm"]["ari_belly_x_tail_at_k4"],
    "chosen_k": ref["gmm"]["chosen_k"],
    "abstains": ref["gate"]["abstains"],
}

# --------------------------------------------------------------------------- save
results["reference"] = reference_ceilings()
log("dip-ranked (instead of variance-ranked) 2-D morphospace ...")
results["dip_ranked"] = {}
for fname in FEAT_ORDER:
    d, X2d, labd = dip_ranked_view(npz[f"scores10__{fname}"])
    results["dip_ranked"][fname] = d
    npz[f"X2dip__{fname}"] = X2d
    npz[f"gmmdip__{fname}"] = labd
    log(f"  {fname}: PCs {d['picked_components']} k*={d['chosen_k']} "
        f"ARI4={d['ari_belly_x_tail']:.3f}")

results["meta"] = {
    "n_specimens": int(N),
    "n_atlas_faces": int(fcd.colors.shape[1]),
    "headline_feature": HEADLINE_FEAT,
    "features": FEAT_ORDER,
    "reducers": list(REDUCERS),
    "params": PARAMS,
    "generator_params": GEN_PARAMS,
    "donor_proxy_params": DONOR_PROXY,
    "gmm": "GaussianMixture(covariance_type='full', n_init=5, random_state=0), BIC over k=1..8",
    "impute_artifacts": False,
    "n_dip_shuffles": N_DIP_SHUFFLE,
    "small_n_draws": SMALL_DRAWS,
    "chance_balanced_accuracy": 0.5,
    "seconds": None,
}
results["meta"]["seconds"] = float(time.time() - T0)

with open(OUT_JSON, "w") as fh:
    json.dump(py(results), fh, indent=2)
np.savez_compressed(OUT_NPZ, **npz)
log(f"wrote {OUT_JSON} and {OUT_NPZ}")

# --------------------------------------------------------------------------- summary
def fmt(x, w=6, p=3):
    return " " * w if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:{w}.{p}f}"


print("\n" + "=" * 118)
print("E3  ColorAtlas built-in exploratory analysis on the 250-specimen atlas output")
print("=" * 118)

print("\n--- A. supervised ceiling of each descriptor (balanced acc, 5-fold x 4; chance .500) ---")
print(f"{'descriptor':24s} {'dim':>6s} " + " ".join(f"{f:>16s}" for f in C.FACTORS))
for fname in FEAT_ORDER:
    e = results["features"][fname]
    nan2 = [float("nan"), float("nan")]
    cells = " ".join(f"{e['cv_full'].get(f, nan2)[0]:8.3f}/{e['cv_full_null'].get(f, nan2)[0]:<7.3f}"
                     for f in C.FACTORS)
    print(f"{fname:24s} {e['dim']:6d} {cells}")
print("                                  (value / shuffled-label null)")

print("\n--- B. 2-D morphospace: variance kept, auto-clustering, and the gate ---")
hdr = (f"{'feature':22s} {'red':6s} {'evr12':>6s} {'k*':>3s} {'ARI4':>6s} {'ARI8':>6s} "
       f"{'ARI@k4':>7s} {'sil':>6s} {'sigP':>7s} {'consK':>6s} {'minJac':>7s} {'abst':>5s}")
print(hdr)
for fname in FEAT_ORDER:
    for rname in REDUCERS:
        e = results["combos"][f"{fname}|{rname}"]
        g, ga = e["gmm"], e["gate"]
        ev = e.get("evr_pc1_pc2")
        print(f"{fname:22s} {rname:6s} {fmt(ev):>6s} {g['chosen_k']:3d} "
              f"{g['ari_belly_x_tail']:6.3f} {g['ari_belly_x_tail_x_stripe']:6.3f} "
              f"{g['ari_belly_x_tail_at_k4']:7.3f} {g['silhouette']:6.3f} "
              f"{ga['sigclust_p']:7.4f} {ga['consensus_chosen_k']:6d} {ga['jaccard_min']:7.3f} "
              f"{str(ga['abstains']):>5s}")

print("\n--- C. balanced accuracy on the leading components (value / shuffled null) ---")
print(f"{'feature':22s} {'red':6s} {'dims':>5s} " + " ".join(f"{f:>16s}" for f in C.FACTORS))
for fname in FEAT_ORDER:
    for rname in REDUCERS:
        e = results["combos"][f"{fname}|{rname}"]
        for d in ("top2", "top4", "top10"):
            if d not in e["cv"]:
                continue
            cells = " ".join(f"{e['cv'][d][f][0]:8.3f}/{e['cv_null'][d][f][0]:<7.3f}"
                             for f in C.FACTORS)
            print(f"{fname:22s} {rname:6s} {d:>5s} {cells}")

print("\n--- D. what each PC actually tracks (|Pearson r|, PCA) ---")
for fname in FEAT_ORDER:
    e = results["combos"][f"{fname}|PCA10"]
    tops = ", ".join(f"PC{j+1}:{p}({r:.2f})" for j, (p, r) in enumerate(e["top_param_per_comp"]))
    gtops = ", ".join(f"PC{j+1}:{p}({r:.2f})"
                      for j, (p, r) in enumerate(e["top_generator_param_per_comp"]))
    print(f"{fname:22s} best overall : {tops}")
    print(f"{'':22s} best COLOUR  : {gtops}")
    print(f"{'':22s} donor eta per PC = {np.round(e['donor_eta_per_comp'], 2)}")

print("\n--- D2. how much of the belly x tail target is recoverable at all ---")
_r = results["reference"]
for c, v in _r["gt_marginal_dip"].items():
    print(f"  dip on the GENERATING parameter {c:16s} = {v['dip']:.4f}  p={v['p']:.3g}"
          + ("   <- genuinely bimodal" if v["p"] < 0.05 else "   <- unimodal, no real split"))
print(f"  ARI vs belly x tail of a PERFECT belly-only split : "
      f"{_r['ari_vs_belly_x_tail']['perfect_belly_only']:.3f}")
print(f"  ARI vs belly x tail of a PERFECT tail-only  split : "
      f"{_r['ari_vs_belly_x_tail']['perfect_tail_only']:.3f}")
print(f"  ARI vs belly x tail of random 4-way labels        : "
      f"{_r['ari_vs_belly_x_tail']['random_4way']:.3f}")
print(f"  ORACLE (GMM+BIC on the generator's own belly/tail hues): "
      f"k*={_r['oracle_on_true_hues']['gmm_bic_chosen_k']} "
      f"ARI={_r['oracle_on_true_hues']['ari_gmm']:.3f} "
      f"(forced k=4 kmeans ARI={_r['oracle_on_true_hues']['ari_kmeans_k4']:.3f})")

print("\n--- E. Hartigan dip on PC1..PC6 vs column-shuffled null (200 shuffles) ---")
print(f"{'feature':22s} " + " ".join(f"{'PC'+str(j+1):>14s}" for j in range(6)))
for fname in FEAT_ORDER:
    d = results["dip"][fname]
    cells = " ".join(f"{d['dip'][j]:6.4f}{'*' if d['exceeds_null_p95'][j] else ' '}/{d['shuffle_null_p95'][j]:6.4f}"
                     for j in range(6))
    print(f"{fname:22s} {cells}")
print("  ( * = dip exceeds the 95th percentile of the column-shuffled null )")

print("\n--- F. small-n regime, headline morphospace (area_hist K=24 -> PCA 2-D) ---")
print(f"{'n':>5s} {'ARI(b x t)':>12s} {'ARI@k=4':>12s} {'k* mean':>8s} {'evr12':>6s} "
      f"{'abst_cons':>10s} {'abst_sig':>9s} {'abst_either':>12s}")
for n in SMALL_N:
    s = results["small_n"][str(n)]
    print(f"{n:5d} {s['ari_bt_mean']:6.3f}+-{s['ari_bt_sd']:<5.3f} "
          f"{s['ari_bt_k4_mean']:6.3f}+-{s['ari_bt_k4_sd']:<5.3f} {s['chosen_k_mean']:8.2f} "
          f"{s['evr12_mean']:6.3f} {s['abstain_rate_consensus']:10.2f} "
          f"{s['abstain_rate_sigclust']:9.2f} {s['abstain_rate_either']:12.2f}")
r = results["small_n"]["full_250_reference"]
print(f"{250:5d} {r['ari_bt']:6.3f}       {r['ari_bt_k4']:6.3f}       {r['chosen_k']:8d}")

print("\n--- G. dip-ranked vs variance-ranked 2-D morphospace (unsupervised axis choice) ---")
print(f"{'feature':22s} {'PCs':>8s} {'k*':>3s} {'ARI4':>6s} {'ARI_belly':>9s} "
      f"{'ARI_tail':>8s} {'sil':>6s} | default PC1-2 ARI4")
for fname in FEAT_ORDER:
    d = results["dip_ranked"][fname]
    base = results["combos"][f"{fname}|PCA10"]["gmm"]
    print(f"{fname:22s} {str(d['picked_components']):>8s} {d['chosen_k']:3d} "
          f"{d['ari_belly_x_tail']:6.3f} {d['ari_belly']:9.3f} {d['ari_tail']:8.3f} "
          f"{d['silhouette']:6.3f} | {base['ari_belly_x_tail']:.3f}")
print(f"\ntotal {results['meta']['seconds']:.0f}s")
