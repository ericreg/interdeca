"""E6 — negative results and honest limits.

An explicit, quantitative account of what does NOT work on the ColorAtlas output of the
250-specimen synthetic cichlid benchmark. Six independent probes:

  (1) UNSUPERVISED DISCOVERY WALL — GMM+BIC ARI on the top-2 PCs, best-single-PC AUC over
      PC1..PC10, Hartigan dip of the most bimodal axis vs a column-shuffled null, and
      HDBSCAN cluster purity for the two rarest/subtlest classes (cheeks, 5 stripes).
  (2) ICA IS NOT A LEVER — FastICA(whiten) spans exactly the PCA subspace, so a linear
      classifier and a full-covariance GMM cannot tell them apart; measured by per-factor
      balanced accuracy, principal angles, projector residual, and single-component AUC.
  (3) SEGMENT-THEN-SPECTRAL — projecting an 8-colour one-hot indicator field onto the atlas
      Laplacian eigenbasis, versus continuous spectral coefficients and segment+spatial.
  (4) EXPLAINED VARIANCE IS THE WRONG STOPPING RULE — accuracy-vs-#components sweeps against
      the 90/95/99% cumulative-explained-variance dimensions.
  (5) BAKE ARTIFACTS AND ROBUSTNESS — face exclusion, median imputation, and Gaussian colour
      noise at 2/5/10 dE.
  (6) SMALL-n COLLAPSE — n in {25,40,80,150,250}, 30 draws each, plus how often an
      unsupervised dip gate would have claimed clusters.

Ground truth is used for VALIDATION ONLY.
"""
from __future__ import annotations

import json
import os
import time
import warnings

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "4")           # this box runs several experiments in parallel

import numpy as np
import scipy.sparse as sp
from scipy.linalg import subspace_angles
from sklearn.decomposition import PCA, FastICA
from sklearn.metrics import adjusted_rand_score, roc_auc_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

import common as C
from fishpipe import features, segment, spatial, spectral, textons
from fishpipe.features import rgb_to_lab

warnings.filterwarnings("ignore")
T0 = time.time()
SEED = 0
NAME = "e6_negative_results"
DESC_CACHE = C.CACHE / f"{NAME}_desc.npz"
# QUICK=1 runs the identical code path with tiny loop counts — a smoke test only, never a
# reportable result (the JSON it writes is overwritten by the full run).
QUICK = os.environ.get("E6_QUICK") == "1"
OUT_NAME = NAME + ("_QUICK" if QUICK else "")     # a smoke test must never overwrite results
PART = C.CACHE / (f"{NAME}_partial{'_quick' if QUICK else ''}.json")
PART_A = C.CACHE / (f"{NAME}_partial{'_quick' if QUICK else ''}_arrays.npz")


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


def save_partial(res, arrays):
    """Checkpoint after every item so a killed run resumes instead of restarting."""
    PART.write_text(json.dumps(jsonable(res), indent=2))
    np.savez_compressed(PART_A, **arrays)
    log("checkpoint saved:", sorted(k for k in res if k.startswith("item")))


def load_partial():
    if not PART.exists():
        return {}, {}
    r = json.loads(PART.read_text())
    a = {}
    if PART_A.exists():
        z = np.load(PART_A, allow_pickle=True)
        a = {k: z[k] for k in z.files}
    return r, a


def getk(d, k):
    """Dict lookup tolerant of the int->str key change a JSON checkpoint round-trip makes."""
    return d[k] if k in d else d[str(k)]


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, (np.floating, float)):
        v = float(o)
        return None if not np.isfinite(v) else v
    if isinstance(o, (np.integer, int)):
        return int(o)
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return jsonable(o.tolist())
    return o


# =====================================================================================
# fast, cache-free re-implementations (needed because fishpipe's vertex_lab is DISK-CACHED
# and would silently return the CLEAN atlas signal for perturbed colour tensors)
# =====================================================================================
def face_to_vertex_matrix(mesh, areas, face_mask=None) -> sp.csr_matrix:
    """(Nv, Nf) row-normalised area-weighted face->vertex averaging operator."""
    fv = mesh.face_v
    nv = mesh.vertices.shape[0]
    nf = fv.shape[0]
    keep = np.ones(nf, bool) if face_mask is None else np.asarray(face_mask, bool)
    fid = np.repeat(np.arange(nf), 3)
    vid = fv.ravel()
    w = np.repeat(areas, 3)
    m = keep[fid]
    W = sp.coo_matrix((w[m], (vid[m], fid[m])), shape=(nv, nf)).tocsr()
    rs = np.asarray(W.sum(1)).ravel()
    rs[rs == 0] = 1.0
    return sp.diags(1.0 / rs) @ W


def spectral_from_lab(LAB, W, Uk) -> np.ndarray:
    """Replicates spectral.spectral_coeffs(channels=('L','a','b')) from a Lab face tensor."""
    N = LAB.shape[0]
    k = Uk.shape[1]
    out = np.zeros((N, 3 * k))
    for i in range(N):
        V = W @ LAB[i]                                    # (Nv, 3)
        V = V - V.mean(0, keepdims=True)
        c = V.T @ Uk                                      # (3, k)
        out[i] = c.reshape(-1)
    return out


def area_hist_from_lab(LAB, areas, n_clusters=24, seed=0) -> np.ndarray:
    """Replicates features.area_hist(color_space='lab', unit_norm=True) from a Lab tensor."""
    from sklearn.cluster import KMeans
    N, Nf, _ = LAB.shape
    rng = np.random.default_rng(seed)
    pooled = LAB.reshape(N * Nf, 3)
    fit_idx = rng.choice(pooled.shape[0], size=min(200_000, pooled.shape[0]), replace=False)
    km = KMeans(n_clusters=n_clusters, n_init=5, random_state=seed).fit(pooled[fit_idx])
    X = np.zeros((N, n_clusters))
    for i in range(N):
        lab = km.predict(LAB[i])
        X[i] = np.bincount(lab, weights=areas, minlength=n_clusters)
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n


# =====================================================================================
# generic evaluation helpers
# =====================================================================================
def cv(X, y):
    return C.balanced_cv(X, y)


def cv_null(X, y, n_shuffle=5, seed=0):
    n_shuffle = 2 if QUICK else n_shuffle
    rng = np.random.default_rng(seed)
    s = [C.balanced_cv(X, rng.permutation(y))[0] for _ in range(n_shuffle)]
    return float(np.mean(s)), float(np.std(s))


def factor_row(X, labels, with_null=False):
    out = {}
    for f in C.FACTORS:
        a, sd = cv(X, labels[f])
        out[f] = {"acc": a, "sd": sd}
        if with_null:
            na, nsd = cv_null(X, labels[f])
            out[f]["null"] = na
            out[f]["null_sd"] = nsd
    return out


def pc_scores(X, n=10, seed=0):
    Z = StandardScaler().fit_transform(np.asarray(X, float))
    n = int(min(n, min(Z.shape)))
    p = PCA(n_components=n, random_state=seed).fit(Z)
    return p.transform(Z), p


def auc_abs(y, s):
    a = roc_auc_score(y, s)
    return max(a, 1 - a)


def best_single_axis_auc(S, y):
    aucs = [auc_abs(y, S[:, j]) for j in range(S.shape[1])]
    j = int(np.argmax(aucs))
    return float(aucs[j]), j, [float(v) for v in aucs]


def max_dip(S):
    import diptest
    d = [diptest.diptest(np.ascontiguousarray(S[:, j], dtype=float)) for j in range(S.shape[1])]
    dips = np.array([x[0] for x in d])
    pvals = np.array([x[1] for x in d])
    j = int(np.argmax(dips))
    return float(dips[j]), float(pvals[j]), j, dips


def column_shuffle(X, rng):
    """Permute every feature column independently -> destroys all between-feature
    covariance while preserving each marginal (the team's existing dip null)."""
    return rng.permuted(np.asarray(X, float), axis=0)


# =====================================================================================
def main():
    log("loading data")
    fcd, art, mesh = C.atlas_data(force=False, impute=False)
    df, labels = C.load_gt(list(fcd.names))
    A = C.face_adjacency(mesh, C.ATLAS_DS.cache_dir)
    N, Nf, _ = fcd.colors.shape
    areas = fcd.areas
    res, arrays = load_partial()
    if res:
        log("resuming from checkpoint with", sorted(k for k in res if k.startswith("item")))
    res["meta"] = {"n_specimens": int(N), "n_faces": int(Nf),
                   "class_counts": {f: np.bincount(labels[f]).tolist() for f in C.FACTORS},
                   "chance": 0.5}

    log("Lab tensor")
    LAB = rgb_to_lab(fcd.colors)                                     # (N, Nf, 3) float64
    Wfv = face_to_vertex_matrix(mesh, areas)
    w_eig, U = spectral.laplacian_eigenbasis(300, mesh=mesh, cache_dir=C.ATLAS_DS.cache_dir)

    # ---------------- descriptors (cached) ----------------
    if DESC_CACHE.exists():
        z = np.load(DESC_CACHE)
        D = {k: z[k] for k in z.files}
        log("descriptors from cache:", list(D))
    else:
        D = {}
        t = time.time(); D["area_hist24"] = features.area_hist(fcd, n_clusters=24)
        log("area_hist24", D["area_hist24"].shape, f"{time.time()-t:.0f}s")
        t = time.time(); D["area_hist64"] = features.area_hist(fcd, n_clusters=64)
        log("area_hist64", D["area_hist64"].shape, f"{time.time()-t:.0f}s")
        t = time.time()
        D["spec40"] = spectral.spectral_coeffs(fcd, k=40, mesh=mesh,
                                               cache_dir=C.ATLAS_DS.cache_dir)
        log("spec40", D["spec40"].shape, f"{time.time()-t:.0f}s")
        t = time.time()
        D["spec100"] = spectral.spectral_coeffs(fcd, k=100, mesh=mesh,
                                                cache_dir=C.ATLAS_DS.cache_dir)
        log("spec100", D["spec100"].shape, f"{time.time()-t:.0f}s")
        t = time.time()
        D["texton_bow"] = textons.texton_descriptor(fcd, K=128, mode="bow",
                                                    cache_dir=C.ATLAS_DS.cache_dir)
        log("texton_bow", D["texton_bow"].shape, f"{time.time()-t:.0f}s")
        np.savez_compressed(DESC_CACHE, **D)

    # sanity: my cache-free re-implementations must reproduce fishpipe on CLEAN data
    chk_spec = spectral_from_lab(LAB, Wfv, U[:, :40].astype(np.float64))
    chk_ah = area_hist_from_lab(LAB, areas, 24, seed=0)
    rep = {
        "spectral_reimpl_max_abs_diff": float(np.abs(chk_spec - D["spec40"]).max()),
        "spectral_reimpl_rel": float(np.abs(chk_spec - D["spec40"]).max()
                                     / (np.abs(D["spec40"]).max() + 1e-12)),
        "area_hist_reimpl_max_abs_diff": float(np.abs(chk_ah - D["area_hist24"]).max()),
    }
    res["reimplementation_check"] = rep
    log("reimpl check", rep)

    MAIN = ["area_hist24", "spec40", "texton_bow"]
    if "headline" not in res:
        log("headline table")
        res["headline"] = {k: factor_row(D[k], labels, with_null=True) for k in MAIN}
        save_partial(res, arrays)
    for k in MAIN:
        log("  ", k, {f: round(res["headline"][k][f]["acc"], 3) for f in C.FACTORS})

    arrays = arrays if arrays else {}

    # =================================================================================
    # (1) UNSUPERVISED DISCOVERY WALL
    # =================================================================================
    if "item1_discovery_wall" in res:
        log("=== skipping item1_discovery_wall (already in checkpoint)")
    else:
        log("=== item 1: unsupervised discovery wall")
        item1 = {}
        B_SHUF = 5 if QUICK else 200
        for key in MAIN:
            X = D[key]
            S10, pca = pc_scores(X, 10, seed=SEED)
            entry = {"pca_evr_top10": [float(v) for v in pca.explained_variance_ratio_]}

            # --- GMM + BIC on the top-2 PCs
            S2 = S10[:, :2]
            bics = {}
            best = (np.inf, None)
            for k in range(1, 9):
                gm = GaussianMixture(k, covariance_type="full", n_init=10, random_state=SEED,
                                     reg_covar=1e-6).fit(S2)
                b = float(gm.bic(S2))
                bics[k] = b
                if b < best[0]:
                    best = (b, gm)
            gm = best[1]
            z = gm.predict(S2)
            entry["gmm_bic"] = {
                "bic_by_k": bics, "chosen_k": int(gm.n_components),
                "ari": {f: float(adjusted_rand_score(labels[f], z)) for f in C.FACTORS},
            }
            rng = np.random.default_rng(1)
            n_null_ari = 20 if QUICK else 200
            null_ari = {f: [float(adjusted_rand_score(rng.permutation(labels[f]), z))
                            for _ in range(n_null_ari)] for f in C.FACTORS}
            entry["gmm_bic"]["ari_null_mean"] = {f: float(np.mean(v)) for f, v in null_ari.items()}
            entry["gmm_bic"]["ari_null_p95"] = {f: float(np.percentile(v, 95))
                                                for f, v in null_ari.items()}

            # --- best single PC AUC over PC1..PC10 (+ label-shuffle null of the SAME max stat)
            rng = np.random.default_rng(2)
            entry["best_pc_auc"] = {}
            for f in C.FACTORS:
                a, j, allauc = best_single_axis_auc(S10, labels[f])
                nulls = [best_single_axis_auc(S10, rng.permutation(labels[f]))[0]
                         for _ in range(B_SHUF)]
                entry["best_pc_auc"][f] = {
                    "auc": a, "pc": int(j + 1), "per_pc": allauc,
                    "null_mean": float(np.mean(nulls)), "null_p95": float(np.percentile(nulls, 95)),
                    "p_perm": float((1 + np.sum(np.asarray(nulls) >= a)) / (B_SHUF + 1)),
                }

            # --- Hartigan dip of the most bimodal PC axis vs a column-shuffled null
            dip, dip_p_analytic, jdip, dips = max_dip(S10)
            rng = np.random.default_rng(3)
            nulls = []
            for _ in range(B_SHUF):
                Xs = column_shuffle(X, rng)
                Ss, _ = pc_scores(Xs, 10, seed=SEED)
                nulls.append(max_dip(Ss)[0])
            nulls = np.asarray(nulls)
            entry["dip"] = {
                "max_dip": dip, "axis_pc": int(jdip + 1), "p_analytic": dip_p_analytic,
                "per_pc_dip": [float(v) for v in dips],
                "null_mean": float(nulls.mean()), "null_p95": float(np.percentile(nulls, 95)),
                "p_colshuffle": float((1 + np.sum(nulls >= dip)) / (B_SHUF + 1)),
                "n_null": int(B_SHUF),
            }
            arrays[f"dipnull_{key}"] = nulls

            # --- HDBSCAN purity for cheeks and 5-stripe
            import hdbscan
            cl = hdbscan.HDBSCAN(min_cluster_size=10, min_samples=5).fit(S10)
            lab_h = cl.labels_
            n_clu = int(len(set(lab_h[lab_h >= 0])))
            entry["hdbscan"] = {"n_clusters": n_clu,
                                "min_cluster_size": 10,
                                "cluster_sizes": [int((lab_h == c).sum())
                                                  for c in sorted(set(lab_h[lab_h >= 0]))],
                                "noise_frac": float((lab_h < 0).mean()),
                                "ari": {f: float(adjusted_rand_score(labels[f], lab_h))
                                        for f in C.FACTORS}}
            for f in ("cheeks", "stripe"):
                y = labels[f]
                prev = float(y.mean())
                bestf1 = {"purity": prev, "recall": 1.0, "f1": 0.0, "cluster": None, "size": 0}
                max_pur = prev
                for c in sorted(set(lab_h[lab_h >= 0])):
                    m = lab_h == c
                    tp = int((y[m] == 1).sum())
                    prec = tp / max(m.sum(), 1)
                    rec = tp / max(int(y.sum()), 1)
                    f1 = 0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)
                    max_pur = max(max_pur, prec)
                    if f1 > bestf1["f1"]:
                        bestf1 = {"purity": float(prec), "recall": float(rec), "f1": float(f1),
                                  "cluster": int(c), "size": int(m.sum())}
                bestf1["prevalence_null_purity"] = prev
                bestf1["max_purity_any_cluster"] = float(max_pur)
                bestf1["enrichment_over_prevalence"] = float(max_pur / prev) if prev > 0 else None
                entry["hdbscan"][f] = bestf1
            item1[key] = entry
            arrays[f"pc10_{key}"] = S10
            log("  ", key, "GMM k=", entry["gmm_bic"]["chosen_k"],
                "ARI", {f: round(entry["gmm_bic"]["ari"][f], 2) for f in C.FACTORS},
                "dip p_shuf=", round(entry["dip"]["p_colshuffle"], 3))
        res["item1_discovery_wall"] = item1
        save_partial(res, arrays)

    # =================================================================================
    # (2) ICA IS NOT A LEVER
    # =================================================================================
    if "item2_ica" in res:
        log("=== skipping item2_ica (already in checkpoint)")
    else:
        log("=== item 2: ICA vs PCA")
        item2 = {}
        for key in ["spec40", "area_hist24"]:
            X = StandardScaler().fit_transform(np.asarray(D[key], float))
            sub = {}
            for n in ((2, 10) if QUICK else (2, 5, 10, 20, 30)):
                if n > min(X.shape):
                    continue
                p = PCA(n_components=n, random_state=SEED).fit(X)
                Sp = p.transform(X)
                ica = FastICA(n_components=n, whiten="unit-variance", random_state=SEED,
                              max_iter=2000, tol=1e-4).fit(X)
                Si = ica.transform(X)
                # subspace comparison: row space of components_ in both cases
                Bp = np.linalg.qr(p.components_.T)[0]
                Bi = np.linalg.qr(ica.components_.T)[0]
                ang = subspace_angles(Bp, Bi)
                Pp, Pi = Bp @ Bp.T, Bi @ Bi.T
                # PCA-whitened scores differ from the ICA sources by an exact ROTATION, so a
                # full-covariance GMM is mathematically equivariant between them; any partition
                # difference there is EM local optima, not information.
                Spw = Sp / Sp.std(0, keepdims=True)
                row = {
                    "max_principal_angle_deg": float(np.degrees(np.max(ang))),
                    "mean_principal_angle_deg": float(np.degrees(np.mean(ang))),
                    "projector_residual_fro": float(np.linalg.norm(Pp - Pi)),
                    "projector_residual_rel": float(np.linalg.norm(Pp - Pi) / np.sqrt(n)),
                    "acc_pca": {}, "acc_ica": {},
                    "gmm_ari_pca": {}, "gmm_ari_ica": {},
                    "best_comp_auc_pca": {}, "best_comp_auc_ica": {},
                    "single_auc_ica_minus_pca": {},
                }
                gmP = GaussianMixture(2, covariance_type="full", n_init=10, random_state=SEED).fit(Sp)
                gmI = GaussianMixture(2, covariance_type="full", n_init=10, random_state=SEED).fit(Si)
                gmW = GaussianMixture(2, covariance_type="full", n_init=10,
                                      random_state=SEED).fit(Spw)
                gp, gi, gw = gmP.predict(Sp), gmI.predict(Si), gmW.predict(Spw)
                row["gmm_pca_vs_ica_ari"] = float(adjusted_rand_score(gp, gi))
                row["gmm_pcawhitened_vs_ica_ari"] = float(adjusted_rand_score(gw, gi))
                # log-likelihoods are comparable only after the change-of-variables constant; we
                # report the raw values plus the Jacobian-corrected PCA value for transparency
                sign, logdet = np.linalg.slogdet(Sp.T @ Sp / Sp.shape[0])
                row["gmm_mean_loglik_pca"] = float(gmP.score(Sp))
                row["gmm_mean_loglik_ica"] = float(gmI.score(Si))
                row["gmm_mean_loglik_pca_whitened"] = float(gmW.score(Spw))
                row["logdet_pca_cov"] = float(logdet)
                for f in C.FACTORS:
                    row["acc_pca"][f] = cv(Sp, labels[f])[0]
                    row["acc_ica"][f] = cv(Si, labels[f])[0]
                    row["gmm_ari_pca"][f] = float(adjusted_rand_score(labels[f], gp))
                    row["gmm_ari_ica"][f] = float(adjusted_rand_score(labels[f], gi))
                    row["best_comp_auc_pca"][f] = best_single_axis_auc(Sp, labels[f])[0]
                    row["best_comp_auc_ica"][f] = best_single_axis_auc(Si, labels[f])[0]
                    row["single_auc_ica_minus_pca"][f] = float(row["best_comp_auc_ica"][f]
                                                               - row["best_comp_auc_pca"][f])
                row["max_abs_acc_diff"] = float(max(abs(row["acc_pca"][f] - row["acc_ica"][f])
                                                    for f in C.FACTORS))
                row["max_abs_single_auc_diff"] = float(
                    max(abs(row["best_comp_auc_pca"][f] - row["best_comp_auc_ica"][f])
                        for f in C.FACTORS))
                row["best_single_auc_ica_gain"] = float(max(row["single_auc_ica_minus_pca"][f]
                                                            for f in C.FACTORS))
                sub[n] = row
                log("  ", key, f"n={n}", "maxang(deg)=", round(row["max_principal_angle_deg"], 4),
                    "|dacc|max=", round(row["max_abs_acc_diff"], 3),
                    "|dAUC|max=", round(row["max_abs_single_auc_diff"], 3))
            item2[key] = sub
        res["item2_ica"] = item2
        save_partial(res, arrays)

    # =================================================================================
    # (3) SEGMENT-THEN-SPECTRAL
    # =================================================================================
    if "item3_segment_then_spectral" in res:
        log("=== skipping item3_segment_then_spectral (already in checkpoint)")
    else:
        log("=== item 3: segment-then-spectral")
        t = time.time()
        seg = segment.segment(fcd, n_colors=8, smooth_iters=0, mesh_adjacency=A)
        log("  segmentation done", f"{time.time()-t:.0f}s")
        K_SEG = 8
        K_EIG = 100
        Uk = U[:, :K_EIG].astype(np.float64)
        onehot_spec = np.zeros((N, K_SEG * K_EIG))
        seg_area = np.zeros((N, K_SEG))
        for i in range(N):
            F = np.zeros((Nf, K_SEG))
            F[np.arange(Nf), seg.labels[i]] = 1.0
            seg_area[i] = np.bincount(seg.labels[i], weights=areas, minlength=K_SEG)
            V = Wfv @ F                                        # (Nv, 8) indicator field
            V = V - V.mean(0, keepdims=True)
            onehot_spec[i] = (V.T @ Uk).reshape(-1)
        seg_area = seg_area / seg_area.sum(1, keepdims=True)

        spatial.principal_axis = lambda: C.principal_axis(mesh)   # atlas-safe (see guardrails)
        t = time.time()
        seg_spatial = spatial.spatial_descriptor(fcd, seg, mesh_adjacency=A)
        log("  spatial descriptor", seg_spatial.shape, f"{time.time()-t:.0f}s")

        item3 = {
            "segment_onehot_spectral_k100": factor_row(onehot_spec, labels, with_null=True),
            "segment_area_fractions": factor_row(seg_area, labels),
            "segment_spatial": factor_row(seg_spatial, labels, with_null=True),
            "continuous_spectral_k40": res["headline"]["spec40"],
            "continuous_spectral_k100": factor_row(D["spec100"], labels),
            "dims": {"segment_onehot_spectral_k100": int(onehot_spec.shape[1]),
                     "segment_area_fractions": int(seg_area.shape[1]),
                     "segment_spatial": int(seg_spatial.shape[1]),
                     "continuous_spectral_k40": int(D["spec40"].shape[1]),
                     "continuous_spectral_k100": int(D["spec100"].shape[1])},
        }
        item3["delta_vs_continuous_k100"] = {
            f: float(item3["segment_onehot_spectral_k100"][f]["acc"]
                     - item3["continuous_spectral_k100"][f]["acc"]) for f in C.FACTORS}
        res["item3_segment_then_spectral"] = item3
        arrays["seg_onehot_spectral"] = onehot_spec
        arrays["seg_spatial"] = seg_spatial
        for k in ("segment_onehot_spectral_k100", "continuous_spectral_k100", "segment_spatial"):
            log("  ", k, {f: round(item3[k][f]["acc"], 3) for f in C.FACTORS})
        save_partial(res, arrays)

    # =================================================================================
    # (4) EXPLAINED VARIANCE IS THE WRONG STOPPING RULE
    # =================================================================================
    if "item4_explained_variance" in res:
        log("=== skipping item4_explained_variance (already in checkpoint)")
    else:
        log("=== item 4: EV is the wrong stopping rule")
        item4 = {}
        for key in ["area_hist64", "spec40", "area_hist24"]:
            X = StandardScaler().fit_transform(np.asarray(D[key], float))
            # full spectrum for the explained-variance rule, sweep only the first `pmax`
            pfull = PCA(n_components=None, random_state=SEED).fit(X)
            cum = np.cumsum(pfull.explained_variance_ratio_)
            pmax = int(min(6 if QUICK else 60, len(cum)))
            S = pfull.transform(X)[:, :pmax]
            ev_dim = {str(int(q * 100)): (int(np.argmax(cum >= q) + 1) if (cum >= q).any() else None)
                      for q in (0.90, 0.95, 0.99)}
            curves = {f: [] for f in C.FACTORS}
            sds = {f: [] for f in C.FACTORS}
            for n in range(1, pmax + 1):
                for f in C.FACTORS:
                    a_, s_ = cv(S[:, :n], labels[f])
                    curves[f].append(a_)
                    sds[f].append(s_)
            sat = {}
            for f in C.FACTORS:
                a = np.asarray(curves[f])
                amax = float(a.max())
                n_at_max = int(np.argmax(a) + 1)
                d1 = int(np.argmax(a >= amax - 0.01) + 1)                       # within 1 pt
                sd_at_max = float(sds[f][n_at_max - 1])
                d_sd = int(np.argmax(a >= amax - sd_at_max) + 1)                # within 1 CV sd
                gap = amax - 0.5
                d90 = int(np.argmax((a - 0.5) >= 0.9 * gap) + 1) if gap > 0 else None
                sat[f] = {"acc_max": amax, "n_at_max": n_at_max, "sd_at_max": sd_at_max,
                          "sat_dim_within_1pt": d1, "sat_dim_within_1sd": d_sd,
                          "sat_dim_90pct_of_gap": d90,
                          "cum_evr_at_sat_dim": float(cum[d1 - 1]),
                          "acc_at_ev90": (float(a[ev_dim["90"] - 1])
                                          if ev_dim["90"] and ev_dim["90"] <= pmax else None),
                          "acc_at_ev95": (float(a[ev_dim["95"] - 1])
                                          if ev_dim["95"] and ev_dim["95"] <= pmax else None),
                          "acc_at_ev99": (float(a[ev_dim["99"] - 1])
                                          if ev_dim["99"] and ev_dim["99"] <= pmax else None)}
            item4[key] = {"n_features": int(X.shape[1]), "sweep_max": pmax,
                          "cum_evr": [float(v) for v in cum], "ev_dim": ev_dim,
                          "acc_curve": {f: [float(v) for v in curves[f]] for f in C.FACTORS},
                          "sd_curve": {f: [float(v) for v in sds[f]] for f in C.FACTORS},
                          "saturation": sat,
                          "mismatch_ev95_over_sat": {
                              f: (float(ev_dim["95"] / sat[f]["sat_dim_within_1pt"])
                                  if ev_dim["95"] else None) for f in C.FACTORS}}
            arrays[f"sweep_{key}"] = np.array([curves[f] for f in C.FACTORS])
            arrays[f"cumevr_{key}"] = cum
            log("  ", key, "EV dims", ev_dim, "sat",
                {f: sat[f]["sat_dim_within_1pt"] for f in C.FACTORS})
        res["item4_explained_variance"] = item4
        save_partial(res, arrays)

    # =================================================================================
    # (5) BAKE ARTIFACTS AND ROBUSTNESS
    # =================================================================================
    if "item5_artifacts_robustness" in res:
        log("=== skipping item5_artifacts_robustness (already in checkpoint)")
    else:
        log("=== item 5: artifacts and robustness")
        item5 = {}
        e1p = C.RESULTS / "e1_fidelity.json"
        e1a = C.RESULTS / "e1_fidelity_arrays.npz"

        def headline_from_lab(lab_tensor, face_keep=None, tag=""):
            if face_keep is None:
                ah = area_hist_from_lab(lab_tensor, areas, 24, seed=0)
                W = Wfv
            else:
                ah = area_hist_from_lab(lab_tensor[:, face_keep, :], areas[face_keep], 24, seed=0)
                # excluded faces get zero weight in the face->vertex operator, so their colour
                # never enters the spectral projection (no copy of the 300 MB tensor needed)
                W = face_to_vertex_matrix(mesh, areas, face_mask=face_keep)
            sp40 = spectral_from_lab(lab_tensor, W, U[:, :40].astype(np.float64))
            return {"area_hist24": factor_row(ah, labels), "spec40": factor_row(sp40, labels)}

        # baseline recomputed through the SAME cache-free code path used for every perturbation,
        # so the deltas below cannot be contaminated by an implementation difference
        base = headline_from_lab(LAB)
        item5["baseline"] = {k: {f: base[k][f]["acc"] for f in C.FACTORS} for k in base}
        item5["baseline_fishpipe"] = {k: {f: res["headline"][k][f]["acc"] for f in C.FACTORS}
                                      for k in ("area_hist24", "spec40")}
        item5["baseline_cv_sd"] = {k: {f: base[k][f]["sd"] for f in C.FACTORS} for k in base}

        def cond_block(r):
            """acc / cv-sd / delta-vs-baseline block for one perturbation condition."""
            return {
                "acc": {k: {f: r[k][f]["acc"] for f in C.FACTORS} for k in r},
                "acc_cv_sd": {k: {f: r[k][f]["sd"] for f in C.FACTORS} for k in r},
                "delta": {k: {f: float(r[k][f]["acc"] - base[k][f]["acc"]) for f in C.FACTORS}
                          for k in r},
                # None where the baseline CV sd is exactly 0 (belly is perfectly separated
                # in every fold) — a delta cannot be expressed in units of zero
                "delta_in_baseline_sd_units": {
                    k: {f: (float((r[k][f]["acc"] - base[k][f]["acc"]) / base[k][f]["sd"])
                            if base[k][f]["sd"] > 1e-6 else None)
                        for f in C.FACTORS} for k in r},
            }

        # ---- (a) exclude faces flagged as bake holes by E1 (reference-based ground truth)
        if e1a.exists() and e1p.exists():
            z1 = np.load(e1a, allow_pickle=True)
            e1j = json.loads(e1p.read_text())
            fidx = z1["face_idx"]
            holes = z1["holes"]
            hole_faces = np.unique(fidx[holes.any(0)])
            keep = np.ones(Nf, bool)
            keep[hole_faces] = False
            r = headline_from_lab(LAB, face_keep=keep, tag="e1holes")
            item5["e1_hole_exclusion"] = {
                "n_e1_specimens_scored": int(holes.shape[0]),
                "n_e1_faces_scored": int(fidx.size),
                "n_atlas_faces_excluded": int(hole_faces.size),
                "frac_atlas_faces_excluded": float(hole_faces.size / Nf),
                "e1_true_bake_hole_frac": e1j["artifacts"]["true_bake_hole_frac"],
                "hole_face_specimen_pairs": int(holes.sum()),
                "hole_frac_face_specimen_pairs": float(holes.mean()),
                "coverage": (f"E1 scored {holes.shape[0]} specimens x {fidx.size} faces "
                             f"against the render reference"),
                "caveat": ("exclusion is the UNION over specimens of reference-verified hole "
                           "faces, so it also deletes those faces' good colour in the "
                           "specimens where the bake succeeded"),
                **cond_block(r),
            }
            log("  e1 hole exclusion", item5["e1_hole_exclusion"]["n_atlas_faces_excluded"],
                "faces", item5["e1_hole_exclusion"]["delta"])
        else:
            item5["e1_hole_exclusion"] = "SKIPPED — results/e1_fidelity*.{json,npz} not found"

        # ---- (b) exclude every face the module's own detector ever flagged (upper bound)
        keep2 = ~art.any(0)
        r2 = headline_from_lab(LAB, face_keep=keep2)
        item5["module_flag_exclusion"] = {
            "n_atlas_faces_excluded": int((~keep2).sum()),
            "frac_atlas_faces_excluded": float((~keep2).mean()),
            "note": "art flags near-black faces; on this specimen the fish has genuinely black "
                    "stripes, so this exclusion also deletes real pigment (upper bound)",
            **cond_block(r2),
        }
        log("  module-flag exclusion", item5["module_flag_exclusion"]["delta"])

        # ---- (c) median-impute flagged faces (the module's own repair)
        col_imp = fcd.colors.copy()
        for f_ in np.where(art.any(0))[0]:
            bad = art[:, f_]
            good = ~bad
            if good.any():
                col_imp[bad, f_, :] = np.median(col_imp[good, f_, :], axis=0).astype(np.uint8)
        LAB_imp = rgb_to_lab(col_imp)
        r3 = headline_from_lab(LAB_imp)
        item5["median_imputation"] = {
            "n_face_specimen_pairs_imputed": int(art.sum()),
            "frac_face_specimen_pairs": float(art.mean()),
            **cond_block(r3),
        }
        log("  median imputation", item5["median_imputation"]["delta"])
        del col_imp, LAB_imp

        # ---- (c2/c3) reference-verified holes: population-median repair, and an ORACLE
        # repair that writes back the colour the bake should have produced. The oracle is the
        # strict upper bound on how much bake holes could ever be worth.
        if e1a.exists() and e1p.exists() and z1["holes"].shape[0] == N:
            H = np.zeros((N, Nf), bool)
            H[:, fidx] = z1["holes"]
            LAB_h = LAB.copy()
            any_h = np.where(H.any(0))[0]
            for f_ in any_h:
                bad = H[:, f_]
                good = ~bad
                if good.any():
                    LAB_h[bad, f_, :] = np.median(LAB_h[good, f_, :], axis=0)
            r4 = headline_from_lab(LAB_h)
            item5["true_hole_median_imputation"] = {
                "n_face_specimen_pairs_imputed": int(H.sum()),
                "frac_face_specimen_pairs": float(H.mean()),
                "n_atlas_faces_touched": int(any_h.size),
                **cond_block(r4),
            }
            log("  true-hole median imputation",
                item5["true_hole_median_imputation"]["delta"])
            del LAB_h

            ref_lab = z1["ref_lab"]                       # (N, Nf, 3) float32 reference colour
            LAB_o = LAB.copy()
            LAB_o[H] = ref_lab[H].astype(np.float64)
            r5 = headline_from_lab(LAB_o)
            item5["oracle_hole_repair"] = {
                "note": "hole faces overwritten with the reference colour the bake should have "
                        "written — the best any artifact repair could possibly do",
                "n_face_specimen_pairs_repaired": int(H.sum()),
                **cond_block(r5),
            }
            log("  oracle hole repair", item5["oracle_hole_repair"]["delta"])
            del LAB_o, ref_lab, H
        else:
            msg = ("SKIPPED — e1 arrays absent or scored on a specimen subset "
                   "(per-specimen hole repair needs all N specimens)")
            item5["true_hole_median_imputation"] = msg
            item5["oracle_hole_repair"] = msg

        # ---- (d) additive Gaussian colour noise, sd in dE units (RMS Lab displacement)
        noise = {}
        for dE in (2.0, 5.0, 10.0):
            rng = np.random.default_rng(7)
            sigma = dE / np.sqrt(3.0)                     # RMS Euclidean Lab displacement = dE
            LABn = LAB + rng.normal(0.0, sigma, size=LAB.shape)
            rn = headline_from_lab(LABn)
            noise[dE] = {
                "sigma_per_channel": float(sigma),
                "realised_rms_dE": float(np.sqrt((((LABn - LAB) ** 2).sum(-1)).mean())),
                **cond_block(rn),
            }
            log("  noise", dE, "dE", noise[dE]["delta"])
            del LABn
        item5["gaussian_noise"] = noise
        res["item5_artifacts_robustness"] = item5
        save_partial(res, arrays)

    # =================================================================================
    # (6) SMALL-n COLLAPSE
    # =================================================================================
    if "item6_small_n" in res:
        log("=== skipping item6_small_n (already in checkpoint)")
    else:
        log("=== item 6: small-n collapse")
        best_desc = {}
        for f in C.FACTORS:
            best_desc[f] = max(MAIN, key=lambda k: res["headline"][k][f]["acc"])
        log("  best descriptor per factor (selected on FULL-n GT — noted as optimistic):",
            best_desc)

        def cv_small(X, y, seed):
            """Balanced-accuracy CV that degrades gracefully at tiny n."""
            cnt = np.bincount(y, minlength=2)
            if cnt.min() < 3:
                return None
            k = 5 if cnt.min() >= 5 else 3
            return C.balanced_cv(X, y, n_splits=k, n_repeats=4, seed=seed)[0]

        def dip_gate(X, n_null=100, seed=0, n_pc=5):
            """The team's existing gate: max dip over PC1..PC5 vs a column-shuffled null."""
            S, _ = pc_scores(X, n_pc, seed=0)
            d = max_dip(S)[0]
            rng = np.random.default_rng(seed)
            nulls = np.empty(n_null)
            for b in range(n_null):
                Xs = column_shuffle(X, rng)
                Ss, _ = pc_scores(Xs, n_pc, seed=0)
                nulls[b] = max_dip(Ss)[0]
            p = (1 + np.sum(nulls >= d)) / (n_null + 1)
            return float(d), float(p)

        ns = [25, 250] if QUICK else [25, 40, 80, 150, 250]
        n_draws = 3 if QUICK else 30
        item6 = {"n_draws": n_draws, "best_descriptor": best_desc, "per_n": {}}
        rng_master = np.random.default_rng(11)
        gate_descs = ["spec40", "area_hist24"]
        for n in ns:
            accs = {f: [] for f in C.FACTORS}
            nulls_ = {f: [] for f in C.FACTORS}
            skipped = {f: 0 for f in C.FACTORS}
            gate_fired = {g: [] for g in gate_descs}
            gate_p = {g: [] for g in gate_descs}
            for b in range(n_draws):
                idx = (np.arange(N) if n == N
                       else rng_master.choice(N, size=n, replace=False))
                for f in C.FACTORS:
                    X = D[best_desc[f]][idx]
                    y = labels[f][idx]
                    a = cv_small(X, y, seed=b)
                    if a is None:
                        skipped[f] += 1
                    else:
                        accs[f].append(a)
                        ysh = np.random.default_rng(1000 + b).permutation(y)
                        an = cv_small(X, ysh, seed=b)
                        if an is not None:
                            nulls_[f].append(an)
                for g in gate_descs:
                    d, p = dip_gate(D[g][idx], n_null=10 if QUICK else 100, seed=100 + b)
                    gate_fired[g].append(p < 0.05)
                    gate_p[g].append(p)
            item6["per_n"][n] = {
                "acc_mean": {f: float(np.mean(accs[f])) if accs[f] else None for f in C.FACTORS},
                "acc_sd": {f: float(np.std(accs[f])) if accs[f] else None for f in C.FACTORS},
                "acc_p05": {f: float(np.percentile(accs[f], 5)) if accs[f] else None
                            for f in C.FACTORS},
                "null_mean": {f: float(np.mean(nulls_[f])) if nulls_[f] else None
                              for f in C.FACTORS},
                "n_usable_draws": {f: len(accs[f]) for f in C.FACTORS},
                "skipped_draws_too_few_positives": skipped,
                "gate_fire_frac": {g: float(np.mean(gate_fired[g])) for g in gate_descs},
                "gate_p_median": {g: float(np.median(gate_p[g])) for g in gate_descs},
            }
            log("  n=", n, {f: round(item6["per_n"][n]["acc_mean"][f], 3)
                            for f in C.FACTORS if item6["per_n"][n]["acc_mean"][f] is not None},
                "gate fires", item6["per_n"][n]["gate_fire_frac"])
            arrays[f"smalln_{n}"] = np.array([accs[f] + [np.nan] * (n_draws - len(accs[f]))
                                              for f in C.FACTORS], dtype=float)
        item6["gate_descriptors"] = gate_descs
        item6["gate_definition"] = ("max Hartigan dip over PC1..PC5 vs 100 column-shuffled "
                                    "nulls; 'fires' = p < 0.05, i.e. the analyst would have "
                                    "claimed the morphospace contains clusters")
        item6["note_n250"] = ("at n=250 every draw is the whole dataset, so the spread reflects "
                              "CV-seed variation only, not sampling variation")

        # ---- how optimistic is it to reuse a codebook fitted on all 250 specimens?
        # area_hist's palette is population-fitted; a real n=25 study would fit it on its own 25.
        # (spectral_coeffs needs NO population fit — it is a fixed linear operator on the atlas.)
        log("  refit-within-draw control for area_hist24")
        refit = {}
        rng_r = np.random.default_rng(23)
        for n in ns:
            pooled_acc = {f: [] for f in C.FACTORS}
            refit_acc = {f: [] for f in C.FACTORS}
            for b in range(2 if QUICK else 10):
                idx = np.arange(N) if n == N else rng_r.choice(N, size=n, replace=False)
                Xr = area_hist_from_lab(LAB[idx], areas, 24, seed=b)      # fitted on the draw only
                Xp = D["area_hist24"][idx]                                # fitted on all 250
                for f in C.FACTORS:
                    y = labels[f][idx]
                    a1 = cv_small(Xp, y, seed=b)
                    a2 = cv_small(Xr, y, seed=b)
                    if a1 is not None and a2 is not None:
                        pooled_acc[f].append(a1)
                        refit_acc[f].append(a2)
            refit[n] = {
                "pooled_codebook_acc": {f: (float(np.mean(pooled_acc[f])) if pooled_acc[f] else None)
                                        for f in C.FACTORS},
                "within_draw_codebook_acc": {f: (float(np.mean(refit_acc[f])) if refit_acc[f]
                                                 else None) for f in C.FACTORS},
                "optimism": {f: (float(np.mean(pooled_acc[f]) - np.mean(refit_acc[f]))
                                 if pooled_acc[f] else None) for f in C.FACTORS},
                "n_draws": 2 if QUICK else 10,
            }
            log("    n=", n, "optimism", {f: (round(refit[n]["optimism"][f], 3)
                                              if refit[n]["optimism"][f] is not None else None)
                                          for f in C.FACTORS})
        item6["codebook_refit_control"] = refit
        res["item6_small_n"] = item6
        save_partial(res, arrays)

    # =================================================================================
    # re-bind from `res` so the summary prints identically on a fresh run and on a resume
    item1 = res["item1_discovery_wall"]
    item2 = res["item2_ica"]
    item3 = res["item3_segment_then_spectral"]
    item4 = res["item4_explained_variance"]
    item5 = res["item5_artifacts_robustness"]
    item6 = res["item6_small_n"]
    best_desc = item6["best_descriptor"]
    ns = [int(k) for k in item6["per_n"]]
    res["seconds"] = float(time.time() - T0)
    (C.RESULTS / f"{OUT_NAME}.json").write_text(json.dumps(jsonable(res), indent=2))
    np.savez_compressed(C.RESULTS / f"{OUT_NAME}_arrays.npz", **arrays)
    log("wrote", C.RESULTS / f"{OUT_NAME}.json")

    # ------------------------------------------------------------------ summary table
    print("\n" + "=" * 92)
    print("E6 — NEGATIVE RESULTS AND HONEST LIMITS   (chance = 0.500 for every factor)")
    print("=" * 92)
    hdr = f"{'':34s}" + "".join(f"{f:>13s}" for f in C.FACTORS)
    print("\n-- HEADLINE SUPERVISED ACCURACY (null on shuffled labels in brackets)")
    print(hdr)
    for k in MAIN:
        print(f"{k:34s}" + "".join(
            f"{res['headline'][k][f]['acc']:.3f}[{res['headline'][k][f]['null']:.2f}]".rjust(13)
            for f in C.FACTORS))

    print("\n-- (1) UNSUPERVISED DISCOVERY WALL")
    print(hdr)
    for k in MAIN:
        e = item1[k]
        print(f"{k+' GMM+BIC ARI (k=%d)' % e['gmm_bic']['chosen_k']:34s}"
              + "".join(f"{e['gmm_bic']['ari'][f]:13.3f}" for f in C.FACTORS))
        print(f"{k+' best PC1-10 AUC':34s}"
              + "".join(f"{e['best_pc_auc'][f]['auc']:13.3f}" for f in C.FACTORS))
        print(f"{'   (shuffled-label null)':34s}"
              + "".join(f"{e['best_pc_auc'][f]['null_mean']:13.3f}" for f in C.FACTORS))
    for k in MAIN:
        e = item1[k]["dip"]
        print(f"{k+' max dip PC1-10':34s} dip={e['max_dip']:.4f} (PC{e['axis_pc']}) "
              f"null={e['null_mean']:.4f} p_colshuffle={e['p_colshuffle']:.3f} "
              f"p_analytic={e['p_analytic']:.3f}")
    for k in MAIN:
        h = item1[k]["hdbscan"]
        print(f"{k+' HDBSCAN':34s} nclu={h['n_clusters']} noise={h['noise_frac']:.2f} "
              f"cheeks purity={h['cheeks']['purity']:.2f}/F1={h['cheeks']['f1']:.2f} "
              f"(prev {h['cheeks']['prevalence_null_purity']:.2f})  "
              f"stripe purity={h['stripe']['purity']:.2f}/F1={h['stripe']['f1']:.2f} "
              f"(prev {h['stripe']['prevalence_null_purity']:.2f})")

    print("\n-- (2) ICA vs PCA  (spec40)")
    print(f"{'n':>4s} {'maxAngle_deg':>13s} {'projResid':>10s} {'max|dAcc|':>10s} "
          f"{'max|dAUC1|':>11s} {'GMM ARI(pca,ica)':>18s}")
    for n, row in item2["spec40"].items():
        print(f"{int(n):>4} {row['max_principal_angle_deg']:13.5f} "
              f"{row['projector_residual_fro']:10.2e} {row['max_abs_acc_diff']:10.3f} "
              f"{row['max_abs_single_auc_diff']:11.3f} {row['gmm_pca_vs_ica_ari']:18.3f}")

    print("\n-- (3) SEGMENT-THEN-SPECTRAL")
    print(f"{'':34s}" + "".join(f"{f:>13s}" for f in C.FACTORS) + f"{'dims':>7s}")
    for k in ("segment_onehot_spectral_k100", "continuous_spectral_k100",
              "continuous_spectral_k40", "segment_spatial", "segment_area_fractions"):
        print(f"{k:34s}" + "".join(f"{item3[k][f]['acc']:13.3f}" for f in C.FACTORS)
              + f"{item3['dims'][k]:7d}")

    print("\n-- (4) SATURATION DIM vs EXPLAINED-VARIANCE DIM")
    for k, e in item4.items():
        print(f"{k:12s} p={e['n_features']:3d}  EV90={e['ev_dim']['90']}  "
              f"EV95={e['ev_dim']['95']}  EV99={e['ev_dim']['99']}")
        for f in C.FACTORS:
            sa = e["saturation"][f]
            print(f"    {f:8s} max={sa['acc_max']:.3f}@{sa['n_at_max']:<3d} "
                  f"sat(1pt)={sa['sat_dim_within_1pt']:<3d} sat(1sd)={sa['sat_dim_within_1sd']:<3d} "
                  f"EVR@sat={sa['cum_evr_at_sat_dim']:.3f}  "
                  f"acc@EV90={sa['acc_at_ev90']}  acc@EV95={sa['acc_at_ev95']}")

    print("\n-- (5) ROBUSTNESS (delta balanced accuracy vs baseline)")
    for cond in ("e1_hole_exclusion", "module_flag_exclusion", "median_imputation",
                 "true_hole_median_imputation", "oracle_hole_repair"):
        v = item5[cond]
        if isinstance(v, str):
            print(f"{cond:26s} {v}")
            continue
        for k in ("area_hist24", "spec40"):
            print(f"{cond+'/'+k:34s}" + "".join(f"{v['delta'][k][f]:+13.3f}"
                                                for f in C.FACTORS))
    for dE, v in item5["gaussian_noise"].items():
        for k in ("area_hist24", "spec40"):
            print(f"{f'noise {dE} dE/{k}':34s}" + "".join(f"{v['delta'][k][f]:+13.3f}"
                                                          for f in C.FACTORS))

    print("\n-- (6) SMALL-n COLLAPSE (mean +/- sd over 30 draws; best descriptor per factor)")
    print(f"{'n':>5s}" + "".join(f"{f:>18s}" for f in C.FACTORS)
          + f"{'gate:spec40':>13s}{'gate:areahist':>15s}")
    for n in ns:
        e = getk(item6["per_n"], n)
        cells = []
        for f in C.FACTORS:
            m, s = e["acc_mean"][f], e["acc_sd"][f]
            cells.append("      n/a         " if m is None else f"{m:.3f}+/-{s:.3f}".rjust(18))
        print(f"{n:>5d}" + "".join(cells)
              + f"{e['gate_fire_frac']['spec40']:13.2f}"
              + f"{e['gate_fire_frac']['area_hist24']:15.2f}")
    print("   (null, shuffled labels, per factor)")
    for n in ns:
        e = getk(item6["per_n"], n)
        cells = []
        for f in C.FACTORS:
            m = e["null_mean"][f]
            cells.append("      n/a         " if m is None else f"{m:.3f}".rjust(18))
        print(f"{n:>5d}" + "".join(cells))
    print(f"\nbest descriptor per factor: {best_desc}")
    print(f"total {time.time() - T0:.0f}s")


if __name__ == "__main__":
    main()
