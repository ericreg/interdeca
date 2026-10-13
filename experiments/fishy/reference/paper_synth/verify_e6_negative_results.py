"""ADVERSARIAL VERIFICATION of e6_negative_results.

Independent re-derivation. Imports ONLY `common as C` (the shared harness) and fishpipe
library primitives — never e6_negative_results.py. Every estimator (CV loop, PCA scores,
GMM+BIC, AUC search, dip null, ICA comparison, EV sweep, small-n draws) is re-written here
from scratch.

Usage:  ../.venv/bin/python verify_e6_negative_results.py <stage>
        stages: data desc sup unsup ica ev smalln seg all
"""
from __future__ import annotations

import json
import os
import sys
import time

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

import numpy as np
import scipy.sparse as sp

import common as C

T0 = time.time()
OUT = {}
SCRATCH = str(C.CACHE / "verify_e6")
os.makedirs(SCRATCH, exist_ok=True)


def log(*a):
    print(f"[{time.time()-T0:7.1f}s]", *a, flush=True)


# ------------------------------------------------------------------ my own CV
def my_bal_acc(y_true, y_pred):
    """Balanced accuracy from scratch (mean per-class recall)."""
    cls = np.unique(y_true)
    return float(np.mean([(y_pred[y_true == c] == c).mean() for c in cls]))


def my_cv(X, y, n_splits=5, n_repeats=4, seed=0):
    """Repeated stratified k-fold, standardise INSIDE the fold, balanced logistic."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import RepeatedStratifiedKFold
    X = np.asarray(X, float)
    y = np.asarray(y)
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    sc = []
    for tr, te in cv.split(X, y):
        mu = X[tr].mean(0)
        sd = X[tr].std(0)
        sd[sd == 0] = 1.0
        clf = LogisticRegression(max_iter=4000, C=1.0, class_weight="balanced")
        clf.fit((X[tr] - mu) / sd, y[tr])
        sc.append(my_bal_acc(y[te], clf.predict((X[te] - mu) / sd)))
    return float(np.mean(sc)), float(np.std(sc))


def my_pcs(X, n=10):
    Z = np.asarray(X, float)
    Z = (Z - Z.mean(0)) / np.where(Z.std(0) == 0, 1.0, Z.std(0))
    Zc = Z - Z.mean(0)
    U, s, Vt = np.linalg.svd(Zc, full_matrices=False)
    S = U[:, :n] * s[:n]
    evr = (s ** 2) / (s ** 2).sum()
    return S, evr


def my_auc(y, s):
    """AUC by rank formula, orientation-free."""
    y = np.asarray(y)
    order = np.argsort(s, kind="mergesort")
    r = np.empty(len(s), float)
    ss = np.asarray(s, float)[order]
    ranks = np.arange(1, len(s) + 1, dtype=float)
    # average ties
    i = 0
    while i < len(ss):
        j = i
        while j + 1 < len(ss) and ss[j + 1] == ss[i]:
            j += 1
        ranks[i:j + 1] = ranks[i:j + 1].mean()
        i = j + 1
    r[order] = ranks
    n1 = int((y == 1).sum())
    n0 = len(y) - n1
    a = (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
    return max(a, 1 - a)


def load_all():
    fcd, art, mesh = C.atlas_data(force=False, impute=False)
    df, labels = C.load_gt(list(fcd.names))
    return fcd, art, mesh, df, labels


# ================================================================== stages
def stage_data():
    fcd, art, mesh, df, labels = load_all()
    import pandas as pd
    raw = pd.read_csv(C.GT_CSV)
    names = list(fcd.names)
    OUT["n_specimens"] = len(names)
    OUT["names_unique"] = len(set(names)) == len(names)
    OUT["names_sorted"] = names == sorted(names)
    # independent join: reconstruct labels straight from the csv, keyed by name
    idx = {t: i for i, t in enumerate(raw["sample_tag"])}
    pos = np.array([idx[n] for n in names])
    OUT["all_names_in_csv"] = bool(len(pos) == len(names))
    stripe_mine = (raw["stripe_count"].to_numpy()[pos] == 5).astype(int)
    cheek_mine = raw["rosy_cheeks_present"].to_numpy()[pos].astype(int)
    OUT["stripe_label_match"] = bool((stripe_mine == labels["stripe"]).all())
    OUT["cheeks_label_match"] = bool((cheek_mine == labels["cheeks"]).all())
    # belly/tail: 2-means on hue -> check the split is unambiguous (gap between groups)
    for f, col in (("belly", "belly_hue"), ("tail", "tail_hue")):
        h = raw[col].to_numpy()[pos]
        y = labels[f]
        lo, hi = h[y == 0], h[y == 1]
        OUT[f"{f}_hue_gap"] = float(hi.min() - lo.max())
        OUT[f"{f}_hue_within_sd"] = [float(lo.std()), float(hi.std())]
        OUT[f"{f}_counts"] = [int((y == 0).sum()), int((y == 1).sum())]
    OUT["class_counts"] = {f: np.bincount(labels[f]).tolist() for f in C.FACTORS}
    # is the mesh really the atlas?
    OUT["n_faces"] = int(fcd.colors.shape[1])
    OUT["n_verts"] = int(mesh.vertices.shape[0])
    A = C.face_adjacency(mesh, C.ATLAS_DS.cache_dir)
    OUT["adjacency_shape_matches_faces"] = bool(A.shape[0] == fcd.colors.shape[1])
    OUT["artifact_frac"] = float(art.mean())
    # correlations among the 4 planted factors (are they independent?)
    L = np.array([labels[f] for f in C.FACTORS], float)
    Cm = np.corrcoef(L)
    OUT["factor_label_corr"] = {f"{a}-{b}": float(Cm[i, j])
                                for i, a in enumerate(C.FACTORS)
                                for j, b in enumerate(C.FACTORS) if i < j}
    log("data", json.dumps(OUT, indent=1))


def my_area_hist(LAB, areas, k=24, seed=0, replace=False):
    from sklearn.cluster import KMeans
    N, Nf, _ = LAB.shape
    rng = np.random.default_rng(seed)
    pooled = LAB.reshape(N * Nf, 3)
    fi = rng.choice(pooled.shape[0], size=min(200_000, pooled.shape[0]), replace=replace)
    km = KMeans(n_clusters=k, n_init=5, random_state=seed).fit(pooled[fi])
    X = np.zeros((N, k))
    for i in range(N):
        X[i] = np.bincount(km.predict(LAB[i]), weights=areas, minlength=k)
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n


def my_spec(LAB, mesh, areas, k=40):
    """Own face->vertex aggregation (dense accumulate, not sparse) + own GFT."""
    fv = mesh.face_v
    nv = mesh.vertices.shape[0]
    varea = np.zeros(nv)
    np.add.at(varea, fv.ravel(), np.repeat(areas, 3))
    varea[varea == 0] = 1.0
    w, U = np.load(C.ATLAS_DS.cache_dir / "lap_w_300.npy"), np.load(C.ATLAS_DS.cache_dir / "lap_U_300.npy")
    Uk = U[:, :k].astype(np.float64)
    N = LAB.shape[0]
    out = np.zeros((N, 3 * k))
    blocks = []
    for c in range(3):
        M = np.zeros((N, k))
        for i in range(N):
            acc = np.zeros(nv)
            np.add.at(acc, fv.ravel(), np.repeat(areas * LAB[i, :, c], 3))
            v = acc / varea
            v = v - v.mean()
            M[i] = v @ Uk
        blocks.append(M)
    out = np.concatenate(blocks, axis=1)
    return out


def stage_desc():
    from fishpipe.features import rgb_to_lab
    fcd, art, mesh, df, labels = load_all()
    LAB = rgb_to_lab(fcd.colors)
    areas = fcd.areas
    z = np.load(C.CACHE / "e6_negative_results_desc.npz")
    ah = my_area_hist(LAB, areas, 24, seed=0)
    OUT["area_hist24_max_abs_diff_vs_agent"] = float(np.abs(ah - z["area_hist24"]).max())
    ah7 = my_area_hist(LAB, areas, 24, seed=7)
    OUT["area_hist24_seed7_selfconsistency"] = float(np.abs(np.sort(ah7, 1) - np.sort(ah, 1)).max())
    sp40 = my_spec(LAB, mesh, areas, 40)
    OUT["spec40_max_abs_diff_vs_agent"] = float(np.abs(sp40 - z["spec40"]).max())
    OUT["spec40_rel_diff"] = float(np.abs(sp40 - z["spec40"]).max() / np.abs(z["spec40"]).max())
    np.savez_compressed(f"{SCRATCH}/verify_desc.npz", area_hist24=ah, area_hist24_seed7=ah7,
                        spec40=sp40, texton_bow=z["texton_bow"], spec100=z["spec100"],
                        area_hist64=z["area_hist64"])
    log("desc", json.dumps(OUT, indent=1))


def _desc():
    z = np.load(f"{SCRATCH}/verify_desc.npz")
    return {k: z[k] for k in z.files}


def stage_sup():
    fcd, art, mesh, df, labels = load_all()
    D = _desc()
    OUT["supervised"] = {}
    for key in ("area_hist24", "spec40", "texton_bow", "area_hist24_seed7"):
        row = {}
        for f in C.FACTORS:
            a, s = my_cv(D[key], labels[f])
            row[f] = {"acc": round(a, 4), "sd": round(s, 4)}
        OUT["supervised"][key] = row
        log("  sup", key, {f: row[f]["acc"] for f in C.FACTORS})
    # shuffled-label nulls, 10 permutations (agent used 5)
    rng = np.random.default_rng(101)
    OUT["null"] = {}
    for key in ("area_hist24", "spec40"):
        row = {}
        for f in C.FACTORS:
            v = [my_cv(D[key], rng.permutation(labels[f]))[0] for _ in range(10)]
            row[f] = {"null_mean": round(float(np.mean(v)), 4), "null_sd": round(float(np.std(v)), 4),
                      "null_max": round(float(np.max(v)), 4)}
        OUT["null"][key] = row
        log("  null", key, {f: row[f]["null_mean"] for f in C.FACTORS})
    log("sup done")


def stage_unsup():
    from sklearn.metrics import adjusted_rand_score
    from sklearn.mixture import GaussianMixture
    fcd, art, mesh, df, labels = load_all()
    D = _desc()
    OUT["unsup"] = {}
    for key in ("area_hist24", "spec40", "texton_bow"):
        S10, evr = my_pcs(D[key], 10)
        e = {"evr10": [round(float(v), 4) for v in evr[:10]]}
        # --- GMM + BIC over k=1..8 on top-2 PCs
        S2 = S10[:, :2]
        best = (np.inf, None)
        bic = {}
        for k in range(1, 9):
            gm = GaussianMixture(k, covariance_type="full", n_init=10, random_state=0,
                                 reg_covar=1e-6).fit(S2)
            b = float(gm.bic(S2))
            bic[k] = round(b, 1)
            if b < best[0]:
                best = (b, gm)
        zlab = best[1].predict(S2)
        e["gmm_k"] = int(best[1].n_components)
        e["gmm_ari"] = {f: round(float(adjusted_rand_score(labels[f], zlab)), 4) for f in C.FACTORS}
        # --- best single PC AUC + 200-permutation null of the SAME max statistic
        rng = np.random.default_rng(2026)
        e["best_pc_auc"] = {}
        for f in C.FACTORS:
            aucs = [my_auc(labels[f], S10[:, j]) for j in range(10)]
            a = max(aucs)
            nulls = []
            for _ in range(200):
                ys = rng.permutation(labels[f])
                nulls.append(max(my_auc(ys, S10[:, j]) for j in range(10)))
            nulls = np.asarray(nulls)
            e["best_pc_auc"][f] = {
                "auc": round(float(a), 4), "pc": int(np.argmax(aucs) + 1),
                "null_mean": round(float(nulls.mean()), 4),
                "null_p95": round(float(np.percentile(nulls, 95)), 4),
                "p_perm": round(float((1 + (nulls >= a).sum()) / 201), 4)}
        # --- dip vs column-shuffle null
        import diptest
        dips = [diptest.diptest(np.ascontiguousarray(S10[:, j]))[0] for j in range(10)]
        d = float(max(dips))
        rng2 = np.random.default_rng(3131)
        X = np.asarray(D[key], float)
        nl = []
        for _ in range(200):
            Xs = rng2.permuted(X, axis=0)
            Ss, _ = my_pcs(Xs, 10)
            nl.append(max(diptest.diptest(np.ascontiguousarray(Ss[:, j]))[0] for j in range(10)))
        nl = np.asarray(nl)
        e["dip"] = {"max_dip": round(d, 4), "axis_pc": int(np.argmax(dips) + 1),
                    "null_mean": round(float(nl.mean()), 4),
                    "null_p95": round(float(np.percentile(nl, 95)), 4),
                    "p_colshuffle": round(float((1 + (nl >= d).sum()) / 201), 4)}
        # --- HDBSCAN
        import hdbscan
        cl = hdbscan.HDBSCAN(min_cluster_size=10, min_samples=5).fit(S10)
        lh = cl.labels_
        e["hdbscan"] = {"n_clusters": int(len(set(lh[lh >= 0]))),
                        "noise_frac": round(float((lh < 0).mean()), 3),
                        "ari": {f: round(float(adjusted_rand_score(labels[f], lh)), 4)
                                for f in C.FACTORS}}
        for f in ("cheeks", "stripe"):
            y = labels[f]
            prev = float(y.mean())
            mp = prev
            for c in sorted(set(lh[lh >= 0])):
                m = lh == c
                mp = max(mp, (y[m] == 1).sum() / max(m.sum(), 1))
            e["hdbscan"][f + "_max_purity"] = round(float(mp), 4)
            e["hdbscan"][f + "_prev"] = round(prev, 4)
            e["hdbscan"][f + "_enrichment"] = round(float(mp / prev), 4)
        OUT["unsup"][key] = e
        log("  unsup", key, json.dumps(e["gmm_ari"]), "dip_p", e["dip"]["p_colshuffle"])
    log("unsup done")


def stage_ica():
    from scipy.linalg import subspace_angles
    from sklearn.decomposition import PCA, FastICA
    from sklearn.metrics import adjusted_rand_score
    from sklearn.mixture import GaussianMixture
    fcd, art, mesh, df, labels = load_all()
    D = _desc()
    OUT["ica"] = {}
    for key in ("spec40", "area_hist24"):
        X = np.asarray(D[key], float)
        X = (X - X.mean(0)) / np.where(X.std(0) == 0, 1, X.std(0))
        sub = {}
        for n in (2, 10, 20, 30):
            if n > min(X.shape):
                continue
            p = PCA(n_components=n, random_state=0).fit(X)
            Sp = p.transform(X)
            ica = FastICA(n_components=n, whiten="unit-variance", random_state=0,
                          max_iter=2000, tol=1e-4).fit(X)
            Si = ica.transform(X)
            Bp = np.linalg.qr(p.components_.T)[0]
            Bi = np.linalg.qr(ica.components_.T)[0]
            ang = float(np.degrees(np.max(subspace_angles(Bp, Bi))))
            resid = float(np.linalg.norm(Bp @ Bp.T - Bi @ Bi.T))
            Spw = Sp / Sp.std(0, keepdims=True)
            gp = GaussianMixture(2, covariance_type="full", n_init=10, random_state=0).fit(Sp).predict(Sp)
            gi = GaussianMixture(2, covariance_type="full", n_init=10, random_state=0).fit(Si).predict(Si)
            gw = GaussianMixture(2, covariance_type="full", n_init=10, random_state=0).fit(Spw).predict(Spw)
            row = {"max_angle_deg": ang, "proj_resid": resid,
                   "gmm_pca_vs_ica_ari": round(float(adjusted_rand_score(gp, gi)), 4),
                   "gmm_pcawhite_vs_ica_ari": round(float(adjusted_rand_score(gw, gi)), 4),
                   "dacc": {}, "dauc": {}}
            for f in C.FACTORS:
                ap = my_cv(Sp, labels[f])[0]
                ai = my_cv(Si, labels[f])[0]
                row["dacc"][f] = round(float(ai - ap), 4)
                bp = max(my_auc(labels[f], Sp[:, j]) for j in range(n))
                bi = max(my_auc(labels[f], Si[:, j]) for j in range(n))
                row["dauc"][f] = round(float(bi - bp), 4)
            row["max_abs_dacc"] = round(max(abs(v) for v in row["dacc"].values()), 4)
            sub[n] = row
            log("  ica", key, n, ang, row["max_abs_dacc"], row["dauc"])
        OUT["ica"][key] = sub
    log("ica done")


def stage_ev():
    from sklearn.decomposition import PCA
    fcd, art, mesh, df, labels = load_all()
    D = _desc()
    OUT["ev"] = {}
    for key in ("spec40", "area_hist24"):
        X = np.asarray(D[key], float)
        X = (X - X.mean(0)) / np.where(X.std(0) == 0, 1, X.std(0))
        p = PCA(n_components=None, random_state=0).fit(X)
        cum = np.cumsum(p.explained_variance_ratio_)
        pmax = int(min(60, len(cum)))
        S = p.transform(X)[:, :pmax]
        evd = {q: int(np.argmax(cum >= q) + 1) for q in (0.90, 0.95, 0.99)}
        curves = {}
        for f in C.FACTORS:
            curves[f] = [my_cv(S[:, :n], labels[f])[0] for n in range(1, pmax + 1)]
        sat = {}
        for f in C.FACTORS:
            a = np.asarray(curves[f])
            amax = float(a.max())
            sat[f] = {"acc_max": round(amax, 4), "n_at_max": int(np.argmax(a) + 1),
                      "sat_within_1pt": int(np.argmax(a >= amax - 0.01) + 1),
                      "acc_at_ev95": round(float(a[evd[0.95] - 1]), 4)}
        OUT["ev"][key] = {"ev_dim": {str(k): v for k, v in evd.items()}, "saturation": sat}
        log("  ev", key, OUT["ev"][key])
    log("ev done")


def stage_smalln():
    fcd, art, mesh, df, labels = load_all()
    D = _desc()
    best = {"belly": "area_hist24", "tail": "area_hist24", "stripe": "spec40", "cheeks": "spec40"}
    N = len(labels["belly"])
    rng = np.random.default_rng(4242)
    OUT["smalln"] = {}
    for n in (25, 80):
        acc = {f: [] for f in C.FACTORS}
        nul = {f: [] for f in C.FACTORS}
        skipped = {f: 0 for f in C.FACTORS}
        for b in range(30):
            idx = rng.choice(N, size=n, replace=False)
            for f in C.FACTORS:
                y = labels[f][idx]
                cnt = np.bincount(y, minlength=2)
                if cnt.min() < 3:
                    skipped[f] += 1
                    continue
                k = 5 if cnt.min() >= 5 else 3
                acc[f].append(my_cv(D[best[f]][idx], y, n_splits=k, n_repeats=4, seed=b)[0])
                nul[f].append(my_cv(D[best[f]][idx], np.random.default_rng(9000 + b).permutation(y),
                                    n_splits=k, n_repeats=4, seed=b)[0])
        OUT["smalln"][n] = {f: {"acc_mean": round(float(np.mean(acc[f])), 4),
                                "acc_sd": round(float(np.std(acc[f])), 4),
                                "null_mean": round(float(np.mean(nul[f])), 4),
                                "skipped": skipped[f]} for f in C.FACTORS}
        log("  smalln", n, OUT["smalln"][n])
    # ---- codebook refit optimism, with a PAIRED sd so the claim can be judged
    from fishpipe.features import rgb_to_lab
    LAB = rgb_to_lab(fcd.colors)
    areas = fcd.areas
    rng2 = np.random.default_rng(5150)
    n = 25
    diffs = {f: [] for f in C.FACTORS}
    for b in range(10):
        idx = rng2.choice(N, size=n, replace=False)
        Xr = my_area_hist(LAB[idx], areas, 24, seed=b)
        Xp = D["area_hist24"][idx]
        for f in C.FACTORS:
            y = labels[f][idx]
            cnt = np.bincount(y, minlength=2)
            if cnt.min() < 3:
                continue
            k = 5 if cnt.min() >= 5 else 3
            a1 = my_cv(Xp, y, n_splits=k, n_repeats=4, seed=b)[0]
            a2 = my_cv(Xr, y, n_splits=k, n_repeats=4, seed=b)[0]
            diffs[f].append(a1 - a2)
    OUT["codebook_optimism_n25"] = {
        f: {"mean": round(float(np.mean(diffs[f])), 4),
            "sd": round(float(np.std(diffs[f], ddof=1)), 4),
            "se": round(float(np.std(diffs[f], ddof=1) / np.sqrt(len(diffs[f]))), 4),
            "n": len(diffs[f])} for f in C.FACTORS}
    log("  codebook optimism", json.dumps(OUT["codebook_optimism_n25"]))


def stage_seg():
    """Item 3: segment-then-spectral, recomputed independently."""
    from fishpipe import segment
    from fishpipe.features import rgb_to_lab
    fcd, art, mesh, df, labels = load_all()
    D = _desc()
    areas = fcd.areas
    A = C.face_adjacency(mesh, C.ATLAS_DS.cache_dir)
    seg = segment.segment(fcd, n_colors=8, smooth_iters=0, mesh_adjacency=A)
    log("  segmented")
    fv = mesh.face_v
    nv = mesh.vertices.shape[0]
    Nf = fv.shape[0]
    N = fcd.colors.shape[0]
    varea = np.zeros(nv)
    np.add.at(varea, fv.ravel(), np.repeat(areas, 3))
    varea[varea == 0] = 1.0
    U = np.load(C.ATLAS_DS.cache_dir / "lap_U_300.npy")[:, :100].astype(np.float64)
    K = 8
    oh = np.zeros((N, K * 100))
    segarea = np.zeros((N, K))
    for i in range(N):
        lab = seg.labels[i]
        segarea[i] = np.bincount(lab, weights=areas, minlength=K)
        Vc = np.zeros((nv, K))
        for c in range(K):
            m = lab == c
            acc = np.zeros(nv)
            np.add.at(acc, fv[m].ravel(), np.repeat(areas[m], 3))
            Vc[:, c] = acc / varea
        Vc = Vc - Vc.mean(0, keepdims=True)
        oh[i] = (Vc.T @ U).reshape(-1)
    segarea = segarea / segarea.sum(1, keepdims=True)
    OUT["seg"] = {"onehot_spec_k100": {}, "seg_area": {}, "cont_spec_k100": {}}
    for f in C.FACTORS:
        OUT["seg"]["onehot_spec_k100"][f] = round(my_cv(oh, labels[f])[0], 4)
        OUT["seg"]["seg_area"][f] = round(my_cv(segarea, labels[f])[0], 4)
        OUT["seg"]["cont_spec_k100"][f] = round(my_cv(D["spec100"], labels[f])[0], 4)
    OUT["seg"]["delta_onehot_minus_cont"] = {
        f: round(OUT["seg"]["onehot_spec_k100"][f] - OUT["seg"]["cont_spec_k100"][f], 4)
        for f in C.FACTORS}
    log("  seg", json.dumps(OUT["seg"], indent=1))


STAGES = {"data": stage_data, "desc": stage_desc, "sup": stage_sup, "unsup": stage_unsup,
          "ica": stage_ica, "ev": stage_ev, "smalln": stage_smalln, "seg": stage_seg}

if __name__ == "__main__":
    want = sys.argv[1:] or ["all"]
    if want == ["all"]:
        want = list(STAGES)
    for s in want:
        log("=== stage", s)
        STAGES[s]()
    p = f"{SCRATCH}/verify_e6_{'_'.join(want)}.json"
    with open(p, "w") as fh:
        json.dump(OUT, fh, indent=1, default=float)
    log("wrote", p)
    print(json.dumps(OUT, indent=1, default=float))
