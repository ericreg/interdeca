"""ADVERSARIAL re-derivation of e5_expert_guided headline numbers.

Independent implementation: nothing is imported from e5_expert_guided.py or e5_build_feats.py.
Only `common as C` (data harness) and the fishpipe library modules the method itself is about.
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (adjusted_rand_score, balanced_accuracy_score,
                             roc_auc_score, silhouette_score)
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import common as C

T0 = time.time()
OUT = {}


def log(m):
    print(f"[{time.time()-T0:7.1f}s] {m}", flush=True)


PART = sys.argv[1] if len(sys.argv) > 1 else "all"

# --------------------------------------------------------------------- data
d = np.load(C.CACHE / "e5_base_feats.npz", allow_pickle=True)
names, H, Tex = list(d["names"]), d["hist"], d["texton"]
df, labels = C.load_gt(names)
N = len(names)
joint8 = C.joint_label(labels, factors=("belly", "tail", "stripe"))
log(f"N={N} H={H.shape} Tex={Tex.shape}")

# ---- provenance / ordering checks -----------------------------------------
if PART in ("all", "prov"):
    import scipy.sparse as sp
    fcd, _art, mesh = C.atlas_data()
    same = list(fcd.names) == names
    A = sp.load_npz(C.ATLAS_DS.cache_dir / "face_adjacency.npz")
    OUT["prov"] = {
        "names_match_fcd": bool(same),
        "n_faces_mesh": int(mesh.face_v.shape[0]),
        "n_faces_colors": int(fcd.colors.shape[1]),
        "adjacency_shape": int(A.shape[0]),
        "adjacency_matches_mesh": bool(A.shape[0] == mesh.face_v.shape[0]),
        "gt_row0_name": str(df["sample_tag"].iloc[0]), "feat_row0_name": str(names[0]),
        "belly_hue_grp0_mean": float(df["belly_hue"].to_numpy()[labels["belly"] == 0].mean()),
        "belly_hue_grp1_mean": float(df["belly_hue"].to_numpy()[labels["belly"] == 1].mean()),
        "tail_hue_grp0_mean": float(df["tail_hue"].to_numpy()[labels["tail"] == 0].mean()),
        "tail_hue_grp1_mean": float(df["tail_hue"].to_numpy()[labels["tail"] == 1].mean()),
        "cheek_hue_pos_mean": float(df["rosy_cheeks_hue"].to_numpy()[labels["cheeks"] == 1].mean()),
        "cheek_hue_pos_sd": float(df["rosy_cheeks_hue"].to_numpy()[labels["cheeks"] == 1].std()),
        "class_counts": {k: np.bincount(v).tolist() for k, v in labels.items()},
    }
    del fcd
    log(f"prov {OUT['prov']}")

# ---- representation --------------------------------------------------------
Xcat = np.hstack([StandardScaler().fit_transform(H), StandardScaler().fit_transform(Tex)])
_sc = StandardScaler().fit(Xcat)
_pc = PCA(n_components=24, random_state=0).fit(_sc.transform(Xcat))
Z = _pc.transform(_sc.transform(Xcat))
log(f"Z {Z.shape} evr {_pc.explained_variance_ratio_.sum():.4f}")

TAGS = ["purple_belly", "blue_belly", "green_tail", "cyan_tail", "5_stripes", "rosy_cheeks"]
Y = np.stack([labels["belly"] == 1, labels["belly"] == 0, labels["tail"] == 0,
              labels["tail"] == 1, labels["stripe"] == 1, labels["cheeks"] == 1],
             axis=1).astype(int)
TAGCOL = {"belly": [0, 1], "tail": [2, 3], "stripe": [4], "cheeks": [5]}

# compare with the agent's stored Z
try:
    az = np.load(C.RESULTS / "e5_expert_guided_arrays.npz", allow_pickle=True)
    Za = az["Z"]
    cor = [abs(np.corrcoef(Z[:, k], Za[:, k])[0, 1]) for k in range(24)]
    OUT["Z_axis_abscorr_min"] = float(np.min(cor))
    OUT["tag_matrix_match"] = bool((az["tag_matrix"] == Y).all())
    OUT["joint8_match"] = bool((az["joint8"] == joint8).all())
except Exception as e:  # pragma: no cover
    OUT["Z_compare_error"] = str(e)


def mycv(X, y, n_splits=5, n_repeats=4, seed=0, Cc=1.0):
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    s = []
    for tr, te in cv.split(X, y):
        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(max_iter=4000, C=Cc, class_weight="balanced"))
        clf.fit(X[tr], y[tr])
        s.append(balanced_accuracy_score(y[te], clf.predict(X[te])))
    return float(np.mean(s)), float(np.std(s))


# ===================================================================== ceilings
if PART in ("all", "ceil"):
    OUT["ceiling_Z"] = {f: mycv(Z, labels[f])[0] for f in C.FACTORS}
    OUT["ceiling_Z_shuffled"] = {
        f: mycv(Z, np.random.default_rng(7).permutation(labels[f]))[0] for f in C.FACTORS}
    OUT["ceiling_Z_shuffled_10seed"] = {
        f: float(np.mean([mycv(Z, np.random.default_rng(1000 + s).permutation(labels[f]))[0]
                          for s in range(10)])) for f in C.FACTORS}
    log(f"ceilZ {OUT['ceiling_Z']}")
    log(f"nullZ {OUT['ceiling_Z_shuffled_10seed']}")
    fcd, _a, mesh = C.atlas_data()
    from fishpipe import spectral
    SPEC = spectral.spectral_coeffs(fcd, k=40, mesh=mesh, cache_dir=C.ATLAS_DS.cache_dir)
    REG = C.region_summary(fcd, mesh, n_regions=128, stat="mean")
    del fcd
    OUT["ceiling_spectral40"] = {f: mycv(SPEC, labels[f])[0] for f in C.FACTORS}
    OUT["ceiling_region128mean"] = {f: mycv(REG, labels[f])[0] for f in C.FACTORS}
    np.save(C.CACHE / "verify_e5_REG.npy", REG)
    log(f"ceil spec {OUT['ceiling_spectral40']} reg {OUT['ceiling_region128mean']}")

# ===================================================================== helpers
def fit_models(Zs, Ys, lab):
    ms = []
    for t in range(Ys.shape[1]):
        yt = Ys[lab, t]
        if len(np.unique(yt)) < 2:
            ms.append(("const", int(yt[0]) if len(yt) else 0))
        else:
            m = make_pipeline(StandardScaler(),
                              LogisticRegression(max_iter=4000, C=0.5, class_weight="balanced"))
            m.fit(Zs[lab], yt)
            ms.append(("clf", m))
    return ms


def scores(ms, Zs):
    S = np.full((len(Zs), len(ms)), 0.5)
    for t, (k, o) in enumerate(ms):
        if k == "clf":
            S[:, t] = o.predict_proba(Zs)[:, 1]
    return S


def acc_on(ms, Zs, Ys, idx):
    out = np.full(Ys.shape[1], np.nan)
    for t, (k, o) in enumerate(ms):
        yt = Ys[idx, t]
        if len(np.unique(yt)) < 2:
            continue
        p = o.predict(Zs[idx]) if k == "clf" else np.full(len(idx), o)
        out[t] = balanced_accuracy_score(yt, p)
    return out


def gmm_ari(S, y, k=8, seed=0, cov="full"):
    if len(S) < k + 2:
        return np.nan
    Ss = StandardScaler().fit_transform(S)
    try:
        g = GaussianMixture(k, covariance_type=cov, random_state=seed, n_init=3,
                            reg_covar=1e-4).fit(Ss)
        return float(adjusted_rand_score(y, g.predict(Ss)))
    except Exception:
        return np.nan


BUD = [8, 16, 24, 40, 56, 80, 120, 160]


def tag_curve(Zs, Ys, joint, budgets, seed, acq, pool=None, eval_fixed=None, cov="full"):
    """pool: indices eligible for labelling (default all). eval_fixed: fixed eval indices."""
    rng = np.random.default_rng(1000 * seed + (0 if acq == "random" else 1))
    n = len(Zs)
    pool = np.arange(n) if pool is None else np.asarray(pool)
    lab = list(rng.permutation(pool)[:budgets[0]])
    accs, aris, accs_fix = [], [], []
    for M in budgets:
        if M > len(lab):
            unl = np.setdiff1d(pool, np.array(lab, int))
            if acq == "random":
                pick = rng.choice(unl, size=min(M - len(lab), len(unl)), replace=False)
            else:
                mm = fit_models(Zs, Ys, np.array(lab, int))
                p = scores(mm, Zs[unl])
                unc = np.abs(2 * p - 1).mean(1)
                pick = unl[np.argsort(unc)[:M - len(lab)]]
            lab.extend(int(i) for i in pick)
        li = np.array(sorted(set(lab)), int)
        ui = np.setdiff1d(pool, li)
        mm = fit_models(Zs, Ys, li)
        accs.append(acc_on(mm, Zs, Ys, ui))
        aris.append(gmm_ari(scores(mm, Zs[ui]), joint[ui], seed=seed, cov=cov))
        if eval_fixed is not None:
            accs_fix.append(acc_on(mm, Zs, Ys, np.asarray(eval_fixed)))
    return np.array(accs), np.array(aris), (np.array(accs_fix) if eval_fixed is not None else None)


def per_factor(acc):  # (seed,budget,tag) -> dict
    return {f: np.nanmean(np.nanmean(acc[:, :, cols], axis=2), axis=0).tolist()
            for f, cols in TAGCOL.items()}


# ===================================================================== A
if PART in ("all", "A"):
    OUT["A"] = {}
    for acq in ("random", "uncertainty"):
        aa, ar = [], []
        for sd in range(6):
            a, r, _ = tag_curve(Z, Y, joint8, BUD, sd, acq)
            aa.append(a); ar.append(r)
        aa = np.array(aa); ar = np.array(ar)
        OUT["A"][acq] = {
            "mean_over_6tags": np.nanmean(np.nanmean(aa, 2), 0).tolist(),
            "mean_over_4factors": np.mean([v for v in per_factor(aa).values()], 0).tolist(),
            "per_factor": per_factor(aa),
            "ari8": np.nanmean(ar, 0).tolist(),
        }
        log(f"A {acq}: tag6 {[round(x,3) for x in OUT['A'][acq]['mean_over_6tags']]}")
        log(f"A {acq}: ari8 {[round(x,3) for x in OUT['A'][acq]['ari8']]}")
    # null
    aa, ar = [], []
    for sd in range(6):
        pi = np.random.default_rng(500 + sd).permutation(N)
        a, r, _ = tag_curve(Z, Y[pi], joint8[pi], BUD, sd, "random")
        aa.append(a); ar.append(r)
    OUT["A"]["null"] = {"mean_over_6tags": np.nanmean(np.nanmean(np.array(aa), 2), 0).tolist(),
                        "ari8": np.nanmean(np.array(ar), 0).tolist()}
    log(f"A null: {[round(x,3) for x in OUT['A']['null']['mean_over_6tags']]}")
    # M=0 unsupervised reference
    OUT["A"]["ari8_M0_Z"] = gmm_ari(Z, joint8, k=8, seed=0)
    OUT["A"]["ari8_M0_Z_seeds"] = [gmm_ari(Z, joint8, k=8, seed=s) for s in range(6)]
    # what does a PERFECT belly x tail 4-partition score against joint8?
    bt = labels["belly"] * 2 + labels["tail"]
    OUT["A"]["ari8_of_perfect_bellyxtail"] = float(adjusted_rand_score(joint8, bt))
    OUT["A"]["ari8_of_perfect_joint8"] = 1.0
    log(f"A M0 ari {OUT['A']['ari8_M0_Z']:.3f}; perfect belly x tail vs joint8 "
        f"{OUT['A']['ari8_of_perfect_bellyxtail']:.3f}")

# ============================================ A-bias: fixed common eval set
if PART in ("all", "Abias"):
    rng0 = np.random.default_rng(12345)
    res = {}
    for acq in ("random", "uncertainty"):
        fx, insample = [], []
        for sd in range(6):
            r = np.random.default_rng(9000 + sd)
            held = np.sort(r.choice(N, 60, replace=False))
            pool = np.setdiff1d(np.arange(N), held)
            a, _, af = tag_curve(Z, Y, joint8, [8, 16, 24, 40, 56, 80, 120], sd, acq,
                                 pool=pool, eval_fixed=held)
            fx.append(af); insample.append(a)
        res[acq] = {"fixed_heldout_mean_tag": np.nanmean(np.nanmean(np.array(fx), 2), 0).tolist(),
                    "remaining_pool_mean_tag": np.nanmean(np.nanmean(np.array(insample), 2), 0).tolist(),
                    "fixed_heldout_per_factor": {f: np.nanmean(np.nanmean(np.array(fx)[:, :, c], 2), 0).tolist()
                                                 for f, c in TAGCOL.items()}}
        log(f"Abias {acq} fixed {[round(x,3) for x in res[acq]['fixed_heldout_mean_tag']]}")
        log(f"Abias {acq} pool  {[round(x,3) for x in res[acq]['remaining_pool_mean_tag']]}")
    OUT["A_bias_fixed_eval"] = {"budgets": [8, 16, 24, 40, 56, 80, 120], **res}

# ===================================================================== B
if PART in ("all", "B"):
    from fishpipe import semisup
    MP = [0, 10, 20, 30, 50, 80]

    def rnd_pairs(n, need, rng, q):
        out, qs = [], set(q)
        while len(out) < need:
            i, j = int(rng.integers(0, n)), int(rng.integers(0, n))
            if i == j:
                continue
            pr = (min(i, j), max(i, j))
            if pr not in qs:
                qs.add(pr); out.append(pr)
        return out

    def pair_curve(X, y, seed, mode, yq=None):
        yq = y if yq is None else yq
        rng = np.random.default_rng(2000 * seed + (0 if mode == "random" else 1))
        q = []
        L = np.eye(X.shape[1])[:, :4]
        out = []
        for m in MP:
            need = m - len(q)
            if need > 0:
                q.extend(rnd_pairs(len(X), need, rng, q) if mode == "random"
                         else semisup.active_pairs(X, L, need, q, rng, n_ensemble=12, k=2))
            ml, cl = semisup._pairs_from_labels(q, yq)
            L = semisup.learn_warp(X, ml, cl, n_dim=4, reg=1.0)
            pred = semisup.cluster_in_warp(X, L, k=2, seed=seed)
            out.append(float(adjusted_rand_score(y, pred)))
        return np.array(out)

    OUT["B"] = {"m_pairs": MP}
    for fac in ("stripe", "belly"):
        y = labels[fac]
        for mode in ("random", "active"):
            r = np.array([pair_curve(Z, y, sd, mode) for sd in range(6)])
            OUT["B"][f"{fac}_{mode}"] = np.nanmean(r, 0).tolist()
            log(f"B {fac} {mode} {[round(x,3) for x in OUT['B'][f'{fac}_{mode}']]}")
        rn = np.array([pair_curve(Z, y, sd, "random",
                                  yq=np.random.default_rng(900 + sd).permutation(y))
                       for sd in range(6)])
        OUT["B"][f"{fac}_null"] = np.nanmean(rn, 0).tolist()
        log(f"B {fac} null {[round(x,3) for x in OUT['B'][f'{fac}_null']]}")

# ===================================================================== C
if PART in ("all", "C"):
    ch = labels["cheeks"].astype(bool)
    lof = LocalOutlierFactor(n_neighbors=20).fit(Z)
    s_lof = -lof.negative_outlier_factor_
    nn = NearestNeighbors(n_neighbors=11).fit(Z)
    dd, _ = nn.kneighbors(Z)
    s_knn = dd[:, 1:].mean(1)

    def cost(order, k):
        h = np.cumsum(ch[order])
        w = np.where(h >= k)[0]
        return int(w[0] + 1) if len(w) else np.nan

    res = {"prevalence": float(ch.mean())}
    for nm, s in (("lof", s_lof), ("knn", s_knn)):
        o = np.argsort(s)[::-1]
        res[nm] = {k: cost(o, k) for k in (1, 3, 5)}
        res[nm + "_rev"] = {k: cost(np.argsort(s), k) for k in (1, 3, 5)}
        res[nm + "_auc_for_cheek"] = float(roc_auc_score(ch, s))
    rr = np.array([[cost(np.random.default_rng(4000 + sd).permutation(N), k) for k in (1, 3, 5)]
                   for sd in range(20)], float)
    res["random_mean"] = rr.mean(0).tolist()
    # BIG random reference (10k orderings) + one-sided p-values for the deterministic orders
    big = np.array([[cost(np.random.default_rng(50000 + sd).permutation(N), k) for k in (1, 3, 5)]
                    for sd in range(2000)], float)
    res["random_mean_2000"] = big.mean(0).tolist()
    for nm in ("lof", "knn", "lof_rev", "knn_rev"):
        v = [res[nm][k] for k in (1, 3, 5)]
        res[nm + "_p_worse_than_random"] = [float((big[:, i] >= v[i]).mean()) for i in range(3)]
        res[nm + "_p_better_than_random"] = [float((big[:, i] <= v[i]).mean()) for i in range(3)]
    log(f"C {json.dumps(res)}")
    OUT["C"] = res

# ===================================================================== D
if PART in ("all", "D"):
    OUT["D"] = {}
    for acq in ("random", "uncertainty"):
        ins, out_, ari = [], [], []
        for sd in range(12):
            rs = np.random.default_rng(7000 + sd)
            sub = np.sort(rs.choice(N, 40, replace=False))
            oi = np.setdiff1d(np.arange(N), sub)
            sc = StandardScaler().fit(Xcat[sub])
            p40 = PCA(n_components=24, random_state=0).fit(sc.transform(Xcat[sub]))
            Z40 = p40.transform(sc.transform(Xcat[sub]))
            Zout = p40.transform(sc.transform(Xcat[oi]))
            Zboth = np.vstack([Z40, Zout])
            Yboth = np.vstack([Y[sub], Y[oi]])
            a, r, ao = tag_curve(Z40, Y[sub], joint8[sub], [8, 16, 24], sd, acq, cov="diag",
                                 eval_fixed=None)
            # held-out eval: refit models on the labelled study rows, score the 210 outsiders
            rng = np.random.default_rng(1000 * sd + (0 if acq == "random" else 1))
            ins.append(a); ari.append(r)
            # recompute held-out by re-running with eval on outsiders via combined array
            a2, _, ao2 = tag_curve(Zboth, Yboth, np.concatenate([joint8[sub], joint8[oi]]),
                                   [8, 16, 24], sd, acq, pool=np.arange(40), cov="diag",
                                   eval_fixed=np.arange(40, 250))
            out_.append(ao2)
        ins = np.array(ins); out_ = np.array(out_); ari = np.array(ari)
        OUT["D"][acq] = {"instudy_mean_tag": np.nanmean(np.nanmean(ins, 2), 0).tolist(),
                         "heldout210_mean_tag": np.nanmean(np.nanmean(out_, 2), 0).tolist(),
                         "instudy_per_factor": per_factor(ins),
                         "ari8": np.nanmean(ari, 0).tolist()}
        log(f"D {acq} in {[round(x,3) for x in OUT['D'][acq]['instudy_mean_tag']]} "
            f"held {[round(x,3) for x in OUT['D'][acq]['heldout210_mean_tag']]} "
            f"ari {[round(x,3) for x in OUT['D'][acq]['ari8']]}")
    aa = []
    for sd in range(12):
        rs = np.random.default_rng(7000 + sd)
        sub = np.sort(rs.choice(N, 40, replace=False))
        sc = StandardScaler().fit(Xcat[sub])
        p40 = PCA(n_components=24, random_state=0).fit(sc.transform(Xcat[sub]))
        Z40 = p40.transform(sc.transform(Xcat[sub]))
        pi = np.random.default_rng(800 + sd).permutation(40)
        a, _, _ = tag_curve(Z40, Y[sub][pi], joint8[sub][pi], [8, 16, 24], sd, "random", cov="diag")
        aa.append(a)
    OUT["D"]["null_mean_tag"] = np.nanmean(np.nanmean(np.array(aa), 2), 0).tolist()
    log(f"D null {[round(x,3) for x in OUT['D']['null_mean_tag']]}")

# ===================================================================== silhouettes
if PART in ("all", "sil"):
    import umap
    U = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=0).fit_transform(Z)
    OUT["sil_umap2_M0"] = {k: float(silhouette_score(U, labels[k])) for k in C.FACTORS}
    OUT["sil_umap2_M0"]["joint8"] = float(silhouette_score(U, joint8))
    log(f"sil M0 {OUT['sil_umap2_M0']}")

print("\n===JSON===")
print(json.dumps(OUT, indent=1, default=float))
