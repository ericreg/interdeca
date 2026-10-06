"""ADVERSARIAL re-derivation of E3 headline numbers. Independent of e3_builtin_eda.py.

Stages (select with argv[1]): gt | feat | cvfull | smalln | all
Only `common as C` is imported from the harness; every estimator below is written here.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

from skimage import color as skcolor  # noqa: E402
from sklearn.cluster import KMeans  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import adjusted_rand_score, balanced_accuracy_score, silhouette_score  # noqa: E402
from sklearn.mixture import GaussianMixture  # noqa: E402
from sklearn.model_selection import RepeatedStratifiedKFold  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from threadpoolctl import threadpool_limits  # noqa: E402

T0 = time.time()
OUT = C.RESULTS / "verify_e3.json"
RES: dict = {}


def log(*a):
    print(f"[{time.time()-T0:7.1f}s]", *a, flush=True)


# --------------------------------------------------------------- my own estimators
def my_area_hist(colors, areas, K=24, seed=0, npool=200_000):
    """Module's area-weighted colour-composition descriptor, re-implemented here."""
    N, Nf, _ = colors.shape
    lab = skcolor.rgb2lab(colors.astype(np.float64) / 255.0)      # (N,Nf,3)
    pooled = lab.reshape(N * Nf, 3)
    rng = np.random.default_rng(seed)
    idx = rng.choice(pooled.shape[0], size=min(npool, pooled.shape[0]), replace=False)
    with threadpool_limits(limits=8):
        km = KMeans(n_clusters=K, n_init=5, random_state=seed).fit(pooled[idx])
    Cn = km.cluster_centers_                                       # (K,3)
    X = np.zeros((N, K))
    sq = (Cn ** 2).sum(1)
    for i in range(N):
        d = sq[None, :] - 2.0 * lab[i] @ Cn.T                      # argmin of ||x-c||^2
        a = np.argmin(d, axis=1)
        X[i] = np.bincount(a, weights=areas, minlength=K)
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n


def my_flatten(colors, sub, seed=42):
    rng = np.random.default_rng(seed)
    idx = rng.choice(colors.shape[1], size=sub, replace=False)
    idx.sort()
    return skcolor.rgb2lab(colors[:, idx, :].astype(np.float64) / 255.0).reshape(len(colors), -1)


def my_bcv(X, y, n_splits=5, n_repeats=4, seed=0):
    X = np.asarray(X, float)
    y = np.asarray(y)
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    s = []
    with threadpool_limits(limits=1):
        for tr, te in cv.split(X, y):
            clf = make_pipeline(StandardScaler(),
                                LogisticRegression(max_iter=4000, C=1.0, class_weight="balanced"))
            clf.fit(X[tr], y[tr])
            s.append(balanced_accuracy_score(y[te], clf.predict(X[te])))
    return float(np.mean(s)), float(np.std(s))


def my_gmm_bic(X2, ks=range(1, 9), seed=0):
    bics, mods = [], {}
    with threadpool_limits(limits=1):
        for k in ks:
            g = GaussianMixture(k, covariance_type="full", n_init=5, random_state=seed).fit(X2)
            bics.append(g.bic(X2))
            mods[k] = g
        ks = list(ks)
        kk = ks[int(np.argmin(bics))]
        lab = mods[kk].predict(X2) if kk > 1 else np.zeros(len(X2), int)
    return kk, lab, np.array(bics)


def absr(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    return abs(float(np.corrcoef(a, b)[0, 1]))


# --------------------------------------------------------------- data
log("loading atlas ...")
fcd, art, mesh = C.atlas_data()
names = list(fcd.names)
df, labels = C.load_gt(names)
N = len(names)
y_bt = C.joint_label(labels, ("belly", "tail"))
donor = df["specimen_index"].to_numpy().astype(int)
log(f"N={N} faces={fcd.colors.shape[1]}")

STAGE = sys.argv[1] if len(sys.argv) > 1 else "all"

# =============================================================== A. ground-truth-only
if STAGE in ("gt", "all"):
    import diptest
    log("--- A. ground truth / oracle ---")
    a = {}
    # ordering sanity: names <-> csv row alignment
    a["names_match_csv_tag"] = bool((df["sample_tag"].to_numpy() == np.array(names)).all())
    a["n_unique_names"] = int(len(set(names)))
    for c in ("belly_hue", "tail_hue"):
        d, p = diptest.diptest(np.ascontiguousarray(df[c].to_numpy(float)))
        a[f"dip_{c}"] = [float(d), float(p)]
    a["perfect_belly_only_ari"] = float(adjusted_rand_score(y_bt, labels["belly"]))
    a["perfect_tail_only_ari"] = float(adjusted_rand_score(y_bt, labels["tail"]))
    G = df[["belly_hue", "tail_hue"]].to_numpy(float)
    G = (G - G.mean(0)) / G.std(0)
    k_o, lab_o, bic_o = my_gmm_bic(G)
    a["oracle_gmm_k"] = int(k_o)
    a["oracle_gmm_ari"] = float(adjusted_rand_score(y_bt, lab_o))
    with threadpool_limits(limits=1):
        km4 = KMeans(4, n_init=10, random_state=0).fit_predict(G)
        gm4 = GaussianMixture(4, covariance_type="full", n_init=5, random_state=0).fit(G)
    a["oracle_kmeans_k4_ari"] = float(adjusted_rand_score(y_bt, km4))
    # the combos force k=4 with a GMM, not KMeans -> is the quoted ceiling protocol-matched?
    a["oracle_gmm_k4_ari"] = float(adjusted_rand_score(y_bt, gm4.predict(G)))
    # generator's LATENT mode assignment for belly/tail, replayed from the generator RNG
    try:
        rng_np = np.random
        rng_np.seed(42)
        n = N
        _ = rng_np.normal(0, 1, n); _ = rng_np.normal(0, 1, n); _ = rng_np.normal(0, 1, n)
        _ = rng_np.choice([4, 5], n, p=[0.8, 0.2])
        _ = rng_np.normal(0, 1, n); _ = rng_np.normal(0, 1, n); _ = rng_np.normal(0, 1, n)
        bc = rng_np.uniform(-0.1, 0.1, size=2); bassign = rng_np.randint(0, 2, size=n)
        bshift = bc[bassign] + rng_np.normal(0.0, 0.015, size=n)
        _ = np.clip(rng_np.normal(1 - 0.05, 0.05, n), 0, 1)
        _ = rng_np.normal(0, 1, n)
        tc = rng_np.uniform(-0.1, 0.1, size=2); tassign = rng_np.randint(0, 2, size=n)
        tshift = tc[tassign] + rng_np.normal(0.0, 0.015, size=n)
        a["latent_replay_ok"] = None
        a["belly_mode_centers"] = [float(v) for v in bc]
        a["tail_mode_centers"] = [float(v) for v in tc]
        a["belly_mode_gap_in_sd"] = float(abs(bc[0] - bc[1]) / 0.015)
        a["tail_mode_gap_in_sd"] = float(abs(tc[0] - tc[1]) / 0.015)
    except Exception as e:  # pragma: no cover
        a["latent_replay_error"] = str(e)
    RES["A_gt"] = a
    print(json.dumps(a, indent=1))

# =============================================================== B. descriptor + morphospace
if STAGE in ("feat", "all"):
    log("--- B. area_hist K=24 rebuilt here ---")
    b = {}
    X24 = my_area_hist(fcd.colors, fcd.areas, K=24, seed=0)
    log(f"  built {X24.shape}")
    p = PCA(n_components=10, random_state=0).fit(X24)
    Z = p.transform(X24)
    b["evr_pc1"] = float(p.explained_variance_ratio_[0])
    b["evr_pc1_pc2"] = float(p.explained_variance_ratio_[:2].sum())
    X2 = Z[:, :2]
    kk, lab, bics = my_gmm_bic(X2)
    b["gmm_chosen_k"] = int(kk)
    b["ari_belly_x_tail"] = float(adjusted_rand_score(y_bt, lab))
    b["ari_belly"] = float(adjusted_rand_score(labels["belly"], lab))
    b["ari_tail"] = float(adjusted_rand_score(labels["tail"], lab))
    b["silhouette"] = float(silhouette_score(X2, lab))
    with threadpool_limits(limits=1):
        g4 = GaussianMixture(4, covariance_type="full", n_init=5, random_state=0).fit(X2)
    b["ari_bt_at_k4"] = float(adjusted_rand_score(y_bt, g4.predict(X2)))
    # supervised readout on PC1-2 + its own shuffled null
    rng = np.random.default_rng(0)
    b["cv_top2"] = {}
    b["cv_top2_null"] = {}
    for f in C.FACTORS:
        b["cv_top2"][f] = list(my_bcv(X2, labels[f]))
        b["cv_top2_null"][f] = list(my_bcv(X2, rng.permutation(labels[f])))
    # what does each PC track?
    b["absr_PC_vs_belly_hue"] = [absr(Z[:, j], df["belly_hue"]) for j in range(10)]
    b["absr_PC_vs_tail_hue"] = [absr(Z[:, j], df["tail_hue"]) for j in range(10)]
    # dip-ranked axes
    import diptest
    dips = np.array([diptest.dipstat(np.ascontiguousarray(Z[:, j])) for j in range(10)])
    pick = np.sort(np.argsort(dips)[::-1][:2])
    b["dip_per_pc"] = [float(v) for v in dips]
    b["dip_picked_pcs"] = [int(v + 1) for v in pick]
    Xd = Z[:, pick]
    kd, labd, _ = my_gmm_bic(Xd)
    b["dip_chosen_k"] = int(kd)
    b["dip_ari_belly_x_tail"] = float(adjusted_rand_score(y_bt, labd))
    with threadpool_limits(limits=1):
        gd4 = GaussianMixture(4, covariance_type="full", n_init=5, random_state=0).fit(Xd)
    b["dip_ari_bt_at_k4"] = float(adjusted_rand_score(y_bt, gd4.predict(Xd)))
    b["dip_cv_top2"] = {f: list(my_bcv(Xd, labels[f])) for f in C.FACTORS}
    # donor correlation ratio of PC1/PC2
    def eta(x, g):
        x = np.asarray(x, float); tot = ((x - x.mean()) ** 2).sum()
        return float(np.sqrt(sum(len(x[g == u]) * (x[g == u].mean() - x.mean()) ** 2
                                 for u in np.unique(g)) / tot))
    b["donor_eta_pc12_area_hist"] = [eta(Z[:, 0], donor), eta(Z[:, 1], donor)]
    # robustness: different palette seed
    X24b = my_area_hist(fcd.colors, fcd.areas, K=24, seed=7)
    pb = PCA(n_components=10, random_state=0).fit(X24b)
    Zb = pb.transform(X24b)
    kb, labb, _ = my_gmm_bic(Zb[:, :2])
    b["seed7_evr_pc1_pc2"] = float(pb.explained_variance_ratio_[:2].sum())
    b["seed7_chosen_k"] = int(kb)
    b["seed7_ari_belly_x_tail"] = float(adjusted_rand_score(y_bt, labb))
    b["seed7_cv_top2_tail"] = list(my_bcv(Zb[:, :2], labels["tail"]))
    b["seed7_cv_top2_belly"] = list(my_bcv(Zb[:, :2], labels["belly"]))
    dips_b = np.array([diptest.dipstat(np.ascontiguousarray(Zb[:, j])) for j in range(10)])
    pick_b = np.sort(np.argsort(dips_b)[::-1][:2])
    b["seed7_dip_picked_pcs"] = [int(v + 1) for v in pick_b]
    kdb, labdb, _ = my_gmm_bic(Zb[:, pick_b])
    b["seed7_dip_ari"] = float(adjusted_rand_score(y_bt, labdb))
    b["seed7_dip_cv_tail"] = list(my_bcv(Zb[:, pick_b], labels["tail"]))
    # spatial_flatten PCA donor eta (headline: 0.86 / 0.15)
    XF = my_flatten(fcd.colors, 20000)
    pf = PCA(n_components=10, random_state=0).fit(XF)
    ZF = pf.transform(XF)
    b["donor_eta_pc12_flatten20000"] = [eta(ZF[:, 0], donor), eta(ZF[:, 1], donor)]
    b["flatten_evr_pc1_pc2"] = float(pf.explained_variance_ratio_[:2].sum())
    RES["B_feat"] = b
    print(json.dumps(b, indent=1))
    np.save(C.CACHE / "verify_e3_X24.npy", X24)

# =============================================================== C. full-descriptor CV
if STAGE in ("cvfull", "all"):
    log("--- C. full-descriptor supervised ceilings ---")
    c = {}
    Xp = C.CACHE / "verify_e3_X24.npy"
    X24 = np.load(Xp) if Xp.exists() else my_area_hist(fcd.colors, fcd.areas, K=24, seed=0)
    rng = np.random.default_rng(1)
    c["area_hist_K24"] = {}
    c["area_hist_K24_null"] = {}
    for f in C.FACTORS:
        c["area_hist_K24"][f] = list(my_bcv(X24, labels[f]))
        c["area_hist_K24_null"][f] = list(my_bcv(X24, rng.permutation(labels[f])))
        log(f"  K24 {f}: {c['area_hist_K24'][f][0]:.3f} null {c['area_hist_K24_null'][f][0]:.3f}")
    XF = my_flatten(fcd.colors, 20000)
    c["spatial_flatten_20000"] = {}
    c["spatial_flatten_20000_null"] = {}
    for f in C.FACTORS:
        c["spatial_flatten_20000"][f] = list(my_bcv(XF, labels[f]))
        c["spatial_flatten_20000_null"][f] = list(my_bcv(XF, rng.permutation(labels[f])))
        log(f"  flat20k {f}: {c['spatial_flatten_20000'][f][0]:.3f} "
            f"null {c['spatial_flatten_20000_null'][f][0]:.3f}")
    RES["C_cvfull"] = c
    print(json.dumps(c, indent=1))

# =============================================================== D. small n
if STAGE in ("smalln", "all"):
    from fishpipe import gating
    from fishpipe.data import FaceColorData
    log("--- D. small-n (own draws, own descriptor) ---")
    d = {}
    NDRAW = int(sys.argv[2]) if len(sys.argv) > 2 else 12
    for n in (25, 40):
        aris, absten_s, absten_c, ks = [], [], [], []
        for dd in range(NDRAW):
            rng = np.random.default_rng(1000 + 7 * dd + n)
            idx = rng.choice(N, size=n, replace=False)
            Xs = my_area_hist(fcd.colors[idx], fcd.areas, K=24, seed=0)
            Zs = PCA(n_components=min(10, n - 1), random_state=0).fit_transform(Xs)
            X2 = Zs[:, :2]
            kk, lab, _ = my_gmm_bic(X2)
            with threadpool_limits(limits=1):
                sc = gating.sigclust(X2, n_sim=1000, seed=0)
                cg = gating.consensus_gate(X2, seed=0)
            aris.append(adjusted_rand_score(y_bt[idx], lab))
            absten_s.append(sc["pvalue"] >= 0.05)
            absten_c.append(cg["chosen_k"] == 1)
            ks.append(kk)
        d[str(n)] = {"n_draws": NDRAW, "ari_mean": float(np.mean(aris)),
                     "ari_sd": float(np.std(aris, ddof=1)),
                     "abstain_sigclust": float(np.mean(absten_s)),
                     "abstain_consensus": float(np.mean(absten_c)),
                     "chosen_k_mean": float(np.mean(ks))}
        log(f"  n={n}: {d[str(n)]}")
    RES["D_smalln"] = d

# =============================================================== E. calibrated nulls / K sweep
if STAGE in ("nulls", "all"):
    log("--- E. 50-permutation nulls + K sweep ---")
    e = {}
    Xp = C.CACHE / "verify_e3_X24.npy"
    X24 = np.load(Xp) if Xp.exists() else my_area_hist(fcd.colors, fcd.areas, K=24, seed=0)
    Z = PCA(n_components=10, random_state=0).fit_transform(X24)
    X2 = Z[:, :2]
    rng = np.random.default_rng(123)
    NPERM = 50
    e["perm_null"] = {}
    for f in C.FACTORS:
        obs = my_bcv(X2, labels[f])[0]
        nulls = [my_bcv(X2, rng.permutation(labels[f]))[0] for _ in range(NPERM)]
        e["perm_null"][f] = {"obs": float(obs), "null_mean": float(np.mean(nulls)),
                             "null_sd": float(np.std(nulls)),
                             "null_p95": float(np.percentile(nulls, 95)),
                             "p_perm": float((1 + sum(x >= obs for x in nulls)) / (NPERM + 1))}
        log(f"  {f}: obs={obs:.3f} null={np.mean(nulls):.3f}+-{np.std(nulls):.3f} "
            f"p95={np.percentile(nulls,95):.3f} p={e['perm_null'][f]['p_perm']:.3f}")
    # K sweep: is the 2-D view really "worse" at larger K, or only lower EVR?
    e["K_sweep"] = {}
    for K in (16, 24, 30):
        XK = X24 if K == 24 else my_area_hist(fcd.colors, fcd.areas, K=K, seed=0)
        pk = PCA(n_components=10, random_state=0).fit(XK)
        ZK = pk.transform(XK)
        kk, lab, _ = my_gmm_bic(ZK[:, :2])
        e["K_sweep"][K] = {
            "evr12": float(pk.explained_variance_ratio_[:2].sum()),
            "ari_bt": float(adjusted_rand_score(y_bt, lab)),
            "cv_top2_belly": my_bcv(ZK[:, :2], labels["belly"])[0],
            "cv_top2_tail": my_bcv(ZK[:, :2], labels["tail"])[0],
            "cv_top2_stripe": my_bcv(ZK[:, :2], labels["stripe"])[0],
            "max_absr_pc12_belly_hue": max(absr(ZK[:, j], df["belly_hue"]) for j in range(2)),
            "max_absr_pc12_tail_hue": max(absr(ZK[:, j], df["tail_hue"]) for j in range(2)),
        }
        log(f"  K={K}: {e['K_sweep'][K]}")
    RES["E_nulls"] = e

# --------------------------------------------------------------- save
old = json.load(open(OUT)) if OUT.exists() else {}
old.update(RES)
json.dump(old, open(OUT, "w"), indent=2, default=float)
log(f"wrote {OUT}")
