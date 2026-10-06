"""INDEPENDENT verification of e4_byoa_descriptors.

Does NOT import e4_byoa_descriptors. Re-derives descriptors, CV protocol, nulls, ARI,
LODO, continuous recovery, cheek detectability and the disentanglement claim from scratch
(only `common as C` is used, for data loading).

usage:  python verify_e4_byoa_descriptors.py [stage ...]
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import rankdata
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import adjusted_rand_score, balanced_accuracy_score
from sklearn.model_selection import KFold, LeaveOneGroupOut, RepeatedStratifiedKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import common as C

SCRATCH = C.CACHE / "verify_e4"
SCRATCH.mkdir(parents=True, exist_ok=True)
OUT = {}
T0 = time.time()


def log(m):
    print(f"[{time.time()-T0:7.1f}s] {m}", flush=True)


# ------------------------------------------------------------------ my own protocol
def bacc_cv(X, y, n_splits=5, n_repeats=4, seed=0):
    X = np.asarray(X, float); y = np.asarray(y)
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    s = []
    for tr, te in cv.split(X, y):
        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(max_iter=4000, C=1.0, class_weight="balanced"))
        clf.fit(X[tr], y[tr])
        s.append(balanced_accuracy_score(y[te], clf.predict(X[te])))
    return float(np.mean(s)), float(np.std(s))


def bacc_null(X, y, n_shuffles=6, seed=999):
    """my own shuffled-label null, different seed / different #shuffles"""
    rng = np.random.default_rng(seed)
    X = np.asarray(X, float); y = np.asarray(y)
    s = []
    for i in range(n_shuffles):
        yp = rng.permutation(y)
        cv = StratifiedKFold(5, shuffle=True, random_state=1000 + i)
        for tr, te in cv.split(X, yp):
            clf = make_pipeline(StandardScaler(),
                                LogisticRegression(max_iter=4000, C=1.0,
                                                   class_weight="balanced"))
            clf.fit(X[tr], yp[tr])
            s.append(balanced_accuracy_score(yp[te], clf.predict(X[te])))
    return float(np.mean(s)), float(np.std(s))


def lodo(X, y, groups):
    X = np.asarray(X, float); y = np.asarray(y)
    s = []
    for tr, te in LeaveOneGroupOut().split(X, y, groups):
        if len(np.unique(y[te])) < 2 or len(np.unique(y[tr])) < 2:
            continue
        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(max_iter=4000, C=1.0, class_weight="balanced"))
        clf.fit(X[tr], y[tr])
        s.append(balanced_accuracy_score(y[te], clf.predict(X[te])))
    return float(np.mean(s)), float(np.std(s)), len(s)


def gmm_ari(X, ydict, max_k=6, seed=0):
    from sklearn.decomposition import PCA
    from sklearn.mixture import GaussianMixture
    Z = PCA(2, random_state=seed).fit_transform(StandardScaler().fit_transform(np.asarray(X, float)))
    best, bb, bl = 1, np.inf, np.zeros(len(Z), int)
    for k in range(1, max_k + 1):
        gm = GaussianMixture(k, covariance_type="full", n_init=10, random_state=seed,
                             reg_covar=1e-5).fit(Z)
        b = gm.bic(Z)
        if b < bb:
            bb, best, bl = b, k, gm.predict(Z)
    return best, {f: float(adjusted_rand_score(v, bl)) for f, v in ydict.items()}


def auc1(x, y):
    y = np.asarray(y).astype(bool)
    r = rankdata(np.asarray(x, float))
    n1 = int(y.sum()); n0 = len(y) - n1
    a = (r[y].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0)
    return float(max(a, 1 - a))


def oof_ridge(X, y, n_seeds=4):
    X = np.asarray(X, float); y = np.asarray(y, float)
    pred = np.zeros(len(y))
    for s in range(n_seeds):
        p = np.zeros(len(y))
        for tr, te in KFold(5, shuffle=True, random_state=s).split(X):
            m = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 5, 25)))
            m.fit(X[tr], y[tr]); p[te] = m.predict(X[te])
        pred += p / n_seeds
    return float(1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def oof_proj(X, y, n_seeds=4, seed0=0):
    X = np.asarray(X, float); y = np.asarray(y)
    oof = np.zeros(len(y))
    for s in range(n_seeds):
        p = np.zeros(len(y))
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed0 + s).split(X, y):
            sc = StandardScaler().fit(X[tr])
            c = LogisticRegression(max_iter=6000, C=1.0, class_weight="balanced")
            c.fit(sc.transform(X[tr]), y[tr])
            p[te] = c.decision_function(sc.transform(X[te]))
        oof += p / n_seeds
    return oof


def pr(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    if a.std() < 1e-12 or b.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def partial(a, b, ctrl):
    a, b, c = (np.asarray(v, float) for v in (a, b, ctrl))
    ra = a - np.polyval(np.polyfit(c, a, 1), c)
    rb = b - np.polyval(np.polyfit(c, b, 1), c)
    return pr(ra, rb)


# ------------------------------------------------------------------ data
log("loading atlas ...")
fcd, art, mesh = C.atlas_data()
df, labels = C.load_gt(list(fcd.names))
N = fcd.colors.shape[0]
donor = df["specimen_index"].to_numpy().astype(int)
FACTORS = ("belly", "tail", "stripe", "cheeks")
log(f"N={N} faces={fcd.colors.shape[1]} verts={mesh.vertices.shape[0]}")

# --- independent GT join check (do not trust C.load_gt ordering blindly)
import pandas as pd
raw = pd.read_csv(C.GT_CSV)
m = dict(zip(raw["sample_tag"], raw.index))
idx = np.array([m[n] for n in fcd.names])
assert (raw.loc[idx, "sample_tag"].to_numpy() == np.array(list(fcd.names))).all()
for col in ("belly_hue", "tail_hue", "stripe_count", "rosy_cheeks_present", "specimen_index"):
    assert np.allclose(raw.loc[idx, col].to_numpy(), df[col].to_numpy()), col
OUT["gt_join_ok"] = True
OUT["label_stripe_matches_count5"] = bool(
    (labels["stripe"] == (raw.loc[idx, "stripe_count"].to_numpy() == 5).astype(int)).all())
OUT["class_balance"] = {f: [int((labels[f] == 0).sum()), int((labels[f] == 1).sum())]
                        for f in FACTORS}

# belly/tail label = kmeans threshold: where is the split, is there a gap?
for f, col in (("belly", "belly_hue"), ("tail", "tail_hue")):
    v = df[col].to_numpy(); o = np.argsort(v); vs = v[o]; ls = labels[f][o]
    ch = np.where(np.diff(ls) != 0)[0]
    OUT[f"{f}_label_boundary_gap"] = float(vs[ch[0] + 1] - vs[ch[0]]) if len(ch) == 1 else None
    OUT[f"{f}_max_gap_in_sorted_hue"] = float(np.max(np.diff(vs)))
    OUT[f"{f}_n_label_changes_in_sorted_hue"] = int(len(ch))

# is donor predictable from the GT generative params (i.e. donor/factor confound)?
gtp = df[["base_color_hue", "base_color_sat", "base_color_val", "stripe_count",
          "stripe_spacing", "stripe_width", "stripe_longitudinal_offset", "belly_hue",
          "belly_strength", "belly_translation", "tail_hue", "tail_strength",
          "rosy_cheeks_present", "rosy_cheeks_hue", "rosy_cheeks_strength",
          "rosy_cheeks_translation_y", "rosy_cheeks_translation_z"]].to_numpy(float)
OUT["donor_from_GTparams_acc"] = bacc_cv(gtp, donor)[0]
for f in FACTORS:
    tab = np.zeros((7, 2))
    for g in range(7):
        for c in (0, 1):
            tab[g, c] = ((donor == g) & (labels[f] == c)).sum()
    from scipy.stats import chi2_contingency
    OUT[f"donor_vs_{f}_chi2_p"] = float(chi2_contingency(tab)[1])

stages = set(sys.argv[1:]) or {"desc", "cheek", "disent", "spec"}

# ------------------------------------------------------------------ my descriptors
lab_all = None


def get_lab():
    global lab_all
    if lab_all is None:
        p = SCRATCH / "lab.npy"
        if p.exists():
            lab_all = np.load(p, mmap_mode="r")
        else:
            from fishpipe.features import rgb_to_lab
            lab_all = rgb_to_lab(fcd.colors).astype(np.float32)
            np.save(p, lab_all)
    return lab_all


def my_hist24():
    """area-weighted 24-cluster Lab histogram; MY seed/subsample (not the library's)."""
    p = SCRATCH / "M1_hist24.npy"
    if p.exists():
        return np.load(p)
    lab = np.asarray(get_lab(), np.float64)
    rng = np.random.default_rng(7)
    pooled = lab.reshape(-1, 3)
    fit = rng.choice(pooled.shape[0], size=150_000, replace=False)
    km = KMeans(24, n_init=6, random_state=7).fit(pooled[fit])
    areas = fcd.areas
    X = np.zeros((N, 24))
    for i in range(N):
        l = km.predict(lab[i])
        h = np.bincount(l, weights=areas, minlength=24)
        X[i] = h / h.sum()
    np.save(p, X)
    return X


def my_region128():
    """area-weighted mean Lab in 128 centroid-KMeans regions; MY seed."""
    p = SCRATCH / "M2_region128.npy"
    if p.exists():
        return np.load(p)
    cents = mesh.face_centroids()
    reg = KMeans(128, n_init=4, random_state=17).fit_predict(cents)
    lab = np.asarray(get_lab(), np.float64)
    areas = fcd.areas
    w = np.bincount(reg, weights=areas, minlength=128); w[w == 0] = 1
    X = np.zeros((N, 128, 3))
    for i in range(N):
        for c in range(3):
            X[i, :, c] = np.bincount(reg, weights=areas * lab[i, :, c], minlength=128) / w
    X = X.reshape(N, -1)
    np.save(p, X)
    return X


def my_flat4000():
    p = SCRATCH / "M3_flat4000.npy"
    if p.exists():
        return np.load(p)
    rng = np.random.default_rng(2024)
    idx = np.sort(rng.choice(fcd.colors.shape[1], size=4000, replace=False))
    X = np.asarray(get_lab()[:, idx, :], np.float64).reshape(N, -1)
    np.save(p, X)
    return X


if "desc" in stages:
    log("building MY descriptors ...")
    M = {"M1_hist24": my_hist24(), "M2_region128mean": my_region128(),
         "M3_flat4000": my_flat4000()}
    res = {}
    for n, X in M.items():
        log(f"  {n} {X.shape}")
        r = {"dim": int(X.shape[1])}
        k, aris = gmm_ari(X, labels)
        r["gmm_ncomp"] = k
        r["gmm_ari"] = aris
        for f in FACTORS:
            a, sd = bacc_cv(X, labels[f])
            na, _ = bacc_null(X, labels[f])
            r[f] = {"acc": a, "sd": sd, "null": na}
            log(f"    {f:7s} acc={a:.3f} null={na:.3f} ari={aris[f]:+.3f}")
        r["donor7"] = bacc_cv(X, donor)[0]
        r["donor7_null"] = bacc_null(X, donor)[0]
        log(f"    donor7 acc={r['donor7']:.3f} null={r['donor7_null']:.3f}")
        for f in FACTORS:
            a, sd, nf = lodo(X, labels[f], donor)
            r[f]["lodo"] = a; r[f]["lodo_sd"] = sd; r[f]["lodo_folds"] = nf
        log("    LODO " + " ".join(f"{f}={r[f]['lodo']:.3f}" for f in FACTORS))
        res[n] = r
    # continuous recovery on region means
    cont = {}
    for tgt in ("belly_hue", "tail_hue", "base_color_hue", "stripe_spacing",
                "stripe_width", "stripe_longitudinal_offset"):
        cont[tgt] = oof_ridge(M["M2_region128mean"], df[tgt].to_numpy())
    res["M2_continuous_R2"] = cont
    log("  cont R2 " + " ".join(f"{k}={v:+.2f}" for k, v in cont.items()))
    OUT["my_descriptors"] = res
    json.dump(OUT, open(SCRATCH / "out.json", "w"), indent=1)

# ------------------------------------------------------------------ cheeks
if "cheek" in stages:
    log("cheek control ...")
    lab = np.asarray(get_lab(), np.float64)
    w = fcd.areas / fcd.areas.sum()
    y = labels["cheeks"]
    ch = np.sqrt(lab[..., 1] ** 2 + lab[..., 2] ** 2)
    ha = np.degrees(np.arctan2(lab[..., 2], lab[..., 1]))
    cheekres = {}
    for tag, (cmin, hmax) in {"agent_c15_h40": (15, 40), "c10_h50": (10, 50),
                              "c5_h60": (5, 60), "c25_h30": (25, 30),
                              "c8_h70": (8, 70)}.items():
        red = (ch > cmin) & (np.abs(ha) < hmax)
        rf = (red * w[None, :]).sum(1)
        cheekres[tag] = {"auc": auc1(rf, y), "present_mean": float(rf[y == 1].mean()),
                         "absent_mean": float(rf[y == 0].mean()),
                         "frac_present_zero": float((rf[y == 1] <= rf[y == 0].max()).mean())}
        log(f"  atlas red rule {tag:14s} AUC={cheekres[tag]['auc']:.3f} "
            f"pres={cheekres[tag]['present_mean']*100:.3f}% abs={cheekres[tag]['absent_mean']*100:.4f}%")
    # nearest-planted-pigment rule from common.py (population constants, not tuned here)
    pm = C.pigment_masks(lab)
    cheek_frac = (pm["cheek"] * w[None, :]).sum(1)
    cheekres["common_pigment_rule"] = {"auc": auc1(cheek_frac, y),
                                       "present_mean": float(cheek_frac[y == 1].mean()),
                                       "absent_mean": float(cheek_frac[y == 0].mean())}
    log(f"  atlas C.pigment_masks cheek AUC={cheekres['common_pigment_rule']['auc']:.3f}")
    # ---- THE control the agent did not run: is the patch present PRE-module?
    nat = C.native_face_colors(verbose=False)
    assert list(nat["names"]) == list(fcd.names)
    nat_cheek = nat["masks"][:, C.PIGMENTS.index("cheek")]
    cheekres["NATIVE_pre_module_cheek_frac"] = {
        "auc": auc1(nat_cheek, y),
        "present_mean": float(nat_cheek[y == 1].mean()),
        "absent_mean": float(nat_cheek[y == 0].mean()),
        "n_present_with_zero_native": int((nat_cheek[y == 1] <= nat_cheek[y == 0].max()).sum()),
        "n_present": int((y == 1).sum()),
    }
    log(f"  NATIVE (pre-module) cheek AUC={cheekres['NATIVE_pre_module_cheek_frac']['auc']:.3f} "
        f"pres={nat_cheek[y==1].mean()*100:.3f}% abs={nat_cheek[y==0].mean()*100:.4f}%")
    # atlas red_frac (agent rule) vs native, restricted to present specimens
    red = (ch > 15) & (np.abs(ha) < 40)
    rf = (red * w[None, :]).sum(1)
    p = y == 1
    cheekres["corr_atlas_vs_native_present"] = pr(rf[p], nat_cheek[p])
    cheekres["native_vs_tz_present"] = pr(nat_cheek[p], df["rosy_cheeks_translation_z"].to_numpy()[p])
    cheekres["atlasred_vs_tz_present"] = pr(rf[p], df["rosy_cheeks_translation_z"].to_numpy()[p])
    thr = rf[~p].max()
    lost = p & (rf <= thr)
    cheekres["n_present_no_atlas_red"] = int(lost.sum())
    cheekres["native_frac_of_those_lost_mean"] = float(nat_cheek[lost].mean())
    cheekres["native_frac_of_kept_mean"] = float(nat_cheek[p & (rf > thr)].mean())
    log(f"  {int(lost.sum())} present specimens have NO atlas red; their NATIVE cheek "
        f"fraction mean={nat_cheek[lost].mean()*100:.4f}% vs kept {nat_cheek[p&(rf>thr)].mean()*100:.4f}%")
    OUT["cheek"] = cheekres
    json.dump(OUT, open(SCRATCH / "out.json", "w"), indent=1)

# ------------------------------------------------------------------ disentanglement
if "disent" in stages:
    log("disentanglement ...")
    cnt = df["stripe_count"].to_numpy(float)
    spc = df["stripe_spacing"].to_numpy(float)
    wid = df["stripe_width"].to_numpy(float)
    off = df["stripe_longitudinal_offset"].to_numpy(float)
    extent = (cnt - 1) * np.abs(spc)          # geometric extent of the stripe field
    d = {"gt": {"r_count_spacing": pr(cnt, spc),
                "r_extent_count": pr(extent, cnt),
                "r_extent_spacing": pr(extent, spc)}}
    DC = Path("cache/e4_desc")
    cand = {"MY_M3_flat4000": my_flat4000(), "MY_M2_region128": my_region128()}
    for nm in ("D10c_blobs", "D10b_endler", "D1_area_hist24", "D2_spatial_flatten"):
        f = DC / f"{nm}.npy"
        if f.exists():
            cand["cached_" + nm] = np.load(f)
    for nm, X in cand.items():
        proj = oof_proj(X, labels["stripe"])
        row = {
            "acc": bacc_cv(X, labels["stripe"])[0],
            "r_count": abs(pr(proj, cnt)), "r_spacing": abs(pr(proj, spc)),
            "r_width": abs(pr(proj, wid)), "r_offset": abs(pr(proj, off)),
            "r_extent": abs(pr(proj, extent)),
            "spacing_given_count": abs(partial(proj, spc, cnt)),
            "count_given_spacing": abs(partial(proj, cnt, spc)),
        }
        d[nm] = row
        log(f"  {nm:24s} acc={row['acc']:.3f} |r| cnt={row['r_count']:.3f} "
            f"spc={row['r_spacing']:.3f} EXTENT={row['r_extent']:.3f} "
            f"cnt|spc={row['count_given_spacing']:.3f} spc|cnt={row['spacing_given_count']:.3f}")
    OUT["disent"] = d
    json.dump(OUT, open(SCRATCH / "out.json", "w"), indent=1)

# ------------------------------------------------------------------ spectral k sweep
if "spec" in stages:
    log("spectral k sweep (own projection onto cached atlas eigenbasis) ...")
    import scipy.sparse as sp
    U = np.load(C.ATLAS_DS.cache_dir / "lap_U_300.npy")
    wev = np.load(C.ATLAS_DS.cache_dir / "lap_w_300.npy")
    nv = mesh.vertices.shape[0]
    assert U.shape[0] == nv, (U.shape, nv)
    # verify the cached basis really is the atlas mesh's normalised-Laplacian eigenbasis
    fv = mesh.face_v
    e = np.vstack([fv[:, [0, 1]], fv[:, [1, 2]], fv[:, [2, 0]]])
    e = np.vstack([e, e[:, ::-1]])
    A = sp.coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(nv, nv)).tocsr()
    A.data[:] = 1.0
    A = ((A + A.T) > 0).astype(np.float64)
    deg = np.asarray(A.sum(1)).ravel()
    D = sp.diags(1.0 / np.sqrt(np.maximum(deg, 1e-12)))
    L = (sp.identity(nv) - D @ A @ D).tocsr()
    k = 5
    resid = np.abs(L @ U[:, :k] - U[:, :k] * wev[:k]).max()
    OUT["eigenbasis_residual_max"] = float(resid)
    log(f"  eigenbasis check |LU-Uw|max={resid:.2e}  (must be ~0)")
    # my own vertex Lab (area-weighted from faces)
    pth = SCRATCH / "vlab.npy"
    if pth.exists():
        vlab = np.load(pth, mmap_mode="r")
    else:
        lab = np.asarray(get_lab(), np.float64)
        areas = fcd.areas
        varea = np.zeros(nv); np.add.at(varea, fv.ravel(), np.repeat(areas, 3))
        varea[varea == 0] = 1
        vlab = np.zeros((N, nv, 3), np.float32)
        for i in range(N):
            for c in range(3):
                acc = np.zeros(nv)
                np.add.at(acc, fv.ravel(), np.repeat(areas * lab[i, :, c], 3))
                vlab[i, :, c] = acc / varea
        np.save(pth, vlab)
    sig = {c: np.asarray(vlab[..., i], np.float64) for i, c in enumerate("Lab")}
    for c in sig:
        sig[c] = sig[c] - sig[c].mean(1, keepdims=True)
    coef = {c: sig[c] @ U[:, :250] for c in sig}
    sweep = {}
    for kk in (1, 3, 10, 20, 40, 80, 150, 250):
        Xk = np.concatenate([coef[c][:, :kk] for c in "Lab"], 1)
        row = {}
        for f in FACTORS:
            a, _ = bacc_cv(Xk, labels[f])
            row[f] = a
        sweep[kk] = row
        log(f"  k={kk:3d} " + " ".join(f"{f}={row[f]:.3f}" for f in FACTORS))
    OUT["k_sweep"] = sweep
    XL = coef["L"][:, :40]
    OUT["D7_Lonly_k40"] = {f: bacc_cv(XL, labels[f])[0] for f in FACTORS}
    OUT["D7_Lonly_k40_null"] = {f: bacc_null(XL, labels[f])[0] for f in FACTORS}
    log("  D7 L-only k40 " + " ".join(f"{f}={OUT['D7_Lonly_k40'][f]:.3f}" for f in FACTORS))
    X40 = np.concatenate([coef[c][:, :40] for c in "Lab"], 1)
    OUT["D5_k40_Lab"] = {f: bacc_cv(X40, labels[f])[0] for f in FACTORS}
    OUT["D5_k40_Lab_donor7"] = bacc_cv(X40, donor)[0]
    log("  D5 k40 Lab " + " ".join(f"{f}={OUT['D5_k40_Lab'][f]:.3f}" for f in FACTORS)
        + f" donor={OUT['D5_k40_Lab_donor7']:.3f}")
    json.dump(OUT, open(SCRATCH / "out.json", "w"), indent=1)

json.dump(OUT, open(SCRATCH / "out.json", "w"), indent=1)
print(json.dumps(OUT, indent=1))
log("done")
