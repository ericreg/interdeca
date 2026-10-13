"""E5 - expert-guided (human-in-the-loop) analysis of the ColorAtlas module output.

How much *expert input* is needed to surface structure that unsupervised analysis of the
atlas output cannot? All "expert" responses here are SIMULATED from the generator's ground
truth (a perfect, noiseless, instantly-available oracle). This is an upper bound on what a
real annotator could deliver; see CAVEATS in the JSON.

Representation (identical everywhere):
    Z = PCA(24) of standardise( [ standardise(area_hist(24)) | standardise(textonBoW(128)) ] )

Parts
  A  TAG feedback (6 binary factor tags), budgets M, random vs uncertainty acquisition
  B  PAIRWISE must-link/cannot-link constraints -> semisup.learn_warp -> GMM
  C  COLD-START discovery of the cheek patch by unsupervised novelty ranking
  D  SMALL-n replication of A at n=40 (the realistic study size)
"""
from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import adjusted_rand_score, balanced_accuracy_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import common as C
from e5_build_feats import load_feats

warnings.filterwarnings("ignore")

NAME = "e5_expert_guided"
RESULTS = C.RESULTS
T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


def jsonify(o):
    if isinstance(o, dict):
        return {str(k): jsonify(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonify(v) for v in o]
    if isinstance(o, (np.floating, float)):
        v = float(o)
        return None if not np.isfinite(v) else v
    if isinstance(o, (np.integer, int)):
        return int(o)
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return jsonify(o.tolist())
    return o


def ms(a, axis=0):
    """(mean, sd) ignoring NaNs; returns python floats/lists."""
    a = np.asarray(a, float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = np.nanmean(a, axis=axis)
        s = np.nanstd(a, axis=axis)
    return m, s


# ===========================================================================
# 0. data + representation
# ===========================================================================
log("loading cached descriptors ...")
names, H, Tex = load_feats()
df, labels = C.load_gt(list(names))
N = len(names)
joint8 = C.joint_label(labels, factors=("belly", "tail", "stripe"))

# blue<->purple / green<->cyan: verified against the generator's hue values
#   belly group 0 mean hue 0.564 (blue) vs group 1 mean 0.693 (purple)
#   tail  group 0 mean hue 0.359 (green) vs group 1 mean 0.406 (cyan)
TAGS = ["purple_belly", "blue_belly", "green_tail", "cyan_tail", "5_stripes", "rosy_cheeks"]
Y = np.stack([
    labels["belly"] == 1,
    labels["belly"] == 0,
    labels["tail"] == 0,
    labels["tail"] == 1,
    labels["stripe"] == 1,
    labels["cheeks"] == 1,
], axis=1).astype(int)                       # (N, 6) tag matrix
TAG2FACTOR = {"purple_belly": "belly", "blue_belly": "belly", "green_tail": "tail",
              "cyan_tail": "tail", "5_stripes": "stripe", "rosy_cheeks": "cheeks"}

Xcat_raw = np.hstack([StandardScaler().fit_transform(H),
                      StandardScaler().fit_transform(Tex)])
_sc_all = StandardScaler().fit(Xcat_raw)
_pca_all = PCA(n_components=24, random_state=0).fit(_sc_all.transform(Xcat_raw))
Z = _pca_all.transform(_sc_all.transform(Xcat_raw))          # (250, 24)
log(f"Z {Z.shape}  explained var {_pca_all.explained_variance_ratio_.sum():.3f}")

# --- fully-supervised ceiling of the PRESCRIBED representation Z (validation only).
# Also of two SPATIALLY AWARE atlas descriptors, because the whole point of the atlas is that
# it puts every specimen in one coordinate frame; Z (a colour-composition + local-texture
# descriptor) throws that frame away.  This ceiling table is what bounds every expert-guided
# curve below: no amount of expert feedback can extract an axis the representation lacks.
log("representation ceilings (supervised CV, validation only) ...")
fcd, _art, mesh = C.atlas_data()
from fishpipe import spectral                                               # noqa: E402
SPEC = spectral.spectral_coeffs(fcd, k=40, mesh=mesh, cache_dir=C.ATLAS_DS.cache_dir)
REG = C.region_summary(fcd, mesh, n_regions=128, stat="mean")
Z_spatial = PCA(n_components=24, random_state=0).fit_transform(
    StandardScaler().fit_transform(REG))                 # dimension-matched to Z
del fcd

REPS = {"Z_hist+texton": Z, "spectral40": SPEC, "region128mean": REG, "Z_spatial(PCA24 of region128)": Z_spatial}
CEIL = {}
for rn, Xr in REPS.items():
    CEIL[rn] = {}
    for f in C.FACTORS:
        a, s = C.balanced_cv(Xr, labels[f])
        # the shuffled-label null is only needed for the PRESCRIBED representation; on the
        # 120/384-dim diagnostic descriptors a random-label logistic fit never converges and
        # costs minutes for a number we already know is ~0.5
        an = (C.balanced_cv(Xr, np.random.default_rng(7).permutation(labels[f]))[0]
              if rn == "Z_hist+texton" else None)
        CEIL[rn][f] = {"cv_bal_acc": a, "cv_sd": s, "cv_bal_acc_shuffled": an}
    log("  " + f"{rn:32s}" + " ".join(f"{f}={CEIL[rn][f]['cv_bal_acc']:.3f}" for f in C.FACTORS))

BUDGETS = [8, 16, 24, 40, 56, 80, 120, 160]
N_SEEDS_A = 6
N_SEEDS_B = 6
N_SEEDS_C = 20
N_SEEDS_D = 12                       # more seeds in D: only 16-32 eval specimens -> noisy


# ===========================================================================
# helpers shared by A and D
# ===========================================================================
def fit_tag_models(Zs, Ys, lab_idx):
    """One balanced logistic model per tag on the labelled subset.
    Returns list of (predict_proba_fn) handling degenerate (single-class) tags."""
    models = []
    for t in range(Ys.shape[1]):
        yt = Ys[lab_idx, t]
        if len(np.unique(yt)) < 2:
            const = int(yt[0]) if len(yt) else 0
            models.append(("const", const))
        else:
            clf = make_pipeline(StandardScaler(),
                                LogisticRegression(max_iter=4000, C=0.5,
                                                   class_weight="balanced"))
            clf.fit(Zs[lab_idx], yt)
            models.append(("clf", clf))
    return models


def tag_scores(models, Zs):
    """(n, n_tags) predicted P(tag=1). Degenerate tags -> 0.5 (zero information)."""
    S = np.full((Zs.shape[0], len(models)), 0.5)
    for t, (kind, obj) in enumerate(models):
        if kind == "clf":
            S[:, t] = obj.predict_proba(Zs)[:, 1]
        else:
            S[:, t] = 0.5
    return S


def eval_tag_accuracy(models, Zs, Ys, idx):
    """Balanced accuracy per tag on `idx`. NaN if the eval set is single-class."""
    out = np.full(Ys.shape[1], np.nan)
    for t, (kind, obj) in enumerate(models):
        yt = Ys[idx, t]
        if len(np.unique(yt)) < 2:
            continue
        pred = obj.predict(Zs[idx]) if kind == "clf" else np.full(len(idx), obj)
        out[t] = balanced_accuracy_score(yt, pred)
    return out


def gmm_ari(S, y_true, k=8, seed=0, cov="full"):
    if len(S) < k + 2 or len(np.unique(y_true)) < 2:
        return np.nan
    Ss = StandardScaler().fit_transform(S)
    try:
        gm = GaussianMixture(k, covariance_type=cov, random_state=seed, n_init=3,
                             reg_covar=1e-4).fit(Ss)
        return float(adjusted_rand_score(y_true, gm.predict(Ss)))
    except Exception:
        return np.nan


def sil(Y2, y_true):
    try:
        if len(np.unique(y_true)) < 2 or len(y_true) <= len(np.unique(y_true)):
            return np.nan
        if not np.isfinite(Y2).all():
            return np.nan
        return float(silhouette_score(Y2, y_true))
    except Exception:
        return np.nan


SIL_KEYS = ["joint8"] + list(C.FACTORS)


def sil_multi(Y2, idx):
    """Silhouette of a 2-D display w.r.t. the 8-way joint label AND each single factor."""
    return [sil(Y2, joint8[idx])] + [sil(Y2, labels[f][idx]) for f in C.FACTORS]


def umap2(S, seed=0, n_neighbors=15):
    import umap
    nn = int(min(n_neighbors, max(2, len(S) - 1)))
    try:
        return umap.UMAP(n_components=2, n_neighbors=nn, min_dist=0.1,
                         random_state=seed, verbose=False).fit_transform(S)
    except Exception:
        return np.full((len(S), 2), np.nan)


def pca2(S):
    if S.shape[0] < 3:
        return np.full((len(S), 2), np.nan)
    return PCA(n_components=2, random_state=0).fit_transform(StandardScaler().fit_transform(S))


def run_tag_curve(Zs, Ys, joint, gidx, budgets, seed, acquisition, do_umap=True,
                  gmm_cov="full", Z_out=None, Ys_out=None):
    """One tag-feedback learning curve.

    Zs/Ys      : representation + tag matrix of the *study* specimens
    gidx       : global specimen index of each row of Zs (so silhouettes can be scored
                 against every factor, incl. under the label permutation used for nulls)
    Z_out/Ys_out: optional held-out specimens OUTSIDE the study (used only in part D)
    Returns dict of arrays indexed by budget.
    """
    rng = np.random.default_rng(1000 * seed + (0 if acquisition == "random" else 1))
    n = Zs.shape[0]
    gidx = np.asarray(gidx, int)
    perm = rng.permutation(n)
    lab = list(perm[:budgets[0]])                       # both strategies share the seed set

    res = {"bal_acc": [], "ari8": [], "sil_pca": [], "sil_umap": [],
           "bal_acc_out": [], "n_lab": []}
    best = None
    for bi, M in enumerate(budgets):
        if M > len(lab):
            need = M - len(lab)
            unl = np.setdiff1d(np.arange(n), np.array(lab, int))
            if acquisition == "random":
                pick = rng.choice(unl, size=min(need, len(unl)), replace=False)
            else:
                models_now = fit_tag_models(Zs, Ys, np.array(lab, int))
                p = tag_scores(models_now, Zs[unl])
                unc = np.abs(2 * p - 1).mean(1)          # 0 = maximally uncertain
                pick = unl[np.argsort(unc)[:min(need, len(unl))]]
            lab.extend(int(i) for i in pick)
        lab_idx = np.array(sorted(set(lab)), int)
        unl_idx = np.setdiff1d(np.arange(n), lab_idx)

        models = fit_tag_models(Zs, Ys, lab_idx)
        acc = eval_tag_accuracy(models, Zs, Ys, unl_idx)
        S_unl = tag_scores(models, Zs[unl_idx])
        ari = gmm_ari(S_unl, joint[unl_idx], k=8, seed=seed, cov=gmm_cov)
        s_p = sil_multi(pca2(S_unl), gidx[unl_idx])
        s_u = (sil_multi(umap2(S_unl, seed=seed), gidx[unl_idx]) if do_umap
               else [np.nan] * len(SIL_KEYS))

        acc_out = np.full(Ys.shape[1], np.nan)
        if Z_out is not None:
            acc_out = eval_tag_accuracy(models, Z_out, Ys_out, np.arange(len(Z_out)))

        res["bal_acc"].append(acc)
        res["ari8"].append(ari)
        res["sil_pca"].append(s_p)
        res["sil_umap"].append(s_u)
        res["bal_acc_out"].append(acc_out)
        res["n_lab"].append(len(lab_idx))

        if do_umap and (best is None or (np.isfinite(ari) and ari > best["ari"])):
            S_all = tag_scores(models, Zs)
            best = {"ari": ari if np.isfinite(ari) else -1.0, "M": M,
                    "S_all": S_all, "pca2_all": pca2(S_all),
                    "umap2_all": umap2(S_all, seed=seed),
                    "lab_mask": np.isin(np.arange(n), lab_idx)}
    for k in ("bal_acc", "bal_acc_out", "sil_pca", "sil_umap"):
        res[k] = np.asarray(res[k], float)               # (n_budget, n_tag) / (n_budget, 5)
    for k in ("ari8", "n_lab"):
        res[k] = np.asarray(res[k], float)
    return res, best


# ===========================================================================
# A. TAG FEEDBACK
# ===========================================================================
log("=== A: tag feedback (n=250) ===")
A_acc, A_ari, A_sp, A_su = {}, {}, {}, {}
best_display = None
ALL_IDX = np.arange(N)
for acq in ("random", "uncertainty"):
    accs, aris, sps, sus = [], [], [], []
    for sd in range(N_SEEDS_A):
        r, bst = run_tag_curve(Z, Y, joint8, ALL_IDX, BUDGETS, sd, acq, do_umap=True)
        accs.append(r["bal_acc"]); aris.append(r["ari8"])
        sps.append(r["sil_pca"]); sus.append(r["sil_umap"])
        if bst is not None and (best_display is None or bst["ari"] > best_display["ari"]):
            best_display = dict(bst, acq=acq, seed=sd)
        log(f"  A {acq:11s} seed {sd}: acc@8={np.nanmean(accs[-1][0]):.3f} "
            f"acc@160={np.nanmean(accs[-1][-1]):.3f} ari@160={aris[-1][-1]:.3f}")
    A_acc[acq] = np.asarray(accs)      # (seed, budget, tag)
    A_ari[acq] = np.asarray(aris)
    A_sp[acq] = np.asarray(sps)        # (seed, budget, 5) -> [joint8, belly, tail, stripe, cheeks]
    A_su[acq] = np.asarray(sus)

# --- null: shuffled tags (random acquisition only)
log("A: null (shuffled tags) ...")
A_acc_null, A_ari_null = [], []
for sd in range(N_SEEDS_A):
    pi = np.random.default_rng(500 + sd).permutation(N)
    r, _ = run_tag_curve(Z, Y[pi], joint8[pi], pi, BUDGETS, sd, "random", do_umap=False)
    A_acc_null.append(r["bal_acc"]); A_ari_null.append(r["ari8"])
A_acc_null = np.asarray(A_acc_null)
A_ari_null = np.asarray(A_ari_null)

# --- SUPPLEMENTARY: the same tag curve on the spatially aware representation.
# This is a MECHANISM test, not a tuned alternative: part B asks whether expert feedback can
# turn a supervised-recoverable-but-unsupervised-invisible axis into a clusterable one, and
# that question is only well posed in a space where the axis is actually present.
log("A-supp: tag feedback on Z_spatial (random acquisition) ...")
A_acc_sp, A_ari_sp = [], []
for sd in range(N_SEEDS_A):
    r, _ = run_tag_curve(Z_spatial, Y, joint8, ALL_IDX, BUDGETS, sd, "random", do_umap=False)
    A_acc_sp.append(r["bal_acc"]); A_ari_sp.append(r["ari8"])
A_acc_sp = np.asarray(A_acc_sp)
A_ari_sp = np.asarray(A_ari_sp)

# --- M = 0 reference: purely unsupervised use of Z
Z_ari0 = gmm_ari(Z, joint8, k=8, seed=0)
Z_sp0 = sil_multi(pca2(Z), ALL_IDX)
Z_su0 = sil_multi(umap2(Z, seed=0), ALL_IDX)
Zs_ari0 = gmm_ari(Z_spatial, joint8, k=8, seed=0)
log(f"A: unsupervised M=0 reference  ari8={Z_ari0:.3f} (Z_spatial {Zs_ari0:.3f}) "
    f"sil_pca={Z_sp0[0]:.3f} sil_umap={Z_su0[0]:.3f}")


def factor_curve(acc):        # (seed, budget, tag) -> dict factor -> (mean, sd) per budget
    out = {}
    for f in C.FACTORS:
        cols = [i for i, t in enumerate(TAGS) if TAG2FACTOR[t] == f]
        m, s = ms(np.nanmean(acc[:, :, cols], axis=2), axis=0)
        out[f] = {"mean": m.tolist(), "sd": s.tolist()}
    return out


# ===========================================================================
# B. PAIRWISE CONSTRAINTS
# ===========================================================================
log("=== B: pairwise constraints ===")
from fishpipe import semisup                                                # noqa: E402

M_PAIRS = [0, 10, 20, 30, 50, 80]


def incremental_random_pairs(n, need, rng, queried):
    out, qs = [], set(queried)
    guard = 0
    while len(out) < need and guard < 100000:
        guard += 1
        i, j = int(rng.integers(0, n)), int(rng.integers(0, n))
        if i == j:
            continue
        pr = (min(i, j), max(i, j))
        if pr not in qs:
            qs.add(pr); out.append(pr)
    return out


def run_pair_curve(X, y, seed, mode, m_list, n_dim=4, k=2):
    rng = np.random.default_rng(2000 * seed + (0 if mode == "random" else 1))
    n = X.shape[0]
    queried = []
    L = np.eye(X.shape[1])[:, :n_dim]
    ari_all, ari_free, n_ml, n_cl = [], [], [], []
    for m in m_list:
        need = m - len(queried)
        if need > 0:
            if mode == "random":
                new = incremental_random_pairs(n, need, rng, queried)
            else:
                new = semisup.active_pairs(X, L, need, queried, rng, n_ensemble=12, k=k)
            queried.extend(new)
        ml, cl = semisup._pairs_from_labels(queried, y)
        L = semisup.learn_warp(X, ml, cl, n_dim=n_dim, reg=1.0)
        pred = semisup.cluster_in_warp(X, L, k=k, seed=seed)
        ari_all.append(float(adjusted_rand_score(y, pred)))
        used = np.zeros(n, bool)
        for i, j in queried:
            used[i] = used[j] = True
        free = ~used
        ari_free.append(float(adjusted_rand_score(y[free], pred[free]))
                        if free.sum() > 10 and len(np.unique(y[free])) > 1 else np.nan)
        n_ml.append(len(ml)); n_cl.append(len(cl))
    return (np.array(ari_all), np.array(ari_free), np.array(n_ml), np.array(n_cl))


B = {}
for fac in ("stripe", "belly"):
    y = labels[fac]
    B[fac] = {}
    for mode in ("random", "active"):
        aa, af, nml, ncl = [], [], [], []
        for sd in range(N_SEEDS_B):
            a, f_, m_, c_ = run_pair_curve(Z, y, sd, mode, M_PAIRS)
            aa.append(a); af.append(f_); nml.append(m_); ncl.append(c_)
        B[fac][mode] = {"ari_all": np.asarray(aa), "ari_free": np.asarray(af),
                        "n_ml": np.asarray(nml), "n_cl": np.asarray(ncl)}
        log(f"  B {fac:6s} {mode:6s}: ARI " +
            " ".join(f"{m}:{v:.3f}" for m, v in zip(M_PAIRS, np.nanmean(aa, 0))))
    # null: constraints derived from SHUFFLED labels, but ARI still scored against the TRUE
    # factor (so a useful warp cannot be manufactured out of meaningless expert answers)
    null_a = []
    for sd in range(N_SEEDS_B):
        rng = np.random.default_rng(2000 * sd)
        ysh = np.random.default_rng(900 + sd).permutation(y)
        queried, L, row = [], np.eye(Z.shape[1])[:, :4], []
        for m in M_PAIRS:
            need = m - len(queried)
            if need > 0:
                queried.extend(incremental_random_pairs(N, need, rng, queried))
            ml, cl = semisup._pairs_from_labels(queried, ysh)
            L = semisup.learn_warp(Z, ml, cl, n_dim=4, reg=1.0)
            pred = semisup.cluster_in_warp(Z, L, k=2, seed=sd)
            row.append(float(adjusted_rand_score(y, pred)))
        null_a.append(row)
    B[fac]["null"] = {"ari_all": np.asarray(null_a)}
    log(f"  B {fac:6s} NULL  : ARI " +
        " ".join(f"{m}:{v:.3f}" for m, v in zip(M_PAIRS, np.nanmean(null_a, 0))))

# --- SUPPLEMENTARY B on the spatially aware representation (mechanism test, see A-supp)
log("B-supp: pairwise constraints on Z_spatial ...")
B_supp = {}
for fac in ("stripe", "belly"):
    y = labels[fac]
    B_supp[fac] = {}
    for mode in ("random", "active"):
        aa, af = [], []
        for sd in range(N_SEEDS_B):
            a, f_, _, _ = run_pair_curve(Z_spatial, y, sd, mode, M_PAIRS)
            aa.append(a); af.append(f_)
        B_supp[fac][mode] = {"ari_all": np.asarray(aa), "ari_free": np.asarray(af)}
        log(f"  B-supp {fac:6s} {mode:6s}: ARI " +
            " ".join(f"{m}:{v:.3f}" for m, v in zip(M_PAIRS, np.nanmean(aa, 0))))


# ===========================================================================
# C. COLD-START DISCOVERY OF THE CHEEK PATCH
# ===========================================================================
log("=== C: cold-start cheek discovery ===")
cheek = labels["cheeks"].astype(bool)
TARGETS = (1, 3, 5)


def labels_to_find(order, pos_mask, targets=TARGETS):
    """#specimens an expert must inspect, following `order`, to see the k-th positive."""
    hits = np.cumsum(pos_mask[order])
    out = {}
    for k in targets:
        w = np.where(hits >= k)[0]
        out[k] = int(w[0] + 1) if len(w) else np.nan
    return out


# unsupervised novelty scores in Z
lof = LocalOutlierFactor(n_neighbors=20)
lof.fit(Z)
score_lof = -lof.negative_outlier_factor_
nn = NearestNeighbors(n_neighbors=11).fit(Z)
d, _ = nn.kneighbors(Z)
score_knn = d[:, 1:].mean(1)

C_res = {"prevalence": float(cheek.mean()), "n_positive": int(cheek.sum())}
for nm, sc in (("lof", score_lof), ("knn_dist", score_knn)):
    order = np.argsort(sc)[::-1]                       # most novel first
    C_res[nm] = labels_to_find(order, cheek)
    C_res[nm + "_reverse"] = labels_to_find(np.argsort(sc), cheek)   # least novel first
rand_rows = []
for sd in range(N_SEEDS_C):
    o = np.random.default_rng(4000 + sd).permutation(N)
    rand_rows.append([labels_to_find(o, cheek)[k] for k in TARGETS])
rand_rows = np.asarray(rand_rows, float)
C_res["random"] = {str(k): {"mean": float(rand_rows[:, i].mean()),
                            "sd": float(rand_rows[:, i].std()),
                            "median": float(np.median(rand_rows[:, i]))}
                   for i, k in enumerate(TARGETS)}
C_res["random_analytic_first"] = float((N + 1) / (cheek.sum() + 1))

# BONUS (not in the brief): realistic HITL loop = find one positive by chance, then rank the
# rest by similarity to it in Z ("more like this").
exp_rows = []
for sd in range(N_SEEDS_C):
    o = np.random.default_rng(4000 + sd).permutation(N)
    first = int(np.where(cheek[o])[0][0])              # `first` wasted probes, then the seed
    seed_spec = o[first]
    already = set(o[:first].tolist())                  # already inspected (all negatives)
    rest = np.array([i for i in o if i != seed_spec and i not in already])
    dist = np.linalg.norm(Z[rest] - Z[seed_spec], axis=1)
    order = np.concatenate([[seed_spec], rest[np.argsort(dist)]])
    cost = labels_to_find(order, cheek)
    # total expert labels = the `first` random probes already paid + position in the new order
    exp_rows.append([cost[k] + first for k in TARGETS])
exp_rows = np.asarray(exp_rows, float)
C_res["similarity_expansion"] = {str(k): {"mean": float(exp_rows[:, i].mean()),
                                          "sd": float(exp_rows[:, i].std())}
                                 for i, k in enumerate(TARGETS)}
log(f"  C: prevalence {cheek.mean():.3f}; lof {C_res['lof']}; random(1,3,5) "
    f"{[round(rand_rows[:, i].mean(), 2) for i in range(3)]}")


# ===========================================================================
# D. SMALL-n REPLICATION OF A  (n = 40)
# ===========================================================================
log("=== D: small-n (n=40) tag feedback ===")
BUDGETS_D = [8, 16, 24]
D_acc, D_ari, D_sp, D_su, D_accout = {}, {}, {}, {}, {}
for acq in ("random", "uncertainty"):
    accs, aris, sps, sus, aouts = [], [], [], [], []
    for sd in range(N_SEEDS_D):
        rs = np.random.default_rng(7000 + sd)
        sub = np.sort(rs.choice(N, 40, replace=False))
        out_idx = np.setdiff1d(np.arange(N), sub)
        # refit scaler+PCA on the 40 study specimens only (the realistic small-n bottleneck)
        sc40 = StandardScaler().fit(Xcat_raw[sub])
        p40 = PCA(n_components=24, random_state=0).fit(sc40.transform(Xcat_raw[sub]))
        Z40 = p40.transform(sc40.transform(Xcat_raw[sub]))
        Zout = p40.transform(sc40.transform(Xcat_raw[out_idx]))
        r, _ = run_tag_curve(Z40, Y[sub], joint8[sub], sub, BUDGETS_D, sd, acq,
                             do_umap=True, gmm_cov="diag",
                             Z_out=Zout, Ys_out=Y[out_idx])
        accs.append(r["bal_acc"]); aris.append(r["ari8"])
        sps.append(r["sil_pca"]); sus.append(r["sil_umap"]); aouts.append(r["bal_acc_out"])
    D_acc[acq] = np.asarray(accs); D_ari[acq] = np.asarray(aris)
    D_sp[acq] = np.asarray(sps); D_su[acq] = np.asarray(sus)
    D_accout[acq] = np.asarray(aouts)
    log(f"  D {acq:11s}: in-study acc@24={np.nanmean(D_acc[acq][:, -1, :]):.3f} "
        f"held-out acc@24={np.nanmean(D_accout[acq][:, -1, :]):.3f}")

# D null
D_acc_null = []
for sd in range(N_SEEDS_D):
    rs = np.random.default_rng(7000 + sd)
    sub = np.sort(rs.choice(N, 40, replace=False))
    out_idx = np.setdiff1d(np.arange(N), sub)
    sc40 = StandardScaler().fit(Xcat_raw[sub])
    p40 = PCA(n_components=24, random_state=0).fit(sc40.transform(Xcat_raw[sub]))
    Z40 = p40.transform(sc40.transform(Xcat_raw[sub]))
    pi = np.random.default_rng(800 + sd).permutation(40)
    r, _ = run_tag_curve(Z40, Y[sub][pi], joint8[sub][pi], sub[pi], BUDGETS_D, sd, "random",
                         do_umap=False, gmm_cov="diag")
    D_acc_null.append(r["bal_acc"])
D_acc_null = np.asarray(D_acc_null)


# ===========================================================================
# assemble results
# ===========================================================================
log("assembling results ...")


def budget_to_reach(acc, budgets, ceil_rep="Z_hist+texton", frac=0.9):
    """Smallest tag budget whose mean balanced accuracy reaches `frac` x the fully-supervised
    ceiling of the representation. None = never reached within the budget grid."""
    fc = factor_curve(acc)
    out = {}
    for f in C.FACTORS:
        tgt = frac * CEIL[ceil_rep][f]["cv_bal_acc"]
        m = np.asarray(fc[f]["mean"], float)
        w = np.where(m >= tgt)[0]
        out[f] = {"target_bal_acc": float(tgt),
                  "budget": int(budgets[w[0]]) if len(w) else None,
                  "best_reached": float(np.nanmax(m))}
    return out


def sil_block(s):
    """s: (seed, budget, 5) -> {key: {'mean': [...per budget], 'sd': [...]}}"""
    m, sd = ms(s, axis=0)
    return {k: {"mean": m[:, i].tolist(), "sd": sd[:, i].tolist()}
            for i, k in enumerate(SIL_KEYS)}


def curve_block(acc, ari, sp, su, budgets, accout=None):
    per_tag_m, per_tag_s = ms(acc, axis=0)
    blk = {
        "budgets": budgets,
        "per_tag_bal_acc_mean": per_tag_m.tolist(),
        "per_tag_bal_acc_sd": per_tag_s.tolist(),
        "per_factor_bal_acc": factor_curve(acc),
        "mean_over_tags_bal_acc": ms(np.nanmean(acc, axis=2), axis=0)[0].tolist(),
        "ari8_mean": ms(ari)[0].tolist(), "ari8_sd": ms(ari)[1].tolist(),
        "sil_pca2": sil_block(sp),
        "sil_umap2": sil_block(su),
    }
    if accout is not None:
        blk["per_factor_bal_acc_heldout"] = factor_curve(accout)
        blk["mean_over_tags_bal_acc_heldout"] = ms(np.nanmean(accout, axis=2), axis=0)[0].tolist()
    return blk


res = {
    "meta": {
        "n_specimens": int(N), "representation": "PCA24 of std[std(area_hist24)|std(textonBoW128)]",
        "pca_explained_var": float(_pca_all.explained_variance_ratio_.sum()),
        "tags": TAGS, "budgets": BUDGETS, "budgets_smalln": BUDGETS_D,
        "n_seeds": {"A": N_SEEDS_A, "B": N_SEEDS_B, "C": N_SEEDS_C, "D": N_SEEDS_D},
        "class_balance": {k: np.bincount(v).tolist() for k, v in labels.items()},
        "chance_bal_acc": 0.5,
        "silhouette_keys": SIL_KEYS,
        "representation_ceiling": CEIL,
        "expert_simulation": "ALL expert responses are read off the generator ground truth: "
                             "noiseless, always available, zero disagreement. Real annotators "
                             "are noisier, so every curve here is an UPPER BOUND.",
    },
    "A_tags": {
        "unsupervised_reference_M0": {
            "ari8_Z": Z_ari0, "ari8_Z_spatial": Zs_ari0,
            "sil_pca2": dict(zip(SIL_KEYS, Z_sp0)), "sil_umap2": dict(zip(SIL_KEYS, Z_su0)),
            "note": "GMM(8)/displays computed on Z itself, no expert input"},
        "supplementary_Z_spatial_random": {
            "budgets": BUDGETS,
            "per_factor_bal_acc": factor_curve(A_acc_sp),
            "mean_over_tags_bal_acc": ms(np.nanmean(A_acc_sp, axis=2), axis=0)[0].tolist(),
            "ari8_mean": ms(A_ari_sp)[0].tolist(), "ari8_sd": ms(A_ari_sp)[1].tolist(),
            "note": "same protocol on PCA24 of region128mean; GT-informed choice, reported as a "
                    "representation-ceiling diagnostic, NOT as a tuned method"},
        "random": curve_block(A_acc["random"], A_ari["random"], A_sp["random"], A_su["random"], BUDGETS),
        "uncertainty": curve_block(A_acc["uncertainty"], A_ari["uncertainty"],
                                   A_sp["uncertainty"], A_su["uncertainty"], BUDGETS),
        "null_shuffled_tags": {
            "budgets": BUDGETS,
            "per_factor_bal_acc": factor_curve(A_acc_null),
            "mean_over_tags_bal_acc": ms(np.nanmean(A_acc_null, axis=2), axis=0)[0].tolist(),
            "ari8_mean": ms(A_ari_null)[0].tolist(),
        },
        "best_display": {"acq": best_display["acq"], "seed": int(best_display["seed"]),
                         "M": int(best_display["M"]), "ari8": float(best_display["ari"])},
        "budget_to_reach_90pct_of_ceiling": {
            "random": budget_to_reach(A_acc["random"], BUDGETS),
            "uncertainty": budget_to_reach(A_acc["uncertainty"], BUDGETS),
            "supplementary_Z_spatial_random":
                budget_to_reach(A_acc_sp, BUDGETS, ceil_rep="Z_spatial(PCA24 of region128)"),
        },
    },
    "B_pairs": {"m_pairs": M_PAIRS},
    "C_cold_start": C_res,
    "D_smalln": {
        "n_study": 40,
        "random": curve_block(D_acc["random"], D_ari["random"], D_sp["random"], D_su["random"],
                              BUDGETS_D, accout=D_accout["random"]),
        "uncertainty": curve_block(D_acc["uncertainty"], D_ari["uncertainty"], D_sp["uncertainty"],
                                   D_su["uncertainty"], BUDGETS_D, accout=D_accout["uncertainty"]),
        "null_shuffled_tags": {
            "budgets": BUDGETS_D,
            "per_factor_bal_acc": factor_curve(D_acc_null),
            "mean_over_tags_bal_acc": ms(np.nanmean(D_acc_null, axis=2), axis=0)[0].tolist(),
        },
    },
}
for fac in ("stripe", "belly"):
    res["B_pairs"][fac] = {}
    for mode in ("random", "active"):
        d = B[fac][mode]
        res["B_pairs"][fac][mode] = {
            "ari_all_mean": ms(d["ari_all"])[0].tolist(), "ari_all_sd": ms(d["ari_all"])[1].tolist(),
            "ari_unconstrained_mean": ms(d["ari_free"])[0].tolist(),
            "ari_unconstrained_sd": ms(d["ari_free"])[1].tolist(),
            "n_must_link_mean": ms(d["n_ml"])[0].tolist(),
            "n_cannot_link_mean": ms(d["n_cl"])[0].tolist(),
        }
    res["B_pairs"][fac]["null_shuffled"] = {
        "ari_all_mean": ms(B[fac]["null"]["ari_all"])[0].tolist(),
        "ari_all_sd": ms(B[fac]["null"]["ari_all"])[1].tolist(),
        "note": "constraints derived from SHUFFLED labels, ARI still scored against the true "
                "factor: measures how much a meaningless warp DESTROYS pre-existing structure",
    }
    res["B_pairs"][fac]["supplementary_Z_spatial"] = {
        mode: {"ari_all_mean": ms(B_supp[fac][mode]["ari_all"])[0].tolist(),
               "ari_all_sd": ms(B_supp[fac][mode]["ari_all"])[1].tolist(),
               "ari_unconstrained_mean": ms(B_supp[fac][mode]["ari_free"])[0].tolist()}
        for mode in ("random", "active")}

with open(RESULTS / f"{NAME}.json", "w") as fh:
    json.dump(jsonify(res), fh, indent=2)

np.savez_compressed(
    RESULTS / f"{NAME}_arrays.npz",
    names=names, Z=Z, Z_spatial=Z_spatial, joint8=joint8, tag_matrix=Y, tags=np.array(TAGS),
    sil_keys=np.array(SIL_KEYS),
    **{f"labels_{k}": v for k, v in labels.items()},
    budgets=np.array(BUDGETS), budgets_d=np.array(BUDGETS_D), m_pairs=np.array(M_PAIRS),
    A_acc_supp_spatial=A_acc_sp, A_ari_supp_spatial=A_ari_sp,
    A_acc_random=A_acc["random"], A_acc_uncertainty=A_acc["uncertainty"],
    A_ari_random=A_ari["random"], A_ari_uncertainty=A_ari["uncertainty"],
    A_silpca_random=A_sp["random"], A_silpca_uncertainty=A_sp["uncertainty"],
    A_silumap_random=A_su["random"], A_silumap_uncertainty=A_su["uncertainty"],
    A_acc_null=A_acc_null, A_ari_null=A_ari_null,
    A_ref_M0_ari=np.array([Z_ari0, Zs_ari0]),
    A_ref_M0_sil_pca=np.array(Z_sp0), A_ref_M0_sil_umap=np.array(Z_su0),
    best_S_all=best_display["S_all"], best_pca2=best_display["pca2_all"],
    best_umap2=best_display["umap2_all"], best_lab_mask=best_display["lab_mask"],
    **{f"B_{fac}_{mode}_{k}": B[fac][mode][k]
       for fac in ("stripe", "belly") for mode in ("random", "active")
       for k in ("ari_all", "ari_free", "n_ml", "n_cl")},
    **{f"B_{fac}_null_ari": B[fac]["null"]["ari_all"] for fac in ("stripe", "belly")},
    **{f"Bsupp_{fac}_{mode}_{k}": B_supp[fac][mode][k]
       for fac in ("stripe", "belly") for mode in ("random", "active")
       for k in ("ari_all", "ari_free")},
    C_random_costs=rand_rows, C_expansion_costs=exp_rows,
    C_score_lof=score_lof, C_score_knn=score_knn, C_cheek=cheek,
    D_acc_random=D_acc["random"], D_acc_uncertainty=D_acc["uncertainty"],
    D_accout_random=D_accout["random"], D_accout_uncertainty=D_accout["uncertainty"],
    D_ari_random=D_ari["random"], D_ari_uncertainty=D_ari["uncertainty"],
    D_silpca_random=D_sp["random"], D_silumap_random=D_su["random"],
    D_silpca_uncertainty=D_sp["uncertainty"], D_silumap_uncertainty=D_su["uncertainty"],
    D_acc_null=D_acc_null,
)


# ===========================================================================
# human-readable summary
# ===========================================================================
def fm(x, w=6, p=3):
    return ("{:>%d.%df}" % (w, p)).format(x) if np.isfinite(x) else " " * (w - 3) + "nan"


print("\n" + "=" * 96)
print("E5  EXPERT-GUIDED ANALYSIS OF THE ATLAS OUTPUT   (expert = noiseless ground-truth oracle)")
print("=" * 96)
print("Z = PCA24 of [area_hist24 | textonBoW128]; N=250; chance balanced accuracy = 0.500")
print("\n--- REPRESENTATION CEILING (supervised 5x4 CV; validation only) ---")
print(f"  {'representation':32s}" + "".join(f"{f:>10s}" for f in C.FACTORS))
for rn in REPS:
    print(f"  {rn:32s}" + "".join(fm(CEIL[rn][f]['cv_bal_acc'], 10) for f in C.FACTORS))
print(f"  {'(shuffled-label null)':32s}" +
      "".join(fm(CEIL['Z_hist+texton'][f]['cv_bal_acc_shuffled'], 10) for f in C.FACTORS))

print("\n--- A. TAG FEEDBACK (n=250): balanced accuracy on the UNLABELLED remainder ---")
hdr = ("  M  " + "".join(f"{f:>9s}" for f in C.FACTORS)
       + "     mean    ARI8   silP8   silU8  silP_be  silP_st")
for acq in ("random", "uncertainty"):
    print(f"[{acq}]"); print(hdr)
    fc = factor_curve(A_acc[acq])
    mo = np.nanmean(A_acc[acq], axis=2)
    for bi, M in enumerate(BUDGETS):
        row = f"{M:>4d} " + "".join(fm(fc[f]['mean'][bi], 9) for f in C.FACTORS)
        row += fm(np.nanmean(mo[:, bi]), 9) + fm(np.nanmean(A_ari[acq][:, bi]), 8)
        row += fm(np.nanmean(A_sp[acq][:, bi, 0]), 8) + fm(np.nanmean(A_su[acq][:, bi, 0]), 8)
        row += fm(np.nanmean(A_sp[acq][:, bi, 1]), 9) + fm(np.nanmean(A_sp[acq][:, bi, 3]), 9)
        print(row)
print("[null: shuffled tags, random acq]"); print(hdr)
fcn = factor_curve(A_acc_null)
for bi, M in enumerate(BUDGETS):
    print(f"{M:>4d} " + "".join(fm(fcn[f]['mean'][bi], 9) for f in C.FACTORS)
          + fm(np.nanmean(np.nanmean(A_acc_null, 2)[:, bi]), 9)
          + fm(np.nanmean(A_ari_null[:, bi]), 8))
print(f"   0 (no expert input, unsupervised Z)                              "
      f"{fm(Z_ari0, 8)}{fm(Z_sp0[0], 8)}{fm(Z_su0[0], 8)}{fm(Z_sp0[1], 9)}{fm(Z_sp0[3], 9)}")
print("[SUPPLEMENTARY: same protocol on Z_spatial = PCA24(region128mean), random acq]")
fcs = factor_curve(A_acc_sp)
print(hdr)
for bi, M in enumerate(BUDGETS):
    print(f"{M:>4d} " + "".join(fm(fcs[f]['mean'][bi], 9) for f in C.FACTORS)
          + fm(np.nanmean(np.nanmean(A_acc_sp, 2)[:, bi]), 9)
          + fm(np.nanmean(A_ari_sp[:, bi]), 8))
print(f"   0 (no expert input, unsupervised Z_spatial)                      {fm(Zs_ari0, 8)}")

print("\n--- B. PAIRWISE CONSTRAINTS: ARI of GMM(2) in the warped space ---")
print("  m  " + "".join(f"{s:>12s}" for s in
                        ["stripe/rand", "stripe/act", "stripe/null",
                         "belly/rand", "belly/act", "belly/null"]))
for mi, m in enumerate(M_PAIRS):
    row = f"{m:>4d}"
    for fac in ("stripe", "belly"):
        for mode in ("random", "active"):
            row += fm(np.nanmean(B[fac][mode]["ari_all"][:, mi]), 12)
        row += fm(np.nanmean(B[fac]["null"]["ari_all"][:, mi]), 12)
    print(row)
print("  (ARI on specimens NOT in any queried pair, m=80: " +
      ", ".join(f"{fac}/{mode}={np.nanmean(B[fac][mode]['ari_free'][:, -1]):.3f}"
                for fac in ("stripe", "belly") for mode in ("random", "active")) + ")")
print("[SUPPLEMENTARY: same on Z_spatial = PCA24(region128mean)]")
print("  m  " + "".join(f"{s:>12s}" for s in
                        ["stripe/rand", "stripe/act", "belly/rand", "belly/act"]))
for mi, m in enumerate(M_PAIRS):
    row = f"{m:>4d}"
    for fac in ("stripe", "belly"):
        for mode in ("random", "active"):
            row += fm(np.nanmean(B_supp[fac][mode]["ari_all"][:, mi]), 12)
    print(row)

print("\n--- C. COLD-START: labels needed to find the k-th cheek-positive specimen ---")
print(f"  cheek prevalence {cheek.mean():.3f} ({cheek.sum()}/250)")
print("  strategy               k=1     k=3     k=5")
for nm in ("lof", "knn_dist", "lof_reverse", "knn_dist_reverse"):
    print(f"  {nm:20s}" + "".join(f"{C_res[nm][k]:>8}" for k in TARGETS))
print("  random (20 seeds)   " + "".join(
    f"{C_res['random'][str(k)]['mean']:>8.2f}" for k in TARGETS))
print("  sim.expansion(bonus)" + "".join(
    f"{C_res['similarity_expansion'][str(k)]['mean']:>8.2f}" for k in TARGETS))

print("\n--- TAG BUDGET NEEDED to reach 90% of that representation's supervised ceiling ---")
for nm, blk in res["A_tags"]["budget_to_reach_90pct_of_ceiling"].items():
    print(f"  {nm:32s}" + "".join(
        f"{str(blk[f]['budget']):>10s}" for f in C.FACTORS))
print(f"  {'(factor)':32s}" + "".join(f"{f:>10s}" for f in C.FACTORS))

print("\n--- D. SMALL-n (study of 40): balanced accuracy ---")
print("  M  " + "".join(f"{f:>9s}" for f in C.FACTORS) + "     mean  [held-out 210]   ARI8")
for acq in ("random", "uncertainty"):
    print(f"[{acq}]")
    fc = factor_curve(D_acc[acq]); fo = np.nanmean(D_accout[acq], axis=2)
    mo = np.nanmean(D_acc[acq], axis=2)
    for bi, M in enumerate(BUDGETS_D):
        print(f"{M:>4d} " + "".join(fm(fc[f]['mean'][bi], 9) for f in C.FACTORS)
              + fm(np.nanmean(mo[:, bi]), 9) + fm(np.nanmean(fo[:, bi]), 16)
              + fm(np.nanmean(D_ari[acq][:, bi]), 8))
print("[null: shuffled tags]")
fcn = factor_curve(D_acc_null)
for bi, M in enumerate(BUDGETS_D):
    print(f"{M:>4d} " + "".join(fm(fcn[f]['mean'][bi], 9) for f in C.FACTORS)
          + fm(np.nanmean(np.nanmean(D_acc_null, 2)[:, bi]), 9))

print("\nwrote", RESULTS / f"{NAME}.json", "and", RESULTS / f"{NAME}_arrays.npz")
log("done")
