"""ADVERSARIAL re-derivation of E2 headline numbers. Does NOT import e2_preservation."""
from __future__ import annotations

import json
import os
import sys
import time

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import numpy as np
import pandas as pd

import common as C
from fishpipe import features, spectral
from fishpipe.features import rgb_to_lab
from fishpipe.mesh import load_obj, sample_face_colors

from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import balanced_accuracy_score, r2_score
from sklearn.model_selection import RepeatedStratifiedKFold, KFold, GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed

OUT = {}
t00 = time.time()


# ------------------------------------------------------------------ my own estimators
def my_bacc(X, y, n_splits=5, n_repeats=4, seed=0, groups=None):
    X = np.asarray(X, float); y = np.asarray(y)
    if groups is None:
        cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
        splits = list(cv.split(X, y))
    else:
        splits = list(GroupKFold(n_splits=n_splits).split(X, y, groups))
    sc = []
    for tr, te in splits:
        mu = X[tr].mean(0); sd = X[tr].std(0); sd[sd == 0] = 1.0
        clf = LogisticRegression(max_iter=4000, C=1.0, class_weight="balanced")
        clf.fit((X[tr] - mu) / sd, y[tr])
        sc.append(balanced_accuracy_score(y[te], clf.predict((X[te] - mu) / sd)))
    return float(np.mean(sc)), float(np.std(sc))


def my_r2(X, y, n_splits=5, n_repeats=4, seed=0):
    X = np.asarray(X, float); y = np.asarray(y, float)
    al = np.logspace(-3, 6, 19)
    out = []
    for rep in range(n_repeats):
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=seed + rep)
        oof = np.zeros_like(y)
        for tr, te in cv.split(X):
            m = make_pipeline(StandardScaler(), RidgeCV(alphas=al))
            m.fit(X[tr], y[tr])
            oof[te] = m.predict(X[te])
        out.append(r2_score(y, oof))
    return float(np.mean(out)), float(np.std(out))


# ------------------------------------------------------------------ my own descriptors
def my_axial(lab, areas, t, nb=32):
    """area-weighted mean Lab in nb bins of t (implemented with np.add.at, not bincount)."""
    edges = np.linspace(0, 1, nb + 1)
    bi = np.clip(np.searchsorted(edges, t, side="right") - 1, 0, nb - 1)
    num = np.zeros((nb, 3)); den = np.zeros(nb)
    np.add.at(den, bi, areas)
    for c in range(3):
        np.add.at(num[:, c], bi, areas * lab[:, c])
    den[den == 0] = 1.0
    return (num / den[:, None]).reshape(-1)


def my_canon(S3):
    """Sign-align (N,32,3) axial profiles to the population mean; label-free.
    Different implementation: z-score per channel, greedy loop to fixpoint on raw dot."""
    X = S3.copy().astype(float)
    N = X.shape[0]
    m0 = X.reshape(-1, 3).mean(0); s0 = X.reshape(-1, 3).std(0) + 1e-12
    flip = np.zeros(N, bool)
    for _ in range(50):
        Z = (X - m0) / s0
        ref = Z.mean(0); ref = ref - ref.mean(0, keepdims=True)
        ch = 0
        for i in range(N):
            a = Z[i] - Z[i].mean(0, keepdims=True)
            if np.sum(a[::-1] * ref) > np.sum(a * ref):
                X[i] = X[i][::-1]; flip[i] = not flip[i]; ch += 1
        if ch == 0:
            break
    return X.reshape(N, -1), flip


def my_hist(lab, areas, pal):
    d = ((lab[:, None, :] - pal[None]) ** 2).sum(-1)
    l = np.argmin(d, 1)
    h = np.zeros(pal.shape[0]); np.add.at(h, l, areas)
    return h / h.sum()


# ------------------------------------------------------------------ my own pigment rules
def my_pig_frac(lab, areas):
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    m = {
        "base": (a <= 5) & (a >= -28) & (b >= 0) & (L >= 30),
        "belly": (b < -3) & (L >= 30),
        "tail": (a < -28) & (b >= 0) & (L >= 30),
        "cheek": (a > 5) & (b >= 0) & (L >= 30),
        "stripe": L < 30,
    }
    tot = areas.sum()
    return np.array([(areas * m[p]).sum() / tot for p in ("base", "belly", "tail", "cheek", "stripe")])


# =================================================================== load
fcd, art, mesh = C.atlas_data(verbose=False) if "verbose" in C.atlas_data.__code__.co_varnames \
    else C.atlas_data()
names = list(fcd.names)
df, labels = C.load_gt(names)
nat = C.native_face_colors(verbose=False)
N = len(names)
donor = df["specimen_index"].to_numpy().astype(int)

OUT["order_ok"] = bool(list(nat["names"]) == names)
OUT["gt_join_ok"] = bool((df["sample_tag"].tolist() == names))
OUT["n"] = N
OUT["n_donor"] = int(len(np.unique(donor)))
print("order_ok", OUT["order_ok"], "gt_join_ok", OUT["gt_join_ok"], N, OUT["n_donor"])

# ---- spot check: is nat['spatial'] really from the native donor meshes?
import imageio.v2 as imageio
chk = []
for i in [0, 77, 200]:
    nm = names[i]
    m = load_obj(C.SRC_MESHES / f"{nm}.obj")
    tex = imageio.imread(C.SRC_IMAGES / f"{nm}.png")
    rgb = sample_face_colors(m, tex)
    ar = m.face_areas().astype(float)
    lb = rgb_to_lab(rgb).astype(float)
    cc = m.face_centroids(); cc = cc - cc.mean(0)
    _, _, vt = np.linalg.svd(cc, full_matrices=False)
    t = cc @ vt[0]; t = (t - t.min()) / (t.max() - t.min() + 1e-12)
    mine = my_axial(lb, ar, t)
    ref = nat["spatial"][i]
    chk.append(float(np.max(np.abs(mine - ref))))
    if i == 0:
        OUT["native_nfaces_example"] = int(m.n_faces)
        OUT["native_pig_recheck_maxabs"] = float(np.max(np.abs(my_pig_frac(lb, ar) - nat["masks"][i])))
OUT["native_spatial_recompute_maxabs"] = chk
print("native spatial recompute maxabs", chk, "pig", OUT.get("native_pig_recheck_maxabs"))

# =================================================================== descriptors
atlas_lab = rgb_to_lab(fcd.colors)
areas = fcd.areas
cents = mesh.face_centroids()
cc = cents - cents.mean(0)
_, _, vt = np.linalg.svd(cc, full_matrices=False)
axis = vt[0]
tt = cc @ axis
tt = (tt - tt.min()) / (tt.max() - tt.min() + 1e-12)

post_axial = np.stack([my_axial(atlas_lab[i], areas, tt) for i in range(N)])
post_hist_matched = np.stack([my_hist(atlas_lab[i], areas, nat["palette"]) for i in range(N)])
post_area = np.stack([my_pig_frac(atlas_lab[i], areas) for i in range(N)])
pre_area = nat["masks"]
pre_hist = nat["hist"]
pre_spatial = nat["spatial"]
pre_canon, flip = my_canon(pre_spatial.reshape(N, 32, 3))
OUT["n_flipped"] = int(flip.sum())
print("flipped", int(flip.sum()))

post_flat = features.spatial_flatten(fcd, color_space="lab", subsample=4000)
post_hist24 = features.area_hist(fcd, n_clusters=24)
post_spec = spectral.spectral_coeffs(fcd, k=40, mesh=mesh, cache_dir=C.ATLAS_DS.cache_dir)

# ---- artifact-free variant of post_axial (adversarial: do bake holes drive the gain?)
good = ~art  # (N,Nf)
post_axial_noart = np.zeros_like(post_axial)
nb = 32
edges = np.linspace(0, 1, nb + 1)
bi = np.clip(np.searchsorted(edges, tt, side="right") - 1, 0, nb - 1)
for i in range(N):
    g = good[i]
    num = np.zeros((nb, 3)); den = np.zeros(nb)
    np.add.at(den, bi[g], areas[g])
    for c in range(3):
        np.add.at(num[:, c], bi[g], areas[g] * atlas_lab[i, g, c])
    den[den == 0] = 1.0
    post_axial_noart[i] = (num / den[:, None]).reshape(-1)
OUT["artifact_frac_mean"] = float(art.mean())
del atlas_lab

BL = {"pre_hist": pre_hist, "post_hist": post_hist24, "post_hist_matched": post_hist_matched,
      "pre_spatial": pre_spatial, "pre_spatial_canon": pre_canon, "post_axial": post_axial,
      "post_spectral": post_spec, "post_spatial_flat": post_flat,
      "post_axial_noart": post_axial_noart}

# =================================================================== classification
FAC = ("belly", "tail", "stripe", "cheeks")
jobs = []
for b in BL:
    for f in FAC + ("donor",):
        jobs.append((b, f, None))
        for s in range(3):
            jobs.append((b, f, s))


def _y(f):
    return donor if f == "donor" else labels[f]


def _job(b, f, s):
    y = _y(f)
    if s is not None:
        y = np.random.default_rng(s).permutation(y)
    return my_bacc(BL[b], y)


res = Parallel(n_jobs=12, verbose=0)(delayed(_job)(b, f, s) for b, f, s in jobs)
acc = {}
for (b, f, s), (a, sd) in zip(jobs, res):
    d = acc.setdefault(b, {}).setdefault(f, {"null": []})
    if s is None:
        d["acc"], d["sd"] = a, sd
    else:
        d["null"].append(a)
for b in acc:
    for f in acc[b]:
        acc[b][f]["null_mean"] = float(np.mean(acc[b][f].pop("null")))
OUT["acc"] = acc

# ---- robustness: different CV seed
OUT["acc_seed7"] = {}
for b in ("pre_spatial_canon", "post_axial", "pre_hist", "post_hist_matched"):
    OUT["acc_seed7"][b] = {f: my_bacc(BL[b], _y(f), seed=7)[0] for f in FAC}

# ---- adversarial: donor-blocked (GroupKFold) CV
OUT["acc_donorblocked"] = {}
for b in ("pre_spatial_canon", "post_axial", "pre_hist", "post_hist_matched", "post_spatial_flat"):
    OUT["acc_donorblocked"][b] = {f: my_bacc(BL[b], _y(f), groups=donor)[0] for f in FAC}

# =================================================================== regression
CT = ("belly_hue", "tail_hue", "base_color_hue", "base_color_sat", "base_color_val",
      "stripe_spacing", "stripe_width", "stripe_longitudinal_offset")
rjobs = [(b, t, s) for b in BL for t in CT for s in (None, 0)]


def _rjob(b, t, s):
    y = df[t].to_numpy(float)
    if s is not None:
        y = np.random.default_rng(s).permutation(y)
    return my_r2(BL[b], y)


rres = Parallel(n_jobs=12, verbose=0)(delayed(_rjob)(b, t, s) for b, t, s in rjobs)
reg = {}
for (b, t, s), (r, sd) in zip(rjobs, rres):
    d = reg.setdefault(b, {}).setdefault(t, {})
    if s is None:
        d["r2"] = r
    else:
        d["null"] = r
OUT["reg"] = reg

# =================================================================== pigment agreement
def dcenter(v):
    o = v.astype(float).copy()
    for k in np.unique(donor):
        m = donor == k
        o[m] -= o[m].mean()
    return o


pig = {}
for p, nmp in enumerate(("base", "belly", "tail", "cheek", "stripe")):
    a, b = pre_area[:, p], post_area[:, p]
    pig[nmp] = {
        "r": float(np.corrcoef(a, b)[0, 1]),
        "r_within": float(np.corrcoef(dcenter(a), dcenter(b))[0, 1]),
        "bias": float((b - a).mean()),
        "rel_bias_pct": float(100 * (b - a).mean() / a.mean()),
        "pre_mean": float(a.mean()), "post_mean": float(b.mean()),
        "frac_var_donor_pre": float(np.var([np.mean(a[donor == k]) for k in donor]) / np.var(a)),
    }
OUT["pig"] = pig
OUT["post_area_vs_agent_maxabs"] = None
try:
    z = np.load(C.RESULTS / "e2_preservation_arrays.npz", allow_pickle=True)
    OUT["post_area_vs_agent_maxabs"] = float(np.abs(z["post_area"] - post_area).max())
    OUT["post_axial_vs_agent_maxabs"] = float(np.abs(z["post_axial"] - post_axial).max())
    OUT["canon_vs_agent_maxabs"] = float(np.abs(z["pre_spatial_canon"] - pre_canon).max())
    OUT["hist_matched_vs_agent_maxabs"] = float(np.abs(z["post_hist_matched"] - post_hist_matched).max())
except Exception as e:
    OUT["npz_err"] = str(e)

# ---- cheek detection floor claim
ch = labels["cheeks"].astype(bool)
OUT["cheek_present_n"] = int(ch.sum())
OUT["cheek_zero_area_among_present"] = int((pre_area[ch, 3] <= 0).sum())
OUT["cheek_area_mean_when_present"] = float(pre_area[ch, 3].mean())

# ---- correlation of pre-canonical axial orientation with population mean (surprise #2)
Z = (pre_spatial.reshape(N, 32, 3) - pre_spatial.reshape(-1, 3).mean(0)) / (
    pre_spatial.reshape(-1, 3).std(0) + 1e-12)
ref = Z.mean(0); ref = ref - ref.mean(0, keepdims=True)
cs = np.array([np.corrcoef((Z[i] - Z[i].mean(0)).ravel(), ref.ravel())[0, 1] for i in range(N)])
OUT["frac_pos_corr_pre_raw"] = float((cs > 0).mean())
Zc = (pre_canon.reshape(N, 32, 3) - pre_canon.reshape(-1, 3).mean(0)) / (
    pre_canon.reshape(-1, 3).std(0) + 1e-12)
refc = Zc.mean(0); refc = refc - refc.mean(0, keepdims=True)
csc = np.array([np.corrcoef((Zc[i] - Zc[i].mean(0)).ravel(), refc.ravel())[0, 1] for i in range(N)])
OUT["frac_pos_corr_pre_canon"] = float((csc > 0).mean())

# ---- donor / parameter independence
from scipy import stats as st
dep = {}
for col in CT + ("stripe_count", "rosy_cheeks_present"):
    g = [df[col].to_numpy()[donor == k] for k in np.unique(donor)]
    dep[col] = float(st.f_oneway(*g)[1])
for f in FAC:
    ct = np.array([[int(((donor == k) & (labels[f] == c)).sum()) for c in np.unique(labels[f])]
                   for k in np.unique(donor)])
    dep["label_" + f] = float(st.chi2_contingency(ct)[1])
OUT["donor_indep_p"] = dep

OUT["seconds"] = round(time.time() - t00, 1)
(C.RESULTS / "verify_e2.json").write_text(json.dumps(OUT, indent=1, default=float))
print(json.dumps({k: v for k, v in OUT.items() if k not in ("acc", "reg")}, indent=1, default=float))
print("done", OUT["seconds"])
