"""Follow-up adversarial checks for E2: paired-fold tests, best-POST selection bias,
and isolating palette-vs-normalisation in the composition 'surprise'."""
from __future__ import annotations
import json, os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "2")
import numpy as np
import common as C
from fishpipe.features import rgb_to_lab
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import balanced_accuracy_score, r2_score
from sklearn.model_selection import RepeatedStratifiedKFold, KFold, GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MiniBatchKMeans, KMeans
from scipy import stats as st

OUT = {}
z = np.load(C.RESULTS / "e2_preservation_arrays.npz", allow_pickle=True)
names = list(z["names"]); donor = z["donor"]; N = len(names)
labels = {f: z["label_" + f] for f in ("belly", "tail", "stripe", "cheeks")}
BL = {k: z[k] for k in ("pre_hist", "post_hist", "post_hist_matched", "pre_spatial",
                        "pre_spatial_canon", "post_axial", "post_spectral")}


def fold_scores(X, y, seed=0, groups=None):
    X = np.asarray(X, float)
    if groups is None:
        sp = list(RepeatedStratifiedKFold(n_splits=5, n_repeats=4, random_state=seed).split(X, y))
    else:
        sp = list(GroupKFold(n_splits=5).split(X, y, groups))
    out = []
    for tr, te in sp:
        mu = X[tr].mean(0); sd = X[tr].std(0); sd[sd == 0] = 1.0
        c = LogisticRegression(max_iter=4000, class_weight="balanced")
        c.fit((X[tr] - mu) / sd, y[tr])
        out.append(balanced_accuracy_score(y[te], c.predict((X[te] - mu) / sd)))
    return np.array(out)


# ---- paired per-fold tests for the two headline gains + the transfer costs
OUT["paired"] = {}
pairs = [("post_axial", "pre_spatial_canon", "stripe"), ("post_axial", "pre_spatial_canon", "cheeks"),
         ("post_spectral", "pre_spatial_canon", "cheeks"), ("post_axial", "pre_spatial_canon", "belly"),
         ("post_axial", "pre_spatial_canon", "tail"),
         ("post_hist_matched", "pre_hist", "belly"), ("post_hist_matched", "pre_hist", "tail"),
         ("post_hist_matched", "pre_hist", "stripe"), ("post_hist_matched", "pre_hist", "cheeks")]
for a, b, f in pairs:
    y = labels[f]
    sa, sb = fold_scores(BL[a], y), fold_scores(BL[b], y)
    d = sa - sb
    t, p = st.ttest_rel(sa, sb)
    OUT["paired"][f"{a}-{b}|{f}"] = {"mean_diff": float(d.mean()), "sd_diff": float(d.std(ddof=1)),
                                     "t": float(t), "p": float(p),
                                     "ci95": [float(d.mean() - 1.96 * d.std(ddof=1) / np.sqrt(len(d))),
                                              float(d.mean() + 1.96 * d.std(ddof=1) / np.sqrt(len(d)))]}

# ---- donor-blocked + seed sweep for post_spectral cheeks (the +0.127 bestPOST claim)
OUT["spectral_cheeks"] = {
    "seed0": float(fold_scores(BL["post_spectral"], labels["cheeks"], 0).mean()),
    "seed7": float(fold_scores(BL["post_spectral"], labels["cheeks"], 7).mean()),
    "donorblocked": float(fold_scores(BL["post_spectral"], labels["cheeks"], groups=donor).mean()),
    "pre_canon_donorblocked": float(fold_scores(BL["pre_spatial_canon"], labels["cheeks"], groups=donor).mean()),
}
OUT["spectral_stripe"] = {
    "seed0": float(fold_scores(BL["post_spectral"], labels["stripe"], 0).mean()),
    "donorblocked": float(fold_scores(BL["post_spectral"], labels["stripe"], groups=donor).mean()),
}

# ---- best-of-3 selection bias: how much does max-over-3 inflate under the null?
rng = np.random.default_rng(1)
infl = []
for _ in range(200):
    s = rng.normal(0, 0.05, 3)          # 3 post descriptors, sd ~ observed acc_sd/sqrt-ish
    infl.append(s.max() - s.mean())
OUT["max_of_3_expected_inflation_sd0.05"] = float(np.mean(infl))

# ---- palette vs normalisation for base_color_hue
y = z["param_base_color_hue"]


def r2cv(X, y, seed=0):
    al = np.logspace(-3, 6, 19); out = []
    for rep in range(4):
        cv = KFold(5, shuffle=True, random_state=seed + rep); oof = np.zeros_like(y, float)
        for tr, te in cv.split(X):
            m = make_pipeline(StandardScaler(), RidgeCV(alphas=al)); m.fit(X[tr], y[tr])
            oof[te] = m.predict(X[te])
        out.append(r2_score(y, oof))
    return float(np.mean(out))


fcd, art, mesh = C.atlas_data()
lab = rgb_to_lab(fcd.colors); areas = fcd.areas
pooled = lab.reshape(-1, 3)
sub = np.random.default_rng(0).choice(pooled.shape[0], 200000, replace=False)


def hist_pal(pal, L1=True):
    H = np.zeros((N, pal.shape[0]))
    for i in range(N):
        d = ((lab[i][:, None, :] - pal[None]) ** 2).sum(-1)
        l = np.argmin(d, 1)
        h = np.zeros(pal.shape[0]); np.add.at(h, l, areas)
        H[i] = h / h.sum() if L1 else h / np.linalg.norm(h)
    return H


res = {"post_hist_asis_L2": r2cv(z["post_hist"], y),
       "post_hist_renorm_L1": r2cv(z["post_hist"] / z["post_hist"].sum(1, keepdims=True), y),
       "pre_hist_L1": r2cv(z["pre_hist"], y),
       "pre_hist_L2": r2cv(z["pre_hist"] / np.linalg.norm(z["pre_hist"], axis=1, keepdims=True), y),
       "post_natpalette_L1": r2cv(z["post_hist_matched"], y)}
for seed in (0, 1, 2):
    km = MiniBatchKMeans(24, random_state=seed, n_init=10, batch_size=4096).fit(pooled[sub])
    res[f"post_atlasPalette_MBK_seed{seed}_L1"] = r2cv(hist_pal(km.cluster_centers_), y)
km = KMeans(24, n_init=5, random_state=1).fit(pooled[sub])
res["post_atlasPalette_KMeans_seed1_L1"] = r2cv(hist_pal(km.cluster_centers_), y)
# native palette applied to NATIVE colours is pre_hist; atlas palette on atlas colours above.
OUT["base_hue_palette_probe"] = res
# how many atlas palette clusters are consumed by near-black (bake holes)?
km0 = KMeans(24, n_init=5, random_state=0).fit(pooled[sub])
OUT["atlas_palette_dark_clusters"] = int((km0.cluster_centers_[:, 0] < 30).sum())
pal_nat = z["palette"]
OUT["native_palette_dark_clusters"] = int((pal_nat[:, 0] < 30).sum())

(C.RESULTS / "verify_e2_followup.json").write_text(json.dumps(OUT, indent=1, default=float))
print(json.dumps(OUT, indent=1, default=float))
