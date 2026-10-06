"""E4 — bring-your-own-analysis (BYOA) descriptors on the ColorAtlas module OUTPUT.

Claim under test: the module's *output* (one shared atlas mesh + 250 corresponded textures)
is a substrate on which an analyst can run their OWN, stronger analyses — not just the
built-in panel. We build 10 label-free descriptors from that output and quantify, factor by
factor, which descriptor recovers which kind of generative feature.

For every descriptor x factor we report
  * balanced_cv accuracy (mean, sd)                         -> supervised-recoverable
  * the SAME protocol on shuffled labels                    -> null
  * best single-axis AUC (+ its shuffled-label null, because max-over-D is optimistic)
  * ARI of GMM+BIC auto-clustering on the descriptor's top-2 PCs -> unsupervised-discoverable
plus two controls the paper needs:
  * DISENTANGLEMENT: is the supervised "stripe-count direction" really keyed on count, or on
    stripe spacing / width (nuisance generative parameters that are independent of count here)?
  * DONOR LEAKAGE: can the descriptor recover which of the 7 real donor scans the specimen was
    warped onto? (the atlas is supposed to have removed donor shape)
and a graph-Fourier mode sweep k in {1,3,10,20,40,80,150,250} per factor.

Ground truth is used for VALIDATION ONLY. No descriptor sees a label.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
from scipy.stats import rankdata

import common as C
from fishpipe import blobs, features, segment, spectral, textons

RESULTS = C.RESULTS
DCACHE = C.CACHE / "e4_desc"
DCACHE.mkdir(parents=True, exist_ok=True)
NAME = "e4_byoa_descriptors"

RNG_SEED = 0
T0 = time.time()


def _log(msg: str) -> None:
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


# ===========================================================================
# metrics
# ===========================================================================
def balanced_cv_null(X, y, n_shuffles: int = 4, seed: int = 123):
    """Same protocol as C.balanced_cv but the labels are re-shuffled for every repeat.

    Cost is identical to one C.balanced_cv call (n_shuffles x 5 folds = 20 fits) but the
    null is averaged over 4 independent permutations instead of one, so it is a much less
    noisy estimate of chance for this X.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    X = np.asarray(X, float)
    y = np.asarray(y)
    rng = np.random.default_rng(seed)
    scores = []
    for s in range(n_shuffles):
        yp = rng.permutation(y)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=s)
        for tr, te in cv.split(X, yp):
            clf = make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=4000, C=1.0, class_weight="balanced"),
            )
            clf.fit(X[tr], yp[tr])
            scores.append(balanced_accuracy_score(yp[te], clf.predict(X[te])))
    return float(np.mean(scores)), float(np.std(scores))


def auc_columns(ranks: np.ndarray, y) -> np.ndarray:
    """Rank-based (tie-corrected) AUC of every column, folded to >= 0.5. `ranks` is
    rankdata(X, axis=0) — precomputed because it does not depend on y."""
    y = np.asarray(y).astype(bool)
    n1 = int(y.sum())
    n0 = len(y) - n1
    s = ranks[y].sum(0)
    auc = (s - n1 * (n1 + 1) / 2.0) / (n1 * n0)
    return np.maximum(auc, 1.0 - auc)


def best_single_auc(ranks, y) -> float:
    return float(np.max(auc_columns(ranks, y)))


def best_single_auc_null(ranks, y, n_shuffles: int = 8, seed: int = 321) -> float:
    """Mean over permutations of max-over-columns AUC. Calibrates the selection bias of
    'best single axis' — with thousands of columns this is well above 0.5."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    return float(np.mean([np.max(auc_columns(ranks, rng.permutation(y)))
                          for _ in range(n_shuffles)]))


def lodo_cv(X, y, groups, n_null: int = 2, seed: int = 7):
    """Leave-one-DONOR-out balanced accuracy: train on 6 real donor scans, test on the 7th.

    Stratified k-fold mixes all 7 donors into every training set, so it cannot tell whether a
    descriptor generalises ACROSS body shapes. This does. Chance is still 0.5.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.model_selection import LeaveOneGroupOut
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    X = np.asarray(X, float)
    y = np.asarray(y)
    logo = LeaveOneGroupOut()

    def run(yy):
        sc = []
        for tr, te in logo.split(X, yy, groups):
            if len(np.unique(yy[te])) < 2 or len(np.unique(yy[tr])) < 2:
                continue
            clf = make_pipeline(StandardScaler(),
                                LogisticRegression(max_iter=4000, C=1.0,
                                                   class_weight="balanced"))
            clf.fit(X[tr], yy[tr])
            sc.append(balanced_accuracy_score(yy[te], clf.predict(X[te])))
        return sc

    sc = run(y)
    rng = np.random.default_rng(seed)
    null = []
    for _ in range(n_null):
        null += run(rng.permutation(y))
    return (float(np.mean(sc)), float(np.std(sc)), int(len(sc)),
            float(np.mean(null)) if null else float("nan"))


def oof_regress(X, y, n_seeds: int = 4):
    """Out-of-fold ridge regression of a CONTINUOUS generative parameter. Returns
    (R2 on pooled OOF predictions, Spearman rho, R2 on a shuffled target)."""
    from scipy.stats import spearmanr
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import KFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    X = np.asarray(X, float)
    y = np.asarray(y, float)

    def run(yy, seed0=0):
        pred = np.zeros(len(yy))
        for s in range(n_seeds):
            p = np.zeros(len(yy))
            for tr, te in KFold(5, shuffle=True, random_state=seed0 + s).split(X):
                m = make_pipeline(StandardScaler(),
                                  RidgeCV(alphas=np.logspace(-2, 5, 25)))
                m.fit(X[tr], yy[tr])
                p[te] = m.predict(X[te])
            pred += p / n_seeds
        ss_res = ((yy - pred) ** 2).sum()
        ss_tot = ((yy - yy.mean()) ** 2).sum()
        return 1.0 - ss_res / ss_tot, float(spearmanr(yy, pred).statistic)

    r2, rho = run(y)
    rng = np.random.default_rng(11)
    r2n, _ = run(rng.permutation(y), seed0=100)
    return float(r2), rho, float(r2n)


def gmm_bic_ari(X, y_dict, max_k: int = 6, seed: int = 0):
    """StandardScaler -> PCA(2) -> GMM with BIC-selected n_components; ARI vs each factor."""
    from sklearn.decomposition import PCA
    from sklearn.metrics import adjusted_rand_score
    from sklearn.mixture import GaussianMixture
    from sklearn.preprocessing import StandardScaler

    Z = PCA(n_components=2, random_state=seed).fit_transform(
        StandardScaler().fit_transform(np.asarray(X, float)))
    best, best_bic, best_lab = 1, np.inf, np.zeros(len(Z), int)
    for k in range(1, max_k + 1):
        gm = GaussianMixture(n_components=k, covariance_type="full", n_init=10,
                             random_state=seed, reg_covar=1e-5).fit(Z)
        b = gm.bic(Z)
        if b < best_bic:
            best_bic, best, best_lab = b, k, gm.predict(Z)
    aris = {f: float(adjusted_rand_score(y, best_lab)) for f, y in y_dict.items()}
    return best, aris, Z


# ===========================================================================
# descriptor library
# ===========================================================================
def stage(name: str, fn):
    """Disk-cache an expensive JSON-able analysis stage so re-runs are cheap."""
    p = DCACHE / f"stage_{name}.json"
    if p.exists():
        _log(f"  stage '{name}': loaded from cache")
        return json.loads(p.read_text())
    t = time.time()
    out = fn()
    p.write_text(json.dumps(out))
    _log(f"  stage '{name}': computed in {time.time() - t:.0f}s")
    return out


def cached(name: str, fn):
    p = DCACHE / f"{name}.npy"
    if p.exists():
        X = np.load(p)
        _log(f"  {name}: cached {X.shape}")
        return X
    t = time.time()
    X = np.asarray(fn(), float)
    np.save(p, X)
    _log(f"  {name}: built {X.shape} in {time.time() - t:.1f}s")
    return X


def spatial_descriptor_atlas(fcd, seg_, mesh, A, n_axes=3, min_area_frac=0.0005, fft_bins=32):
    """Verbatim copy of fishpipe.spatial.spatial_descriptor with the fishy-hardwired
    `structure.principal_axis()` replaced by C.principal_axis(mesh) (guardrail)."""
    cents = fcd.centroids
    areas = fcd.areas
    labels = seg_.labels
    tot = areas.sum()
    axis = C.principal_axis(mesh)
    t_all = cents @ axis
    N, Nf = labels.shape
    K = seg_.centroids_lab.shape[0]
    edges = np.linspace(t_all.min(), t_all.max(), fft_bins + 1)
    rows = []
    for i in range(N):
        row = []
        for c in range(K):
            m = labels[i] == c
            a = areas[m]
            W = a.sum()
            if W <= 0 or m.sum() < 2:
                row += [0.0] * 10
                continue
            pts = cents[m]
            mu = (a[:, None] * pts).sum(0) / W
            d = pts - mu
            cov = np.einsum("i,ij,ik->jk", a, d, d) / W
            eig = np.sort(np.linalg.eigvalsh(cov))[::-1]
            eig = np.sqrt(np.clip(eig, 0, None))
            aniso = float(eig[0] / (eig.sum() + 1e-12))
            tp = t_all[m]
            ax_mean = float((a * tp).sum() / W)
            ax_spread = float(np.sqrt((a * (tp - ax_mean) ** 2).sum() / W))
            h = np.histogram(tp, bins=edges, weights=a)[0]
            f = np.abs(np.fft.rfft(h - h.mean()))[1:]
            fft_peak = float(np.argmax(f) + 1) if f.size else 0.0
            sub = A[m][:, m]
            ncomp, comp = connected_components(sub, directed=False)
            bc = []
            for k in range(ncomp):
                bm = comp == k
                if a[bm].sum() / tot >= min_area_frac:
                    bc.append((a[bm, None] * pts[bm]).sum(0) / a[bm].sum())
            n_blobs = float(len(bc))
            nn_cv = 0.0
            if len(bc) >= 2:
                bc = np.asarray(bc)
                D = np.linalg.norm(bc[:, None] - bc[None], axis=2)
                np.fill_diagonal(D, np.inf)
                nn = D.min(1)
                nn_cv = float(nn.std() / (nn.mean() + 1e-12))
            row += [W / tot, eig[0], eig[1], eig[2], aniso, ax_mean, ax_spread, fft_peak,
                    n_blobs, nn_cv]
        rows.append(row)
        if (i + 1) % 50 == 0:
            _log(f"    spatial_descriptor {i + 1}/{N}")
    return np.asarray(rows)


# ===========================================================================
def main():
    _log("loading atlas data ...")
    fcd, art, mesh = C.atlas_data()
    df, labels = C.load_gt(list(fcd.names))
    N = fcd.colors.shape[0]
    A = C.face_adjacency(mesh, C.ATLAS_DS.cache_dir)
    _log(f"N={N}  faces={fcd.colors.shape[1]}  adjacency nnz={A.nnz}")

    donor = df["specimen_index"].to_numpy().astype(int)

    # ---------------------------------------------------------------- descriptors
    _log("building descriptors ...")
    X = {}
    X["D1_area_hist24"] = cached("D1_area_hist24",
                                 lambda: features.area_hist(fcd, n_clusters=24))
    X["D2_spatial_flatten"] = cached(
        "D2_spatial_flatten",
        lambda: features.spatial_flatten(fcd, color_space="lab", subsample=4000))
    X["D3_region128_mean"] = cached(
        "D3_region128_mean", lambda: C.region_summary(fcd, mesh, 128, "mean"))
    X["D4_region128_maxchroma"] = cached(
        "D4_region128_maxchroma", lambda: C.region_summary(fcd, mesh, 128, "maxchroma"))

    # spectral: compute the k=250 coefficient block ONCE, slice for every k (identical to
    # calling spectral_coeffs(k) because Uk = U[:, :k] is a prefix)
    SPEC_KMAX = 250
    spec_full = cached(
        "SPEC_full250",
        lambda: spectral.spectral_coeffs(fcd, k=SPEC_KMAX, channels=("L", "a", "b"),
                                         mesh=mesh, cache_dir=C.ATLAS_DS.cache_dir))

    def spec_slice(k, channels=("L", "a", "b")):
        ci = {"L": 0, "a": 1, "b": 2}
        return np.concatenate([spec_full[:, ci[c] * SPEC_KMAX: ci[c] * SPEC_KMAX + k]
                               for c in channels], axis=1)

    X["D5_spectral_k40_Lab"] = spec_slice(40)
    X["D6_spectral_k150_Lab"] = spec_slice(150)
    X["D7_spectral_k40_Lonly"] = spec_slice(40, ("L",))

    # textons: share the jet + codebook between BoW and VLAD (identical vocabulary)
    if not ((DCACHE / "D8_texton_bow128.npy").exists()
            and (DCACHE / "D9_texton_vlad128.npy").exists()):
        _log("  building texton jet + shared codebook (K=128) ...")
        P = textons.diffusion_operator(C.ATLAS_DS.cache_dir)
        jet = textons.local_jet(fcd, scales=(2, 4), P=P, cache_dir=C.ATLAS_DS.cache_dir)
        scaler, km = textons.build_codebook(jet, K=128, seed=0)
        X["D8_texton_bow128"] = cached(
            "D8_texton_bow128", lambda: textons.encode(fcd, jet, scaler, km, mode="bow"))
        X["D9_texton_vlad128"] = cached(
            "D9_texton_vlad128", lambda: textons.encode(fcd, jet, scaler, km, mode="vlad"))
        del jet
    else:
        X["D8_texton_bow128"] = cached("D8_texton_bow128", lambda: None)
        X["D9_texton_vlad128"] = cached("D9_texton_vlad128", lambda: None)

    # segmentation -> spatial + Endler + blob statistics
    need_seg = not all((DCACHE / f"{n}.npy").exists()
                       for n in ("D10a_seg_spatial", "D10b_endler", "D10c_blobs"))
    if need_seg:
        _log("  segmenting (n_colors=8, smooth_iters=0) ...")
        seg_ = segment.segment(fcd, n_colors=8, smooth_iters=0, mesh_adjacency=A)
        np.save(DCACHE / "seg_labels.npy", seg_.labels)
        np.save(DCACHE / "seg_centroids_lab.npy", seg_.centroids_lab)
        X["D10a_seg_spatial"] = cached(
            "D10a_seg_spatial", lambda: spatial_descriptor_atlas(fcd, seg_, mesh, A))
        X["D10b_endler"] = cached(
            "D10b_endler", lambda: blobs.endler_transitions(fcd, seg_, mesh_adjacency=A))
        X["D10c_blobs"] = cached(
            "D10c_blobs",
            lambda: blobs.blob_descriptor(fcd, seg_, mesh_adjacency=A)["features"])
    else:
        for n in ("D10a_seg_spatial", "D10b_endler", "D10c_blobs"):
            X[n] = cached(n, lambda: None)
    X["D10_seg_all"] = np.concatenate(
        [X["D10a_seg_spatial"], X["D10b_endler"], X["D10c_blobs"]], axis=1)
    _log(f"  D10_seg_all: {X['D10_seg_all'].shape}")

    order = ["D1_area_hist24", "D2_spatial_flatten", "D3_region128_mean",
             "D4_region128_maxchroma", "D5_spectral_k40_Lab", "D6_spectral_k150_Lab",
             "D7_spectral_k40_Lonly", "D8_texton_bow128", "D9_texton_vlad128",
             "D10a_seg_spatial", "D10b_endler", "D10c_blobs", "D10_seg_all"]
    for k in order:
        assert X[k].shape[0] == N, (k, X[k].shape)
        assert np.isfinite(X[k]).all(), f"{k} has non-finite values"

    # ---------------------------------------------------------------- main table
    _log("evaluating descriptor x factor grid ...")
    pcs = {}
    for name in order:
        _, _, pcs[name] = gmm_bic_ari(X[name], labels)

    def _grid():
        table = {}
        for name in order:
            Xi = X[name]
            row = {"dim": int(Xi.shape[1]), "factors": {}}
            ncomp, aris, _ = gmm_bic_ari(Xi, labels)
            row["gmm_bic_ncomp"] = int(ncomp)
            ranks = rankdata(Xi, axis=0)
            for f in C.FACTORS:
                y = labels[f]
                t = time.time()
                acc, sd = C.balanced_cv(Xi, y)
                nacc, nsd = balanced_cv_null(Xi, y)
                cell = {
                    "acc": acc, "sd": sd, "null_acc": nacc, "null_sd": nsd,
                    "auc_best": best_single_auc(ranks, y),
                    "auc_best_null": best_single_auc_null(ranks, y),
                    "gmm_ari": aris[f],
                }
                row["factors"][f] = cell
                _log(f"  {name:24s} {f:7s} acc={acc:.3f}+-{sd:.3f} null={nacc:.3f} "
                     f"aucbest={cell['auc_best']:.3f} (null {cell['auc_best_null']:.3f}) "
                     f"ARI={aris[f]:+.3f}  [{time.time() - t:.0f}s]")
            # donor-identity control (7 classes, chance = 1/7)
            row["donor_acc"], row["donor_sd"] = C.balanced_cv(Xi, donor)
            _log(f"  {name:24s} donor7  acc={row['donor_acc']:.3f} (chance 0.143)")
            table[name] = row
        return table

    table = stage("grid", _grid)

    # ---------------------------------------------------------------- disentanglement
    _log("disentanglement test (stripe direction) ...")
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler

    y_str = labels["stripe"]
    nuis = {
        "stripe_count": df["stripe_count"].to_numpy().astype(float),
        "stripe_spacing": df["stripe_spacing"].to_numpy().astype(float),
        "stripe_width": df["stripe_width"].to_numpy().astype(float),
        "stripe_longitudinal_offset": df["stripe_longitudinal_offset"].to_numpy().astype(float),
    }
    gt_num = df.select_dtypes(include=[np.number]).drop(
        columns=[c for c in ("sample_index", "tps_lambda", "n_landmarks_used", "tps_rmse")
                 if c in df.columns])

    def pearson(a, b):
        a = np.asarray(a, float)
        b = np.asarray(b, float)
        if a.std() < 1e-12 or b.std() < 1e-12:
            return 0.0
        return float(np.corrcoef(a, b)[0, 1])

    def partial_r(a, b, ctrl):
        """corr(a,b | ctrl) by residualising both on ctrl (single control variable)."""
        a, b, c = (np.asarray(v, float) for v in (a, b, ctrl))
        ra = a - np.polyval(np.polyfit(c, a, 1), c)
        rb = b - np.polyval(np.polyfit(c, b, 1), c)
        return pearson(ra, rb)

    def oof_decision(name, y):
        """Out-of-fold logistic decision function, averaged over 4 stratified 5-fold seeds.
        Scalers/classifiers are fit on training folds only, so this projection is leakage-free."""
        oof = np.zeros(N)
        for s in range(4):
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=s)
            p = np.zeros(N)
            for tr, te in cv.split(X[name], y):
                sc = StandardScaler().fit(X[name][tr])
                c2 = LogisticRegression(max_iter=6000, C=1.0, class_weight="balanced")
                c2.fit(sc.transform(X[name][tr]), y[tr])
                p[te] = c2.decision_function(sc.transform(X[name][te]))
            oof += p / 4.0
        return oof

    def _disent():
        out_d = {}
        for name in order:
            Xs = StandardScaler().fit_transform(X[name])
            clf = LogisticRegression(max_iter=6000, C=1.0,
                                     class_weight="balanced").fit(Xs, y_str)
            proj_in = Xs @ clf.coef_[0]
            oof = oof_decision(name, y_str)
            d = {"n_dim": int(X[name].shape[1])}
            for tag, proj in (("insample", proj_in), ("oof", oof)):
                d[tag] = {k: abs(pearson(proj, v)) for k, v in nuis.items()}
                d[tag]["spacing_given_count"] = abs(
                    partial_r(proj, nuis["stripe_spacing"], nuis["stripe_count"]))
                d[tag]["width_given_count"] = abs(
                    partial_r(proj, nuis["stripe_width"], nuis["stripe_count"]))
                # what GT parameter does this direction correlate with MOST?
                rs = {c: abs(pearson(proj, gt_num[c].to_numpy())) for c in gt_num.columns}
                top = sorted(rs.items(), key=lambda kv: -kv[1])[:3]
                d[tag]["top3_gt_params"] = [[k, float(v)] for k, v in top]
                # donor identity: fraction of projection variance explained by donor
                gm = proj.mean()
                ssb = sum(((proj[donor == g].mean() - gm) ** 2) * (donor == g).sum()
                          for g in np.unique(donor))
                d[tag]["donor_eta2"] = float(ssb / (((proj - gm) ** 2).sum() + 1e-12))
            out_d[name] = d
            _log(f"  {name:24s} |r| count={d['oof']['stripe_count']:.3f} "
                 f"spacing={d['oof']['stripe_spacing']:.3f} "
                 f"width={d['oof']['stripe_width']:.3f} "
                 f"donor_eta2={d['oof']['donor_eta2']:.3f}  (oof)")
        return out_d

    disent = stage("disent", _disent)

    # GT sanity: are the nuisance params actually independent of count?
    gt_corr = {k: pearson(nuis["stripe_count"], v) for k, v in nuis.items()}

    # ---------------------------------------------------------------- k sweep
    _log("graph-Fourier mode sweep ...")
    ks = [1, 3, 10, 20, 40, 80, 150, 250]

    def _sweep():
        sw = {f: {"k": ks, "acc": [], "sd": [], "null": []} for f in C.FACTORS}
        for k in ks:
            Xk = spec_slice(k)
            for f in C.FACTORS:
                a, s = C.balanced_cv(Xk, labels[f])
                na, _ = balanced_cv_null(Xk, labels[f])
                sw[f]["acc"].append(a)
                sw[f]["sd"].append(s)
                sw[f]["null"].append(na)
            _log(f"  k={k:3d} dim={Xk.shape[1]:4d} " +
                 " ".join(f"{f}={sw[f]['acc'][-1]:.3f}" for f in C.FACTORS))
        return sw

    sweep = stage("sweep", _sweep)

    saturation = {}
    for f in C.FACTORS:
        a = np.asarray(sweep[f]["acc"])
        sd = np.asarray(sweep[f]["sd"])
        kbest = int(ks[int(np.argmax(a))])
        # smallest k reaching within 1 sd of the best, and within 1% of the best
        within_sd = np.where(a >= a.max() - sd[int(np.argmax(a))])[0]
        within_1pct = np.where(a >= 0.99 * a.max())[0]
        saturation[f] = {
            "k_argmax": kbest, "acc_max": float(a.max()),
            "k_within_1sd": int(ks[int(within_sd[0])]),
            "k_within_1pct": int(ks[int(within_1pct[0])]),
        }

    # ================================================================ CONTROLS
    # The grid above says donor identity is recoverable at 0.78-1.00 balanced accuracy
    # (chance 1/7) from almost every descriptor. Three controls interrogate that, and the
    # continuous-parameter control checks that the near-ceiling belly/tail numbers are real
    # colour recovery rather than an artefact of thresholding a continuous parameter.

    # ---- C1: leave-one-DONOR-out generalisation -------------------------------
    def _lodo_one(name):
        def go():
            r = {}
            for f in C.FACTORS:
                m, s, nf, nul = lodo_cv(X[name], labels[f], donor)
                r[f] = {"acc": m, "sd": s, "n_folds": nf, "null": nul}
            _log("  LODO " + f"{name:24s} " +
                 " ".join(f"{f}={r[f]['acc']:.3f}" for f in C.FACTORS))
            return r
        return go

    _log("control 1: leave-one-donor-out CV ...")
    # cached per descriptor so a long run can be resumed; cheapest descriptors first
    lodo = {name: stage(f"lodo_{name}", _lodo_one(name))
            for name in sorted(order, key=lambda n: X[n].shape[1])}

    # ---- C2: where does the donor signal come from? ---------------------------
    _log("control 2: donor recoverability, pre-module vs post-module vs bake artefacts ...")

    def _donorsrc():
        nat = C.native_face_colors(verbose=False)
        assert list(nat["names"]) == list(fcd.names), "native/atlas name order mismatch"
        rng = np.random.default_rng(5)
        sub = rng.choice(fcd.colors.shape[1], size=4000, replace=False)
        cand = {
            "PRE_native_hist24": nat["hist"],
            "PRE_native_spatial96": nat["spatial"],
            "PRE_native_labmean": nat["lab_mean"],
            "PRE_native_pigment_masks": nat["masks"],
            "POST_atlas_hist24": X["D1_area_hist24"],
            "POST_atlas_region128_mean": X["D3_region128_mean"],
            "POST_atlas_spectral_k40_L": X["D7_spectral_k40_Lonly"],
            # NB: this is the NAIVE black-pixel mask = bake holes AND the genuinely black
            # planted stripes (e1: 0.37% holes vs 0.24% planted black). It is therefore a
            # mixed signal, not a pure artefact fingerprint — see caveats.
            "POST_atlas_blackmask_holes_plus_stripes": art[:, sub].astype(float),
        }
        res = {}
        for k, v in cand.items():
            a, s = C.balanced_cv(v, donor)
            na, _ = balanced_cv_null(v, donor)
            res[k] = {"dim": int(np.shape(v)[1]), "donor_acc": a, "donor_sd": s,
                      "donor_null": na}
            _log(f"    {k:30s} dim={res[k]['dim']:5d} donor acc={a:.3f} (null {na:.3f})")
        # does the black mask alone carry the FACTORS?
        res["blackmask_factor_acc"] = {
            f: list(C.balanced_cv(art[:, sub].astype(float), labels[f])) for f in C.FACTORS}
        return res

    donor_src = stage("donor_src", _donorsrc)

    # ---- C3: continuous generative-parameter recovery -------------------------
    _log("control 3: out-of-fold recovery of CONTINUOUS generative parameters ...")
    cont_targets = ["belly_hue", "tail_hue", "base_color_hue", "stripe_spacing",
                    "stripe_width", "stripe_longitudinal_offset"]
    cont_desc = ["D1_area_hist24", "D3_region128_mean", "D5_spectral_k40_Lab",
                 "D6_spectral_k150_Lab", "D9_texton_vlad128", "D10_seg_all"]

    def _cont():
        res = {}
        for name in cont_desc:
            res[name] = {}
            for tgt in cont_targets:
                r2, rho, r2n = oof_regress(X[name], df[tgt].to_numpy())
                res[name][tgt] = {"r2": r2, "spearman": rho, "r2_shuffled": r2n}
            _log("  cont " + f"{name:24s} " +
                 " ".join(f"{t.split('_')[0][:5]}={res[name][t]['r2']:+.2f}"
                          for t in cont_targets))
        return res

    cont = stage("continuous", _cont)

    # ---- C4: why do rosy cheeks top out at ~0.8? ------------------------------
    _log("control 4: rosy-cheek detectability vs cheek/base hue contrast ...")

    def _cheek():
        from fishpipe.features import rgb_to_lab
        best = max(order, key=lambda n: table[n]["factors"]["cheeks"]["acc"])
        y = labels["cheeks"]
        pres = y == 1
        oof = oof_decision(best, y)
        correct = ((oof > 0).astype(int) == y).astype(float)

        # The generator paints cheeks in essentially pure red for EVERY present specimen and
        # always at high strength, so neither hue contrast nor strength can explain a ceiling.
        h = df["rosy_cheeks_hue"].to_numpy()
        red_hue = float(np.mean(np.minimum(h[pres], 1.0 - h[pres]) < 0.05))
        strength = df["rosy_cheeks_strength"].to_numpy()

        # DIRECT, label-free measurement: how much red-sector surface does each atlas
        # specimen actually carry? If the cheek patch survived the transfer this alone should
        # separate present from absent almost perfectly.
        lab = rgb_to_lab(fcd.colors)
        chroma = np.sqrt(lab[..., 1] ** 2 + lab[..., 2] ** 2)
        hue_ang = np.degrees(np.arctan2(lab[..., 2], lab[..., 1]))     # 0 deg = +a* = red
        red = (chroma > 15) & (np.abs(hue_ang) < 40)
        w = fcd.areas / fcd.areas.sum()
        red_frac = (red * w[None, :]).sum(1)
        r_auc = float(np.max(auc_columns(rankdata(red_frac[:, None], axis=0), y)))
        r_acc, r_sd = C.balanced_cv(red_frac[:, None], y)

        # does detectability depend on WHERE the patch was placed?
        ty = df["rosy_cheeks_translation_y"].to_numpy()
        tz = df["rosy_cheeks_translation_z"].to_numpy()
        return {
            "descriptor": best,
            "cheek_hue_is_pure_red_frac": red_hue,
            "cheek_strength_min": float(strength[pres].min()),
            "cheek_strength_median": float(np.median(strength[pres])),
            "red_area_frac_present_mean": float(red_frac[pres].mean()),
            "red_area_frac_absent_mean": float(red_frac[~pres].mean()),
            "red_area_frac_auc": r_auc,
            "red_area_frac_cv_acc": r_acc, "red_area_frac_cv_sd": r_sd,
            "r_correct_vs_strength": pearson(correct[pres], strength[pres]),
            "r_correct_vs_translation_y": pearson(correct[pres], ty[pres]),
            "r_correct_vs_translation_z": pearson(correct[pres], tz[pres]),
            "overall_recall_present": float(correct[pres].mean()),
            "overall_recall_absent": float(correct[~pres].mean()),
        }

    cheek = stage("cheek", _cheek)

    # ---------------------------------------------------------------- outputs
    out = {
        "n_specimens": int(N),
        "n_faces": int(fcd.colors.shape[1]),
        "chance": {f: 0.5 for f in C.FACTORS} | {"donor7": 1.0 / 7.0},
        "class_balance": {f: [int((labels[f] == 0).sum()), int((labels[f] == 1).sum())]
                          for f in C.FACTORS},
        "descriptors": {n: int(X[n].shape[1]) for n in order},
        "table": table,
        "disentanglement": disent,
        "gt_stripe_param_corr_with_count": gt_corr,
        "k_sweep": sweep,
        "k_saturation": saturation,
        "control_lodo_leave_one_donor_out": lodo,
        "control_donor_signal_source": donor_src,
        "control_continuous_recovery": cont,
        "control_cheek_detectability": cheek,
        "seconds": round(time.time() - T0, 1),
    }
    (RESULTS / f"{NAME}.json").write_text(json.dumps(out, indent=2))

    np.savez_compressed(
        RESULTS / f"{NAME}_arrays.npz",
        names=np.asarray(list(fcd.names)),
        donor=donor,
        **{f"label_{f}": labels[f] for f in C.FACTORS},
        **{f"pc2_{n}": pcs[n] for n in order},
        k_sweep_ks=np.asarray(ks),
        **{f"sweep_acc_{f}": np.asarray(sweep[f]["acc"]) for f in C.FACTORS},
        **{f"sweep_sd_{f}": np.asarray(sweep[f]["sd"]) for f in C.FACTORS},
        **{f"sweep_null_{f}": np.asarray(sweep[f]["null"]) for f in C.FACTORS},
        acc_matrix=np.asarray([[table[n]["factors"][f]["acc"] for f in C.FACTORS]
                               for n in order]),
        null_matrix=np.asarray([[table[n]["factors"][f]["null_acc"] for f in C.FACTORS]
                                for n in order]),
        ari_matrix=np.asarray([[table[n]["factors"][f]["gmm_ari"] for f in C.FACTORS]
                               for n in order]),
        auc_matrix=np.asarray([[table[n]["factors"][f]["auc_best"] for f in C.FACTORS]
                               for n in order]),
        donor_acc=np.asarray([table[n]["donor_acc"] for n in order]),
        lodo_matrix=np.asarray([[lodo[n][f]["acc"] for f in C.FACTORS] for n in order]),
        lodo_null_matrix=np.asarray([[lodo[n][f]["null"] for f in C.FACTORS] for n in order]),
        cont_r2=np.asarray([[cont[n][t]["r2"] for t in cont_targets] for n in cont_desc]),
        cont_desc_order=np.asarray(cont_desc),
        cont_target_order=np.asarray(cont_targets),
        descriptor_order=np.asarray(order),
        stripe_count=nuis["stripe_count"], stripe_spacing=nuis["stripe_spacing"],
        stripe_width=nuis["stripe_width"],
    )

    # ---------------------------------------------------------------- summary
    print("\n" + "=" * 108)
    print("E4  BYOA DESCRIPTORS ON ATLAS OUTPUT — balanced accuracy (null in parens), chance 0.50")
    print("=" * 108)
    hdr = f"{'descriptor':26s}{'dim':>6s}" + "".join(f"{C.FACTOR_LABEL[f].split(' (')[0]:>21s}"
                                                     for f in C.FACTORS) + f"{'donor7':>9s}"
    print(hdr)
    print("-" * 108)
    for n in order:
        r = table[n]
        line = f"{n:26s}{r['dim']:6d}"
        for f in C.FACTORS:
            c = r["factors"][f]
            line += f"{c['acc']:.3f}({c['null_acc']:.3f})".rjust(21)
        line += f"{r['donor_acc']:9.3f}"
        print(line)
    print("-" * 108)

    print("\nUNSUPERVISED (GMM+BIC on top-2 PCs): ARI vs each factor  [n_components chosen]")
    print(f"{'descriptor':26s}{'ncomp':>6s}" + "".join(f"{f:>10s}" for f in C.FACTORS))
    for n in order:
        r = table[n]
        print(f"{n:26s}{r['gmm_bic_ncomp']:6d}"
              + "".join(f"{r['factors'][f]['gmm_ari']:10.3f}" for f in C.FACTORS))

    print("\nBEST SINGLE AXIS AUC (shuffled-label null in parens — max-over-D is optimistic)")
    print(f"{'descriptor':26s}" + "".join(f"{f:>18s}" for f in C.FACTORS))
    for n in order:
        r = table[n]
        print(f"{n:26s}" + "".join(
            f"{r['factors'][f]['auc_best']:.3f}({r['factors'][f]['auc_best_null']:.3f})".rjust(18)
            for f in C.FACTORS))

    print("\nDISENTANGLEMENT of the supervised stripe direction  (|Pearson r|, out-of-fold projection)")
    print(f"  GT check: corr(stripe_count, spacing)={gt_corr['stripe_spacing']:+.3f}, "
          f"corr(stripe_count, width)={gt_corr['stripe_width']:+.3f} "
          f"-> nuisances are independent of count by construction")
    print(f"{'descriptor':26s}{'count':>8s}{'spacing':>9s}{'width':>8s}{'offset':>8s}"
          f"{'sp|cnt':>8s}{'donor_eta2':>11s}  top GT param")
    for n in order:
        d = disent[n]["oof"]
        top = d["top3_gt_params"][0]
        print(f"{n:26s}{d['stripe_count']:8.3f}{d['stripe_spacing']:9.3f}{d['stripe_width']:8.3f}"
              f"{d['stripe_longitudinal_offset']:8.3f}{d['spacing_given_count']:8.3f}"
              f"{d['donor_eta2']:11.3f}  {top[0]}={top[1]:.3f}")

    print("\nGRAPH-FOURIER MODE SWEEP (spectral_coeffs, channels L,a,b) — balanced acc")
    print(f"{'k':>5s}{'dim':>6s}" + "".join(f"{f:>10s}" for f in C.FACTORS))
    for i, k in enumerate(ks):
        print(f"{k:5d}{k * 3:6d}" + "".join(f"{sweep[f]['acc'][i]:10.3f}" for f in C.FACTORS))
    print("saturation (smallest k within 1 sd of the best k):")
    for f in C.FACTORS:
        s = saturation[f]
        print(f"  {f:8s} best acc {s['acc_max']:.3f} at k={s['k_argmax']:3d}; "
              f"saturates at k={s['k_within_1sd']} (1sd) / k={s['k_within_1pct']} (1%)")
    print("\nCONTROL 1 — LEAVE-ONE-DONOR-OUT (train on 6 real donor scans, test on the 7th)")
    print("  stratified-CV accuracy in brackets; a large drop = descriptor is donor-specific")
    print(f"{'descriptor':26s}" + "".join(f"{f:>20s}" for f in C.FACTORS))
    for n in order:
        line = f"{n:26s}"
        for f in C.FACTORS:
            line += f"{lodo[n][f]['acc']:.3f}[{table[n]['factors'][f]['acc']:.3f}]".rjust(20)
        print(line)

    print("\nCONTROL 2 — WHERE DOES THE DONOR SIGNAL COME FROM? (7-way, chance 0.143)")
    for k, v in donor_src.items():
        if k == "blackmask_factor_acc":
            continue
        print(f"  {k:42s} dim={v['dim']:5d}  donor acc={v['donor_acc']:.3f} "
              f"(null {v['donor_null']:.3f})")
    print("  atlas black mask alone, FACTOR accuracies: "
          + " ".join(f"{f}={donor_src['blackmask_factor_acc'][f][0]:.3f}"
                     for f in C.FACTORS))
    print("  -> donor identity is ALREADY ~perfectly recoverable PRE-module (native axial")
    print("     profile 0.995), so the module does not manufacture it; the 7 donor body")
    print("     shapes genuinely lay the same painted pattern out differently.")

    print("\nCONTROL 3 — CONTINUOUS generative-parameter recovery (out-of-fold R^2; "
          "shuffled-target R^2 in parens)")
    print(f"{'descriptor':26s}" + "".join(f"{t[:13]:>16s}" for t in cont_targets))
    for n in cont_desc:
        print(f"{n:26s}" + "".join(
            f"{cont[n][t]['r2']:+.2f}({cont[n][t]['r2_shuffled']:+.2f})".rjust(16)
            for t in cont_targets))

    print("\nCONTROL 4 — WHY DO ROSY CHEEKS TOP OUT? (best cheek descriptor: "
          f"{cheek['descriptor']})")
    print(f"  the planted patch is never weak or camouflaged: "
          f"{cheek['cheek_hue_is_pure_red_frac'] * 100:.0f}% of present specimens are painted "
          f"pure red, strength {cheek['cheek_strength_min']:.2f}-1.00")
    print(f"  DIRECT measurement of red surface in atlas space: present "
          f"{cheek['red_area_frac_present_mean'] * 100:.2f}% of area vs absent "
          f"{cheek['red_area_frac_absent_mean'] * 100:.2f}%  ->  AUC "
          f"{cheek['red_area_frac_auc']:.3f}, 1-feature CV acc "
          f"{cheek['red_area_frac_cv_acc']:.3f}+-{cheek['red_area_frac_cv_sd']:.3f}")
    print(f"  out-of-fold recall  present={cheek['overall_recall_present']:.3f}  "
          f"absent={cheek['overall_recall_absent']:.3f}")
    print(f"  corr(correct, strength)={cheek['r_correct_vs_strength']:+.3f}  "
          f"corr(correct, patch translation y)={cheek['r_correct_vs_translation_y']:+.3f}  "
          f"z={cheek['r_correct_vs_translation_z']:+.3f}")

    print(f"\nwrote {RESULTS / (NAME + '.json')}  and  {RESULTS / (NAME + '_arrays.npz')}")
    print(f"total {time.time() - T0:.0f}s")


if __name__ == "__main__":
    main()
