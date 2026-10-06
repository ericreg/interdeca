"""E2 — Information preservation across the ColorAtlas module.

What survives, what is lost, and what is GAINED when 250 specimen textures — each painted on
its own donor mesh in its own UV chart — are transported into one shared atlas space.

The comparison is organised so that every claim rests on a PAIR of numbers produced by the
SAME estimator under the SAME protocol, differing only in which side of the module the data
came from:

  COMPOSITION ("what colours, how much area")  — a statistic that needs no correspondence,
      so it can be computed identically before and after transfer. Any drop here is a pure
      TRANSFER COST.
        pre  : nat['hist']                       (area-weighted Lab palette histogram)
        post : features.area_hist(fcd, 24)       (the module's own version of that statistic)
      Because those two use palettes fitted independently in each space, and because the
      module's version L2-normalises while the native one L1-normalises, we ALSO report a
      strictly matched pair (same palette = the native one, same L1 normalisation) so that
      the composition conclusion cannot be an artifact of palette or scaling choices.

  SPATIAL ("where on the animal")  — this is what the atlas is FOR. Any gain here is what
      correspondence BUYS.
        pre  : nat['spatial']  — 32-bin axial Lab profile along each specimen's OWN principal
               axis. This is the best spatial frame available with no correspondence step.
               Its head/tail sign is set by an SVD whose sign is arbitrary, so we additionally
               report a LABEL-FREE sign-canonicalised version: refusing to fix a fixable
               defect would strawman the pre-module side.
        post : features.spatial_flatten (per-face Lab, 4000 faces), spectral GFT coefficients,
               and — dimension-matched to the pre-module descriptor — a 32-bin axial Lab
               profile along the ATLAS principal axis (96-d, exactly the same statistic as
               `nat['spatial']`, differing only in that the axis and the bins are now shared
               across specimens).

Read-outs: (1) balanced 5x4-fold CV accuracy on the 4 planted categorical factors, always
against a shuffled-label null; (2) cross-validated Ridge R^2 for 8 continuous generating
parameters; (3) Bland-Altman agreement of planted-pigment AREA fractions pre vs post, broken
down by donor shape; (4) a donor-shape confound check — if the atlas leaked geometry into
colour, the donor would be recoverable from post-module colour descriptors.

Ground truth is used for scoring only; no representation ever sees a label.
"""
from __future__ import annotations

import json
import os
import time

# keep BLAS from oversubscribing when we fan jobs out with joblib
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import numpy as np  # noqa: E402

import common as C  # noqa: E402
from fishpipe import features, spectral  # noqa: E402
from fishpipe.features import rgb_to_lab  # noqa: E402

N_SHUFFLE = 3          # shuffled-label repeats used for every null
N_JOBS = 12
CONT_TARGETS = ("belly_hue", "tail_hue", "base_color_hue", "base_color_sat",
                "base_color_val", "stripe_spacing", "stripe_width",
                "stripe_longitudinal_offset")


# --------------------------------------------------------------------------- descriptors
def axial_profile(lab_faces: np.ndarray, areas: np.ndarray, t: np.ndarray,
                  n_bins: int = 32) -> np.ndarray:
    """Area-weighted mean Lab in `n_bins` bins along a normalised axial coordinate t in [0,1].

    Identical arithmetic to the pre-module `nat['spatial']` builder in common.py, so the two
    are the same statistic; only the axis differs (own axis vs shared atlas axis).
    """
    bi = np.clip((t * n_bins).astype(int), 0, n_bins - 1)
    w = np.bincount(bi, weights=areas, minlength=n_bins)
    w[w == 0] = 1.0
    prof = np.stack([np.bincount(bi, weights=areas * lab_faces[:, c], minlength=n_bins) / w
                     for c in range(3)], axis=1)
    return prof.reshape(-1)


def canonicalise_axial_sign(S: np.ndarray, n_iter: int = 25):
    """Label-free head/tail sign alignment of per-specimen axial profiles.

    Each specimen's principal axis comes from an SVD, whose sign is arbitrary, so roughly half
    the pre-module profiles run tail->head and half head->tail. We iteratively flip each
    profile to whichever orientation correlates better with the population mean profile
    (Procrustes-style sign alignment). This uses no labels and no atlas — it is something any
    analyst could do — and it is applied ONLY to the pre-module side, i.e. it can only make
    the pre-module baseline stronger.
    """
    N = S.shape[0]
    X = S.copy()                                          # (N, n_bins, 3)
    mu = X.reshape(-1, 3).mean(0)
    sd = X.reshape(-1, 3).std(0) + 1e-12
    flipped = np.zeros(N, bool)
    for _ in range(n_iter):
        Z = (X - mu) / sd
        m = Z.mean(0)
        m = m - m.mean(0, keepdims=True)
        changed = 0
        for i in range(N):
            a = Z[i] - Z[i].mean(0, keepdims=True)
            b = a[::-1]
            if (b * m).sum() > (a * m).sum():
                X[i] = X[i][::-1]
                flipped[i] = ~flipped[i]
                changed += 1
        if changed == 0:
            break
    return X.reshape(N, -1), flipped


def hist_on_palette(lab_faces: np.ndarray, areas: np.ndarray, palette: np.ndarray) -> np.ndarray:
    """L1-normalised area-weighted histogram over a GIVEN Lab palette (nearest centroid)."""
    d = ((lab_faces[:, None, :] - palette[None, :, :]) ** 2).sum(-1)
    lbl = np.argmin(d, axis=1)
    h = np.bincount(lbl, weights=areas, minlength=palette.shape[0])
    return h / (h.sum() + 1e-12)


# --------------------------------------------------------------------------- estimators
def ridge_r2(X: np.ndarray, y: np.ndarray, n_splits: int = 5, n_repeats: int = 4, seed: int = 0):
    """Cross-validated R^2 of a standardised RidgeCV.

    Out-of-fold predictions are pooled within each repeat and scored once, giving one R^2 per
    repeat; we report mean and sd over repeats. The ridge penalty is selected by RidgeCV's
    internal leave-one-out on the TRAINING fold only, so no test information reaches the fit.
    """
    from sklearn.linear_model import RidgeCV
    from sklearn.metrics import r2_score
    from sklearn.model_selection import KFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    alphas = np.logspace(-3, 6, 19)
    r2s = []
    for rep in range(n_repeats):
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=seed + rep)
        oof = np.zeros_like(y)
        for tr, te in cv.split(X):
            mdl = make_pipeline(StandardScaler(), RidgeCV(alphas=alphas))
            mdl.fit(X[tr], y[tr])
            oof[te] = mdl.predict(X[te])
        r2s.append(r2_score(y, oof))
    return float(np.mean(r2s)), float(np.std(r2s))


def _clf_job(X, y, shuffle_seed):
    if shuffle_seed is None:
        return C.balanced_cv(X, y)
    rng = np.random.default_rng(shuffle_seed)
    return C.balanced_cv(X, rng.permutation(y))


def _reg_job(X, y, shuffle_seed):
    if shuffle_seed is None:
        return ridge_r2(X, y)
    rng = np.random.default_rng(shuffle_seed)
    return ridge_r2(X, rng.permutation(y))


# --------------------------------------------------------------------------- main
def main():
    t0 = time.time()
    from joblib import Parallel, delayed

    fcd, art, mesh = C.atlas_data()
    names = list(fcd.names)
    df, labels = C.load_gt(names)
    nat = C.native_face_colors(verbose=False)
    assert list(nat["names"]) == names, "pre/post specimen order mismatch"
    N = len(names)
    donor = df["specimen_index"].to_numpy().astype(int)

    print(f"[E2] {N} specimens, {mesh.n_faces} atlas faces, {len(np.unique(donor))} donor shapes")

    # ---------------------------------------------------------------- POST descriptors
    print("[E2] building POST-module descriptors ...")
    atlas_lab = rgb_to_lab(fcd.colors)                      # (N, Nf, 3) float64
    areas = fcd.areas

    t = time.time()
    post_hist = features.area_hist(fcd, n_clusters=24)      # (N,24) L2-normalised, own palette
    print(f"  area_hist {post_hist.shape} ({time.time() - t:.1f}s)")
    post_flat = features.spatial_flatten(fcd, color_space="lab", subsample=4000)   # (N,12000)
    post_spec = spectral.spectral_coeffs(fcd, k=40, mesh=mesh, cache_dir=C.ATLAS_DS.cache_dir)
    print(f"  spatial_flatten {post_flat.shape}  spectral {post_spec.shape}")

    # dimension-matched axial profile in the SHARED atlas frame (96-d, same statistic as pre)
    axis = C.principal_axis(mesh)
    cents = mesh.face_centroids()
    tt = (cents - cents.mean(0)) @ axis
    tt = (tt - tt.min()) / (tt.max() - tt.min() + 1e-12)
    post_axial = np.stack([axial_profile(atlas_lab[i], areas, tt) for i in range(N)])
    # palette/normalisation-matched composition: native palette, L1 norm, atlas colours
    post_hist_matched = np.stack([hist_on_palette(atlas_lab[i], areas, nat["palette"])
                                  for i in range(N)])
    print(f"  post_axial {post_axial.shape}  post_hist_matched {post_hist_matched.shape}")

    # ---------------------------------------------------------------- PRE descriptors
    pre_hist = nat["hist"]                                   # (N,24) L1, native palette
    pre_spatial = nat["spatial"]                             # (N,96) own-axis, arbitrary sign
    pre_spatial_canon, flipped = canonicalise_axial_sign(pre_spatial.reshape(N, 32, 3))
    print(f"  pre axial sign-canonicalisation flipped {int(flipped.sum())}/{N} profiles")

    BLOCKS = {
        "pre_hist":            ("PRE",  "composition", pre_hist),
        "post_hist":           ("POST", "composition", post_hist),
        "post_hist_matched":   ("POST", "composition", post_hist_matched),
        "pre_spatial":         ("PRE",  "spatial",     pre_spatial),
        "pre_spatial_canon":   ("PRE",  "spatial",     pre_spatial_canon),
        "post_axial":          ("POST", "spatial",     post_axial),
        "post_spectral":       ("POST", "spatial",     post_spec),
        "post_spatial_flat":   ("POST", "spatial",     post_flat),
    }
    for k, (side, kind, X) in BLOCKS.items():
        assert X.shape[0] == N and np.isfinite(X).all(), f"bad block {k}"

    # ---------------------------------------------------------------- (3) pigment areas
    # done before the parallel sections so the 300 MB per-face Lab tensor can be released
    print("[E2] planted-pigment area agreement (pre vs post) ...")
    pre_area = nat["masks"]                                  # (N,5) PIGMENTS order
    post_area = np.stack([C.pigment_area_fractions(atlas_lab[i], areas,
                                                   C.pigment_lab(df.iloc[i])) for i in range(N)])
    del atlas_lab

    # ---------------------------------------------------------------- (1) classification
    print("[E2] classification (balanced 5-fold x4 CV + shuffled null) ...")
    tasks = []
    for bname in BLOCKS:
        for fac in C.FACTORS + ("donor",):
            for s in [None] + list(range(N_SHUFFLE)):
                tasks.append((bname, fac, s))

    def y_of(fac):
        return donor if fac == "donor" else labels[fac]

    res = Parallel(n_jobs=N_JOBS, verbose=1)(
        delayed(_clf_job)(BLOCKS[b][2], y_of(f), s) for b, f, s in tasks)

    clf = {}
    for (b, f, s), (acc, sd) in zip(tasks, res):
        d = clf.setdefault(b, {}).setdefault(f, {"null": []})
        if s is None:
            d["acc"], d["acc_sd"] = acc, sd
        else:
            d["null"].append(acc)
    for b in clf:
        for f in clf[b]:
            n = clf[b][f].pop("null")
            clf[b][f]["null_acc"] = float(np.mean(n))
            clf[b][f]["null_acc_sd"] = float(np.std(n))

    # ---------------------------------------------------------------- (2) regression
    print("[E2] continuous parameter recovery (RidgeCV, 5-fold x4) ...")
    rtasks = [(b, tg, s) for b in BLOCKS for tg in CONT_TARGETS for s in (None, 0)]
    rres = Parallel(n_jobs=N_JOBS, verbose=1)(
        delayed(_reg_job)(BLOCKS[b][2], df[tg].to_numpy(), s) for b, tg, s in rtasks)
    reg = {}
    for (b, tg, s), (r2, sd) in zip(rtasks, rres):
        d = reg.setdefault(b, {}).setdefault(tg, {})
        if s is None:
            d["r2"], d["r2_sd"] = r2, sd
        else:
            d["null_r2"] = r2

    def _donor_center(v):
        """Subtract the donor-group mean — removes the between-donor component."""
        out = v.astype(float).copy()
        for k in np.unique(donor):
            m = donor == k
            out[m] -= out[m].mean()
        return out

    def _frac_var_donor(v):
        gm = np.array([v[donor == k].mean() for k in np.unique(donor)])
        idx = {k: i for i, k in enumerate(np.unique(donor))}
        between = np.var(np.array([gm[idx[k]] for k in donor]))
        return float(between / (np.var(v) + 1e-30))

    pig = {}
    for p, name in enumerate(C.PIGMENTS):
        a, b = pre_area[:, p], post_area[:, p]
        d = b - a
        if a.std() > 1e-12 and b.std() > 1e-12:
            r = float(np.corrcoef(a, b)[0, 1])
            slope, intercept = np.polyfit(a, b, 1)
        else:
            r, slope, intercept = float("nan"), float("nan"), float("nan")
        # Between-donor structure dominates these area fractions (the 7 donor shapes have very
        # different regional proportions), so a raw pre-vs-post r mostly certifies that the
        # donor CLUSTERS line up. The donor-centred r is the honest per-specimen agreement:
        # does the module preserve how one specimen differs from its own donor's average?
        ac, bc = _donor_center(a), _donor_center(b)
        if ac.std() > 1e-12 and bc.std() > 1e-12:
            r_w = float(np.corrcoef(ac, bc)[0, 1])
            slope_w = float(np.polyfit(ac, bc, 1)[0])
        else:
            r_w, slope_w = float("nan"), float("nan")
        pig[name] = {
            "pearson_r": r, "slope_post_on_pre": float(slope), "intercept": float(intercept),
            "within_donor_pearson_r": r_w, "within_donor_slope": slope_w,
            "frac_var_explained_by_donor_pre": _frac_var_donor(a),
            "frac_var_explained_by_donor_post": _frac_var_donor(b),
            "mean_signed_diff": float(d.mean()), "sd_diff": float(d.std()),
            "loa_lo": float(d.mean() - 1.96 * d.std()), "loa_hi": float(d.mean() + 1.96 * d.std()),
            "mean_abs_diff": float(np.abs(d).mean()),
            "pre_mean": float(a.mean()), "post_mean": float(b.mean()),
            "pre_sd": float(a.std()), "post_sd": float(b.std()),
            "rel_bias_pct": float(100 * d.mean() / (a.mean() + 1e-12)),
        }

    # ---------------------------------------------------------------- per-donor breakdown
    per_donor = {}
    for k in sorted(np.unique(donor)):
        m = donor == k
        per_donor[int(k)] = {
            "n": int(m.sum()),
            **{f"mad_{name}": float(np.abs(post_area[m, p] - pre_area[m, p]).mean())
               for p, name in enumerate(C.PIGMENTS)},
        }
    # does donor explain the pre/post area discrepancy?
    from scipy import stats as _st
    donor_anova = {}
    for p, name in enumerate(C.PIGMENTS):
        g = [np.abs(post_area[donor == k, p] - pre_area[donor == k, p])
             for k in sorted(np.unique(donor))]
        F, pv = _st.f_oneway(*g)
        donor_anova[name] = {"F": float(F), "p": float(pv)}

    # ------------------------------------------------- donor / parameter independence check
    # Load-bearing for the confound argument: donor shape is recoverable from colour on BOTH
    # sides of the module, but that can only distort the factor results if donor is correlated
    # with the generating parameters. The generator assigned donors independently, and we
    # verify it rather than assume it.
    dep = {}
    for col in CONT_TARGETS + ("stripe_count", "rosy_cheeks_present"):
        g = [df[col].to_numpy()[donor == k] for k in np.unique(donor)]
        F, pv = _st.f_oneway(*g)
        dep[col] = {"anova_F": float(F), "anova_p": float(pv)}
    for f in C.FACTORS:
        ct = np.array([[int(((donor == k) & (labels[f] == c)).sum())
                        for c in np.unique(labels[f])] for k in np.unique(donor)])
        chi2, pv = _st.chi2_contingency(ct)[:2]
        dep[f"label_{f}"] = {"chi2": float(chi2), "chi2_p": float(pv)}

    # ---------------------------------------------------------------- headline summary
    # Transfer COST  = matched composition statistic, post minus pre (same palette, same norm).
    # Atlas GAIN     = best POST spatial descriptor minus the strongest (sign-canonicalised)
    #                  PRE spatial descriptor. Using the canonicalised pre-module baseline is
    #                  the conservative choice: it makes the gain smaller, not larger.
    POST_SPATIAL = ("post_axial", "post_spectral", "post_spatial_flat")
    summary = {"transfer_cost_composition": {}, "atlas_gain_spatial": {},
               "atlas_gain_spatial_dim_matched": {}}
    for f in C.FACTORS:
        summary["transfer_cost_composition"][f] = float(
            clf["post_hist_matched"][f]["acc"] - clf["pre_hist"][f]["acc"])
        best = max(clf[b][f]["acc"] for b in POST_SPATIAL)
        summary["atlas_gain_spatial"][f] = float(best - clf["pre_spatial_canon"][f]["acc"])
        summary["atlas_gain_spatial_dim_matched"][f] = float(
            clf["post_axial"][f]["acc"] - clf["pre_spatial_canon"][f]["acc"])
    summary["transfer_cost_composition_r2"] = {
        t: float(reg["post_hist_matched"][t]["r2"] - reg["pre_hist"][t]["r2"])
        for t in CONT_TARGETS}
    summary["atlas_gain_spatial_r2_dim_matched"] = {
        t: float(reg["post_axial"][t]["r2"] - reg["pre_spatial_canon"][t]["r2"])
        for t in CONT_TARGETS}
    summary["donor_recoverability"] = {
        b: float(clf[b]["donor"]["acc"]) for b in BLOCKS}

    # ---------------------------------------------------------------- assemble + report
    out = {
        "summary": summary,
        "n_specimens": N, "n_atlas_faces": int(mesh.n_faces),
        "dims": {k: int(v[2].shape[1]) for k, v in BLOCKS.items()},
        "side": {k: v[0] for k, v in BLOCKS.items()},
        "kind": {k: v[1] for k, v in BLOCKS.items()},
        "classification": clf,
        "regression": reg,
        "pigment_area_agreement": pig,
        "per_donor_mean_abs_area_diff": per_donor,
        "per_donor_anova_on_abs_diff": donor_anova,
        "donor_parameter_independence": dep,
        "pre_axial_profiles_flipped": int(flipped.sum()),
        "chance": {"factors": 0.5, "donor": float(1.0 / len(np.unique(donor)))},
        "seconds": round(time.time() - t0, 1),
    }
    (C.RESULTS / "e2_preservation.json").write_text(json.dumps(out, indent=2))
    np.savez_compressed(
        C.RESULTS / "e2_preservation_arrays.npz",
        pre_area=pre_area, post_area=post_area, donor=donor,
        pigments=np.array(C.PIGMENTS), names=np.array(names),
        pre_hist=pre_hist, post_hist=post_hist, post_hist_matched=post_hist_matched,
        pre_spatial=pre_spatial, pre_spatial_canon=pre_spatial_canon,
        post_axial=post_axial, post_spectral=post_spec,
        flipped=flipped, palette=nat["palette"],
        **{f"label_{k}": v for k, v in labels.items()},
        **{f"param_{t}": df[t].to_numpy() for t in CONT_TARGETS},
    )

    # ---------------------------------------------------------------- printed summary
    W = 20
    print("\n" + "=" * 104)
    print("CLASSIFICATION — balanced accuracy (shuffled-label null in parens); chance .500 / donor .143")
    print("=" * 104)
    hdr = f"{'descriptor':<20}{'side':<6}{'dim':>6}  " + "".join(
        f"{f:>17}" for f in C.FACTORS + ("donor",))
    print(hdr)
    for b, (side, kind, X) in BLOCKS.items():
        row = f"{b:<20}{side:<6}{X.shape[1]:>6}  "
        for f in C.FACTORS + ("donor",):
            c = clf[b][f]
            row += f"{c['acc']:.3f} ({c['null_acc']:.3f})".rjust(17)
        print(row)

    print("\n" + "=" * 104)
    print("CONTINUOUS RECOVERY — cross-validated Ridge R^2 (negative = worse than the mean)")
    print("=" * 104)
    print(f"{'descriptor':<20}" + "".join(f"{t[:11]:>13}" for t in CONT_TARGETS))
    for b in BLOCKS:
        print(f"{b:<20}" + "".join(f"{reg[b][t]['r2']:>13.3f}" for t in CONT_TARGETS))
    print(f"{'(null, shuffled)':<20}" + "".join(
        f"{np.mean([reg[b][t]['null_r2'] for b in BLOCKS]):>13.3f}" for t in CONT_TARGETS))

    print("\n" + "=" * 104)
    print("PLANTED-PIGMENT AREA FRACTION — pre (own mesh) vs post (atlas mean shape)")
    print("=" * 104)
    print(f"{'pigment':<10}{'pre_mean':>10}{'post_mean':>11}{'r':>8}{'slope':>8}"
          f"{'bias':>10}{'rel%':>9}{'LoA':>22}{'r|donor':>10}{'%varDonor':>11}")
    for name, v in pig.items():
        loa = "[%+.4f, %+.4f]" % (v["loa_lo"], v["loa_hi"])
        print(f"{name:<10}{v['pre_mean']:>10.4f}{v['post_mean']:>11.4f}{v['pearson_r']:>8.3f}"
              f"{v['slope_post_on_pre']:>8.3f}{v['mean_signed_diff']:>10.4f}"
              f"{v['rel_bias_pct']:>9.1f}{loa:>22}"
              f"{v['within_donor_pearson_r']:>10.3f}"
              f"{100 * v['frac_var_explained_by_donor_pre']:>11.1f}")
    print("  r|donor = donor-centred correlation (per-specimen agreement with between-donor "
          "structure removed);\n  %varDonor = share of PRE area-fraction variance explained by donor shape.")

    print("\n" + "=" * 104)
    print("HEADLINE")
    print("=" * 104)
    print("transfer cost, composition (post_hist_matched - pre_hist), balanced acc:")
    print("   " + "  ".join(f"{f}={summary['transfer_cost_composition'][f]:+.3f}" for f in C.FACTORS))
    print("atlas gain, spatial, dimension-matched (post_axial - pre_spatial_canon), balanced acc:")
    print("   " + "  ".join(f"{f}={summary['atlas_gain_spatial_dim_matched'][f]:+.3f}"
                            for f in C.FACTORS))
    print("atlas gain, spatial, best POST vs best PRE:")
    print("   " + "  ".join(f"{f}={summary['atlas_gain_spatial'][f]:+.3f}" for f in C.FACTORS))
    print("atlas gain, spatial R^2, dimension-matched:")
    print("   " + "  ".join(f"{t[:12]}={summary['atlas_gain_spatial_r2_dim_matched'][t]:+.3f}"
                            for t in CONT_TARGETS))

    print("\nPER-DONOR mean |post-pre| area fraction")
    print(f"{'donor':<8}{'n':>5}" + "".join(f"{p:>10}" for p in C.PIGMENTS))
    for k, v in per_donor.items():
        print(f"{k:<8}{v['n']:>5}" + "".join(f"{v['mad_' + p]:>10.4f}" for p in C.PIGMENTS))
    print("donor ANOVA on |post-pre|: " + ", ".join(
        f"{k} F={v['F']:.2f} p={v['p']:.2g}" for k, v in donor_anova.items()))
    print(f"\ntotal {out['seconds']}s -> results/e2_preservation.json")


if __name__ == "__main__":
    main()
