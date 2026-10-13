"""E6b — strengthened versions of three limit analyses (adequately powered).

The first pass swept only six principal components, ran three sample draws and six label
permutations, which is too little to support the claims. This re-runs:
  (1) accuracy vs number of principal components, to 60, alongside cumulative explained
      variance — the "explained variance is the wrong stopping rule" claim;
  (2) accuracy vs sample size, n in {25,40,80,150,250} with 30 draws;
  (3) the unsupervised-discovery gate with 200 permutations / column-shuffled nulls.
"""
from __future__ import annotations

import json
import time

import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler

import common as C
from fishpipe import features, spectral

FACTORS = ("belly", "tail", "stripe", "cheeks")


def main():
    t0 = time.time()
    fcd, _, mesh = C.atlas_data()
    names = list(fcd.names)
    df, labels = C.load_gt(names)
    reps = {
        "composition (area histogram, 24)": features.area_hist(fcd, n_clusters=24),
        "regional colour (128 regions)": C.region_summary(fcd, mesh, 128, "mean"),
        "graph-Fourier coefficients (k=40)": spectral.spectral_coeffs(
            fcd, k=40, channels=("L", "a", "b"), mesh=mesh, cache_dir=C.ATLAS_DS.cache_dir),
    }
    out = {"meta": {"n": len(names), "factors": list(FACTORS)}}

    # ---------------------------------------------------------------- (1) dims vs accuracy
    dims = [1, 2, 3, 4, 6, 8, 10, 14, 18, 24, 30, 40, 50, 60]
    item1 = {}
    for rn, X in reps.items():
        Xs = StandardScaler().fit_transform(X)
        p = PCA(n_components=min(60, Xs.shape[1], Xs.shape[0] - 1), random_state=0).fit(Xs)
        Z = p.transform(Xs)
        cum = np.cumsum(p.explained_variance_ratio_)
        ev_dim = {q: (int(np.argmax(cum >= q / 100) + 1) if (cum >= q / 100).any() else None)
                  for q in (90, 95, 99)}
        curves = {f: [] for f in FACTORS}
        for d in dims:
            if d > Z.shape[1]:
                for f in FACTORS:
                    curves[f].append(float("nan"))
                continue
            for f in FACTORS:
                a, _ = C.balanced_cv(Z[:, :d], labels[f], n_repeats=2)
                curves[f].append(a)
        sat = {}
        for f in FACTORS:
            v = np.array(curves[f], float)
            amax = np.nanmax(v)
            k = int(np.argmax(v >= amax - 0.01))
            sat[f] = {"acc_max": float(amax), "sat_dim": int(dims[k])}
        item1[rn] = {"dims": dims, "cum_evr": [float(x) for x in cum[:60]],
                     "ev_dim": ev_dim, "acc": {f: [float(x) for x in curves[f]] for f in FACTORS},
                     "saturation": sat}
        print(f"[dims] {rn}: EV95 at {ev_dim[95]} dims; saturation " +
              " ".join(f"{f}={sat[f]['sat_dim']}({sat[f]['acc_max']:.2f})" for f in FACTORS))
    out["dims_vs_accuracy"] = item1

    # ---------------------------------------------------------------- (2) sample size
    best = {"belly": "composition (area histogram, 24)", "tail": "composition (area histogram, 24)",
            "stripe": "regional colour (128 regions)",
            "cheeks": "graph-Fourier coefficients (k=40)"}
    ns = [25, 40, 80, 150, 250]
    item2 = {}
    rng = np.random.default_rng(0)
    for n in ns:
        rec = {f: [] for f in FACTORS}
        for draw in range(30 if n < 250 else 1):
            idx = rng.choice(len(names), size=n, replace=False) if n < 250 else np.arange(250)
            for f in FACTORS:
                y = labels[f][idx]
                if len(np.unique(y)) < 2 or min(np.bincount(y)) < 3:
                    continue
                a, _ = C.balanced_cv(reps[best[f]][idx], y, n_splits=3, n_repeats=2)
                rec[f].append(a)
        item2[str(n)] = {f: {"mean": float(np.mean(v)) if v else None,
                             "sd": float(np.std(v)) if v else None,
                             "p05": float(np.percentile(v, 5)) if v else None,
                             "n_draws": len(v)} for f, v in rec.items()}
        print(f"[n={n}] " + " ".join(f"{f}={item2[str(n)][f]['mean']:.3f}" for f in FACTORS
                                     if item2[str(n)][f]['mean'] is not None))
    out["sample_size"] = {"best_descriptor": best, "curve": item2}

    # ---------------------------------------------------------------- (3) discovery gate
    import diptest
    item3 = {}
    joint = labels["belly"] * 4 + labels["tail"] * 2 + labels["stripe"]
    for rn, X in reps.items():
        Xs = StandardScaler().fit_transform(X)
        Z = PCA(n_components=10, random_state=0).fit_transform(Xs)
        rec = {}
        # GMM+BIC on the 2-D morphospace a user would look at
        bic = {k: GaussianMixture(k, covariance_type="full", random_state=0,
                                  n_init=3).fit(Z[:, :2]).bic(Z[:, :2]) for k in range(1, 9)}
        kbest = min(bic, key=bic.get)
        lab2 = (GaussianMixture(kbest, covariance_type="full", random_state=0, n_init=3)
                .fit(Z[:, :2]).predict(Z[:, :2])) if kbest > 1 else np.zeros(len(Z), int)
        rec["gmm_bic_k_2d"] = int(kbest)
        rec["ari_2d"] = {f: float(adjusted_rand_score(labels[f], lab2)) for f in FACTORS}
        rec["ari_2d_joint8"] = float(adjusted_rand_score(joint, lab2))
        # best single PC, with a proper permutation null
        pc_auc, pc_idx, pc_p = {}, {}, {}
        for f in FACTORS:
            aucs = [max(roc_auc_score(labels[f], Z[:, j]),
                        1 - roc_auc_score(labels[f], Z[:, j])) for j in range(10)]
            j = int(np.argmax(aucs))
            pc_auc[f], pc_idx[f] = float(aucs[j]), j + 1
            null = []
            r2 = np.random.default_rng(5)
            for _ in range(200):
                yp = r2.permutation(labels[f])
                null.append(max(max(roc_auc_score(yp, Z[:, k]),
                                    1 - roc_auc_score(yp, Z[:, k])) for k in range(10)))
            pc_p[f] = float((np.sum(np.array(null) >= aucs[j]) + 1) / 201)
        rec["best_pc"] = {f: {"auc": pc_auc[f], "pc": pc_idx[f], "p_perm": pc_p[f]}
                          for f in FACTORS}
        # Hartigan dip on each PC vs a column-shuffled null (destroys joint structure)
        dips = [float(diptest.diptest(Z[:, j])[0]) for j in range(10)]
        pvals = [float(diptest.diptest(Z[:, j])[1]) for j in range(10)]
        r3 = np.random.default_rng(11)
        null_dips = []
        for _ in range(200):
            Xn = np.column_stack([r3.permutation(Xs[:, c]) for c in range(Xs.shape[1])])
            Zn = PCA(n_components=10, random_state=0).fit_transform(Xn)
            null_dips.append(max(diptest.diptest(Zn[:, j])[0] for j in range(10)))
        rec["dip"] = {"per_pc": dips, "p_per_pc": pvals,
                      "max_dip": float(max(dips)),
                      "null_max_dip_p95": float(np.percentile(null_dips, 95)),
                      "passes_gate": bool(max(dips) > np.percentile(null_dips, 95))}
        item3[rn] = rec
        print(f"[gate] {rn}: GMM k={kbest} ARI8={rec['ari_2d_joint8']:.3f} | "
              f"maxdip={rec['dip']['max_dip']:.4f} vs null95 {rec['dip']['null_max_dip_p95']:.4f} "
              f"-> {'PASS' if rec['dip']['passes_gate'] else 'ABSTAIN'}")
    out["discovery_gate"] = item3

    out["seconds"] = round(time.time() - t0, 1)
    (C.RESULTS / "e6b_limits.json").write_text(json.dumps(out, indent=2))
    print("saved results/e6b_limits.json in", out["seconds"], "s")


if __name__ == "__main__":
    main()
