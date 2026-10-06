"""E3 — the analysis a user gets from the module's own panel, scored against ground truth.

Reproduces `performPopulationAnalysis`: an area-weighted colour-composition vector per
specimen, then PCA / ICA / UMAP, then the auto-clustering and cluster-trustworthiness checks
a user would rely on. Everything here is label-free; the generating parameters are used only
to score the result afterwards.
"""
from __future__ import annotations

import json
import time

import numpy as np
from sklearn.decomposition import PCA, FastICA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.mixture import GaussianMixture

import common as C
from fishpipe import features

FACTORS = ("belly", "tail", "stripe", "cheeks")
PARAMS = ["belly_hue", "tail_hue", "base_color_hue", "base_color_sat", "base_color_val",
          "stripe_spacing", "stripe_width", "stripe_longitudinal_offset", "belly_strength",
          "belly_translation", "tail_strength", "stripe_count", "rosy_cheeks_present"]


def gmm_bic_select(X, kmax=8, seed=0):
    bic = {}
    for k in range(1, kmax + 1):
        g = GaussianMixture(k, covariance_type="full", random_state=seed, n_init=4).fit(X)
        bic[k] = float(g.bic(X))
    kbest = min(bic, key=bic.get)
    lab = (GaussianMixture(kbest, covariance_type="full", random_state=seed, n_init=4)
           .fit(X).predict(X)) if kbest > 1 else np.zeros(len(X), int)
    return kbest, lab, bic


def main():
    t0 = time.time()
    fcd, _, mesh = C.atlas_data()
    names = list(fcd.names)
    df, labels = C.load_gt(names)
    joint4 = labels["belly"] * 2 + labels["tail"]
    joint8 = joint4 * 2 + labels["stripe"]
    out = {"meta": {"n": len(names), "n_faces": int(mesh.n_faces)}}

    reps = {}
    for K in (16, 24, 30):
        reps[f"area_hist_K{K}"] = features.area_hist(fcd, n_clusters=K)
    reps["subsample_avg_4000"] = features.spatial_flatten(fcd, "lab", subsample=4000)

    table = {}
    coords = {}
    for rn, X in reps.items():
        p = PCA(n_components=min(10, X.shape[1]), random_state=0).fit(X)
        Z = p.transform(X)
        rec = {"evr": [float(v) for v in p.explained_variance_ratio_]}
        # accuracy from the first n components, as a user's downstream test would see it
        rec["acc_by_ncomp"] = {}
        for n in (2, 4, 10):
            n = min(n, Z.shape[1])
            rec["acc_by_ncomp"][str(n)] = {
                f: C.balanced_cv(Z[:, :n], labels[f])[0] for f in FACTORS}
        Xr = X if X.shape[1] <= 512 else PCA(n_components=50, random_state=0).fit_transform(X)
        rec["acc_full"] = {f: C.balanced_cv(Xr, labels[f])[0] for f in FACTORS}
        rec["acc_full_on"] = "raw" if X.shape[1] <= 512 else "top-50 PCs"
        # auto-clustering exactly as the user would run it, on the 2-D morphospace
        k2, lab2, bic = gmm_bic_select(Z[:, :2])
        rec["gmm2d"] = {
            "k": int(k2), "bic": bic,
            "ari_belly_x_tail": float(adjusted_rand_score(joint4, lab2)),
            "ari_joint8": float(adjusted_rand_score(joint8, lab2)),
            "ari_per_factor": {f: float(adjusted_rand_score(labels[f], lab2)) for f in FACTORS},
            "silhouette": float(silhouette_score(Z[:, :2], lab2)) if k2 > 1 else float("nan"),
        }
        # and on the first four components, which is what actually holds the second factor
        k4, lab4, _ = gmm_bic_select(Z[:, :4])
        rec["gmm4d"] = {
            "k": int(k4),
            "ari_belly_x_tail": float(adjusted_rand_score(joint4, lab4)),
            "ari_joint8": float(adjusted_rand_score(joint8, lab4)),
            "ari_per_factor": {f: float(adjusted_rand_score(labels[f], lab4)) for f in FACTORS},
        }
        # PC x generating-parameter correlations
        corr = np.zeros((6, len(PARAMS)))
        for i in range(min(6, Z.shape[1])):
            for j, prm in enumerate(PARAMS):
                corr[i, j] = np.corrcoef(Z[:, i], df[prm].to_numpy(float))[0, 1]
        rec["pc_param_corr"] = corr.tolist()
        table[rn] = rec
        coords[rn] = Z
        print(f"[{rn}] EV1/2 = {rec['evr'][0]:.3f}/{rec['evr'][1]:.3f}  "
              f"GMM-2D k={k2} ARI(belly×tail)={rec['gmm2d']['ari_belly_x_tail']:.3f}  "
              f"GMM-4D k={k4} ARI={rec['gmm4d']['ari_belly_x_tail']:.3f}")
    out["params"] = PARAMS
    out["table"] = table

    # ---- ICA and UMAP on the default representation ---------------------------------
    X = reps["area_hist_K24"]
    ica = FastICA(n_components=10, random_state=0, max_iter=2000, whiten="unit-variance")
    Zi = ica.fit_transform(X)
    ki, labi, _ = gmm_bic_select(Zi[:, :2])
    umap_ok = True
    try:
        import umap
        Zu = umap.UMAP(n_components=2, random_state=42, n_neighbors=15,
                       min_dist=0.1).fit_transform(X)
    except Exception as e:                                              # noqa: BLE001
        umap_ok = False
        Zu = PCA(2, random_state=0).fit_transform(X)
        print("UMAP unavailable:", e)
    ku, labu, _ = gmm_bic_select(Zu)
    out["reducers"] = {
        "ICA10_top2": {"k": int(ki), "ari_belly_x_tail": float(adjusted_rand_score(joint4, labi)),
                       "acc": {f: C.balanced_cv(Zi, labels[f])[0] for f in FACTORS}},
        "UMAP2": {"available": umap_ok, "k": int(ku),
                  "ari_belly_x_tail": float(adjusted_rand_score(joint4, labu)),
                  "acc": {f: C.balanced_cv(Zu, labels[f])[0] for f in FACTORS}},
    }
    print("[ICA]  k=", ki, "ARI", out["reducers"]["ICA10_top2"]["ari_belly_x_tail"])
    print("[UMAP] k=", ku, "ARI", out["reducers"]["UMAP2"]["ari_belly_x_tail"])

    # ---- trustworthiness gate on the default 2-D morphospace ------------------------
    import diptest
    from fishpipe import gating
    Z = coords["area_hist_K24"]
    sig = gating.sigclust(Z[:, :2], n_sim=500, seed=0)
    dips = [float(diptest.diptest(Z[:, j])[0]) for j in range(6)]
    dipp = [float(diptest.diptest(Z[:, j])[1]) for j in range(6)]
    rng = np.random.default_rng(3)
    null_max = []
    for _ in range(200):
        Xn = np.column_stack([rng.permutation(X[:, c]) for c in range(X.shape[1])])
        Zn = PCA(n_components=6, random_state=0).fit_transform(Xn)
        null_max.append(max(diptest.diptest(Zn[:, j])[0] for j in range(6)))
    out["gate"] = {"sigclust_p": float(sig.get("p", float("nan"))),
                   "dip_per_pc": dips, "dip_p_per_pc": dipp,
                   "dip_null_p95": float(np.percentile(null_max, 95)),
                   "passes": bool(max(dips) > np.percentile(null_max, 95))}
    print("[gate] SigClust p =", out["gate"]["sigclust_p"],
          " max dip", max(dips), "vs null95", out["gate"]["dip_null_p95"])

    # ---- the sample sizes real studies have -----------------------------------------
    small = {}
    rng = np.random.default_rng(11)
    X24 = reps["area_hist_K24"]
    for n in (25, 40, 80):
        aris, ks, gates = [], [], []
        for _ in range(30):
            idx = rng.choice(len(names), size=n, replace=False)
            Xi = X24[idx]
            Zi2 = PCA(n_components=2, random_state=0).fit_transform(Xi)
            k, lab, _ = gmm_bic_select(Zi2, kmax=6)
            aris.append(adjusted_rand_score(joint4[idx], lab))
            ks.append(k)
            d = max(diptest.diptest(Zi2[:, j])[0] for j in range(2))
            nulls = []
            for _ in range(40):
                Xn = np.column_stack([rng.permutation(Xi[:, c]) for c in range(Xi.shape[1])])
                Zn = PCA(n_components=2, random_state=0).fit_transform(Xn)
                nulls.append(max(diptest.diptest(Zn[:, j])[0] for j in range(2)))
            gates.append(d > np.percentile(nulls, 95))
        small[str(n)] = {"ari_mean": float(np.mean(aris)), "ari_sd": float(np.std(aris)),
                         "k_mode": int(np.bincount(ks).argmax()),
                         "gate_fire_frac": float(np.mean(gates))}
        print(f"[small-n {n}] ARI {small[str(n)]['ari_mean']:.3f}±{small[str(n)]['ari_sd']:.3f} "
              f"k mode {small[str(n)]['k_mode']} gate fires {100*small[str(n)]['gate_fire_frac']:.0f}%")
    out["small_n"] = small

    np.savez_compressed(C.RESULTS / "e3_module_eda_arrays.npz",
                        pca_K24=coords["area_hist_K24"], ica=Zi, umap=Zu,
                        gmm2d_labels=gmm_bic_select(coords["area_hist_K24"][:, :2])[1],
                        joint4=joint4, joint8=joint8,
                        pc_param_corr=np.array(table["area_hist_K24"]["pc_param_corr"]))
    out["seconds"] = round(time.time() - t0, 1)
    (C.RESULTS / "e3_module_eda.json").write_text(json.dumps(out, indent=2))
    print("saved results/e3_module_eda.json in", out["seconds"], "s")


if __name__ == "__main__":
    main()
