"""E5b — expert-guided analysis, and how much it depends on the representation it sits on.

E5 ran the tag-feedback loop on a colour-composition + texton representation and found that
stripe count plateaued near chance. E4 shows why: that representation simply does not carry
stripe count (0.66-0.74 balanced accuracy even with every label). The interesting question is
therefore not "does expert feedback work" but "expert feedback on WHAT". Here the identical
feedback protocol is run on two representations built from the SAME ColorAtlas output:

  Z_comp : the module's own colour-composition vector (area-weighted 24-bin Lab histogram)
  Z_full : PCA-60 of [composition | regional colour | graph-Fourier coefficients], i.e. what a
           user gets by taking the module's corresponded output into their own analysis

Expert responses are simulated from the generating parameters; that limitation is explicit.
"""
from __future__ import annotations

import json
import time

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import adjusted_rand_score, balanced_accuracy_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

import common as C
from fishpipe import features, spectral

BUDGETS = [8, 16, 24, 40, 56, 80, 120, 160]
TAGS = ["blue_belly", "purple_belly", "green_tail", "cyan_tail", "five_stripes", "rosy_cheeks"]
NSEED = 8


def build_reps(fcd, mesh):
    comp = features.area_hist(fcd, n_clusters=24)
    reg = C.region_summary(fcd, mesh, 128, "mean")
    spec = spectral.spectral_coeffs(fcd, k=40, channels=("L", "a", "b"), mesh=mesh,
                                    cache_dir=C.ATLAS_DS.cache_dir)
    Zc = StandardScaler().fit_transform(comp)
    big = np.hstack([StandardScaler().fit_transform(x) for x in (comp, reg, spec)])
    p = PCA(n_components=60, random_state=0)
    Zf = p.fit_transform(StandardScaler().fit_transform(big))
    return {"Z_comp": Zc, "Z_full": Zf}, float(p.explained_variance_ratio_.sum())


def tag_matrix(labels):
    return np.stack([
        labels["belly"] == 0, labels["belly"] == 1,
        labels["tail"] == 0, labels["tail"] == 1,
        labels["stripe"] == 1, labels["cheeks"] == 1,
    ], axis=1).astype(int)


def fit_tags(Z, Y, idx):
    """One regularised logistic per tag, fitted on the tagged specimens only."""
    models = []
    for t in range(Y.shape[1]):
        y = Y[idx, t]
        if len(np.unique(y)) < 2:
            models.append(None)
            continue
        m = LogisticRegression(max_iter=4000, C=0.5, class_weight="balanced")
        m.fit(Z[idx], y)
        models.append(m)
    return models


def tag_scores(models, Z):
    out = np.zeros((len(Z), len(models)))
    for t, m in enumerate(models):
        if m is not None:
            out[:, t] = m.decision_function(Z)
    return out


def evaluate(models, Z, Y, labels, held):
    res = {}
    pred = {}
    for t, m in enumerate(models):
        pred[t] = m.predict(Z[held]) if m is not None else np.zeros(len(held), int)
    # a factor is called correct when its own tag(s) are predicted correctly
    res["belly"] = balanced_accuracy_score(labels["belly"][held], pred[1])
    res["tail"] = balanced_accuracy_score(labels["tail"][held], pred[3])
    res["stripe"] = balanced_accuracy_score(labels["stripe"][held], pred[4])
    res["cheeks"] = balanced_accuracy_score(labels["cheeks"][held], pred[5])
    S = tag_scores(models, Z)
    joint = labels["belly"][held] * 4 + labels["tail"][held] * 2 + labels["stripe"][held]
    try:
        g = GaussianMixture(8, covariance_type="full", random_state=0, n_init=2).fit(S[held])
        res["ari8"] = adjusted_rand_score(joint, g.predict(S[held]))
    except Exception:
        res["ari8"] = float("nan")
    try:
        res["sil8"] = silhouette_score(PCA(2, random_state=0).fit_transform(S[held]), joint)
    except Exception:
        res["sil8"] = float("nan")
    return res, S


def run_curve(Z, Y, labels, acq="random", seeds=NSEED, budgets=BUDGETS, n_pool=None):
    n = len(Z)
    pool_all = np.arange(n) if n_pool is None else n_pool
    out = {k: {b: [] for b in budgets} for k in ("belly", "tail", "stripe", "cheeks", "ari8", "sil8")}
    for s in range(seeds):
        rng = np.random.default_rng(100 + s)
        tagged = list(rng.choice(pool_all, size=budgets[0], replace=False))
        for bi, b in enumerate(budgets):
            if len(tagged) < b:
                rest = np.setdiff1d(pool_all, tagged)
                need = b - len(tagged)
                if acq == "random" or bi == 0:
                    add = rng.choice(rest, size=need, replace=False)
                else:
                    mods = fit_tags(Z, Y, np.array(tagged))
                    unc = np.zeros(len(rest))
                    for m in mods:
                        if m is None:
                            continue
                        p = m.predict_proba(Z[rest])[:, 1]
                        unc += 1 - np.abs(2 * p - 1)
                    add = rest[np.argsort(-unc)[:need]]
                tagged.extend(list(add))
            idx = np.array(tagged)
            held = np.setdiff1d(pool_all, idx)
            mods = fit_tags(Z, Y, idx)
            r, _ = evaluate(mods, Z, Y, labels, held)
            for k in out:
                out[k][b].append(r[k])
    return {k: {"budgets": budgets,
                "mean": [float(np.mean(v[b])) for b in budgets],
                "sd": [float(np.std(v[b])) for b in budgets]} for k, v in out.items()}


def main():
    t0 = time.time()
    fcd, _, mesh = C.atlas_data()
    names = list(fcd.names)
    df, labels = C.load_gt(names)
    reps, ev = build_reps(fcd, mesh)
    Y = tag_matrix(labels)
    res = {"meta": {"n": len(names), "tags": TAGS, "budgets": BUDGETS, "n_seeds": NSEED,
                    "Z_full_pca60_explained_var": ev,
                    "note": "expert responses simulated from generating parameters"}}

    # ---- representation ceilings (all labels available) -----------------------------
    ceil = {}
    for rn, Z in reps.items():
        ceil[rn] = {}
        for f in ("belly", "tail", "stripe", "cheeks"):
            a, sd = C.balanced_cv(Z, labels[f])
            an, _ = C.balanced_cv(Z, np.random.default_rng(0).permutation(labels[f]))
            ceil[rn][f] = {"acc": a, "sd": sd, "null": an}
        print(f"[ceiling] {rn}: " + " ".join(f"{f}={ceil[rn][f]['acc']:.3f}"
                                            for f in ("belly", "tail", "stripe", "cheeks")))
    res["representation_ceiling"] = ceil

    # ---- unsupervised reference (no expert input at all) ----------------------------
    unsup = {}
    joint = labels["belly"] * 4 + labels["tail"] * 2 + labels["stripe"]
    for rn, Z in reps.items():
        best = {}
        for k in range(2, 9):
            g = GaussianMixture(k, covariance_type="full", random_state=0, n_init=3).fit(Z[:, :10])
            best[k] = g.bic(Z[:, :10])
        kbest = min(best, key=best.get)
        g = GaussianMixture(kbest, covariance_type="full", random_state=0, n_init=3).fit(Z[:, :10])
        lab = g.predict(Z[:, :10])
        unsup[rn] = {"gmm_bic_k": int(kbest),
                     "ari_joint8": float(adjusted_rand_score(joint, lab)),
                     "ari_belly": float(adjusted_rand_score(labels["belly"], lab)),
                     "ari_tail": float(adjusted_rand_score(labels["tail"], lab)),
                     "ari_stripe": float(adjusted_rand_score(labels["stripe"], lab)),
                     "ari_cheeks": float(adjusted_rand_score(labels["cheeks"], lab))}
        print(f"[unsup]  {rn}: k={kbest} ARI8={unsup[rn]['ari_joint8']:.3f} "
              f"belly={unsup[rn]['ari_belly']:.3f} stripe={unsup[rn]['ari_stripe']:.3f}")
    res["unsupervised_reference"] = unsup

    # ---- tag-feedback learning curves ----------------------------------------------
    curves = {}
    for rn, Z in reps.items():
        for acq in ("random", "uncertainty"):
            print(f"[curve] {rn} / {acq} ...")
            curves[f"{rn}|{acq}"] = run_curve(Z, Y, labels, acq=acq)
            m = curves[f"{rn}|{acq}"]
            print("   stripe:", " ".join(f"{x:.2f}" for x in m["stripe"]["mean"]),
                  " ari8:", " ".join(f"{x:.2f}" for x in m["ari8"]["mean"]))
    # shuffled-tag null on the strong representation
    Yn = Y.copy()
    rng = np.random.default_rng(7)
    perm = rng.permutation(len(Yn))
    curves["Z_full|null_shuffled"] = run_curve(reps["Z_full"], Yn[perm], labels,
                                               acq="random", seeds=4)
    res["tag_curves"] = curves

    # ---- pairwise-constraint metric warp -------------------------------------------
    from fishpipe import semisup
    pair_res = {}
    for factor in ("stripe", "belly"):
        y = labels[factor]
        rec = {}
        for m_pairs in (0, 10, 20, 30, 50, 80):
            aris = []
            for s in range(6):
                rng = np.random.default_rng(1000 + s)
                Z = reps["Z_full"]
                if m_pairs == 0:
                    g = GaussianMixture(2, covariance_type="full", random_state=0,
                                        n_init=3).fit(Z)
                    aris.append(adjusted_rand_score(y, g.predict(Z)))
                    continue
                idx = rng.integers(0, len(Z), size=(m_pairs, 2))
                idx = idx[idx[:, 0] != idx[:, 1]]
                ml = [(int(a), int(b)) for a, b in idx if y[a] == y[b]]
                cl = [(int(a), int(b)) for a, b in idx if y[a] != y[b]]
                try:
                    L = semisup.learn_warp(Z, ml, cl, n_dim=4, reg=1.0)
                    Zw = Z @ L
                    g = GaussianMixture(2, covariance_type="full", random_state=0,
                                        n_init=3).fit(Zw)
                    aris.append(adjusted_rand_score(y, g.predict(Zw)))
                except Exception as e:                                  # noqa: BLE001
                    aris.append(float("nan"))
            rec[m_pairs] = {"mean": float(np.nanmean(aris)), "sd": float(np.nanstd(aris))}
            print(f"[pairs] {factor} m={m_pairs}: ARI {rec[m_pairs]['mean']:.3f}")
        pair_res[factor] = rec
    res["pairwise_warp"] = pair_res

    # ---- small-n study (n = 40 specimens total) ------------------------------------
    small = {}
    rng = np.random.default_rng(3)
    for rn, Z in reps.items():
        accs = {k: [] for k in ("belly", "tail", "stripe", "cheeks", "ari8")}
        for s in range(12):
            sub = rng.choice(len(Z), size=40, replace=False)
            cur = run_curve(Z, Y, labels, acq="uncertainty", seeds=1,
                            budgets=[8, 16, 24], n_pool=sub)
            for k in accs:
                accs[k].append(cur[k]["mean"][-1])
        small[rn] = {k: {"mean": float(np.mean(v)), "sd": float(np.std(v))}
                     for k, v in accs.items()}
        print(f"[small-n 40, 24 tags] {rn}: " +
              " ".join(f"{k}={small[rn][k]['mean']:.3f}" for k in accs))
    res["small_n"] = small

    # ---- best display latent for the figure ----------------------------------------
    Z = reps["Z_full"]
    disp = {}
    for b in (0, 24, 80, 160):
        if b == 0:
            S = Z[:, :6]
            tagged = np.array([], int)
        else:
            rng2 = np.random.default_rng(104)
            tagged = rng2.choice(len(Z), size=b, replace=False)
            S = tag_scores(fit_tags(Z, Y, tagged), Z)
        try:
            import umap
            emb = umap.UMAP(n_components=2, random_state=42, min_dist=0.1,
                            n_neighbors=15).fit_transform(S)
        except Exception:
            emb = PCA(2, random_state=0).fit_transform(S)
        disp[str(b)] = {"emb": emb.tolist(), "tagged": tagged.tolist(),
                        "sil8": float(silhouette_score(emb, joint))}
        print(f"[display] {b} tags: silhouette(8 groups) = {disp[str(b)]['sil8']:.3f}")
    np.savez_compressed(C.RESULTS / "e5b_arrays.npz",
                        **{f"emb_{b}": np.array(disp[b]["emb"]) for b in disp},
                        **{f"tagged_{b}": np.array(disp[b]["tagged"]) for b in disp},
                        joint=joint)
    res["display_silhouette"] = {b: disp[b]["sil8"] for b in disp}
    res["seconds"] = round(time.time() - t0, 1)
    (C.RESULTS / "e5b_expert_repr.json").write_text(json.dumps(res, indent=2))
    print("saved results/e5b_expert_repr.json in", res["seconds"], "s")


if __name__ == "__main__":
    main()
