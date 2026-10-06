"""E2b — the same before/after comparison under leave-one-donor-out cross-validation.

Random cross-validation lets a classifier see other specimens painted on the *same* donor scan
as the held-out one. Because a donor's body shape is shared by 28–48 specimens, this is a mild
form of block structure. Leave-one-donor-out is the strict version of the question a user
actually asks — will this generalise to a specimen shape the analysis has never seen? — and it
is the right test of whether the atlas gain is a real gain or a shape-specific memorisation.
"""
from __future__ import annotations

import json

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import common as C
from fishpipe import features

FACTORS = ("belly", "tail", "stripe", "cheeks")
REPS = ["pre_hist", "post_hist_matched", "pre_spatial_canon", "post_axial", "post_spectral"]


def donor_blocked(X, y, donor, seed=0):
    """Balanced accuracy pooled over leave-one-donor-out folds (and per fold)."""
    per = []
    preds = np.zeros(len(y), int)
    for d in np.unique(donor):
        te = donor == d
        tr = ~te
        if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
            continue
        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(max_iter=4000, class_weight="balanced",
                                               random_state=seed))
        clf.fit(X[tr], y[tr])
        p = clf.predict(X[te])
        preds[te] = p
        per.append(balanced_accuracy_score(y[te], p))
    return float(np.mean(per)), float(np.std(per)), len(per)


def main():
    z = np.load(C.RESULTS / "e2_preservation_arrays.npz", allow_pickle=True)
    donor = z["donor"]
    labels = {f: z[f"label_{f}"] for f in FACTORS}
    fcd, _, mesh = C.atlas_data()
    X = {r: z[r] for r in REPS}
    X["post_spatial_flat"] = features.spatial_flatten(fcd, "lab", subsample=1200)

    out = {"n_donor_folds": int(len(np.unique(donor))),
           "donor_sizes": {int(d): int((donor == d).sum()) for d in np.unique(donor)},
           "random_cv": {}, "donor_blocked": {}}
    print(f"{'representation':<22}" + "".join(f"{f:>22}" for f in FACTORS))
    for r, M in X.items():
        row_r, row_d = {}, {}
        line = f"{r:<22}"
        for f in FACTORS:
            a, sd = C.balanced_cv(M, labels[f])
            b, bsd, nf = donor_blocked(M, labels[f], donor)
            an, _ = C.balanced_cv(M, np.random.default_rng(0).permutation(labels[f]))
            bn, _, _ = donor_blocked(M, np.random.default_rng(1).permutation(labels[f]), donor)
            row_r[f] = {"acc": a, "sd": sd, "null": an}
            row_d[f] = {"acc": b, "sd": bsd, "null": bn, "n_folds": nf}
            line += f"{a:.3f} -> {b:.3f} (n{bn:.2f})".rjust(22)
        out["random_cv"][r] = row_r
        out["donor_blocked"][r] = row_d
        print(line)

    summ = {}
    for f in FACTORS:
        summ[f] = {
            "transfer_cost_composition": (out["donor_blocked"]["post_hist_matched"][f]["acc"]
                                          - out["donor_blocked"]["pre_hist"][f]["acc"]),
            "atlas_gain_spatial": (max(out["donor_blocked"][k][f]["acc"]
                                       for k in ("post_axial", "post_spectral",
                                                 "post_spatial_flat"))
                                   - out["donor_blocked"]["pre_spatial_canon"][f]["acc"]),
            "atlas_gain_dim_matched": (out["donor_blocked"]["post_axial"][f]["acc"]
                                       - out["donor_blocked"]["pre_spatial_canon"][f]["acc"]),
        }
    out["summary_donor_blocked"] = summ
    print("\nunder leave-one-donor-out:")
    for f in FACTORS:
        print(f"  {f:8s} transfer cost {summ[f]['transfer_cost_composition']:+.3f}   "
              f"atlas gain {summ[f]['atlas_gain_spatial']:+.3f} "
              f"(dimension-matched {summ[f]['atlas_gain_dim_matched']:+.3f})")

    (C.RESULTS / "e2b_donorblocked.json").write_text(json.dumps(out, indent=2))
    print("saved results/e2b_donorblocked.json")


if __name__ == "__main__":
    main()
