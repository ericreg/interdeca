"""Figure 3 (manuscript Figure 8) — what transfer costs and what correspondence adds."""
from __future__ import annotations

import json

import numpy as np
import matplotlib.pyplot as plt

import common as C
import figstyle as F

F.use_style(8.0)

FAC = ["belly", "tail", "stripe", "cheeks"]
FLAB = {"belly": "belly hue\n(2 groups)", "tail": "tail hue\n(2 groups)",
        "stripe": "stripe count\n(4 vs 5)", "cheeks": "cheek patch\n(present)"}
SHORT = {"belly": "belly hue", "tail": "tail hue", "stripe": "stripe count",
         "cheeks": "cheek patch"}
PLAB = {"belly_hue": "belly hue", "tail_hue": "tail hue", "base_color_hue": "base hue",
        "base_color_sat": "base saturation", "base_color_val": "base value",
        "stripe_spacing": "stripe spacing", "stripe_width": "stripe width",
        "stripe_longitudinal_offset": "stripe offset"}


def title(fig, x, y, letter, text, size=8.4, ls=1.3):
    fig.text(x, y, letter, fontsize=11, fontweight="bold", va="top")
    fig.text(x + 0.030, y, text, fontsize=size, va="top", linespacing=ls)


def main():
    d = json.loads((C.RESULTS / "e2_preservation.json").read_text())
    db = json.loads((C.RESULTS / "e2b_donorblocked.json").read_text())
    lm = json.loads((C.RESULTS / "e2c_landmark_baseline.json").read_text())["detail"]
    clf, reg, summ = d["classification"], d["regression"], d["summary"]
    dbb = db["donor_blocked"]
    dbs = db["summary_donor_blocked"]

    fig = plt.figure(figsize=(7.2, 7.4))

    # ================================================================= A dimension-matched
    title(fig, 0.045, 0.978, "A",
          "Factor recovery before and after transfer")
    axA = fig.add_axes([0.085, 0.735, 0.885, 0.200])
    series = [
        ("pre_hist", "composition, no correspondence (24 bins)", "#c3cbd4"),
        ("post_hist_matched", "composition, after transfer (same bins)", "#7d8b99"),
        ("pre_spatial_canon", "spatial, no correspondence (96 dims, own body axis)", "#9ecae1"),
        ("lm_axial96", "spatial, landmark correspondence (96 dims)", "#4a90c4"),
        ("post_axial", "spatial, atlas correspondence (96 dims)", F.OKABE["blue"]),
    ]
    w = 0.155
    xs = np.arange(len(FAC))
    for j, (key, lab, col) in enumerate(series):
        src = clf if key in clf else lm
        vals = [src[key][f]["acc"] if key in clf else lm[key][f]["random_cv"] for f in FAC]
        errs = [src[key][f]["acc_sd"] if key in clf else lm[key][f]["random_cv_sd"] for f in FAC]
        axA.bar(xs + (j - 2) * w, vals, w, yerr=errs, color=col, label=lab,
                error_kw=dict(lw=0.7, capsize=1.6, ecolor="#4a4a4a"))
    axA.axhline(0.5, color=F.MUTED, lw=0.8, ls=(0, (3, 2)))

    axA.set_xticks(xs)
    axA.set_xticklabels([FLAB[f] for f in FAC])
    axA.set_ylabel("balanced accuracy", labelpad=2)
    axA.set_ylim(0.42, 1.05)
    F.hairline_grid(axA, "y")
    axA.legend(fontsize=6.0, loc="upper center", ncol=2, bbox_to_anchor=(0.5, -0.16),
               handlelength=1.1, columnspacing=1.6, handletextpad=0.4)


    # ================================================================= B donor-blocked
    title(fig, 0.045, 0.618, "B",
          "Leave-one-donor-out cross-validation",
          size=8.0)
    axB = fig.add_axes([0.085, 0.390, 0.395, 0.145])
    for j, (key, lab, col) in enumerate(series):
        vals = [dbb[key][f]["acc"] if key in dbb else lm[key][f]["donor_blocked"] for f in FAC]
        errs = [dbb[key][f]["sd"] if key in dbb else lm[key][f]["donor_blocked_sd"] for f in FAC]
        axB.bar(xs + (j - 2) * w, vals, w, yerr=errs, color=col,
                error_kw=dict(lw=0.6, capsize=1.4, ecolor="#4a4a4a"))
    axB.axhline(0.5, color=F.MUTED, lw=0.8, ls=(0, (3, 2)))
    axB.set_xticks(xs)
    axB.set_xticklabels([SHORT[f] for f in FAC], fontsize=6.6)
    axB.set_ylabel("balanced accuracy", labelpad=2)
    axB.set_ylim(0.42, 1.05)
    F.hairline_grid(axB, "y")


    # ================================================================= C donor leakage
    title(fig, 0.530, 0.618, "C",
          "Donor-identity recoverability",
          size=8.0)
    axC = fig.add_axes([0.715, 0.390, 0.240, 0.145])
    dr = summ["donor_recoverability"]
    rows = [("composition, before", dr["pre_hist"], "#c3cbd4"),
            ("composition, after\n(same bins)", dr["post_hist_matched"], "#7d8b99"),
            ("composition, after\n(refitted bins)", dr["post_hist"], "#5b6b7a"),
            ("spatial profile, before", dr["pre_spatial_canon"], "#9ecae1"),
            ("spatial profile, after", dr["post_axial"], F.OKABE["blue"]),
            ("graph-Fourier, after", dr["post_spectral"], "#004c78")]
    yp = np.arange(len(rows))[::-1]
    for y, (lab, v, col) in zip(yp, rows):
        axC.barh(y, v, height=0.6, color=col)
    axC.axvline(1 / 7, color=F.MUTED, lw=0.8, ls=(0, (3, 2)))
    axC.set_yticks(yp)
    axC.set_yticklabels([r[0] for r in rows], fontsize=5.9, linespacing=1.2)
    axC.set_xlim(0, 1.18)
    axC.set_xlabel("balanced accuracy, 7-way donor identity\n(chance = 0.14)", fontsize=6.4,
                   linespacing=1.25)
    F.hairline_grid(axC, "x")

    # ================================================================= D continuous params
    title(fig, 0.045, 0.335, "D",
          "Continuous generating parameters", size=8.0)
    axD = fig.add_axes([0.165, 0.070, 0.310, 0.185])
    keys = list(PLAB)
    pre = [reg["pre_spatial_canon"][k]["r2"] for k in keys]
    post = [reg["post_axial"][k]["r2"] for k in keys]
    ypos = np.arange(len(keys))[::-1]
    for y, a, b in zip(ypos, pre, post):
        axD.plot([a, b], [y, y], color="#cbd3da", lw=1.6, zorder=1, solid_capstyle="round")
        axD.scatter([a], [y], s=17, color="#9ecae1", zorder=3, linewidths=0)
        axD.scatter([b], [y], s=17, color=F.OKABE["blue"], zorder=3, linewidths=0)
    axD.set_yticks(ypos)
    axD.set_yticklabels([PLAB[k] for k in keys], fontsize=6.6)
    axD.set_xlabel("cross-validated $R^2$")
    axD.set_xlim(-0.02, 1.02)
    F.hairline_grid(axD, "x")
    axD.scatter([], [], s=17, color="#9ecae1", label="before transfer")
    axD.scatter([], [], s=17, color=F.OKABE["blue"], label="after transfer")
    axD.legend(fontsize=6.0, loc="lower left", bbox_to_anchor=(0.01, -0.02),
               handletextpad=0.2, labelspacing=0.2)

    # ================================================================= E coverage bias
    title(fig, 0.545, 0.335, "E", "Coverage bias per pigment", size=8.0)
    axE = fig.add_axes([0.645, 0.070, 0.140, 0.185])
    ag = d["pigment_area_agreement"]
    pigs = ["base", "belly", "tail", "stripe", "cheek"]
    cols = ["#B8A400", F.OKABE["blue"], F.OKABE["green"], "#2b2b2b", F.OKABE["purple"]]
    vals = [ag[p]["rel_bias_pct"] for p in pigs]
    yp = np.arange(len(pigs))[::-1]
    axE.barh(yp, vals, 0.55, color=cols)

    axE.axvline(0, color="#4a4a4a", lw=0.7)
    axE.set_yticks(yp); axE.set_yticklabels(pigs, fontsize=6.6)
    axE.set_xlabel("relative change in\nsurface coverage", fontsize=6.4, linespacing=1.25)
    axE.set_xlim(min(vals) - 8, max(vals) + 9)
    F.hairline_grid(axE, "x")

    # ================================================================= F net effect
    title(fig, 0.815, 0.335, "F", "Net effect,\ndonor-blocked", size=8.0)
    axF = fig.add_axes([0.855, 0.075, 0.105, 0.180])
    gains = [dbs[f]["atlas_gain_dim_matched"] for f in FAC]
    costs = [dbs[f]["transfer_cost_composition"] for f in FAC]
    yp = np.arange(len(FAC))[::-1]
    axF.barh(yp + 0.18, gains, 0.32, color=F.OKABE["blue"])
    axF.barh(yp - 0.18, costs, 0.32, color="#9aa5b1")

    axF.axvline(0, color="#4a4a4a", lw=0.7)
    axF.set_yticks(yp)
    axF.set_yticklabels([SHORT[f].split()[0] for f in FAC], fontsize=6.4)
    axF.set_xlabel("Δ balanced accuracy", fontsize=6.2)
    axF.tick_params(labelsize=5.6)
    axF.set_xlim(-0.08, 0.24)
    axF.scatter([], [], marker="s", s=14, color=F.OKABE["blue"], label="gained")
    axF.scatter([], [], marker="s", s=14, color="#9aa5b1", label="cost")
    axF.legend(fontsize=5.6, loc="upper center", bbox_to_anchor=(0.5, -0.36), ncol=1,
               handletextpad=0.2, labelspacing=0.2)
    F.hairline_grid(axF, "x")

    out = C.FIGURES / "fig3_preservation.png"
    fig.savefig(out, dpi=400)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    main()
