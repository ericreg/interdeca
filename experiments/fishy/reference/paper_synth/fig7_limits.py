"""Figure 7 — Honest limits: what does not work, and where the ceilings are."""
from __future__ import annotations

import json

import numpy as np
import matplotlib.pyplot as plt

import common as C
import figstyle as F

F.use_style(8.0)
FAC = ["belly", "tail", "stripe", "cheeks"]
FLAB = {"belly": "belly hue", "tail": "tail hue", "stripe": "stripe count",
        "cheeks": "cheek patch"}
RSHORT = {"composition (area histogram, 24)": "composition",
          "regional color (128 regions)": "regional color",
          "regional colour (128 regions)": "regional color",
          "graph-Fourier coefficients (k=40)": "graph-Fourier"}


def title(fig, x, y, letter, text, size=8.4, ls=1.3):
    fig.text(x, y, letter, fontsize=11, fontweight="bold", va="top")
    fig.text(x + 0.030, y, text, fontsize=size, va="top", linespacing=ls)


def main():
    b = json.loads((C.RESULTS / "e6b_limits.json").read_text())
    a = json.loads((C.RESULTS / "e6_negative_results.json").read_text())

    fig = plt.figure(figsize=(7.2, 4.7))

    # ================================================================= A dims vs EV
    title(fig, 0.045, 0.968, "A",
          "Components required versus\nexplained variance")
    axA = fig.add_axes([0.100, 0.585, 0.385, 0.290])
    reps = list(b["dims_vs_accuracy"])
    xs = np.arange(len(reps))
    w = 0.19
    for j, f in enumerate(FAC):
        vals = [b["dims_vs_accuracy"][r]["saturation"][f]["sat_dim"] for r in reps]
        axA.bar(xs + (j - 1.5) * w, vals, w, color=F.FACTOR_COLOR[f], label=FLAB[f])
    ev = [b["dims_vs_accuracy"][r]["ev_dim"]["95"] for r in reps]
    for x, e in zip(xs, ev):
        axA.plot([x - 0.46, x + 0.46], [e, e], color=F.INK, lw=1.5, zorder=6)
    axA.plot([], [], color=F.INK, lw=1.5, label="95% of cumulative variance")
    axA.set_xticks(xs)
    axA.set_xticklabels([RSHORT.get(r, r) for r in reps], fontsize=7.0)
    axA.set_ylabel("principal components needed to reach\nthat factor's maximum accuracy",
                   fontsize=7.0, linespacing=1.25)
    axA.set_ylim(0, 78)
    F.hairline_grid(axA, "y")
    axA.legend(fontsize=6.0, loc="upper left", ncol=2, handlelength=1.1, columnspacing=1.0)

    # ================================================================= B sample size
    title(fig, 0.560, 0.968, "B", "Accuracy versus study size", size=8.0)
    axB = fig.add_axes([0.665, 0.585, 0.290, 0.290])
    curve = b["sample_size"]["curve"]
    ns = sorted(int(k) for k in curve)
    for f in FAC:
        mu = [curve[str(n)][f]["mean"] for n in ns]
        sd = [curve[str(n)][f]["sd"] or 0 for n in ns]
        axB.plot(ns, mu, marker="o", ms=3, color=F.FACTOR_COLOR[f], label=FLAB[f])
        axB.fill_between(ns, np.array(mu) - np.array(sd), np.array(mu) + np.array(sd),
                         color=F.FACTOR_COLOR[f], alpha=0.12, linewidth=0)
    axB.axhline(0.5, color=F.MUTED, lw=0.8, ls=(0, (3, 2)))
    axB.axvspan(20, 45, color="#f1f3f5", zorder=0)

    axB.set_xscale("log")
    axB.set_xticks(ns); axB.set_xticklabels([str(n) for n in ns])
    axB.set_xlabel("specimens in the study")
    axB.set_ylabel("balanced accuracy")
    axB.set_ylim(0.44, 1.03)
    F.hairline_grid(axB, "y")
    axB.legend(fontsize=5.7, loc="lower right", ncol=2, handlelength=1.0, columnspacing=0.8,
               handletextpad=0.3, borderaxespad=0.3)

    # ================================================================= C ICA = PCA
    title(fig, 0.045, 0.455, "C", "Principal versus independent\ncomponents",
          size=8.0)
    axC = fig.add_axes([0.100, 0.185, 0.245, 0.175])
    ica = a["item2_ica"]
    rep = "spec40" if "spec40" in ica else list(ica)[0]
    ns_i = sorted(int(k) for k in ica[rep])
    n_use = str(max(ns_i))
    w = 0.36
    xs_c = np.arange(len(FAC))
    axC.bar(xs_c - w / 2, [ica[rep][n_use]["acc_pca"][f] for f in FAC], w,
            color=F.OKABE["blue"], label="principal components")
    axC.bar(xs_c + w / 2, [ica[rep][n_use]["acc_ica"][f] for f in FAC], w,
            color=F.OKABE["vermillion"], label="independent components")
    axC.axhline(0.5, color=F.MUTED, lw=0.8, ls=(0, (3, 2)))
    axC.set_xticks(xs_c)
    axC.set_xticklabels([FLAB[f].split()[0] for f in FAC], fontsize=6.6)

    axC.set_ylabel("balanced accuracy", fontsize=7.0)
    axC.set_ylim(0.45, 1.06)
    maxd = max(ica[rep][str(n)]["max_abs_acc_diff"] for n in ns_i)
    ang = max(ica[rep][str(n)]["max_principal_angle_deg"] for n in ns_i)

    F.hairline_grid(axC, "y")
    axC.legend(fontsize=5.9, loc="lower center", ncol=1, bbox_to_anchor=(0.5, -0.62),
               handlelength=1.0, columnspacing=1.2)

    # ================================================================= D chaining hurts
    title(fig, 0.400, 0.455, "D", "Continuous versus segmented\nspectral input",
          size=8.0)
    axD = fig.add_axes([0.455, 0.185, 0.230, 0.175])
    s3 = a["item3_segment_then_spectral"]
    pairs = [("continuous_spectral_k40", "continuous coefficients", F.OKABE["blue"]),
             ("segment_onehot_spectral_k100", "segmented, then spectral", "#9aa5b1")]
    w = 0.36
    xs = np.arange(len(FAC))
    for j, (k, lab, col) in enumerate(pairs):
        vals = [s3[k][f]["acc"] for f in FAC]
        axD.bar(xs + (j - 0.5) * w, vals, w, color=col, label=lab)
    axD.axhline(0.5, color=F.MUTED, lw=0.8, ls=(0, (3, 2)))
    axD.set_xticks(xs)
    axD.set_xticklabels([FLAB[f].split()[0] for f in FAC], fontsize=6.6)
    axD.set_ylabel("balanced accuracy", fontsize=7.0)
    axD.set_ylim(0.45, 1.06)
    F.hairline_grid(axD, "y")
    axD.legend(fontsize=5.9, loc="lower center", ncol=1, bbox_to_anchor=(0.5, -0.62),
               handlelength=1.0)

    # ================================================================= E gate
    title(fig, 0.720, 0.455, "E", "Cluster agreement per factor", size=8.0)
    axE = fig.add_axes([0.800, 0.185, 0.155, 0.175])
    g = b["discovery_gate"]
    gr = list(g)
    ys = np.arange(len(FAC))[::-1]
    r0 = gr[0]
    vals = [g[r0]["ari_2d"][f] for f in FAC]
    axE.barh(ys, vals, 0.55, color=[F.FACTOR_COLOR[f] for f in FAC])

    axE.set_yticks(ys); axE.set_yticklabels([FLAB[f] for f in FAC], fontsize=6.4)
    axE.set_xlabel("ARI of the automatic clusters\nvs each planted factor", fontsize=6.4,
                   linespacing=1.25)
    axE.set_xlim(-0.02, 0.72)
    F.hairline_grid(axE, "x")


    out = C.FIGURES / "fig7_limits.png"
    fig.savefig(out, dpi=400)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    main()
