"""Figure 6 — Expert-guided analysis of the ColorAtlas output."""
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
STYLE = {
    "Z_comp|random": dict(color="#9aa5b1", ls="-", label="composition vector, random selection"),
    "Z_comp|uncertainty": dict(color="#5b6b7a", ls="--",
                               label="composition vector, uncertainty selection"),
    "Z_full|random": dict(color="#74b3d8", ls="-", label="custom representation, random selection"),
    "Z_full|uncertainty": dict(color=F.OKABE["blue"], ls="--",
                               label="custom representation, uncertainty selection"),
}


def title(fig, x, y, letter, text, size=8.4, ls=1.3):
    fig.text(x, y, letter, fontsize=11, fontweight="bold", va="top")
    fig.text(x + 0.030, y, text, fontsize=size, va="top", linespacing=ls)


def main():
    d = json.loads((C.RESULTS / "e5b_expert_repr.json").read_text())
    cur = d["tag_curves"]
    arr = np.load(C.RESULTS / "e5b_arrays.npz")
    joint = arr["joint"]

    fig = plt.figure(figsize=(7.2, 7.4))

    # ================================================================= A stripe curve
    title(fig, 0.045, 0.975, "A",
          "Stripe-count recovery versus\ntagging budget")
    axA = fig.add_axes([0.095, 0.720, 0.375, 0.175])
    for k, st in STYLE.items():
        m = cur[k]["stripe"]
        axA.plot(m["budgets"], m["mean"], marker="o", ms=2.8, lw=1.5, **st)
        axA.fill_between(m["budgets"], np.array(m["mean"]) - np.array(m["sd"]),
                         np.array(m["mean"]) + np.array(m["sd"]), color=st["color"],
                         alpha=0.12, linewidth=0)
    nl = cur["Z_full|null_shuffled"]["stripe"]
    axA.plot(nl["budgets"], nl["mean"], color=F.MUTED, lw=1.0, ls=(0, (1.5, 1.5)),
             label="shuffled tags (null)")
    axA.axhline(d["representation_ceiling"]["Z_full"]["stripe"]["acc"], color=F.OKABE["blue"],
                lw=0.8, ls=(0, (4, 2)))

    axA.axhline(d["representation_ceiling"]["Z_comp"]["stripe"]["acc"], color="#5b6b7a",
                lw=0.8, ls=(0, (4, 2)))

    axA.set_xlabel("specimens tagged by the expert")
    axA.set_ylabel("stripe-count balanced accuracy\non the untagged specimens", linespacing=1.25)
    axA.set_ylim(0.46, 0.97)
    F.hairline_grid(axA, "y")
    axA.legend(fontsize=5.9, loc="upper center", ncol=2, labelspacing=0.22, handlelength=1.6,
               bbox_to_anchor=(1.30, -0.235), columnspacing=1.4)

    # ================================================================= B joint ARI
    title(fig, 0.535, 0.975, "B", "Eight-group cluster agreement",
          size=8.0)
    axB = fig.add_axes([0.620, 0.720, 0.330, 0.175])
    for k, st in STYLE.items():
        m = cur[k]["ari8"]
        axB.plot(m["budgets"], m["mean"], marker="o", ms=2.8, lw=1.5, **{**st, "label": None})
    axB.axhline(d["unsupervised_reference"]["Z_full"]["ari_joint8"], color=F.MUTED, lw=0.8,
                ls=(0, (4, 2)))

    axB.set_xlabel("specimens tagged by the expert")
    axB.set_ylabel("adjusted Rand index\nvs the 8 planted groups", linespacing=1.25)
    axB.set_ylim(0, 0.78)
    F.hairline_grid(axB, "y")

    # ================================================================= C per-factor
    title(fig, 0.045, 0.616, "C",
          "Per-factor recovery", size=8.0)
    axC = fig.add_axes([0.095, 0.398, 0.375, 0.145])
    m = cur["Z_full|uncertainty"]
    for f in FAC:
        axC.plot(m[f]["budgets"], m[f]["mean"], marker="o", ms=2.8, lw=1.5,
                 color=F.FACTOR_COLOR[f], label=FLAB[f])
        axC.fill_between(m[f]["budgets"], np.array(m[f]["mean"]) - np.array(m[f]["sd"]),
                         np.array(m[f]["mean"]) + np.array(m[f]["sd"]),
                         color=F.FACTOR_COLOR[f], alpha=0.12, linewidth=0)
    axC.axhline(0.5, color=F.MUTED, lw=0.8, ls=(0, (3, 2)))
    axC.set_xlabel("specimens tagged by the expert")
    axC.set_ylabel("balanced accuracy")
    axC.set_ylim(0.45, 1.03)
    F.hairline_grid(axC, "y")
    axC.legend(fontsize=6.0, loc="lower right", ncol=2, handlelength=1.2,
               columnspacing=0.9, handletextpad=0.4, bbox_to_anchor=(1.0, 0.03))

    # ================================================================= D pairwise
    title(fig, 0.535, 0.616, "D", "Pairwise feedback",
          size=8.0)
    axD = fig.add_axes([0.620, 0.398, 0.330, 0.145])
    pw = d["pairwise_warp"]
    for f, col in (("stripe", F.OKABE["vermillion"]), ("belly", F.OKABE["blue"])):
        ms = sorted(int(k) for k in pw[f])
        mu = [pw[f][str(k)]["mean"] for k in ms]
        sd = [pw[f][str(k)]["sd"] for k in ms]
        axD.plot(ms, mu, marker="o", ms=3, lw=1.5, color=col, label=FLAB[f])
        axD.fill_between(ms, np.array(mu) - np.array(sd), np.array(mu) + np.array(sd),
                         color=col, alpha=0.12, linewidth=0)
    axD.set_xlabel("pairs judged by the expert")
    axD.set_ylabel("ARI of clustering\nvs that factor", linespacing=1.25)
    axD.set_ylim(-0.05, 1.05)
    F.hairline_grid(axD, "y")
    axD.legend(fontsize=6.0, loc="center right", handlelength=1.2)

    # ================================================================= E emerging display
    title(fig, 0.045, 0.345, "E",
          "Display latent as tags accumulate", size=8.0)
    pal = ["#3b5bdb", "#748ffc", "#0ca678", "#66d9a0", "#e8590c", "#ffa94d", "#7048e8", "#d0bfff"]
    for j, b in enumerate(["0", "24", "80", "160"]):
        ax = fig.add_axes([0.085 + j * 0.225, 0.075, 0.185, 0.185])
        emb = arr[f"emb_{b}"]
        for g in range(8):
            s = joint == g
            ax.scatter(emb[s, 0], emb[s, 1], s=5.5, color=pal[g], linewidths=0, alpha=0.9)
        tg = arr[f"tagged_{b}"]
        if 0 < len(tg) <= 90:
            ax.scatter(emb[tg, 0], emb[tg, 1], s=15, facecolors="none", edgecolors="#3a3a3a",
                       linewidths=0.35)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color("#c8ced4"); sp.set_linewidth(0.6)
        ax.set_title(f"{b} tags", fontsize=6.3, color=F.INK, loc="left", pad=2)


    out = C.FIGURES / "fig6_expert.png"
    fig.savefig(out, dpi=400)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    main()
