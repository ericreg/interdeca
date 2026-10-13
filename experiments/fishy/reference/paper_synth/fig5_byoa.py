"""Figure 5 — Custom (bring-your-own) analyses of the ColorAtlas output."""
from __future__ import annotations

import json

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import common as C
import figstyle as F

F.use_style(8.0)
FAC = ["belly", "tail", "stripe", "cheeks"]
FLAB = {"belly": "belly hue", "tail": "tail hue", "stripe": "stripe count", "cheeks": "cheek patch"}
DLAB = {
    "D1_area_hist24": "color composition (24 bins)\n— the module's own vector",
    "D2_spatial_flatten": "corresponded face colors",
    "D3_region128_mean": "regional color (128 regions)",
    "D4_region128_maxchroma": "regional peak chroma",
    "D5_spectral_k40_Lab": "graph-Fourier coefficients, k = 40",
    "D6_spectral_k150_Lab": "graph-Fourier coefficients, k = 150",
    "D7_spectral_k40_Lonly": "graph-Fourier, lightness only",
    "D8_texton_bow128": "texton histogram (shared codebook)",
    "D9_texton_vlad128": "texton VLAD encoding",
    "D10a_seg_spatial": "segmentation + spatial arrangement",
    "D10b_endler": "segmentation + color adjacency",
    "D10_seg_all": "segmentation, all statistics",
}


def title(fig, x, y, letter, text, size=8.4, ls=1.3):
    fig.text(x, y, letter, fontsize=11, fontweight="bold", va="top")
    fig.text(x + 0.030, y, text, fontsize=size, va="top", linespacing=ls)


def main():
    d = json.loads((C.RESULTS / "e4_byoa_descriptors.json").read_text())
    t = d["table"]
    keys = [k for k in DLAB if k in t]

    fig = plt.figure(figsize=(7.2, 7.6))

    # ================================================================= A heat map
    title(fig, 0.045, 0.975, "A",
          "Factor recovery by representation")
    axA = fig.add_axes([0.335, 0.640, 0.290, 0.295])
    M = np.array([[t[k]["factors"][f]["acc"] for f in FAC] for k in keys])
    im = axA.imshow(M, cmap="YlGnBu", vmin=0.5, vmax=1.0, aspect="auto")
    axA.set_xticks(range(len(FAC)))
    axA.set_xticklabels([FLAB[f] for f in FAC], rotation=32, ha="right", fontsize=6.6)
    axA.set_yticks(range(len(keys)))
    axA.set_yticklabels([DLAB[k] for k in keys], fontsize=6.0, linespacing=1.15)
    axA.tick_params(length=1.5, pad=1.5)
    for i in range(len(keys)):
        for j in range(len(FAC)):
            axA.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=5.6,
                     color="white" if M[i, j] > 0.82 else F.INK)
    axA.set_xticks(np.arange(-0.5, len(FAC)), minor=True)
    axA.set_yticks(np.arange(-0.5, len(keys)), minor=True)
    axA.grid(which="minor", color="white", linewidth=0.8)
    axA.tick_params(which="minor", length=0)
    cax = fig.add_axes([0.905, 0.640, 0.011, 0.295])
    cb = mpl.colorbar.ColorbarBase(cax, cmap=plt.get_cmap("YlGnBu"),
                                   norm=mpl.colors.Normalize(0.5, 1.0))
    cb.set_label("balanced accuracy", fontsize=6.3, labelpad=2)
    cb.ax.tick_params(labelsize=5.6, length=1.6, pad=1)
    # shuffled-label nulls, one column per factor (they are NOT all 0.5)
    axN = fig.add_axes([0.700, 0.640, 0.185, 0.295])
    N = np.array([[t[k]["factors"][f]["null_acc"] for f in FAC] for k in keys])
    axN.imshow(N, cmap="YlGnBu", vmin=0.5, vmax=1.0, aspect="auto")
    for i in range(len(keys)):
        for j in range(len(FAC)):
            axN.text(j, i, f"{N[i, j]:.2f}", ha="center", va="center", fontsize=5.4,
                     color=F.INK)
    axN.set_xticks(range(len(FAC)))
    axN.set_xticklabels([FLAB[f] for f in FAC], rotation=32, ha="right", fontsize=6.0)
    axN.set_yticks([]); axN.tick_params(length=1.2, pad=1.5)
    axN.set_xticks(np.arange(-0.5, len(FAC)), minor=True)
    axN.set_yticks(np.arange(-0.5, len(keys)), minor=True)
    axN.grid(which="minor", color="white", linewidth=0.8)
    axN.tick_params(which="minor", length=0)
    axN.text(0.5, 1.03, "shuffled labels", transform=axN.transAxes, ha="center",
             fontsize=6.4, color=F.MUTED)
    axA.text(0.5, 1.03, "real labels", transform=axA.transAxes, ha="center",
             fontsize=6.4, color=F.MUTED)

    # ================================================================= B supervised vs unsup
    title(fig, 0.045, 0.590, "B",
          "Supervised recovery versus\nunsupervised cluster agreement", size=8.0)
    axB = fig.add_axes([0.100, 0.375, 0.360, 0.155])
    best_acc, best_ari = [], []
    for f in FAC:
        accs = [t[k]["factors"][f]["acc"] for k in keys]
        aris = [t[k]["factors"][f]["gmm_ari"] for k in keys]
        best_acc.append(max(accs)); best_ari.append(max(aris))
    xs = np.arange(len(FAC))
    for x, a, r in zip(xs, best_acc, best_ari):
        axB.plot([x, x], [r, a], color="#cbd3da", lw=1.6, zorder=1, solid_capstyle="round")
    axB.scatter(xs, best_acc, s=34, color=F.OKABE["blue"], zorder=3, linewidths=0,
                label="best supervised accuracy")
    axB.scatter(xs, best_ari, s=34, color=F.OKABE["vermillion"], zorder=3, linewidths=0,
                marker="D", label="best unsupervised cluster agreement (ARI)")

    axB.axhline(0, color=F.MUTED, lw=0.7, ls=(0, (3, 2)))
    axB.set_xticks(xs); axB.set_xticklabels([FLAB[f] for f in FAC], fontsize=6.8)
    axB.set_ylim(-0.18, 1.16); axB.set_xlim(-0.45, len(FAC) - 0.25)
    axB.set_ylabel("balanced accuracy  /  ARI", fontsize=7.0)
    F.hairline_grid(axB, "y")
    axB.legend(fontsize=5.8, loc="center right", ncol=1, handletextpad=0.2,
               labelspacing=0.25, bbox_to_anchor=(1.0, 0.42))

    # ================================================================= C disentanglement
    title(fig, 0.540, 0.590, "C",
          "Pattern axis versus stripe\nparameters", size=8.0)
    axC = fig.add_axes([0.640, 0.375, 0.310, 0.155])
    dis = d["disentanglement"]
    sel = ["D1_area_hist24", "D5_spectral_k40_Lab", "D3_region128_mean", "D2_spatial_flatten"]
    sel = [s for s in sel if s in dis]
    short = {"D1_area_hist24": "composition", "D5_spectral_k40_Lab": "graph-Fourier",
             "D3_region128_mean": "regional color", "D2_spatial_flatten": "face colors"}
    w = 0.26
    ys = np.arange(len(sel))[::-1]
    for j, (prm, col, lab) in enumerate([("stripe_count", F.OKABE["vermillion"], "stripe count"),
                                         ("stripe_spacing", "#9aa5b1", "stripe spacing"),
                                         ("stripe_width", "#dee2e6", "stripe width")]):
        vals = [dis[s]["oof"][prm] for s in sel]
        axC.barh(ys + (1 - j) * w, vals, w, color=col, label=lab)
    axC.set_yticks(ys); axC.set_yticklabels([short[s] for s in sel], fontsize=6.6)
    axC.set_xlabel("|correlation| of the out-of-fold pattern axis\nwith each generating parameter",
                   fontsize=6.4, linespacing=1.25)
    axC.set_xlim(0, 0.95)
    F.hairline_grid(axC, "x")
    axC.legend(fontsize=5.9, loc="lower right", handlelength=1.0, handletextpad=0.4,
               labelspacing=0.25)

    # ================================================================= D k sweep
    title(fig, 0.045, 0.315, "D",
          "Accuracy versus retained modes", size=8.0)
    axD = fig.add_axes([0.100, 0.075, 0.360, 0.175])
    ks = d["k_sweep"]
    for f in FAC:
        axD.plot(ks[f]["k"], ks[f]["acc"], marker="o", ms=3, color=F.FACTOR_COLOR[f],
                 label=FLAB[f])
        axD.fill_between(ks[f]["k"],
                         np.array(ks[f]["acc"]) - np.array(ks[f]["sd"]),
                         np.array(ks[f]["acc"]) + np.array(ks[f]["sd"]),
                         color=F.FACTOR_COLOR[f], alpha=0.12, linewidth=0)
    sat = d["k_saturation"]
    for f in FAC:
        axD.axvline(sat[f]["k_within_1pct"], color=F.FACTOR_COLOR[f], lw=0.7,
                    ls=(0, (2, 2)), alpha=0.7)
    axD.axhline(0.5, color=F.MUTED, lw=0.8, ls=(0, (3, 2)))
    axD.set_xscale("log")
    axD.set_xlabel("number of graph-Fourier modes retained, $k$")
    axD.set_ylabel("balanced accuracy")
    axD.set_ylim(0.45, 1.04)
    F.hairline_grid(axD, "y")
    axD.legend(fontsize=6.0, loc="lower right", ncol=2, handlelength=1.1,
               handletextpad=0.3, columnspacing=0.8, bbox_to_anchor=(1.0, 0.02))


    # ================================================================= E composition vs custom
    title(fig, 0.540, 0.315, "E",
          "Custom descriptor versus\ncomposition vector", size=8.0)
    axE = fig.add_axes([0.640, 0.075, 0.310, 0.175])
    base = [t["D1_area_hist24"]["factors"][f]["acc"] for f in FAC]
    best = best_acc
    ys = np.arange(len(FAC))[::-1]
    for y, b, m in zip(ys, base, best):
        axE.plot([b, m], [y, y], color="#cbd3da", lw=2.0, solid_capstyle="round", zorder=1)
        axE.scatter([b], [y], s=26, color="#9aa5b1", zorder=3, linewidths=0)
        axE.scatter([m], [y], s=26, color=F.OKABE["blue"], zorder=3, linewidths=0)

    axE.set_yticks(ys); axE.set_yticklabels([FLAB[f] for f in FAC], fontsize=6.8)
    axE.set_xlim(0.45, 1.05)
    axE.set_xlabel("balanced accuracy")
    axE.axvline(0.5, color=F.MUTED, lw=0.8, ls=(0, (3, 2)))
    axE.scatter([], [], s=26, color="#9aa5b1", label="module's composition vector")
    axE.scatter([], [], s=26, color=F.OKABE["blue"], label="best custom descriptor")
    axE.legend(fontsize=5.9, loc="upper center", handletextpad=0.2, labelspacing=0.25,
               bbox_to_anchor=(0.5, -0.24), ncol=2)

    F.hairline_grid(axE, "x")

    out = C.FIGURES / "fig5_byoa.png"
    fig.savefig(out, dpi=400)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    main()
