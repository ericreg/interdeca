"""Figure 4 (manuscript Figure 9) — what the module's own exploratory panel delivers."""
from __future__ import annotations

import json

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import common as C
import figstyle as F

F.use_style(8.0)
FAC = ["belly", "tail", "stripe", "cheeks"]
FLAB = {"belly": "belly hue", "tail": "tail hue", "stripe": "stripe count",
        "cheeks": "cheek patch"}
PLAB = {"belly_hue": "belly hue", "tail_hue": "tail hue", "base_color_hue": "base hue",
        "base_color_sat": "base saturation", "base_color_val": "base value",
        "stripe_spacing": "stripe spacing", "stripe_width": "stripe width",
        "stripe_longitudinal_offset": "stripe offset", "belly_strength": "belly strength",
        "belly_translation": "belly offset", "tail_strength": "tail strength",
        "stripe_count": "stripe count", "rosy_cheeks_present": "cheek present"}


def title(fig, x, y, letter, text, size=8.4, ls=1.3):
    fig.text(x, y, letter, fontsize=11, fontweight="bold", va="top")
    fig.text(x + 0.030, y, text, fontsize=size, va="top", linespacing=ls)


def main():
    d = json.loads((C.RESULTS / "e3_module_eda.json").read_text())
    z = np.load(C.RESULTS / "e3_module_eda_arrays.npz")
    Z = z["pca_K24"]
    lab2 = z["gmm2d_labels"]
    fcd, _, mesh = C.atlas_data()
    names = list(fcd.names)
    df, labels = C.load_gt(names)
    _, thumbs = C.thumbnails(px=260, which="atlas")
    rec = d["table"]["area_hist_K24"]

    fig = plt.figure(figsize=(7.2, 7.2))

    # ================================================================= A the morphospace
    title(fig, 0.045, 0.975, "A",
          "Color morphospace with\nautomatic clustering")
    axA = fig.add_axes([0.085, 0.640, 0.400, 0.275])
    pal = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00", "#7048e8",
           "#868e96"]
    for g in np.unique(lab2):
        s = lab2 == g
        axA.scatter(Z[s, 0], Z[s, 1], s=13, color=pal[g % len(pal)], linewidths=0, alpha=0.85,
                    label=f"cluster {g + 1} (n={int(s.sum())})")
    idx = F.farthest_point_sample(Z[:, :2], 5, seed=3)
    for i in idx:
        F.place_image(axA, thumbs[i], (Z[i, 0], Z[i, 1]), zoom=0.075, frame="#adb5bd",
                      frame_lw=0.5)
    axA.set_xlabel(f"PC1 ({100 * rec['evr'][0]:.1f}% of variance)")
    axA.set_ylabel(f"PC2 ({100 * rec['evr'][1]:.1f}%)")
    F.hairline_grid(axA, "both")
    axA.legend(fontsize=5.3, loc="upper left", ncol=2, handletextpad=0.2, labelspacing=0.18,
               columnspacing=0.6, bbox_to_anchor=(-0.02, 1.005), borderaxespad=0.1)


    # ================================================================= B validation view
    title(fig, 0.565, 0.975, "B", "Same coordinates, colored by\nplanted group",
          size=8.0)
    axB = fig.add_axes([0.635, 0.640, 0.320, 0.275])
    grp = labels["belly"] * 2 + labels["tail"]
    gc = ["#1c4fd6", "#9b6bff", "#0b8f5e", "#8ac926"]
    gn = ["blue belly · green tail", "blue belly · cyan tail",
          "purple belly · green tail", "purple belly · cyan tail"]
    for g in range(4):
        s = grp == g
        axB.scatter(Z[s, 0], Z[s, 1], s=13, color=gc[g], linewidths=0, alpha=0.85, label=gn[g])
    axB.set_xlabel("PC1"); axB.set_ylabel("PC2")
    F.hairline_grid(axB, "both")
    axB.legend(fontsize=5.6, loc="upper left", handletextpad=0.2, labelspacing=0.2)


    # ================================================================= C PC x parameter
    title(fig, 0.045, 0.560, "C", "Component–parameter correlations", size=8.0)
    axC = fig.add_axes([0.175, 0.295, 0.215, 0.180])
    M = np.abs(z["pc_param_corr"])
    params = d["params"]
    im = axC.imshow(M.T, cmap="Greys", vmin=0, vmax=1, aspect="auto")
    axC.set_xticks(range(M.shape[0]))
    axC.set_xticklabels([f"PC{i + 1}" for i in range(M.shape[0])], fontsize=6.0)
    axC.set_yticks(range(len(params)))
    axC.set_yticklabels([PLAB.get(p, p) for p in params], fontsize=5.8)
    axC.tick_params(length=1.4, pad=1.5)
    for i in range(M.shape[0]):
        for j in range(len(params)):
            if M[i, j] > 0.35:
                axC.text(i, j, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=4.8,
                         color="white" if M[i, j] > 0.7 else F.INK)
    cax = fig.add_axes([0.398, 0.295, 0.009, 0.180])
    cb = mpl.colorbar.ColorbarBase(cax, cmap=plt.get_cmap("Greys"),
                                   norm=mpl.colors.Normalize(0, 1))
    cb.set_label("|correlation|", fontsize=6.0, labelpad=2)
    cb.ax.tick_params(labelsize=5.4, length=1.4)

    # ================================================================= D reducers
    title(fig, 0.470, 0.560, "D", "Cluster agreement by reducer",
          size=8.0)
    axD = fig.add_axes([0.575, 0.295, 0.170, 0.180])
    red = [("PCA", rec["gmm2d"]["ari_belly_x_tail"]),
           ("ICA", d["reducers"]["ICA10_top2"]["ari_belly_x_tail"]),
           ("UMAP", d["reducers"]["UMAP2"]["ari_belly_x_tail"]),
           ("PCA,\n4 components", rec["gmm4d"]["ari_belly_x_tail"])]
    ys = np.arange(len(red))[::-1]
    axD.barh(ys, [r[1] for r in red], 0.55, color=["#0072B2", "#56B4E9", "#9aa5b1", "#3b7fa8"])

    axD.set_yticks(ys); axD.set_yticklabels([r[0] for r in red], fontsize=6.2)
    axD.set_xlim(0, 0.62)
    axD.set_xlabel("ARI vs the four planted\ncolor groups", fontsize=6.4, linespacing=1.25)
    F.hairline_grid(axD, "x")

    # ================================================================= E gate
    title(fig, 0.775, 0.560, "E", "Dip statistic per component", size=8.0)
    axE = fig.add_axes([0.830, 0.295, 0.125, 0.180])
    dips = d["gate"]["dip_per_pc"]
    xs = np.arange(len(dips))
    axE.bar(xs, dips, 0.6, color=[F.OKABE["blue"] if v > d["gate"]["dip_null_p95"] else "#c8ced4"
                                  for v in dips])
    axE.axhline(d["gate"]["dip_null_p95"], color=F.INK, lw=1.0, ls=(0, (3, 2)))

    axE.set_xticks(xs); axE.set_xticklabels([f"{i + 1}" for i in xs], fontsize=6.0)
    axE.set_xlabel("principal component", fontsize=6.4)
    axE.set_ylabel("Hartigan dip statistic", fontsize=6.4)
    F.hairline_grid(axE, "y")

    # ================================================================= F small n
    title(fig, 0.045, 0.230, "F", "Cluster agreement at small\nsample size", size=8.0)
    title(fig, 0.510, 0.230, "G", "Gate outcome at small\nsample size", size=8.0)
    axF = fig.add_axes([0.115, 0.070, 0.330, 0.115])
    sn = d["small_n"]
    ns = sorted(int(k) for k in sn)
    mu = [sn[str(n)]["ari_mean"] for n in ns]
    sd = [sn[str(n)]["ari_sd"] for n in ns]
    axF.errorbar(ns, mu, yerr=sd, marker="o", ms=4, lw=1.5, color=F.OKABE["blue"],
                 capsize=2, elinewidth=0.8)
    axF.axhline(rec["gmm2d"]["ari_belly_x_tail"], color=F.MUTED, lw=0.8, ls=(0, (3, 2)))

    axF.set_xscale("log"); axF.set_xticks(ns); axF.set_xticklabels([str(n) for n in ns])
    axF.xaxis.set_minor_locator(plt.NullLocator())
    axF.set_xlabel("specimens in the study")
    axF.set_ylabel("ARI vs the four\nplanted color groups", fontsize=6.6, linespacing=1.25)
    axF.set_ylim(0, 0.55)
    F.hairline_grid(axF, "y")

    axG = fig.add_axes([0.575, 0.070, 0.330, 0.115])
    fire = [100 * sn[str(n)]["gate_fire_frac"] for n in ns]
    kmode = [sn[str(n)]["k_mode"] for n in ns]
    axG.bar(np.arange(len(ns)), fire, 0.5, color="#9aa5b1")

    axG.set_xticks(np.arange(len(ns))); axG.set_xticklabels([str(n) for n in ns])
    axG.set_xlabel("specimens in the study")
    axG.set_ylabel("draws in which the gate\nreports real structure", fontsize=6.6,
                   linespacing=1.25)
    axG.set_ylim(0, 108)
    F.hairline_grid(axG, "y")

    out = C.FIGURES / "fig4_module_eda.png"
    fig.savefig(out, dpi=400)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    main()
