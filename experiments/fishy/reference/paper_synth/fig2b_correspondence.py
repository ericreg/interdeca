"""Figure 8 — anatomical correspondence of the transferred pigments (four panels)."""
from __future__ import annotations

import json

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import common as C
import figstyle as F
from fig2_fidelity import paint, title
from fishpipe.features import rgb_to_lab

F.use_style(8.0)

KEYS = ["belly", "tail", "stripe", "cheek"]
KL = {"belly": "belly patch", "tail": "tail patch", "stripe": "stripes", "cheek": "cheek patch"}


def main():
    fcd, _, mesh = C.atlas_data()
    names = list(fcd.names)
    e1b = json.loads((C.RESULTS / "e1b_correspondence.json").read_text())
    occ = np.load(C.RESULTS / "e1b_arrays.npz")

    h, v, d, fh, fv = C.orientation(mesh, C.atlas_landmarks())
    ax_, flips = (h, v, d), (fh, fv)
    lo, hi = mesh.vertices.min(0), mesh.vertices.max(0)
    pad = 0.02 * (hi - lo)
    bounds = (lo[h] - pad[h], hi[h] + pad[h], lo[v] - pad[v], hi[v] + pad[v])

    fig = plt.figure(figsize=(7.0, 5.2))

    # ---------------------------------------------------------------- A Dice bars
    title(fig, 0.045, 0.975, "A", "Mask overlap between specimens")
    axA = fig.add_axes([0.090, 0.660, 0.520, 0.235])
    xs = np.arange(len(KEYS))
    w = 0.20
    series = [("nativeUV_same_donor", "no correspondence, same donor shape", "#c8d0d8"),
              ("atlas_same_donor", "after ColorAtlas, same donor shape", "#74b3d8"),
              ("nativeUV_diff_donor", "no correspondence, different donor shapes", "#8d99a6"),
              ("atlas_diff_donor", "after ColorAtlas, different donor shapes", F.OKABE["blue"])]
    for j, (key, lab, col) in enumerate(series):
        axA.bar(xs + (j - 1.5) * w, [e1b["correspondence"][k][key] for k in KEYS], w,
                color=col, label=lab)
    for x, k in zip(xs, KEYS):
        c = e1b["correspondence"][k]["chance_dice"]
        axA.plot([x - 2.1 * w, x + 2.1 * w], [c, c], color=F.INK, lw=0.9, ls=(0, (2, 1.6)))
    axA.plot([], [], color=F.INK, lw=0.9, ls=(0, (2, 1.6)), label="chance overlap")
    axA.set_xticks(xs)
    axA.set_xticklabels([KL[k] for k in KEYS])
    axA.set_ylabel("Dice overlap", labelpad=2)
    axA.set_ylim(0, 1.05)
    F.hairline_grid(axA, "y")
    axA.legend(fontsize=6.3, loc="upper center", ncol=2, handlelength=1.0,
               columnspacing=1.2, handletextpad=0.4, bbox_to_anchor=(0.5, -0.16))

    # ---------------------------------------------------------------- B occupancy maps
    title(fig, 0.655, 0.975, "B", "Per-face pigment occupancy")
    for j, k in enumerate(KEYS):
        axf = fig.add_axes([0.745, 0.820 - j * 0.058, 0.215, 0.055])
        axf.imshow(paint(mesh, occ[f"occ_{k}"], "viridis", 0, 1, 240, bounds, ax_, flips),
                   interpolation="antialiased")
        axf.axis("off")
        axf.text(-0.03, 0.5, KL[k], transform=axf.transAxes, fontsize=6.4, color=F.INK,
                 ha="right", va="center")
    caxF = fig.add_axes([0.770, 0.578, 0.170, 0.012])
    cbF = mpl.colorbar.ColorbarBase(caxF, cmap=plt.get_cmap("viridis"),
                                    norm=mpl.colors.Normalize(0, 1), orientation="horizontal")
    cbF.set_label("fraction of specimens", fontsize=6.6, labelpad=1)
    cbF.ax.tick_params(labelsize=6.0, length=1.8, pad=1)

    # ---------------------------------------------------------------- C coverage agreement
    title(fig, 0.045, 0.470, "C", "Pigment coverage before and after transfer")
    axC = fig.add_axes([0.110, 0.105, 0.320, 0.290])
    nat = C.native_face_colors()
    lab = rgb_to_lab(fcd.colors)
    post = np.array([C.pigment_area_fractions(lab[i], fcd.areas) for i in range(len(names))])
    pre = nat["masks"]
    order = [1, 2, 4, 3]
    mk = {1: "o", 2: "s", 4: "^", 3: "D"}
    cl = {1: F.OKABE["blue"], 2: F.OKABE["green"], 4: "#2b2b2b", 3: F.OKABE["purple"]}
    nm = {1: "belly", 2: "tail", 4: "stripes", 3: "cheek"}
    for p in order:
        axC.scatter(100 * pre[:, p], 100 * post[:, p], s=7, marker=mk[p], color=cl[p],
                    alpha=0.6, linewidths=0, label=nm[p])
    axC.plot([1e-3, 60], [1e-3, 60], color=F.MUTED, lw=0.8, ls=(0, (3, 2)), zorder=0)
    axC.set_xscale("symlog", linthresh=0.05)
    axC.set_yscale("symlog", linthresh=0.05)
    axC.set_xlim(-0.004, 60)
    axC.set_ylim(-0.004, 60)
    for a_ in (axC.xaxis, axC.yaxis):
        a_.set_ticks([0, 0.1, 1, 10])
        a_.set_ticklabels(["0", "0.1", "1", "10"])
    axC.set_xlabel("% of surface before transfer")
    axC.set_ylabel("% of surface after transfer")
    axC.legend(fontsize=6.4, loc="upper left", handletextpad=0.2, labelspacing=0.25)
    F.hairline_grid(axC, "both")

    # ---------------------------------------------------------------- D donor matrix
    title(fig, 0.530, 0.470, "D", "Tail-patch overlap between donor scans")
    axD = fig.add_axes([0.640, 0.135, 0.230, 0.260])
    M = np.array([[np.nan if x is None else x for x in row]
                  for row in e1b["correspondence"]["tail"]["donor_matrix"]])
    dn = [n.replace("_", " ") for n in e1b["donor_names"]]
    im = axD.imshow(M, cmap="Blues", vmin=0.6, vmax=1.0)
    axD.set_xticks(range(len(dn)))
    axD.set_yticks(range(len(dn)))
    axD.set_xticklabels(dn, rotation=50, ha="right", fontsize=6.0)
    axD.set_yticklabels(dn, fontsize=6.0)
    axD.tick_params(length=1.6, pad=1)
    cbH = fig.colorbar(im, ax=axD, fraction=0.045, pad=0.04)
    cbH.ax.tick_params(labelsize=6.0, length=1.8)
    cbH.set_label("Dice overlap", fontsize=6.6)

    out = C.FIGURES / "fig2b_correspondence.png"
    fig.savefig(out, dpi=400)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    main()
