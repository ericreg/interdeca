"""Figure 7 — photometric fidelity of the texture transfer (four panels, column width)."""
from __future__ import annotations

import json

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import common as C
import figstyle as F
from fishpipe.render import render_textured_rgba

F.use_style(8.0)


def paint(mesh, scalar, cmap, vmin, vmax, px, bounds, axes, flips):
    from matplotlib.colors import Normalize
    h, v, d = axes[:3]
    rgba = plt.get_cmap(cmap)(Normalize(vmin, vmax)(np.clip(scalar, vmin, vmax)))
    cols = (rgba[:, :3] * 255).astype(np.uint8)
    im = render_textured_rgba(mesh, cols, px=px, h_axis=h, v_axis=v, depth_axis=d,
                              cull=1.0, bounds=bounds, supersample=3, crop=False,
                              align=False)[::3, ::3]
    return C.orient_image(im, *flips)


def title(fig, x, y, letter, text, size=8.0):
    fig.text(x, y, letter, fontsize=10, fontweight="bold", va="top")
    fig.text(x + 0.028, y, text, fontsize=size, va="top", linespacing=1.25)


def main():
    fcd, _, mesh = C.atlas_data()
    names = list(fcd.names)
    df, labels = C.load_gt(names)
    e1 = json.loads((C.RESULTS / "e1_fidelity.json").read_text())
    arr = np.load(C.RESULTS / "e1_fidelity_arrays.npz")
    dE, holes = arr["dE"], arr["holes"]
    _, thumbs = C.thumbnails(px=260, which="atlas")
    nat_names, nat_thumbs = C.thumbnails(px=260, which="native")

    h, v, d, fh, fv = C.orientation(mesh, C.atlas_landmarks())
    ax_, flips = (h, v, d), (fh, fv)
    lo, hi = mesh.vertices.min(0), mesh.vertices.max(0)
    pad = 0.02 * (hi - lo)
    bounds = (lo[h] - pad[h], hi[h] + pad[h], lo[v] - pad[v], hi[v] + pad[v])

    fig = plt.figure(figsize=(7.0, 5.0))

    # ---------------------------------------------------------------- A before / after
    title(fig, 0.045, 0.975, "A", "Specimens before and after transfer")
    donor_of = df["specimen_index"].to_numpy()
    picks = [int(np.where(donor_of == dd)[0][0]) for dd in (0, 2, 5)]
    for j, i in enumerate(picks):
        x0 = 0.185 + j * 0.265
        a1 = fig.add_axes([x0, 0.795, 0.245, 0.135])
        a1.imshow(nat_thumbs[nat_names.index(names[i])], interpolation="antialiased")
        a1.axis("off")
        a2 = fig.add_axes([x0, 0.640, 0.245, 0.135])
        a2.imshow(thumbs[i], interpolation="antialiased")
        a2.axis("off")
    fig.text(0.175, 0.862, "before", fontsize=7.0, ha="right", va="center")
    fig.text(0.175, 0.707, "after", fontsize=7.0, ha="right", va="center")

    # ---------------------------------------------------------------- B error map
    title(fig, 0.045, 0.585, "B", "Median transfer error per atlas face")
    axB = fig.add_axes([0.075, 0.360, 0.400, 0.185])
    axB.imshow(paint(mesh, np.median(dE, 0), "magma", 0, 6, 380, bounds, ax_, flips),
               interpolation="antialiased")
    axB.axis("off")
    cax = fig.add_axes([0.135, 0.335, 0.280, 0.014])
    cb = mpl.colorbar.ColorbarBase(cax, cmap=plt.get_cmap("magma"),
                                   norm=mpl.colors.Normalize(0, 6), orientation="horizontal")
    cb.set_label("ΔE$_{00}$", fontsize=7.0, labelpad=1)
    cb.ax.tick_params(labelsize=6.5, length=2, pad=1)

    # ---------------------------------------------------------------- C error distribution
    title(fig, 0.535, 0.600, "C", "Distribution of transfer error")
    axC = fig.add_axes([0.650, 0.375, 0.300, 0.155])
    dd = np.sort(dE.ravel()[::13])
    axC.plot(dd, 100 * np.arange(1, len(dd) + 1) / len(dd), color=F.OKABE["blue"], lw=1.6)
    for thr in (1.0, 2.3, 5.0):
        axC.axvline(thr, color=F.MUTED, lw=0.7, ls=(0, (2, 2)))
    axC.set_xscale("log")
    axC.set_xlim(0.04, 60)
    axC.set_ylim(0, 104)
    axC.set_xlabel("ΔE$_{00}$ from the source surface", fontsize=7.5)
    axC.set_ylabel("cumulative % of samples", fontsize=7.5)
    F.hairline_grid(axC, "y")

    # ---------------------------------------------------------------- D black-face census
    title(fig, 0.045, 0.280, "D", "Composition of black atlas faces")
    title(fig, 0.535, 0.280, "E", "Unfilled gaps per specimen")
    a = e1["artifacts"]
    v1, v2 = 100 * a["planted_black_frac"], 100 * a["true_bake_hole_frac"]
    axD = fig.add_axes([0.135, 0.190, 0.340, 0.048])
    axD.barh(0, v1, color="#2b2b2b", height=0.7, label="planted stripe pigment")
    axD.barh(0, v2, left=v1, color=F.OKABE["vermillion"], height=0.7, label="unfilled gap")
    axD.set_xlim(0, 0.70)
    axD.set_ylim(-0.6, 0.6)
    axD.set_yticks([])
    axD.set_xlabel("% of all atlas faces", fontsize=7.5)
    for sp in ("left", "right", "top"):
        axD.spines[sp].set_visible(False)
    axD.legend(fontsize=6.5, loc="upper center", ncol=2, bbox_to_anchor=(0.5, -1.35),
               handlelength=1.0, columnspacing=1.4)

    axD2 = fig.add_axes([0.650, 0.105, 0.300, 0.135])
    axD2.hist(100 * holes.mean(1), bins=24, color=F.OKABE["vermillion"], linewidth=0)
    axD2.set_xlabel("% of faces", fontsize=7.5)
    axD2.set_ylabel("specimens", fontsize=7.5)
    F.hairline_grid(axD2, "y")

    out = C.FIGURES / "fig2_fidelity.png"
    fig.savefig(out, dpi=400)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    main()
