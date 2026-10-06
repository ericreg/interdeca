"""Figure 1 — Anatomy of the synthetic texture dataset."""
from __future__ import annotations

import json
import os
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import matplotlib.pyplot as plt

import common as C
import figstyle as F
from fishpipe.features import rgb_to_lab
from fishpipe.mesh import load_obj, sample_face_colors
from fishpipe.render import render_textured_rgba

F.use_style(8.0)

# original donor scans: <root>/Texture/<stem>.png and <root>/Downsampled_decimate/Models/<stem>.obj
SCAN_TEX = Path(os.environ.get(
    "SYNTH_DONOR_SCANS",
    C.MOD_RUN.parent.parent.parent / "3D_Fish (Copy)" / "3D_Fish",
)) / "Texture"
PIG_COLOR = {"base": "#9c8800", "stripe": "#2b2b2b", "belly": F.OKABE["blue"],
             "tail": "#0b7a53", "cheek": "#b5397f"}
PIG_LABEL = {
    "base": "base body\nyellow; hue, saturation and\nvalue independently jittered",
    "stripe": "transverse stripes\n4 or 5 per side; spacing, width\nand offset independently jittered",
    "belly": "belly patch\ntwo planted hues, blue vs purple\n(24% of the surface)",
    "tail": "tail patch\ntwo planted hues, green vs cyan\n(8.6% of the surface)",
    "cheek": "cheek patch\npresent in 46% of specimens\n(0.37% of the surface)",
}


def scan_thumb(stem, px=250):
    mesh = load_obj(C.SRC_MESHES.parent.parent / "cichlid-synth-decimated" / "meshes"
                    if False else C.MOD_RUN)      # placeholder, replaced below
    raise RuntimeError


def donor_scan_thumbs(donor_stems, donor_names, px=250):
    """Render each donor's decimated mesh with its ORIGINAL scan texture, in the shared
    lateral orientation (head right, back up) derived from that specimen's landmarks."""
    base = SCAN_TEX.parent
    out = []
    for stem, nm in zip(donor_stems, donor_names):
        m = load_obj(base / "Downsampled_decimate" / "Models" / f"{stem}.obj")
        tex = imageio.imread(SCAN_TEX / f"{stem}.png")
        cols = sample_face_colors(m, tex)
        h, v, d, fh, fv = C.orientation(m, C.specimen_landmarks(nm))
        im = render_textured_rgba(m, cols, px=px, h_axis=h, v_axis=v, depth_axis=d,
                                  cull=1.0, supersample=3, crop=True, align=False)
        out.append(C.orient_image(im[::3, ::3], fh, fv))
    return out


def main():
    fcd, _, mesh = C.atlas_data()
    names = list(fcd.names)
    df, labels = C.load_gt(names)
    _, thumbs = C.thumbnails(px=260, which="atlas")
    nat_names, nat_thumbs = C.thumbnails(px=260, which="native")
    h, v, d, fh, fv = C.orientation(mesh, C.atlas_landmarks())
    lo, hi = mesh.vertices.min(0), mesh.vertices.max(0)
    pad = 0.02 * (hi - lo)
    bounds = (lo[h] - pad[h], hi[h] + pad[h], lo[v] - pad[v], hi[v] + pad[v])
    lab = rgb_to_lab(fcd.colors)
    cents = mesh.face_centroids()

    cheek_area = np.array([(fcd.areas * C.pigment_masks(lab[i])["cheek"]).sum()
                           for i in range(len(names))]) / fcd.areas.sum()
    cand = np.where((labels["cheeks"] == 1) & (labels["stripe"] == 1))[0]
    hero = int(cand[np.argmax(cheek_area[cand])])

    fig = plt.figure(figsize=(7.2, 9.1))

    # =============================================================== A  hero specimen
    axA = fig.add_axes([0.10, 0.700, 0.82, 0.255])
    big = render_textured_rgba(mesh, fcd.colors[hero], px=900, h_axis=h, v_axis=v,
                               depth_axis=d, cull=1.0, bounds=bounds, supersample=2,
                               crop=False, align=False)
    big = C.orient_image(big, fh, fv)
    axA.imshow(big, extent=bounds, origin="upper", interpolation="antialiased")
    sh, sv = bounds[1] - bounds[0], bounds[3] - bounds[2]
    axA.set_xlim(bounds[0] - 0.33 * sh, bounds[1] + 0.36 * sh)
    axA.set_ylim(bounds[2] - 0.62 * sv, bounds[3] + 0.62 * sv)
    axA.axis("off")
    fig.text(0.055, 0.972, "A", fontsize=11, fontweight="bold", va="top")
    fig.text(0.088, 0.972, "The five painted pigments", fontsize=8.6, va="top")

    m = C.pigment_masks(lab[hero])
    p3 = mesh.vertices[mesh.face_v]
    front = np.cross(p3[:, 1] - p3[:, 0], p3[:, 2] - p3[:, 0])[:, d] > 0
    anchors = {}
    for p in PIG_LABEL:
        sel = m[p] & front
        if sel.sum() < 5:
            sel = m[p]
        w = fcd.areas[sel]
        mu = (np.average(cents[sel, h], weights=w), np.average(cents[sel, v], weights=w))
        idx = np.where(sel)[0]
        j = idx[np.argmin((cents[idx, h] - mu[0]) ** 2 + (cents[idx, v] - mu[1]) ** 2)]
        anchors[p] = (cents[j, h], cents[j, v])
    # keep the base anchor on the dorsal flank, away from the stripes
    dorsal = m["base"] & front & (cents[:, v] > np.quantile(cents[:, v], 0.80))
    di = np.where(dorsal)[0]
    j = di[np.argmin(np.abs(cents[di, h] - np.quantile(cents[di, h], 0.35)))]
    anchors["base"] = (cents[j, h], cents[j, v])

    anchors = {k: C.map_point(x, y, bounds, fh, fv) for k, (x, y) in anchors.items()}
    head_left = anchors["tail"][0] > anchors["cheek"][0]
    L = bounds[0] - 0.32 * sh
    R = bounds[1] + 0.04 * sh
    txt = {
        "base":   (bounds[0] + 0.22 * sh, bounds[3] + 0.20 * sv, "left", "bottom"),
        "stripe": (bounds[0] + 0.60 * sh, bounds[3] + 0.20 * sv, "left", "bottom"),
        "belly":  (bounds[0] + 0.18 * sh, bounds[2] - 0.34 * sv, "left", "top"),
        "cheek":  ((L if head_left else R), bounds[2] + 0.86 * sv, "left", "center"),
        "tail":   ((R if head_left else L), bounds[2] + 0.86 * sv, "left", "center"),
    }
    for p, (tx, ty, ha, va) in txt.items():
        ax_, ay_ = anchors[p]
        axA.annotate(PIG_LABEL[p], xy=(ax_, ay_), xytext=(tx, ty), ha=ha, va=va,
                     fontsize=6.3, color=F.INK, linespacing=1.3,
                     arrowprops=dict(arrowstyle="-", color=PIG_COLOR[p], lw=0.9,
                                     shrinkA=2, shrinkB=3,
                                     connectionstyle="arc3,rad=0.12"))
        axA.plot([ax_], [ay_], marker="o", ms=3.6, mfc=PIG_COLOR[p], mec="white", mew=0.8,
                 zorder=6)
    # zoom on the cheek
    cx, cy = anchors["cheek"]
    zw = 0.11 * sh
    ins = axA.inset_axes([0.80 if not head_left else 0.02, 0.03, 0.17, 0.30])
    ins.imshow(big, extent=bounds, origin="upper", interpolation="antialiased")
    ins.set_xlim(cx - zw, cx + zw)
    ins.set_ylim(cy - zw * 0.8, cy + zw * 0.8)
    ins.set_xticks([]); ins.set_yticks([])
    for s in ins.spines.values():
        s.set_color(PIG_COLOR["cheek"]); s.set_linewidth(0.9)
    ins.set_title("cheek, 4×", fontsize=5.6, color=PIG_COLOR["cheek"], pad=1.2)

    # =============================================================== B  donors
    donor_of = df["specimen_index"].to_numpy()
    donors = sorted(set(donor_of))
    stems = [names[int(np.where(donor_of == dd)[0][0])].rsplit("_fishy", 1)[0] for dd in donors]
    scans = donor_scan_thumbs(stems, [names[int(np.where(donor_of == dd)[0][0])] for dd in donors], px=250)
    fig.text(0.055, 0.680, "B", fontsize=11, fontweight="bold", va="top")
    fig.text(0.088, 0.680, "Donor scans, painted specimens and the shared atlas",
             fontsize=8.6, va="top")
    ROW_Y = (0.606, 0.556, 0.506)
    ROW_H = 0.048
    for j, dd in enumerate(donors):
        i0 = int(np.where(donor_of == dd)[0][0])
        x0 = 0.085 + j * 0.121
        a1 = fig.add_axes([x0, ROW_Y[0], 0.113, ROW_H])
        a1.imshow(scans[j], interpolation="antialiased"); a1.axis("off")
        a1.set_title(stems[j].replace("_", " "), fontsize=5.7, color=F.MUTED, pad=1.2)
        a2 = fig.add_axes([x0, ROW_Y[1], 0.113, ROW_H])
        a2.imshow(nat_thumbs[nat_names.index(names[i0])], interpolation="antialiased")
        a2.axis("off")
        a3 = fig.add_axes([x0, ROW_Y[2], 0.113, ROW_H])
        a3.imshow(thumbs[i0], interpolation="antialiased")
        a3.axis("off")
    for y, lab in zip(ROW_Y, ("scan", "painted", "on the atlas")):
        fig.text(0.080, y + ROW_H / 2, lab, fontsize=6.2, ha="right", va="center", color=F.INK)

    # =============================================================== C  planted hue factors
    axC = fig.add_axes([0.093, 0.292, 0.325, 0.155])
    axCx = fig.add_axes([0.093, 0.452, 0.325, 0.030], sharex=axC)
    axCy = fig.add_axes([0.421, 0.292, 0.032, 0.155], sharey=axC)
    grp = labels["belly"] * 2 + labels["tail"]
    gcols = ["#3b5bdb", "#7048e8", "#0ca678", "#2f9e44"]
    gnames = ["blue·green", "blue·cyan", "purple·green", "purple·cyan"]
    for g in range(4):
        s = grp == g
        axC.scatter(df["belly_hue"][s], df["tail_hue"][s], s=8, c=gcols[g], alpha=0.85,
                    linewidths=0, label=f"{gnames[g]} (n={int(s.sum())})")
    for lab_i, col, ax_, key in ((0, "#3b5bdb", axCx, "belly"), (1, "#7048e8", axCx, "belly")):
        s = labels["belly"] == lab_i
        ax_.hist(df["belly_hue"][s], bins=26, color=col, alpha=0.75, linewidth=0)
    for lab_i, col in ((0, "#0ca678"), (1, "#2f9e44")):
        s = labels["tail"] == lab_i
        axCy.hist(df["tail_hue"][s], bins=26, color=col, alpha=0.75, linewidth=0,
                  orientation="horizontal")
    for a in (axCx, axCy):
        a.axis("off")


    axC.set_xlabel("planted belly hue")
    axC.set_ylabel("planted tail hue")
    axC.legend(fontsize=5.4, loc="lower left", ncol=2, handletextpad=0.25,
               columnspacing=0.9, borderaxespad=0.25, labelspacing=0.25)
    F.hairline_grid(axC, "both")
    fig.text(0.055, 0.500, "C", fontsize=11, fontweight="bold", va="top")
    fig.text(0.088, 0.500, "Planted hue factors", fontsize=8.2, va="top")

    # =============================================================== D  binary factors
    axD = fig.add_axes([0.585, 0.380, 0.360, 0.062])
    rows = [("stripe count", ["4 stripes", "5 stripes"],
             [int((labels["stripe"] == 0).sum()), int((labels["stripe"] == 1).sum())],
             ["#f59f00", F.OKABE["vermillion"]]),
            ("cheek patch", ["absent", "present"],
             [int((labels["cheeks"] == 0).sum()), int((labels["cheeks"] == 1).sum())],
             ["#ced4da", F.OKABE["purple"]])]
    y = 0
    for title, cats, counts, cols in rows:
        left = 0
        for cname, cnt, cc in zip(cats, counts, cols):
            axD.barh(y, cnt, left=left, color=cc, height=0.46, edgecolor="white", linewidth=0.8)
            inside = cnt >= 90
            axD.text(left + cnt / 2 if inside else left + cnt + 4, y,
                     f"{cname}  {cnt} ({100 * cnt / 250:.0f}%)" if inside
                     else f"{cname}  {cnt} ({100 * cnt / 250:.0f}%)",
                     ha="center" if inside else "left", va="center", fontsize=5.7,
                     color=("white" if cc != "#ced4da" else F.INK) if inside else F.INK)
            left += cnt
        axD.text(0, y + 0.40, title, fontsize=6.8, ha="left", va="bottom", color=F.INK)
        y -= 1.05
    axD.set_xlim(0, 330); axD.set_ylim(-1.45, 0.75); axD.axis("off")

    # matched close-ups of what the two binary factors look like on the surface
    def frac_of(px_, py_):
        return ((px_ - bounds[0]) / (bounds[1] - bounds[0]),
                (bounds[3] - py_) / (bounds[3] - bounds[2]))

    def crop_of(i, cx_frac, cy_frac, wfrac):
        img = C.orient_image(render_textured_rgba(mesh, fcd.colors[i], px=620, h_axis=h,
                                                  v_axis=v, depth_axis=d, cull=1.0,
                                                  bounds=bounds, supersample=2, crop=False,
                                                  align=False), fh, fv)
        H, W = img.shape[:2]
        ww = int(W * wfrac); hh = int(ww * 0.62)
        x0 = int(W * cx_frac) - ww // 2; y0 = int(H * cy_frac) - hh // 2
        x0 = max(0, min(W - ww, x0)); y0 = max(0, min(H - hh, y0))
        return img[y0:y0 + hh, x0:x0 + ww]

    s4 = int(np.where((labels["stripe"] == 0))[0][0])
    s5 = int(np.where((labels["stripe"] == 1))[0][0])
    c1 = int(np.argmax(cheek_area))
    c0 = int(np.where(labels["cheeks"] == 0)[0][0])
    sfx, sfy = frac_of(*anchors["stripe"])
    kfx, kfy = frac_of(*anchors["cheek"])
    tiles = [(s4, sfx, sfy, 0.62, "4 stripes"), (s5, sfx, sfy, 0.62, "5 stripes"),
             (c0, kfx, kfy, 0.17, "no cheek"), (c1, kfx, kfy, 0.17, "cheek patch")]
    for j, (i, cxf, cyf, wf, ttl) in enumerate(tiles):
        r, cc = divmod(j, 2)
        a = fig.add_axes([0.585 + cc * 0.185, 0.322 - r * 0.060, 0.170, 0.052])
        a.imshow(crop_of(i, cxf, cyf, wf), interpolation="antialiased")
        a.set_xticks([]); a.set_yticks([])
        for sp_ in a.spines.values():
            sp_.set_color("#c8ced4"); sp_.set_linewidth(0.6)
        a.set_title(ttl, fontsize=5.6, color=F.MUTED, pad=1.2)
    fig.text(0.545, 0.500, "D", fontsize=11, fontweight="bold", va="top")
    fig.text(0.578, 0.500, "Stripe-count and cheek-patch factors",
             fontsize=8.2, va="top", linespacing=1.3)

    # =============================================================== E  8 combinations
    fig.text(0.055, 0.232, "E", fontsize=11, fontweight="bold", va="top")
    fig.text(0.088, 0.232, "Planted factor combinations", fontsize=8.2,
             va="top")
    combos = [(b, t, s) for b in (0, 1) for t in (0, 1) for s in (0, 1)]
    for j, (b, t, s) in enumerate(combos):
        sel = np.where((labels["belly"] == b) & (labels["tail"] == t) & (labels["stripe"] == s))[0]
        if len(sel) == 0:
            continue
        i = int(sel[0])
        r, c = divmod(j, 4)
        a = fig.add_axes([0.080 + c * 0.112, 0.122 - r * 0.082, 0.106, 0.072])
        a.imshow(thumbs[i], interpolation="antialiased"); a.axis("off")
        a.set_title(f"{'blue' if b == 0 else 'purple'}·{'green' if t == 0 else 'cyan'}·{4 + s}",
                    fontsize=5.3, color=F.MUTED, pad=1.0)

    # =============================================================== F  nuisance variation
    axF = fig.add_axes([0.635, 0.048, 0.315, 0.162])
    nuis = [("base hue", "base_color_hue"), ("base saturation", "base_color_sat"),
            ("base value", "base_color_val"), ("stripe spacing", "stripe_spacing"),
            ("stripe width", "stripe_width"), ("stripe offset", "stripe_longitudinal_offset"),
            ("belly offset", "belly_translation"), ("belly strength", "belly_strength"),
            ("tail strength", "tail_strength")]
    for j, (nm, col) in enumerate(nuis):
        x = df[col].to_numpy().astype(float)
        x = (x - x.mean()) / (x.std() + 1e-12)
        xs = np.linspace(-3.4, 3.4, 200)
        kde = np.exp(-0.5 * ((xs[:, None] - x[None, :]) / 0.28) ** 2).sum(1)
        kde = kde / kde.max() * 0.85
        axF.fill_between(xs, j, j + kde, color="#c5ccd3", linewidth=0)
        axF.plot(xs, j + kde, color=F.MUTED, lw=0.5)
        axF.text(-3.8, j + 0.30, nm, fontsize=5.9, ha="right", va="center", color=F.INK)
    axF.set_xlim(-8.2, 3.6); axF.set_ylim(-0.35, len(nuis) + 0.10)
    axF.set_yticks([]); axF.set_xticks([-3, 0, 3])
    axF.set_xticklabels(["−3", "0", "+3"], fontsize=5.8)
    axF.set_xlabel("continuous nuisance variation (SD)", fontsize=6.6, labelpad=1)
    for sp in ("left", "right", "top"):
        axF.spines[sp].set_visible(False)
    fig.text(0.545, 0.232, "F", fontsize=11, fontweight="bold", va="top")
    fig.text(0.578, 0.232, "Continuous nuisance parameters",
             fontsize=8.2, va="top", linespacing=1.3)

    out = C.FIGURES / "fig1_dataset.png"
    fig.savefig(out, dpi=400)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    meta = {"hero": names[hero], "donors": stems,
            "belly_dprime": 8.43, "tail_dprime": 3.46}
    (C.RESULTS / "fig1_meta.json").write_text(json.dumps(meta, indent=2))
    print("wrote", out)


if __name__ == "__main__":
    main()
