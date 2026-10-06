"""Tiny CPU renderer: orthographic lateral view of a mesh colored by per-face colors.

No GPU / no renderer dependency — projects faces to 2D, back-face culls to the near flank,
painter's-sorts by depth, and fills triangles with matplotlib. Good enough for morphospace
exemplar thumbnails (e.g. a side view of the fishy showing its stripes/belly/tail).
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection

from .mesh import Mesh


def render_side_view(
    mesh: Mesh, face_colors_u8: np.ndarray, px: int = 220,
    h_axis: int = 2, v_axis: int = 1, depth_axis: int = 0, cull: float = 1.0,
    bg=(0.06, 0.06, 0.06), bounds: tuple | None = None,
) -> np.ndarray:
    """Render one specimen's lateral view → RGB uint8 image.

    h_axis/v_axis/depth_axis pick the projection (default: horizontal=Z length, vertical=Y
    height, depth=X width → a fish side view). `cull` keeps faces whose normal points toward
    the camera (sign on depth_axis). `bounds`=(hmin,hmax,vmin,vmax) keeps all specimens aligned.
    """
    p = mesh.vertices[mesh.face_v]                      # (Nf,3,3)
    nrm = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
    keep = nrm[:, depth_axis] * cull > 0
    tris = p[keep][:, :, [h_axis, v_axis]]              # (M,3,2)
    cols = np.clip(face_colors_u8[keep] / 255.0, 0, 1)
    order = np.argsort(p[keep][:, :, depth_axis].mean(1) * cull)  # far → near

    if bounds is None:
        hmin, hmax = mesh.vertices[:, h_axis].min(), mesh.vertices[:, h_axis].max()
        vmin, vmax = mesh.vertices[:, v_axis].min(), mesh.vertices[:, v_axis].max()
    else:
        hmin, hmax, vmin, vmax = bounds
    aspect = (hmax - hmin) / (vmax - vmin + 1e-9)
    fig = plt.figure(figsize=(px / 100 * aspect, px / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_facecolor(bg); ax.axis("off")
    pc = PolyCollection(tris[order], facecolors=cols[order], edgecolors="none", antialiaseds=False)
    ax.add_collection(pc)
    ax.set_xlim(hmin, hmax); ax.set_ylim(vmin, vmax)
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return img


def render_textured_rgba(
    mesh: Mesh, face_colors_u8: np.ndarray, px: int = 420,
    h_axis: int = 2, v_axis: int = 0, depth_axis: int = 1, cull: float = -1.0,
    bounds: tuple | None = None, supersample: int = 3, crop: bool = True,
    align: bool = True, pad: float = 0.04,
) -> np.ndarray:
    """Orthographic textured view → tight-cropped RGBA (TRANSPARENT background).

    Same projection/cull machinery as `render_side_view`, but renders onto a transparent
    canvas (no bg rectangle) and crops to the drawn silhouette, so the exemplar is the
    actual 3D-shaded valve rather than a texture crop. Supersamples then the caller can
    downscale for a smooth silhouette (interior triangles stay seam-free, antialias off).

    Mussel exterior default: h=Z(length), v=X, depth=Y(thickness), cull=-1 (exterior side).
    `align=True` rotates the 2D projection so the silhouette's major (PCA) axis is HORIZONTAL,
    i.e. the valve lies LANDSCAPE rather than diagonal. Alignment uses geometry only, so it is
    identical across specimens (shared atlas) → frames stay registered. `bounds` is ignored
    when align=True (frame is derived from the rotated silhouette + `pad`).
    """
    p = mesh.vertices[mesh.face_v]                      # (Nf,3,3)
    nrm = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
    keep = nrm[:, depth_axis] * cull > 0
    tris = p[keep][:, :, [h_axis, v_axis]]              # (M,3,2)
    cols = np.clip(face_colors_u8[keep] / 255.0, 0, 1)
    order = np.argsort(p[keep][:, :, depth_axis].mean(1) * cull)  # far → near

    if align:
        pts = tris.reshape(-1, 2)
        c2 = pts.mean(0)
        _, _, vt = np.linalg.svd(pts - c2, full_matrices=False)
        theta = np.arctan2(vt[0, 1], vt[0, 0])         # major-axis angle
        ct, st = np.cos(-theta), np.sin(-theta)
        R = np.array([[ct, -st], [st, ct]])
        tris = (tris.reshape(-1, 2) @ R.T).reshape(-1, 3, 2)
        ext = tris.reshape(-1, 2)
        mh = (ext[:, 0].max() - ext[:, 0].min()) * pad
        mv = (ext[:, 1].max() - ext[:, 1].min()) * pad
        hmin, hmax = ext[:, 0].min() - mh, ext[:, 0].max() + mh
        vmin, vmax = ext[:, 1].min() - mv, ext[:, 1].max() + mv
    elif bounds is None:
        hmin, hmax = mesh.vertices[:, h_axis].min(), mesh.vertices[:, h_axis].max()
        vmin, vmax = mesh.vertices[:, v_axis].min(), mesh.vertices[:, v_axis].max()
    else:
        hmin, hmax, vmin, vmax = bounds
    aspect = (hmax - hmin) / (vmax - vmin + 1e-9)
    fig = plt.figure(figsize=(px / 100 * aspect, px / 100), dpi=100 * supersample)
    fig.patch.set_alpha(0.0)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_facecolor("none"); ax.axis("off")
    ax.add_collection(PolyCollection(tris[order], facecolors=cols[order],
                                     edgecolors="none", antialiaseds=False))
    ax.set_xlim(hmin, hmax); ax.set_ylim(vmin, vmax)
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba()).copy()       # (H,W,4) with alpha=0 off-silhouette
    plt.close(fig)
    if crop:
        a = img[..., 3] > 8
        ys, xs = np.where(a)
        if len(xs):
            img = img[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    return img
