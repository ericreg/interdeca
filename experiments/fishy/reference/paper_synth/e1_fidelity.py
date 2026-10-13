"""E1 — Does the ColorAtlas pipeline preserve the texture information it transports?

Three independent measurements, all made possible by the synthetic ground truth:

  (1) PHOTOMETRIC fidelity.  For every atlas face we recover the colour that Blender's
      selected-to-active bake *should* have written, by sampling the specimen's own texture
      at the closest point on its aligned source surface, and compare it to what the module
      actually wrote (CIEDE2000).

  (2) ARTIFACT census.  Near-black atlas faces are only bake holes when the reference is NOT
      dark — the synthetic fish also carries genuinely black stripes, so a naive black-pixel
      detector would confuse a planted pigment with a failure. The reference lets us separate
      them.

  (3) ANATOMICAL correspondence.  Each planted pigment (belly, tail, cheek) was painted at a
      homologous anatomical location on seven DIFFERENT donor shapes. If the pipeline
      establishes true correspondence, those patches must land on the SAME atlas faces. We
      measure pairwise Dice overlap in atlas space and contrast it with the overlap of the
      same patches in the specimens' native UV parameterisations (what an analyst would have
      without the module) and with a chance model.
"""
from __future__ import annotations

import json
import time

import imageio.v2 as imageio
import numpy as np
from scipy.spatial import cKDTree

import common as C
from fishpipe.features import rgb_to_lab
from fishpipe.mesh import load_obj

ALIGNED = C.MOD_RUN / "ATLAS" / "alignedModels"
RESAMPLED = C.ATLAS_DIR / "resampledOBJ_withUV"


# --------------------------------------------------------------------------- helpers
def deltaE2000(lab1: np.ndarray, lab2: np.ndarray) -> np.ndarray:
    """Vectorised CIEDE2000 (Sharma et al. 2005). lab*: (..., 3)."""
    L1, a1, b1 = lab1[..., 0], lab1[..., 1], lab1[..., 2]
    L2, a2, b2 = lab2[..., 0], lab2[..., 1], lab2[..., 2]
    C1 = np.hypot(a1, b1)
    C2 = np.hypot(a2, b2)
    Cbar = 0.5 * (C1 + C2)
    G = 0.5 * (1 - np.sqrt(Cbar ** 7 / (Cbar ** 7 + 25.0 ** 7 + 1e-30)))
    a1p, a2p = (1 + G) * a1, (1 + G) * a2
    C1p, C2p = np.hypot(a1p, b1), np.hypot(a2p, b2)
    h1p = np.degrees(np.arctan2(b1, a1p)) % 360
    h2p = np.degrees(np.arctan2(b2, a2p)) % 360
    dLp = L2 - L1
    dCp = C2p - C1p
    dhp = h2p - h1p
    dhp = np.where(dhp > 180, dhp - 360, dhp)
    dhp = np.where(dhp < -180, dhp + 360, dhp)
    dhp = np.where(C1p * C2p == 0, 0.0, dhp)
    dHp = 2 * np.sqrt(C1p * C2p) * np.sin(np.radians(dhp / 2))
    Lbar = 0.5 * (L1 + L2)
    Cbarp = 0.5 * (C1p + C2p)
    hsum = h1p + h2p
    hdiff = np.abs(h1p - h2p)
    hbar = np.where(C1p * C2p == 0, hsum,
                    np.where(hdiff <= 180, hsum / 2,
                             np.where(hsum < 360, (hsum + 360) / 2, (hsum - 360) / 2)))
    T = (1 - 0.17 * np.cos(np.radians(hbar - 30)) + 0.24 * np.cos(np.radians(2 * hbar))
         + 0.32 * np.cos(np.radians(3 * hbar + 6)) - 0.20 * np.cos(np.radians(4 * hbar - 63)))
    dTheta = 30 * np.exp(-(((hbar - 275) / 25) ** 2))
    Rc = 2 * np.sqrt(Cbarp ** 7 / (Cbarp ** 7 + 25.0 ** 7 + 1e-30))
    Sl = 1 + (0.015 * (Lbar - 50) ** 2) / np.sqrt(20 + (Lbar - 50) ** 2)
    Sc = 1 + 0.045 * Cbarp
    Sh = 1 + 0.015 * Cbarp * T
    Rt = -np.sin(np.radians(2 * dTheta)) * Rc
    return np.sqrt((dLp / Sl) ** 2 + (dCp / Sc) ** 2 + (dHp / Sh) ** 2
                   + Rt * (dCp / Sc) * (dHp / Sh))


def _closest_point_uv(query: np.ndarray, src_tris: np.ndarray, src_uvs: np.ndarray,
                      tree: cKDTree, corner_face: np.ndarray, k: int = 6):
    """For each query point, closest point on the source triangle soup -> interpolated UV.

    src_tris (M,3,3), src_uvs (M,3,2). Candidate faces come from the k nearest triangle
    corners; the exact point-triangle closest point is then evaluated for those candidates.
    Returns (uv (Q,2), dist (Q,)).
    """
    _, idx = tree.query(query, k=k, workers=-1)          # (Q,k) corner indices
    cand = corner_face[idx]                              # (Q,k) face indices
    Q, K = cand.shape
    P = query[:, None, :]                                # (Q,1,3)
    A = src_tris[cand, 0]; B = src_tris[cand, 1]; Cc = src_tris[cand, 2]   # (Q,K,3)
    AB = B - A
    AC = Cc - A
    AP = P - A
    d1 = np.einsum("qkc,qkc->qk", AB, AP)
    d2 = np.einsum("qkc,qkc->qk", AC, AP)
    BP = P - B
    d3 = np.einsum("qkc,qkc->qk", AB, BP)
    d4 = np.einsum("qkc,qkc->qk", AC, BP)
    CP = P - Cc
    d5 = np.einsum("qkc,qkc->qk", AB, CP)
    d6 = np.einsum("qkc,qkc->qk", AC, CP)
    va = d3 * d6 - d5 * d4
    vb = d5 * d2 - d1 * d6
    vc = d1 * d4 - d3 * d2
    denom = va + vb + vc
    # barycentric of the closest point, region by region (Ericson, Real-Time Collision Detection)
    u = np.zeros_like(d1); v = np.zeros_like(d1); w = np.zeros_like(d1)
    # region A
    mA = (d1 <= 0) & (d2 <= 0)
    # region B
    mB = (d3 >= 0) & (d4 <= d3)
    # region C
    mC = (d6 >= 0) & (d5 <= d6)
    # edge AB
    mAB = (vc <= 0) & (d1 >= 0) & (d3 <= 0)
    # edge AC
    mAC = (vb <= 0) & (d2 >= 0) & (d6 <= 0)
    # edge BC
    mBC = (va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0)
    interior = ~(mA | mB | mC | mAB | mAC | mBC)
    with np.errstate(divide="ignore", invalid="ignore"):
        t_ab = np.where(mAB, d1 / np.where(d1 - d3 == 0, 1e-30, d1 - d3), 0.0)
        t_ac = np.where(mAC, d2 / np.where(d2 - d6 == 0, 1e-30, d2 - d6), 0.0)
        t_bc = np.where(mBC, (d4 - d3) / np.where((d4 - d3) + (d5 - d6) == 0, 1e-30,
                                                  (d4 - d3) + (d5 - d6)), 0.0)
        vi = np.where(interior, vb / np.where(denom == 0, 1e-30, denom), 0.0)
        wi = np.where(interior, vc / np.where(denom == 0, 1e-30, denom), 0.0)
    v = np.where(mAB, t_ab, np.where(mBC, 1 - t_bc, np.where(interior, vi, np.where(mB, 1.0, 0.0))))
    w = np.where(mAC, t_ac, np.where(mBC, t_bc, np.where(interior, wi, np.where(mC, 1.0, 0.0))))
    v = np.clip(v, 0, 1); w = np.clip(w, 0, 1)
    s = v + w
    over = s > 1
    v = np.where(over, v / np.where(s == 0, 1, s), v)
    w = np.where(over, w / np.where(s == 0, 1, s), w)
    u = 1 - v - w
    closest = A + v[..., None] * AB + w[..., None] * AC            # (Q,K,3)
    dist = np.linalg.norm(closest - P, axis=-1)                    # (Q,K)
    best = np.argmin(dist, axis=1)
    qi = np.arange(Q)
    fbest = cand[qi, best]
    uv = (u[qi, best, None] * src_uvs[fbest, 0]
          + v[qi, best, None] * src_uvs[fbest, 1]
          + w[qi, best, None] * src_uvs[fbest, 2])
    return uv, dist[qi, best]


def _sample_tex(tex: np.ndarray, uv: np.ndarray) -> np.ndarray:
    h, w = tex.shape[:2]
    x = np.clip((np.clip(uv[:, 0], 0, 1) * (w - 1)).astype(np.int64), 0, w - 1)
    y = np.clip(((1 - np.clip(uv[:, 1], 0, 1)) * (h - 1)).astype(np.int64), 0, h - 1)
    return tex[y, x, :3]


# --------------------------------------------------------------------------- main
def main(n_specimens: int | None = None, subsample: int | None = None):
    t0 = time.time()
    fcd, art, amesh = C.atlas_data()
    names = list(fcd.names)
    df, labels = C.load_gt(names)
    N = len(names) if n_specimens is None else min(n_specimens, len(names))
    Nf = amesh.n_faces
    rng = np.random.default_rng(0)
    fidx = np.arange(Nf) if subsample is None else np.sort(
        rng.choice(Nf, size=min(subsample, Nf), replace=False))

    atlas_lab = rgb_to_lab(fcd.colors)                     # (N, Nf, 3)

    dE_all = np.zeros((N, len(fidx)), dtype=np.float32)
    ref_lab_all = np.zeros((N, len(fidx), 3), dtype=np.float32)
    gap = np.zeros((N, len(fidx)), dtype=np.float32)       # closest-point distance
    print(f"[E1] photometric fidelity over {N} specimens x {len(fidx)} faces")
    for i in range(N):
        nm = names[i]
        rs = load_obj(RESAMPLED / f"{nm}_resampled.obj")
        al = load_obj(ALIGNED / f"{nm}_align.obj")
        tex = imageio.imread(C.SRC_IMAGES / f"{nm}.png")
        src_tris = al.vertices[al.face_v]                  # (M,3,3)
        src_uvs = al.uvs[al.face_vt]                       # (M,3,2)
        corners = src_tris.reshape(-1, 3)
        corner_face = np.repeat(np.arange(src_tris.shape[0]), 3)
        tree = cKDTree(corners)
        q = rs.vertices[rs.face_v][fidx].mean(axis=1)      # atlas-face centroid on the specimen
        uv, d = _closest_point_uv(q, src_tris, src_uvs, tree, corner_face)
        ref_rgb = _sample_tex(tex, uv)
        ref_lab = rgb_to_lab(ref_rgb)
        ref_lab_all[i] = ref_lab
        gap[i] = d
        dE_all[i] = deltaE2000(ref_lab, atlas_lab[i, fidx])
        if i % 25 == 0 or i == N - 1:
            print(f"  {i + 1}/{N} {nm}: median dE={np.median(dE_all[i]):.2f} "
                  f"p95={np.percentile(dE_all[i], 95):.2f} gap_p95={np.percentile(d, 95):.2e}")

    # ---------------- photometric summary ----------------
    photo = {
        "median_dE": float(np.median(dE_all)),
        "mean_dE": float(dE_all.mean()),
        "p90_dE": float(np.percentile(dE_all, 90)),
        "p95_dE": float(np.percentile(dE_all, 95)),
        "p99_dE": float(np.percentile(dE_all, 99)),
        "frac_dE_lt_1": float((dE_all < 1).mean()),
        "frac_dE_lt_2p3": float((dE_all < 2.3).mean()),   # 1 JND
        "frac_dE_lt_5": float((dE_all < 5).mean()),
        "per_specimen_median": [float(x) for x in np.median(dE_all, axis=1)],
        "closest_point_gap_p95_mm_units": float(np.percentile(gap, 95)),
    }

    # ---------------- artifact census ----------------
    atl_dark = atlas_lab[:N][:, fidx, 0] < 12
    ref_dark = ref_lab_all[..., 0] < 12
    holes = atl_dark & ~ref_dark                 # black in the atlas but not on the specimen
    planted_black = atl_dark & ref_dark          # the planted stripes
    missed = (~atl_dark) & ref_dark              # stripe present on specimen, lost in atlas
    artifacts = {
        "atlas_black_frac": float(atl_dark.mean()),
        "true_bake_hole_frac": float(holes.mean()),
        "planted_black_frac": float(planted_black.mean()),
        "lost_black_frac": float(missed.mean()),
        "naive_blackpixel_detector_frac": float(atl_dark.mean()),
        "specimens_with_hole_frac_gt_1pct": int((holes.mean(1) > 0.01).sum()),
        "worst_specimen": names[int(np.argmax(holes.mean(1)))],
        "worst_specimen_hole_frac": float(holes.mean(1).max()),
    }

    # ---------------- anatomical correspondence ----------------
    # masks in ATLAS space (shared faces). The native-UV comparison lives in e1b, which
    # restricts it to UV-island coverage; without that restriction the unmapped black
    # background is scored as stripe pigment and the overlap is spuriously inflated.
    def dice(m1, m2, w):
        inter = (w * (m1 & m2)).sum()
        return float(2 * inter / ((w * m1).sum() + (w * m2).sum() + 1e-12))

    areas = fcd.areas[fidx]
    keys = ("belly", "tail", "cheek", "stripe")
    atlas_masks = {k: np.zeros((N, len(fidx)), bool) for k in keys}
    uv_masks = {k: np.zeros((N, 256 * 256), bool) for k in keys}
    for i in range(N):
        refs = C.pigment_lab(df.iloc[i])
        m = C.pigment_masks(atlas_lab[i, fidx], refs)
        for k in keys:
            atlas_masks[k][i] = m[k]
        # native-UV masks: each specimen's own texture on a common 256x256 UV grid — this is
        # the "no correspondence" control: the same image coordinate means nothing across
        # specimens because every donor has its own unwrap.
        tex = imageio.imread(C.SRC_IMAGES / f"{names[i]}.png")
        small = tex[::tex.shape[0] // 256, ::tex.shape[1] // 256][:256, :256]
        mu = C.pigment_masks(rgb_to_lab(small), refs)
        for k in keys:
            uv_masks[k][i] = mu[k].ravel()

    spec_idx = df["specimen_index"].to_numpy()[:N]
    corr = {}
    npairs = 4000
    pi = rng.integers(0, N, npairs)
    pj = rng.integers(0, N, npairs)
    keep = pi != pj
    pi, pj = pi[keep], pj[keep]
    same_donor = spec_idx[pi] == spec_idx[pj]
    for k in keys:
        Ma, Mu = atlas_masks[k], uv_masks[k]
        if k == "cheek":                     # only compare specimens that HAVE the feature
            has = (labels["cheeks"][:N] == 1) & (Ma.sum(1) > 0)
            sel = has[pi] & has[pj]
        else:
            sel = np.ones(len(pi), bool)
        if sel.sum() == 0:
            continue
        a_d = np.array([dice(Ma[x], Ma[y], areas) for x, y in zip(pi[sel], pj[sel])])
        u_d = np.array([dice(Mu[x], Mu[y], np.ones(Mu.shape[1]))
                        for x, y in zip(pi[sel], pj[sel])])
        fa = np.array([(areas * Ma[x]).sum() / areas.sum() for x in range(N)])
        chance = np.array([2 * fa[x] * fa[y] / (fa[x] + fa[y] + 1e-12)
                           for x, y in zip(pi[sel], pj[sel])])
        sd = same_donor[sel]
        corr[k] = {
            "atlas_dice_mean": float(a_d.mean()), "atlas_dice_sd": float(a_d.std()),
            "atlas_dice_same_donor": float(a_d[sd].mean()) if sd.any() else None,
            "atlas_dice_diff_donor": float(a_d[~sd].mean()) if (~sd).any() else None,
            "native_uv_dice_mean": float(u_d.mean()),
            "native_uv_dice_diff_donor": float(u_d[~sd].mean()) if (~sd).any() else None,
            "chance_dice_mean": float(chance.mean()),
            "area_frac_mean": float(fa.mean()),
            "n_pairs": int(sel.sum()),
        }
        print(f"  [{k}] atlas Dice {a_d.mean():.3f} (same-donor {a_d[sd].mean() if sd.any() else float('nan'):.3f} / "
              f"diff-donor {a_d[~sd].mean():.3f}) | native-UV {u_d.mean():.3f} | chance {chance.mean():.3f}")

    out = {"photometric": photo, "artifacts": artifacts, "correspondence": corr,
           "n_specimens": N, "n_faces_scored": int(len(fidx)),
           "seconds": round(time.time() - t0, 1)}
    (C.RESULTS / "e1_fidelity.json").write_text(json.dumps(out, indent=2))
    np.savez_compressed(C.RESULTS / "e1_fidelity_arrays.npz",
                        dE=dE_all, face_idx=fidx, holes=holes, ref_lab=ref_lab_all,
                        atlas_belly=atlas_masks["belly"], atlas_tail=atlas_masks["tail"],
                        atlas_cheek=atlas_masks["cheek"], atlas_stripe=atlas_masks["stripe"],
                        spec_idx=spec_idx)
    print(json.dumps(out, indent=2)[:2500])


if __name__ == "__main__":
    import sys
    ns = int(sys.argv[1]) if len(sys.argv) > 1 else None
    ss = int(sys.argv[2]) if len(sys.argv) > 2 else None
    main(ns, ss)
