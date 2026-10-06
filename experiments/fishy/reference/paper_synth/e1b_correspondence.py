"""E1b — anatomical correspondence, done properly.

Refinements over the first pass:
  * the native-UV control is restricted to texture pixels actually covered by a UV island
    (otherwise the unmapped black background inflates the overlap of the black-stripe mask);
  * the planted positional jitter is separated from pipeline error by contrasting pairs drawn
    from the SAME donor scan (where the only difference is the planted jitter, so their Dice
    is the attainable ceiling) with pairs from DIFFERENT donor scans (ceiling + the cost of
    transporting between shapes);
  * a per-donor 7x7 Dice matrix, and Dice as a function of the planted translation difference;
  * per-face occupancy maps of each planted patch, for the figures.
"""
from __future__ import annotations

import json

import imageio.v2 as imageio
import numpy as np

import common as C
from fishpipe.features import rgb_to_lab
from fishpipe.mesh import load_obj

KEYS = ("belly", "tail", "cheek", "stripe")
GRID = 256


def uv_coverage(mesh, grid: int = GRID, n_samples: int = 12, seed: int = 0) -> np.ndarray:
    """Boolean (grid, grid) mask of texture pixels covered by the mesh's UV islands."""
    rng = np.random.default_rng(seed)
    uv = mesh.face_corner_uvs()                                    # (Nf,3,2)
    r1 = rng.random((uv.shape[0], n_samples, 1))
    r2 = rng.random((uv.shape[0], n_samples, 1))
    su = np.sqrt(r1)
    w0 = 1 - su
    w1 = su * (1 - r2)
    w2 = su * r2
    pts = w0 * uv[:, None, 0] + w1 * uv[:, None, 1] + w2 * uv[:, None, 2]
    pts = pts.reshape(-1, 2)
    x = np.clip((pts[:, 0] * (grid - 1)).astype(int), 0, grid - 1)
    y = np.clip(((1 - pts[:, 1]) * (grid - 1)).astype(int), 0, grid - 1)
    m = np.zeros((grid, grid), bool)
    m[y, x] = True
    return m


def dice(m1, m2, w):
    inter = (w * (m1 & m2)).sum()
    denom = (w * m1).sum() + (w * m2).sum()
    return float(2 * inter / denom) if denom > 0 else np.nan


def main():
    fcd, _, mesh = C.atlas_data()
    names = list(fcd.names)
    df, labels = C.load_gt(names)
    N = len(names)
    areas = fcd.areas
    spec_idx = df["specimen_index"].to_numpy()
    donors = sorted(set(spec_idx))

    print("[E1b] atlas-space masks")
    lab = rgb_to_lab(fcd.colors)
    A = {k: np.zeros((N, mesh.n_faces), bool) for k in KEYS}
    for i in range(N):
        m = C.pigment_masks(lab[i])
        for k in KEYS:
            A[k][i] = m[k]

    print("[E1b] native-UV masks restricted to UV coverage")
    # one coverage mask per donor (all samples from a donor share its mesh/unwrap)
    cov = {}
    for d in donors:
        i0 = int(np.where(spec_idx == d)[0][0])
        cov[d] = uv_coverage(load_obj(C.SRC_MESHES / f"{names[i0]}.obj"))
        print(f"   donor {d}: UV coverage {cov[d].mean() * 100:.1f}% of the texture")
    U = {k: np.zeros((N, GRID * GRID), bool) for k in KEYS}
    covflat = np.zeros((N, GRID * GRID), bool)
    for i in range(N):
        tex = imageio.imread(C.SRC_IMAGES / f"{names[i]}.png")
        # classify pigments at FULL texture resolution, then area-average the binary masks
        # onto the comparison grid: point sampling would alias away the thin stripes and
        # handicap the no-correspondence baseline.
        full = C.pigment_masks(rgb_to_lab(tex))
        blk = tex.shape[0] // GRID
        mu = {k: (v[:GRID * blk, :GRID * blk]
                  .reshape(GRID, blk, GRID, blk).mean(axis=(1, 3)) >= 0.5) for k, v in full.items()}
        c = cov[spec_idx[i]]
        covflat[i] = c.ravel()
        for k in KEYS:
            U[k][i] = (mu[k] & c).ravel()

    rng = np.random.default_rng(1)
    npairs = 6000
    pi = rng.integers(0, N, npairs)
    pj = rng.integers(0, N, npairs)
    ok = pi != pj
    pi, pj = pi[ok], pj[ok]
    same = spec_idx[pi] == spec_idx[pj]

    out = {}
    for k in KEYS:
        if k == "cheek":
            has = (labels["cheeks"] == 1) & (A[k].sum(1) > 0)
            sel = has[pi] & has[pj]
        else:
            sel = np.ones(len(pi), bool)
        a = np.array([dice(A[k][x], A[k][y], areas) for x, y in zip(pi[sel], pj[sel])])
        u = np.array([dice(U[k][x], U[k][y], (covflat[x] & covflat[y]).astype(float))
                      for x, y in zip(pi[sel], pj[sel])])
        fa = (areas[None] * A[k]).sum(1) / areas.sum()
        ch = np.array([2 * fa[x] * fa[y] / (fa[x] + fa[y]) if (fa[x] + fa[y]) > 0 else np.nan
                       for x, y in zip(pi[sel], pj[sel])])
        s = same[sel]
        rec = {
            "atlas_dice": float(np.nanmean(a)), "atlas_dice_sd": float(np.nanstd(a)),
            "atlas_same_donor": float(np.nanmean(a[s])),
            "atlas_diff_donor": float(np.nanmean(a[~s])),
            "transfer_efficiency": float(np.nanmean(a[~s]) / np.nanmean(a[s])),
            "nativeUV_dice": float(np.nanmean(u)),
            "nativeUV_diff_donor": float(np.nanmean(u[~s])),
            "nativeUV_same_donor": float(np.nanmean(u[s])) if s.any() else None,
            "chance_dice": float(np.nanmean(ch)),
            "area_frac_mean": float(fa.mean()), "area_frac_sd": float(fa.std()),
            "n_pairs": int(sel.sum()), "n_same": int(s.sum()),
        }
        # 7x7 donor Dice matrix
        M = np.full((len(donors), len(donors)), np.nan)
        for ii, d1 in enumerate(donors):
            for jj, d2 in enumerate(donors):
                q = (spec_idx[pi[sel]] == d1) & (spec_idx[pj[sel]] == d2)
                if q.sum() > 5:
                    M[ii, jj] = np.nanmean(a[q])
        rec["donor_matrix"] = [[None if np.isnan(v) else float(v) for v in row] for row in M]
        out[k] = rec
        print(f"  {k:7s} atlas {rec['atlas_dice']:.3f} "
              f"(same-donor ceiling {rec['atlas_same_donor']:.3f}, cross-donor "
              f"{rec['atlas_diff_donor']:.3f} = {100 * rec['transfer_efficiency']:.0f}% of ceiling) | "
              f"native-UV {rec['nativeUV_dice']:.3f} "
              f"(same-donor {rec['nativeUV_same_donor']:.3f}, cross-donor "
              f"{rec['nativeUV_diff_donor']:.3f}) | chance {rec['chance_dice']:.3f}")

    # --- how much of the residual is the PLANTED jitter, not the pipeline? -------------
    jit = {}
    for k, col in (("belly", "belly_translation"),
                   ("cheek", "rosy_cheeks_translation_y")):
        if k == "cheek":
            has = (labels["cheeks"] == 1) & (A[k].sum(1) > 0)
            sel = has[pi] & has[pj]
        else:
            sel = np.ones(len(pi), bool)
        v = df[col].to_numpy()
        d_par = np.abs(v[pi[sel]] - v[pj[sel]])
        a = np.array([dice(A[k][x], A[k][y], areas) for x, y in zip(pi[sel], pj[sel])])
        q = np.quantile(d_par, [0.25, 0.75])
        jit[k] = {
            "param": col,
            "dice_small_jitter": float(np.nanmean(a[d_par <= q[0]])),
            "dice_large_jitter": float(np.nanmean(a[d_par >= q[1]])),
            "spearman_dice_vs_jitter": float(
                __import__("scipy.stats", fromlist=["x"]).spearmanr(d_par, a).statistic),
        }
        print(f"  [{k}] Dice with small planted offset {jit[k]['dice_small_jitter']:.3f} vs "
              f"large {jit[k]['dice_large_jitter']:.3f} "
              f"(rho={jit[k]['spearman_dice_vs_jitter']:+.2f})")

    # --- how far is each pigment field from the landmarks that drive both the paint and the
    # module's correspondence?  Faces far from every landmark are the ones where the dense
    # correspondence is genuinely surface-driven rather than landmark-interpolated.
    import json as _json
    from scipy.spatial import cKDTree
    lm = _json.loads((C.ATLAS_DIR / "atlasLM.mrk.json").read_text())
    pts = np.array([cp["position"] for cp in lm["markups"][0]["controlPoints"]], float)
    cents = mesh.face_centroids()
    dmin = cKDTree(pts).query(cents)[0]
    diag = float(np.linalg.norm(mesh.vertices.max(0) - mesh.vertices.min(0)))
    lmdist = {}
    for k in KEYS:
        occ = A[k].mean(0)
        w = occ * areas
        if w.sum() > 0:
            lmdist[k] = {"mean_dist_frac_of_diagonal": float((w * dmin).sum() / w.sum() / diag),
                         "p90_dist_frac_of_diagonal": float(
                             np.quantile(dmin[occ > 0.1], 0.9) / diag) if (occ > 0.1).any() else None}
    lmdist["_all_faces_mean_frac"] = float(dmin.mean() / diag)
    print("  distance to the nearest landmark, as a fraction of the atlas diagonal:")
    for k, v in lmdist.items():
        if isinstance(v, dict):
            print(f"    {k:7s} mean {v['mean_dist_frac_of_diagonal']:.3f}")

    res = {"correspondence": out, "planted_jitter": jit, "landmark_distance": lmdist,
           "donors": [int(d) for d in donors],
           "donor_names": [names[int(np.where(spec_idx == d)[0][0])].rsplit("_fishy", 1)[0]
                           for d in donors]}
    (C.RESULTS / "e1b_correspondence.json").write_text(json.dumps(res, indent=2))
    np.savez_compressed(C.RESULTS / "e1b_arrays.npz",
                        occ_belly=A["belly"].mean(0).astype(np.float32),
                        occ_tail=A["tail"].mean(0).astype(np.float32),
                        occ_cheek=A["cheek"].mean(0).astype(np.float32),
                        occ_stripe=A["stripe"].mean(0).astype(np.float32),
                        spec_idx=spec_idx)
    print("saved results/e1b_correspondence.json")


if __name__ == "__main__":
    main()
