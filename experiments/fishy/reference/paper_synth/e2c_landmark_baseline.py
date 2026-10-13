"""E2c — the STRONG pre-module baseline: a hand-built, landmark-driven correspondence.

A reviewer's objection to E2: the pre-module spatial baseline (`pre_spatial_canon`, a 32-bin
Lab profile along each specimen's OWN principal axis) is a straw man, because any user who has
run ColorAtlas has ALREADY digitised 47 anatomical landmarks on every specimen. The strongest
descriptor available *without* the module is therefore not a self-normalised axis — it is a
dense correspondence interpolated from those landmarks by hand.

This script builds exactly that and scores it with the project's standard protocol:

  reference frame : ONE specimen of donor `Mchenga_m1` (specimen_index 0 — the same reference
                    the generator used), in its NATIVE mesh and NATIVE UV.
  warp            : regularised 3-D thin-plate spline, U(r) = r^2 log(r + eps), lambda = 1e-6,
                    fitted from the reference's 47 landmarks to each specimen's 47 landmarks
                    (shared labels only, consistent order) — the same maths as the generator's
                    `TPS3D` in color_fishy_tps.py (re-implemented here; that module needs bpy).
                    The spline is fitted in a UNIT-SCALE frame (each landmark cloud centred and
                    divided by its RMS radius, the similarity normalisation absorbed by the
                    spline's own affine block). This matters: U(r) = r^2 log r is NOT scale
                    equivariant, so a fixed lambda means different things at different units.
                    The scans live in metre-ish coordinates (body ~0.08-0.19 long), where
                    lambda = 1e-6 is comparable to the kernel entries and visibly over-smooths
                    (landmark-fit RMSE 0.10-0.16% of the body length). The generator fitted its
                    warps in canonical space with the body normalised to unit length, where the
                    same lambda is negligible; we reproduce that convention, and the fit becomes
                    a near-interpolant (RMSE ~1e-7, i.e. 1e-4 % of the body length). Both RMSEs
                    are reported. Using the near-interpolating fit makes this baseline STRONGER,
                    which is the conservative direction for the claim under test.
  transfer        : every reference-mesh VERTEX is pushed through the spline into the target's
                    space, projected to the closest point on the target's own triangle soup, and
                    the target's own 2048^2 texture is read there. Per-face colour = mean of the
                    three corner reads, i.e. the identical face-colour rule as
                    `fishpipe.mesh.sample_face_colors` (= the module's own rule), so the only
                    thing that differs from the PRE-module descriptors is the correspondence.

Descriptors (all indexed by the REFERENCE mesh's 49,999 faces, area-weighted by the REFERENCE
mesh, so they are directly comparable across specimens):
  lm_hist24   (24)   area-weighted Lab composition on the SAME palette as `pre_hist`
  lm_axial96  (96)   32-bin axial Lab profile along the reference mesh's principal axis
                     — dimension-matched to `post_axial`
  lm_flat     (3600) Lab of a fixed random subsample of 1200 reference faces
                     — dimension-matched to `post_spatial_flat` (subsample=1200)

Scored under repeated stratified random CV (`C.balanced_cv`) and leave-one-donor-out, against
`pre_spatial_canon` (weak pre-module baseline) and `post_axial` / `post_spatial_flat` /
`post_hist_matched` (post-module), plus label-shuffled nulls. Finally the planted-pigment Dice
overlap is recomputed in this landmark-warped frame exactly as `e1b_correspondence.py` does it
for the atlas, so the two can be plotted side by side.

CAVEAT that the script verifies and reports rather than hides: the 250 specimens are painted on
only 7 distinct donor scans, and the landmark files are reused verbatim within a donor. There
are therefore only 7 distinct warps, and for the 40 specimens of the reference donor the warp is
the identity — that group is transported exactly, for free. This *favours* the landmark
baseline, which is the conservative direction for the claim being tested.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from scipy.spatial import cKDTree
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import common as C
from e1_fidelity import _closest_point_uv, _sample_tex, deltaE2000
from fishpipe import features
from fishpipe.features import rgb_to_lab
from fishpipe.mesh import load_obj, sample_face_colors

FACTORS = ("belly", "tail", "stripe", "cheeks")
PIG_KEYS = ("belly", "tail", "cheek", "stripe")
TPS_LAMBDA = 1e-6
KNN = 96                 # candidate triangle corners for the closest-point projection
KNN_CHECK = 192          # wider candidate set used to prove the projection has converged
N_BINS = 32
N_FLAT = 1200


# --------------------------------------------------------------------------- TPS
class TPS3D:
    """3-D thin-plate spline, U(r) = r^2 log(r + eps), Tikhonov on the kernel block.

    Verbatim re-implementation of `TPS3D` in color_fishy_tps.py (which cannot be imported
    here: that module imports bpy at load time).
    """

    def __init__(self):
        self.W = None
        self.A = None
        self.ctrl = None

    @staticmethod
    def _U(r):
        eps = 1e-12
        return r ** 2 * np.log(r + eps)

    def fit(self, X, Y, lambda_reg: float = TPS_LAMBDA):
        X = np.asarray(X, np.float64)
        Y = np.asarray(Y, np.float64)
        K = X.shape[0]
        assert X.shape == (K, 3) and Y.shape == (K, 3)
        if K < 4:
            raise ValueError(f"TPS requires >= 4 control points, got {K}")
        sv = np.linalg.svd(X - X.mean(0, keepdims=True), compute_uv=False)
        if sv[2] < 1e-10:
            raise ValueError("control points are (near-)coplanar")
        self.ctrl = X.copy()
        P_mat = np.hstack([np.ones((K, 1)), X])
        d = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(2))
        L = np.zeros((K + 4, K + 4))
        L[:K, :K] = self._U(d) + lambda_reg * np.eye(K)
        L[:K, K:] = P_mat
        L[K:, :K] = P_mat.T
        rhs = np.zeros((K + 4, 3))
        rhs[:K] = Y
        params = np.linalg.solve(L, rhs)
        self.W, self.A = params[:K], params[K:]

    def evaluate(self, P, chunk: int = 20000):
        P = np.asarray(P, np.float64)
        out = np.empty_like(P)
        for s in range(0, P.shape[0], chunk):
            q = P[s:s + chunk]
            d = np.sqrt(((q[:, None, :] - self.ctrl[None, :, :]) ** 2).sum(2))
            out[s:s + chunk] = self._U(d) @ self.W + np.hstack([np.ones((len(q), 1)), q]) @ self.A
        return out


class NormalisedTPS:
    """TPS fitted in a unit-scale frame; the similarity is folded back on evaluation.

    src -> (src - cX)/sX  ==TPS==>  (dst - cY)/sY -> dst
    A similarity is exactly representable by the spline's affine block, so this changes only
    what `lambda` means (it is applied where the point cloud has unit RMS radius, matching the
    generator's canonical convention) — not the family of maps being fitted.
    """

    def __init__(self, src, dst, lambda_reg: float = TPS_LAMBDA, normalise: bool = True):
        src = np.asarray(src, np.float64)
        dst = np.asarray(dst, np.float64)
        if normalise:
            self.cX = src.mean(0)
            self.sX = float(np.sqrt(((src - self.cX) ** 2).sum(1).mean()))
            self.cY = dst.mean(0)
            self.sY = float(np.sqrt(((dst - self.cY) ** 2).sum(1).mean()))
        else:
            self.cX = np.zeros(3); self.sX = 1.0
            self.cY = np.zeros(3); self.sY = 1.0
        self.tps = TPS3D()
        self.tps.fit((src - self.cX) / self.sX, (dst - self.cY) / self.sY, lambda_reg)

    def __call__(self, P):
        return self.tps.evaluate((np.asarray(P, np.float64) - self.cX) / self.sX) * self.sY + self.cY


def load_landmarks(path: Path):
    """(labels, (K,3) positions) from a 3D Slicer markups JSON, defined points only."""
    data = json.loads(Path(path).read_text())
    fid = None
    for m in data.get("markups", []):
        if m.get("type") == "Fiducial":
            fid = m
            break
    if fid is None:
        fid = data["markups"][0]
    labs, pts = [], []
    for cp in fid["controlPoints"]:
        if cp.get("positionStatus", "defined") != "defined":
            continue
        labs.append(str(cp["label"]))
        pts.append([float(v) for v in cp["position"]])
    return labs, np.asarray(pts, np.float64)


# --------------------------------------------------------------------------- descriptors
def axial_profile(lab_faces, areas, t, n_bins: int = N_BINS):
    """Identical arithmetic to `e2_preservation.axial_profile` / `common.native_face_colors`."""
    bi = np.clip((t * n_bins).astype(int), 0, n_bins - 1)
    w = np.bincount(bi, weights=areas, minlength=n_bins)
    w[w == 0] = 1.0
    prof = np.stack([np.bincount(bi, weights=areas * lab_faces[:, c], minlength=n_bins) / w
                     for c in range(3)], axis=1)
    return prof.reshape(-1)


def hist_on_palette(lab_faces, areas, palette):
    """Identical to `e2_preservation.hist_on_palette` (L1-normalised, nearest Lab centroid)."""
    d = ((lab_faces[:, None, :] - palette[None, :, :]) ** 2).sum(-1)
    lbl = np.argmin(d, axis=1)
    h = np.bincount(lbl, weights=areas, minlength=palette.shape[0])
    return h / (h.sum() + 1e-12)


# --------------------------------------------------------------------------- estimators
def donor_blocked(X, y, donor, seed=0):
    """Leave-one-donor-out balanced accuracy (identical to e2b_donorblocked.donor_blocked)."""
    X = np.asarray(X, float)
    y = np.asarray(y)
    per = []
    for d in np.unique(donor):
        te = donor == d
        tr = ~te
        if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
            continue
        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(max_iter=4000, class_weight="balanced",
                                               random_state=seed))
        clf.fit(X[tr], y[tr])
        per.append(balanced_accuracy_score(y[te], clf.predict(X[te])))
    return float(np.mean(per)), float(np.std(per)), len(per)


def dice_matrix(M, areas):
    """(N,N) area-weighted Dice between boolean per-face masks."""
    Mf = M.astype(np.float32)
    Wf = Mf * areas.astype(np.float32)[None, :]
    inter = Mf @ Wf.T                                    # (N,N)
    s = Wf.sum(1)
    den = s[:, None] + s[None, :]
    with np.errstate(invalid="ignore", divide="ignore"):
        D = np.where(den > 0, 2.0 * inter / np.where(den == 0, 1.0, den), np.nan)
    return D, s


# --------------------------------------------------------------------------- main
def main():
    t0 = time.time()
    warn = []

    # ---------------------------------------------------------------- order & ground truth
    fcd, _, amesh = C.atlas_data()
    names = list(fcd.names)
    N = len(names)
    nat = C.native_face_colors(verbose=False)
    assert list(nat["names"]) == names, "native/atlas specimen order mismatch"
    disk = sorted(p.stem for p in C.SRC_IMAGES.glob("*.png"))
    assert disk == names, "fcd.names does not match the on-disk specimen list"
    df, labels = C.load_gt(names)
    assert list(df["sample_tag"]) == names, "GT table not aligned to fcd.names"
    donor = df["specimen_index"].to_numpy().astype(int)

    z = np.load(C.RESULTS / "e2_preservation_arrays.npz", allow_pickle=True)
    assert list(z["names"]) == names, "e2 arrays are in a different specimen order"
    assert (z["donor"] == donor).all(), "e2 donor vector mismatch"
    for f in FACTORS:
        assert (z[f"label_{f}"] == labels[f]).all(), f"label_{f} mismatch vs e2 arrays"
    palette = nat["palette"]
    assert palette.shape == (24, 3)
    print(f"[E2c] {N} specimens, order verified against fcd.names / e2 arrays / samples.csv")

    # ---------------------------------------------------------------- reference specimen
    ref_i = int(np.where(donor == 0)[0][0])
    ref_name = names[ref_i]
    assert ref_name.startswith("Mchenga_m1"), f"reference is not Mchenga_m1: {ref_name}"
    ref_mesh = load_obj(C.SRC_MESHES / f"{ref_name}.obj")
    ref_labs, ref_P = load_landmarks(C.SRC_ROOT / "landmarks" / f"{ref_name}.mrk.json")
    Nf = ref_mesh.n_faces
    ref_areas = ref_mesh.face_areas()
    ref_diag = float(np.linalg.norm(ref_mesh.vertices.max(0) - ref_mesh.vertices.min(0)))
    print(f"[E2c] reference = {ref_name} (donor 0): {ref_mesh.vertices.shape[0]} v / {Nf} f, "
          f"{len(ref_labs)} landmarks, bbox diagonal {ref_diag:.4f}")

    # ---------------------------------------------------------------- landmarks, all 250
    all_lm = [load_landmarks(C.SRC_ROOT / "landmarks" / f"{nm}.mrk.json") for nm in names]
    n_shared = []
    for labs_i, _ in all_lm:
        n_shared.append(len([l for l in ref_labs if l in set(labs_i)]))
    n_shared = np.array(n_shared)
    print(f"[E2c] shared landmark labels with the reference: "
          f"min {n_shared.min()} / median {int(np.median(n_shared))} / max {n_shared.max()}")
    if n_shared.min() < 4:
        warn.append("some specimen shares <4 landmark labels with the reference")

    # landmark files are reused verbatim within a donor -> verify, then exploit
    lm_identical_within_donor = True
    for d in np.unique(donor):
        idx = np.where(donor == d)[0]
        l0, p0 = all_lm[idx[0]]
        for j in idx[1:]:
            lj, pj = all_lm[j]
            if lj != l0 or not np.array_equal(pj, p0):
                lm_identical_within_donor = False
    print(f"[E2c] landmark sets identical within each donor: {lm_identical_within_donor}")

    # ---------------------------------------------------------------- fit 250 TPS warps
    print("[E2c] fitting 250 TPS warps (reference landmarks -> specimen landmarks) ...")
    rmse = np.zeros(N)          # unit-scale fit (the one actually used)
    rmse_world = np.zeros(N)    # same lambda applied in raw scan units, for the record
    warped_by_donor = {}
    warp_disagree = np.zeros(N)
    for i, nm in enumerate(names):
        labs_i, P_i = all_lm[i]
        shared = [l for l in ref_labs if l in set(labs_i)]
        src = ref_P[[ref_labs.index(l) for l in shared]]
        dst = P_i[[labs_i.index(l) for l in shared]]
        warp = NormalisedTPS(src, dst, TPS_LAMBDA, normalise=True)
        rmse[i] = float(np.sqrt(((warp(src) - dst) ** 2).sum(1).mean()))
        warp_w = NormalisedTPS(src, dst, TPS_LAMBDA, normalise=False)
        rmse_world[i] = float(np.sqrt(((warp_w(src) - dst) ** 2).sum(1).mean()))
        d = int(donor[i])
        Vw = warp(ref_mesh.vertices)
        if d not in warped_by_donor:
            warped_by_donor[d] = Vw
        else:
            warp_disagree[i] = float(np.abs(Vw - warped_by_donor[d]).max())
    print(f"[E2c] max within-donor disagreement between independently fitted warps: "
          f"{warp_disagree.max():.3e} (mesh diagonal {ref_diag:.4f})")
    if warp_disagree.max() > 1e-9 * ref_diag:
        warn.append(f"within-donor warps differ by {warp_disagree.max():.2e}")

    rmse_rel = rmse / ref_diag
    same = donor == 0
    print(f"[E2c] TPS landmark-fit RMSE (unit-scale fit, used): reference donor "
          f"{rmse[same].max():.2e} ({100 * rmse_rel[same].max():.6f}% of ref diagonal) | "
          f"other donors median {np.median(rmse[~same]):.2e} max {rmse[~same].max():.2e} "
          f"({100 * rmse_rel[~same].max():.6f}% of ref diagonal)")
    print(f"[E2c]   for the record, the SAME lambda applied in raw scan units over-smooths: "
          f"cross-donor RMSE median {np.median(rmse_world[~same]):.2e} "
          f"({100 * np.median(rmse_world[~same]) / ref_diag:.4f}% of ref diagonal)")
    if rmse[same].max() > 1e-9:
        warn.append("identity warp (reference donor) does not reproduce its own landmarks")
    if rmse_rel.max() > 1e-4:
        warn.append(f"TPS landmark fit RMSE reaches {100 * rmse_rel.max():.4f}% of the "
                    f"mesh diagonal — the spline is not interpolating")

    # ---------------------------------------------------------------- per-donor projection
    print("[E2c] projecting warped reference vertices onto each donor surface ...")
    uv_by_donor, gap_by_donor = {}, {}
    knn_check = {}
    for d in sorted(warped_by_donor):
        nm0 = names[int(np.where(donor == d)[0][0])]
        tmesh = load_obj(C.SRC_MESHES / f"{nm0}.obj")
        tris = tmesh.vertices[tmesh.face_v]
        tuvs = tmesh.uvs[tmesh.face_vt]
        corners = tris.reshape(-1, 3)
        cface = np.repeat(np.arange(tris.shape[0]), 3)
        tree = cKDTree(corners)
        Vw = warped_by_donor[d]
        uv, gap = _closest_point_uv(Vw, tris, tuvs, tree, cface, k=KNN)
        uv_by_donor[d] = uv
        gap_by_donor[d] = gap
        tdiag = float(np.linalg.norm(tmesh.vertices.max(0) - tmesh.vertices.min(0)))
        # convergence proof: a much wider candidate set must find NO closer triangle at all
        _, gap2 = _closest_point_uv(Vw, tris, tuvs, tree, cface, k=KNN_CHECK)
        imp = gap - gap2
        knn_check[int(d)] = {
            "k": KNN, "k_check": KNN_CHECK,
            "max_gap_improvement": float(imp.max()),
            "max_gap_improvement_frac_of_diag": float(imp.max() / tdiag),
            "n_vertices_improved_gt_1e4_diag": int((imp > 1e-4 * tdiag).sum()),
        }
        print(f"   donor {d} ({nm0.rsplit('_fishy', 1)[0]:14s}): gap median "
              f"{np.median(gap) / tdiag * 100:.3f}% p95 {np.percentile(gap, 95) / tdiag * 100:.3f}% "
              f"max {gap.max() / tdiag * 100:.3f}% of its diagonal | k={KNN_CHECK} improves "
              f"{knn_check[int(d)]['n_vertices_improved_gt_1e4_diag']} of "
              f"{len(gap)} vertices (max {imp.max():.1e})")
    n_imp = sum(v["n_vertices_improved_gt_1e4_diag"] for v in knn_check.values())
    n_tot = len(warped_by_donor) * ref_mesh.vertices.shape[0]
    print(f"[E2c] closest-point convergence: k={KNN_CHECK} improves {n_imp}/{n_tot} "
          f"({100 * n_imp / n_tot:.4f}%) of the projections by >1e-4 of a body diagonal")
    if n_imp > 1e-3 * n_tot:
        warn.append(f"closest-point search has not converged at k={KNN}: k={KNN_CHECK} "
                    f"improves {n_imp}/{n_tot} projections")

    gap_all = np.concatenate([gap_by_donor[int(d)] for d in donor])  # per specimen, per vertex
    gap_per_spec_med = np.array([np.median(gap_by_donor[int(d)]) for d in donor])

    # ---------------------------------------------------------------- transfer the textures
    print("[E2c] sampling each specimen's own texture through its landmark warp ...")
    face_v = ref_mesh.face_v
    lab_all = np.zeros((N, Nf, 3), np.float32)
    donor0_dE = []
    for i, nm in enumerate(names):
        tex = imageio.imread(C.SRC_IMAGES / f"{nm}.png")
        vrgb = _sample_tex(tex, uv_by_donor[int(donor[i])])          # (Nv,3) uint8
        frgb = vrgb[face_v].astype(np.float64).mean(1)               # same rule as sample_face_colors
        frgb = np.clip(frgb, 0, 255).astype(np.uint8)
        lab_all[i] = rgb_to_lab(frgb).astype(np.float32)
        if donor[i] == 0:
            native = sample_face_colors(ref_mesh, tex)               # identity warp -> must match
            donor0_dE.append(deltaE2000(rgb_to_lab(native), lab_all[i].astype(np.float64)))
        if i % 50 == 0 or i == N - 1:
            print(f"   {i + 1}/{N} {nm}")
    donor0_dE = np.concatenate(donor0_dE)
    print(f"[E2c] identity-warp check (donor 0, {int(same.sum())} specimens): dE2000 vs the "
          f"native face-colour rule — median {np.median(donor0_dE):.3f}, "
          f"p99 {np.percentile(donor0_dE, 99):.2f}, frac>2.3 {np.mean(donor0_dE > 2.3):.4f}")
    if np.median(donor0_dE) > 0.5:
        warn.append("identity-warp transfer does not reproduce the native face colours")

    # ---------------------------------------------------------------- transport fidelity
    # Independent check that the warp transports COMPOSITION faithfully: the area fraction of
    # each planted pigment measured in the landmark-warped reference frame must track the value
    # measured on the specimen's own native mesh (`common.native_face_colors()['masks']`).
    # This is what catches a warp that smears or duplicates a patch. It also documents a real
    # property of the dataset that would otherwise look like a bug: the reference donor
    # Mchenga_m1 carries a genuinely much smaller painted belly (~7% of body area) than every
    # other donor (~25-35%) — visible identically in the native, atlas and landmark frames — so
    # per-donor area fractions are expected to differ a lot, and cross-donor belly Dice is
    # correspondingly capped.
    print("[E2c] transport fidelity: landmark-frame vs native-frame pigment area fractions")
    lm_area = np.zeros((N, len(C.PIGMENTS)))
    for i in range(N):
        lm_area[i] = C.pigment_area_fractions(lab_all[i].astype(np.float64), ref_areas)
    transport = {}
    for pi_, pig in enumerate(C.PIGMENTS):
        nat_f = nat["masks"][:, pi_]
        lm_f = lm_area[:, pi_]
        ref_grp = donor == 0
        transport[pig] = {
            "identity_donor_native": float(nat_f[ref_grp].mean()),
            "identity_donor_landmark": float(lm_f[ref_grp].mean()),
            "identity_donor_abs_diff": float(np.abs(nat_f[ref_grp] - lm_f[ref_grp]).mean()),
            "other_donors_native": float(nat_f[~ref_grp].mean()),
            "other_donors_landmark": float(lm_f[~ref_grp].mean()),
            "pearson_r_native_vs_landmark": float(np.corrcoef(nat_f, lm_f)[0, 1]),
            "per_donor_native": [float(nat_f[donor == d].mean()) for d in range(7)],
            "per_donor_landmark": [float(lm_f[donor == d].mean()) for d in range(7)],
        }
        print(f"   {pig:7s} identity donor native {transport[pig]['identity_donor_native']:.4f} "
              f"vs landmark {transport[pig]['identity_donor_landmark']:.4f} "
              f"(|diff| {transport[pig]['identity_donor_abs_diff']:.5f}) | other donors "
              f"{transport[pig]['other_donors_native']:.4f} vs "
              f"{transport[pig]['other_donors_landmark']:.4f} | "
              f"r={transport[pig]['pearson_r_native_vs_landmark']:+.3f}")
    if transport["belly"]["identity_donor_abs_diff"] > 1e-3:
        warn.append("identity-warp transport does not reproduce the native belly area fraction")

    # ---------------------------------------------------------------- descriptors
    print("[E2c] building descriptors ...")
    axis = C.principal_axis(ref_mesh)
    cents = ref_mesh.face_centroids()
    tt = (cents - cents.mean(0)) @ axis
    tt = (tt - tt.min()) / (tt.max() - tt.min() + 1e-12)

    lm_hist24 = np.stack([hist_on_palette(lab_all[i].astype(np.float64), ref_areas, palette)
                          for i in range(N)])
    lm_axial96 = np.stack([axial_profile(lab_all[i].astype(np.float64), ref_areas, tt)
                           for i in range(N)])
    flat_idx = np.random.default_rng(42).choice(Nf, size=N_FLAT, replace=False)
    flat_idx.sort()
    lm_flat = lab_all[:, flat_idx, :].astype(np.float64).reshape(N, -1)
    print(f"   lm_hist24 {lm_hist24.shape}  lm_axial96 {lm_axial96.shape}  lm_flat {lm_flat.shape}")

    # ---------------------------------------------------------------- comparison set
    post_flat1200 = features.spatial_flatten(fcd, color_space="lab", subsample=N_FLAT)
    REPS = {
        "lm_hist24": lm_hist24,
        "lm_axial96": lm_axial96,
        "lm_flat": lm_flat,
        "pre_hist": z["pre_hist"],
        "pre_spatial_canon": z["pre_spatial_canon"],
        "post_hist_matched": z["post_hist_matched"],
        "post_axial": z["post_axial"],
        "post_spatial_flat": post_flat1200,
    }
    assert lm_axial96.shape[1] == z["post_axial"].shape[1] == z["pre_spatial_canon"].shape[1]
    assert lm_flat.shape[1] == post_flat1200.shape[1]
    assert lm_hist24.shape[1] == z["pre_hist"].shape[1]

    # ---------------------------------------------------------------- scoring
    print("\n[E2c] scoring (random CV -> leave-one-donor-out; nulls in brackets)")
    hdr = f"{'representation':<20}{'dim':>6}  " + "".join(f"{f:>26}" for f in FACTORS)
    print(hdr)
    print("-" * len(hdr))
    table, detail = {}, {}
    rng_null = np.random.default_rng(0)
    perm_r = {f: rng_null.permutation(N) for f in FACTORS}
    perm_d = {f: np.random.default_rng(1).permutation(N) for f in FACTORS}
    for r, M in REPS.items():
        M = np.asarray(M, float)
        row, det, line = {}, {}, f"{r:<20}{M.shape[1]:>6}  "
        for f in FACTORS:
            y = labels[f]
            a, sd = C.balanced_cv(M, y)
            b, bsd, nfold = donor_blocked(M, y, donor)
            an, _ = C.balanced_cv(M, y[perm_r[f]])
            bn, _, _ = donor_blocked(M, y[perm_d[f]], donor)
            row[f] = {"random_cv": round(a, 4), "donor_blocked": round(b, 4)}
            det[f] = {"random_cv": a, "random_cv_sd": sd, "random_cv_null": an,
                      "donor_blocked": b, "donor_blocked_sd": bsd, "donor_blocked_null": bn,
                      "n_folds": nfold}
            line += f"{a:.3f}[{an:.2f}] -> {b:.3f}[{bn:.2f}]".rjust(26)
        table[r], detail[r] = row, det
        print(line)

    # ---------------------------------------------------------------- Dice, landmark frame
    print("\n[E2c] planted-pigment Dice in the landmark-warped frame (e1b protocol)")
    A = {k: np.zeros((N, Nf), bool) for k in PIG_KEYS}
    for i in range(N):
        m = C.pigment_masks(lab_all[i].astype(np.float64))
        for k in PIG_KEYS:
            A[k][i] = m[k]

    rng = np.random.default_rng(1)          # same seed / #pairs as e1b_correspondence.py
    npairs = 6000
    pi = rng.integers(0, N, npairs)
    pj = rng.integers(0, N, npairs)
    ok = pi != pj
    pi, pj = pi[ok], pj[ok]
    same_pair = donor[pi] == donor[pj]

    try:
        e1b = json.loads((C.RESULTS / "e1b_correspondence.json").read_text())["correspondence"]
    except Exception:
        e1b = {}

    dice_out, dice_short = {}, {}
    for k in PIG_KEYS:
        if k == "cheek":
            has = (labels["cheeks"] == 1) & (A[k].sum(1) > 0)
            sel = has[pi] & has[pj]
        else:
            sel = np.ones(len(pi), bool)
        D, s = dice_matrix(A[k], ref_areas)
        fa = s / ref_areas.sum()
        x, y = pi[sel], pj[sel]
        a = D[x, y]
        ch = np.where(fa[x] + fa[y] > 0, 2 * fa[x] * fa[y] / (fa[x] + fa[y] + 1e-300), np.nan)
        sm = same_pair[sel]
        # the reference donor is transported by the identity warp -> flag it explicitly
        refpair = (donor[x] == 0) & (donor[y] == 0)
        rec = {
            "landmark_dice": float(np.nanmean(a)), "landmark_dice_sd": float(np.nanstd(a)),
            "landmark_same_donor": float(np.nanmean(a[sm])),
            "landmark_diff_donor": float(np.nanmean(a[~sm])),
            "landmark_same_donor_reference_only": float(np.nanmean(a[refpair]))
            if refpair.any() else None,
            "landmark_same_donor_excl_reference": float(np.nanmean(a[sm & ~refpair]))
            if (sm & ~refpair).any() else None,
            "transfer_efficiency": float(np.nanmean(a[~sm]) / np.nanmean(a[sm])),
            "chance_dice": float(np.nanmean(ch)),
            "area_frac_mean": float(fa.mean()), "n_pairs": int(sel.sum()), "n_same": int(sm.sum()),
        }
        if k in e1b:
            rec["atlas_dice"] = e1b[k]["atlas_dice"]
            rec["atlas_same_donor"] = e1b[k]["atlas_same_donor"]
            rec["atlas_diff_donor"] = e1b[k]["atlas_diff_donor"]
            rec["atlas_chance_dice"] = e1b[k]["chance_dice"]
            rec["nativeUV_dice"] = e1b[k]["nativeUV_dice"]
        dice_out[k] = rec
        dice_short[k] = {"landmark_diff_donor": round(rec["landmark_diff_donor"], 4),
                         "landmark_same_donor": round(rec["landmark_same_donor"], 4),
                         "chance": round(rec["chance_dice"], 4)}
        print(f"  {k:7s} landmark {rec['landmark_dice']:.3f} "
              f"(same-donor {rec['landmark_same_donor']:.3f}, cross-donor "
              f"{rec['landmark_diff_donor']:.3f}) | atlas cross-donor "
              f"{rec.get('atlas_diff_donor', float('nan')):.3f} | chance {rec['chance_dice']:.3f}")

    # ---------------------------------------------------------------- save
    out = {
        "reference_specimen": ref_name,
        "reference_donor_index": 0,
        "n_specimens": N, "n_reference_faces": int(Nf),
        "n_reference_vertices": int(ref_mesh.vertices.shape[0]),
        "tps": {
            "lambda": TPS_LAMBDA, "kernel": "r^2 log(r+1e-12)",
            "fitted_in": "unit-RMS-radius frame (generator's canonical convention)",
            "rmse_world_units_same_lambda_median": float(np.median(rmse_world[~same])),
            "rmse_world_units_same_lambda_frac_of_diagonal": float(
                np.median(rmse_world[~same]) / ref_diag),
            "n_shared_landmarks_min": int(n_shared.min()),
            "n_shared_landmarks_max": int(n_shared.max()),
            "landmarks_identical_within_donor": bool(lm_identical_within_donor),
            "max_within_donor_warp_disagreement": float(warp_disagree.max()),
            "rmse_reference_donor_max": float(rmse[same].max()),
            "rmse_other_donors_median": float(np.median(rmse[~same])),
            "rmse_other_donors_max": float(rmse[~same].max()),
            "rmse_max_frac_of_reference_diagonal": float(rmse_rel.max()),
            "reference_bbox_diagonal": ref_diag,
        },
        "projection": {
            "k_neighbours": KNN,
            "knn_robustness_per_donor": knn_check,
            "n_projections_improved_by_wider_k": int(n_imp),
            "n_projections_total": int(n_tot),
            "gap_median_frac_of_ref_diag": float(np.median(gap_all) / ref_diag),
            "gap_p95_frac_of_ref_diag": float(np.percentile(gap_all, 95) / ref_diag),
            "gap_max_frac_of_ref_diag": float(gap_all.max() / ref_diag),
            "identity_warp_dE_median": float(np.median(donor0_dE)),
            "identity_warp_dE_p99": float(np.percentile(donor0_dE, 99)),
            "identity_warp_frac_dE_gt_2p3": float(np.mean(donor0_dE > 2.3)),
        },
        "transport_fidelity": transport,
        "table": table,
        "detail": detail,
        "dice": dice_out,
        "warnings": warn,
        "seconds": round(time.time() - t0, 1),
    }
    (C.RESULTS / "e2c_landmark_baseline.json").write_text(json.dumps(out, indent=2))
    np.savez_compressed(
        C.RESULTS / "e2c_landmark_baseline_arrays.npz",
        names=np.array(names), donor=donor,
        lm_hist24=lm_hist24, lm_axial96=lm_axial96, lm_flat=lm_flat,
        flat_face_idx=flat_idx, ref_areas=ref_areas, ref_axial_t=tt,
        ref_face_centroids=cents.astype(np.float32), palette=palette,
        ref_name=np.array(ref_name), tps_rmse=rmse, tps_rmse_world_units=rmse_world,
        lm_pigment_area_frac=lm_area, native_pigment_area_frac=nat["masks"],
        gap_median_per_specimen=gap_per_spec_med,
        lm_face_lab_mean=lab_all.mean(0).astype(np.float32),
        mask_belly=A["belly"], mask_tail=A["tail"], mask_cheek=A["cheek"],
        mask_stripe=A["stripe"],
        occ_belly=A["belly"].mean(0).astype(np.float32),
        occ_tail=A["tail"].mean(0).astype(np.float32),
        occ_cheek=A["cheek"].mean(0).astype(np.float32),
        occ_stripe=A["stripe"].mean(0).astype(np.float32),
    )
    print("\nwarnings:", warn if warn else "none")
    print(f"saved results/e2c_landmark_baseline.json  ({out['seconds']}s)")


if __name__ == "__main__":
    main()
