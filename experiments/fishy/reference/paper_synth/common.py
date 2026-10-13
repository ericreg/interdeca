"""Shared harness for the SYNTHETIC CICHLID TEXTURE DATASET validation (paper section).

Two data states of the *same* 250 synthetic specimens:

  PRE-module  : native decimated cichlid mesh (7 distinct shapes) + its own synthetic
                2048^2 texture, in its own UV parameterisation.
                -> /mnt/data/ml_data/cichlid-synth-decimated/{meshes,images}

  POST-module : the shared ColorAtlas mean-shape atlas (24,999 v / 49,997 f, one UV
                layout) + 250 atlas-space baked textures produced by the module run.
                -> .../mussels/out/2026_07-16_20_37_11/colorAnalysis/{atlasModelUV.obj,
                   atlasTextures}

Ground truth = the generator's per-sample parameter table (samples.csv), joined by
`sample_tag` == texture filename stem.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

_FR = Path(__file__).resolve().parents[1]           # fishy_research/
if str(_FR / "src") not in sys.path:
    sys.path.insert(0, str(_FR / "src"))

from fishpipe import config as _cfg                                       # noqa: E402
from fishpipe.data import FaceColorData                                   # noqa: E402
from fishpipe.dataset import Dataset, build_face_colors                   # noqa: E402
from fishpipe.features import rgb_to_lab                                  # noqa: E402
from fishpipe.mesh import load_obj, sample_face_colors                    # noqa: E402

# --------------------------------------------------------------------------- paths
HERE = Path(__file__).resolve().parent
CACHE = HERE / "cache"
RESULTS = HERE / "results"
FIGURES = HERE / "figures"
for _d in (CACHE, RESULTS, FIGURES):
    _d.mkdir(parents=True, exist_ok=True)

SRC_ROOT = Path(os.environ.get("SYNTH_SRC", "/mnt/data/ml_data/cichlid-synth-decimated"))
SRC_MESHES = SRC_ROOT / "meshes"
SRC_IMAGES = SRC_ROOT / "images"
GT_CSV = SRC_ROOT / "samples.csv"

MOD_RUN = Path(os.environ.get(
    "SYNTH_MODULE_RUN",
    "/mnt/data/ml_data/color-modeling-pub/mussels/out/2026_07-16_20_37_11",
))
ATLAS_DIR = MOD_RUN / "colorAnalysis"
ATLAS_MESH = ATLAS_DIR / "atlasModelUV.obj"
ATLAS_TEX = ATLAS_DIR / "atlasTextures"
RESAMPLED_OBJ = ATLAS_DIR / "resampledOBJ_withUV"

ATLAS_DS = Dataset(
    name="cichlid_synth_atlas",
    mesh_path=ATLAS_MESH,
    texture_dir=ATLAS_TEX,
    exclude=("average",),
    cache_dir=CACHE / "atlas",
)

# --------------------------------------------------------------------------- GT
FACTORS = ("belly", "tail", "stripe", "cheeks")
FACTOR_LABEL = {
    "belly": "belly hue (2 groups)",
    "tail": "tail hue (2 groups)",
    "stripe": "stripe count (4 vs 5)",
    "cheeks": "rosy cheeks (present)",
}


def _kmeans_1d(x: np.ndarray, k: int = 2) -> np.ndarray:
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=k, n_init=10, random_state=0)
    lab = km.fit_predict(np.asarray(x, float).reshape(-1, 1))
    order = np.argsort(km.cluster_centers_.ravel())
    remap = np.zeros(k, dtype=int)
    remap[order] = np.arange(k)
    return remap[lab]


def load_gt(names: list[str]):
    """Return (params DataFrame reindexed to `names`, labels dict)."""
    import pandas as pd
    df = pd.read_csv(GT_CSV)
    df = df.set_index("sample_tag")
    missing = [n for n in names if n not in df.index]
    if missing:
        raise KeyError(f"{len(missing)} names absent from samples.csv, e.g. {missing[:3]}")
    df = df.loc[names].reset_index()
    labels = {
        "belly": _kmeans_1d(df["belly_hue"].to_numpy()),
        "tail": _kmeans_1d(df["tail_hue"].to_numpy()),
        "stripe": (df["stripe_count"].to_numpy() == 5).astype(int),
        "cheeks": df["rosy_cheeks_present"].to_numpy().astype(int),
    }
    return df, labels


def joint_label(labels: dict, factors=("belly", "tail", "stripe")) -> np.ndarray:
    out = np.zeros(len(next(iter(labels.values()))), dtype=int)
    for f in factors:
        out = out * 2 + labels[f]
    return out


# --------------------------------------------------------------------------- atlas data
def atlas_data(force: bool = False, impute: bool = False):
    """(FaceColorData, artifact_mask, mesh) for the module's atlas-space textures.

    NOTE: `impute=False` by default here — for the fidelity analysis we want to *measure*
    bake artifacts, not silently repair them. Downstream analyses may re-run with imputation.
    """
    fcd, art = build_face_colors(ATLAS_DS, force=force, impute_artifacts=impute, verbose=True)
    mesh = ATLAS_DS.load_mesh()
    return fcd, art, mesh


def atlas_mesh():
    return load_obj(ATLAS_MESH)


# --------------------------------------------------------------------------- native data
def native_face_colors(force: bool = False, verbose: bool = True):
    """PRE-module per-specimen face colours on each specimen's OWN mesh.

    Meshes differ between the 7 donor specimens, so this cannot be a single (N, Nf, 3)
    tensor. We store, per specimen, the ragged per-face Lab colours + areas summarised into
    the quantities every mesh-independent analysis needs:
      * `hist`     : area-weighted colour composition over a SHARED Lab palette (fit once
                     on pooled colours) — the module's own `area_hist` statistic, computed
                     without any atlas.
      * `lab_mean` : area-weighted mean Lab.
      * `masks`    : area fraction of each planted feature (belly/tail/stripe/cheek) by
                     colour rule — a mesh-free readout of what the texture contains.
    Cached to CACHE/native/.
    """
    import imageio.v2 as imageio
    from sklearn.cluster import MiniBatchKMeans

    cdir = CACHE / "native"
    cdir.mkdir(parents=True, exist_ok=True)
    keys = ("names", "hist", "lab_mean", "masks", "palette", "spatial")
    paths = {k: cdir / f"{k}.npy" for k in keys}
    if not force and all(p.exists() for p in paths.values()):
        return {k: np.load(paths[k], allow_pickle=True) for k in keys}

    names = sorted(p.stem for p in SRC_IMAGES.glob("*.png"))
    import pandas as pd
    gt = pd.read_csv(GT_CSV).set_index("sample_tag").loc[names].reset_index()

    n_bins = 32
    # ---- pass 1: pooled sample of Lab colours to fit the shared palette
    rng = np.random.default_rng(0)
    pool = []
    per_spec = {}
    for i, nm in enumerate(names):
        mesh = load_obj(SRC_MESHES / f"{nm}.obj")
        tex = imageio.imread(SRC_IMAGES / f"{nm}.png")
        rgb = sample_face_colors(mesh, tex)                 # (Nf,3) uint8
        areas = mesh.face_areas().astype(np.float64)
        lab = rgb_to_lab(rgb).astype(np.float64)
        cents = mesh.face_centroids()
        # each specimen's OWN principal (antero-posterior) axis — the best spatial frame an
        # analyst has WITHOUT an atlas (no cross-specimen homology, only self-normalised position)
        cc = cents - cents.mean(0)
        _, _, vt = np.linalg.svd(cc, full_matrices=False)
        t = cc @ vt[0]
        t = (t - t.min()) / (t.max() - t.min() + 1e-12)
        per_spec[nm] = (lab, areas, t)
        idx = rng.choice(len(lab), size=min(3000, len(lab)), replace=False)
        pool.append(lab[idx])
        if verbose and (i % 25 == 0 or i == len(names) - 1):
            print(f"  native {i + 1}/{len(names)}: {nm}")
    pool = np.concatenate(pool, 0)
    km = MiniBatchKMeans(n_clusters=24, random_state=0, n_init=10, batch_size=4096).fit(pool)
    palette = km.cluster_centers_

    hist = np.zeros((len(names), 24))
    lab_mean = np.zeros((len(names), 3))
    masks = np.zeros((len(names), 5))
    spatial = np.zeros((len(names), n_bins * 3))
    for i, nm in enumerate(names):
        lab, areas, t = per_spec[nm]
        lbl = km.predict(lab)
        h = np.bincount(lbl, weights=areas, minlength=24)
        hist[i] = h / (h.sum() + 1e-12)
        lab_mean[i] = (lab * areas[:, None]).sum(0) / areas.sum()
        masks[i] = pigment_area_fractions(lab, areas, pigment_lab(gt.iloc[i]))
        bi = np.clip((t * n_bins).astype(int), 0, n_bins - 1)
        w = np.bincount(bi, weights=areas, minlength=n_bins)
        w[w == 0] = 1.0
        prof = np.stack([np.bincount(bi, weights=areas * lab[:, c], minlength=n_bins) / w
                         for c in range(3)], axis=1)
        spatial[i] = prof.reshape(-1)

    out = {"names": np.array(names), "hist": hist, "lab_mean": lab_mean,
           "masks": masks, "palette": palette, "spatial": spatial}
    for k, v in out.items():
        np.save(paths[k], v)
    return out


# --------------------------------------------------------------------------- planted-pigment recovery
# The generator paints five pigments whose exact colours are recorded per specimen in
# samples.csv (base body, belly, tail, cheek + pure black stripes). We therefore do not need
# ad-hoc colour thresholds: each surface element is assigned to its NEAREST planted pigment
# in CIE L*a*b*. The identical rule is applied to native and atlas-space textures, which is
# what makes the pre/post comparison fair.
import colorsys as _colorsys                                                   # noqa: E402

PIGMENTS = ("base", "belly", "tail", "cheek", "stripe")
# generator palette anchors (color_fishy_tps.py)
_BLUE = _colorsys.rgb_to_hsv(39 / 255, 70 / 255, 144 / 255)
_YELLOW = _colorsys.rgb_to_hsv(242 / 255, 255 / 255, 73 / 255)
_GREEN = _colorsys.rgb_to_hsv(21 / 255, 127 / 255, 31 / 255)
_RED = _colorsys.rgb_to_hsv(1.0, 102 / 255, 99 / 255)


def pigment_rgb(row) -> np.ndarray:
    """(5, 3) float RGB in [0,1] of this specimen's planted pigments, in PIGMENTS order."""
    base = _colorsys.hsv_to_rgb(float(row["base_color_hue"]), float(row["base_color_sat"]),
                                float(row["base_color_val"]))
    belly = _colorsys.hsv_to_rgb(float(row["belly_hue"]), _BLUE[1], _BLUE[2])
    tail = _colorsys.hsv_to_rgb(float(row["tail_hue"]), _GREEN[1], _GREEN[2])
    cheek = _colorsys.hsv_to_rgb(float(row["rosy_cheeks_hue"]), _RED[1], _RED[2])
    return np.array([base, belly, tail, cheek, (0.0, 0.0, 0.0)])


def pigment_lab(row) -> np.ndarray:
    from skimage import color as skcolor
    return skcolor.rgb2lab(pigment_rgb(row)[None, :, :])[0]


def pigment_masks(lab: np.ndarray, refs_lab: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """Boolean masks of the four painted patches, from CIE L*a*b* alone.

    The five planted pigments are separated by construction in the (a*, b*) plane — over the
    whole population the base body is the only pigment with strongly positive b* AND negative
    a*, the belly (blue-to-purple mixture) is the only one with negative b*, the tail is the
    only strongly negative a*, the cheek the only positive a* with positive b*, and stripes
    are pure black. The rules below are therefore population-level constants, not tuned
    thresholds, and are applied identically to native and atlas-space textures.
    `refs_lab` (per-specimen exact pigment colours) is accepted for provenance but not needed.
    """
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    return {
        "stripe": L < 30,
        "belly": (b < -3) & (L >= 30),
        "tail": (a < -28) & (b >= 0) & (L >= 30),
        "cheek": (a > 5) & (b >= 0) & (L >= 30),
        "base": (a <= 5) & (a >= -28) & (b >= 0) & (L >= 30),
    }


def pigment_area_fractions(lab: np.ndarray, areas: np.ndarray,
                           refs_lab: np.ndarray | None = None) -> np.ndarray:
    m = pigment_masks(lab, refs_lab)
    tot = areas.sum()
    return np.array([(areas * m[p]).sum() / tot for p in PIGMENTS])


# --------------------------------------------------------------------------- misc
def face_adjacency(mesh, cache_dir: Path):
    """Build/caches the face-adjacency sparse matrix used by textons/segment."""
    import scipy.sparse as sp
    p = Path(cache_dir) / "face_adjacency.npz"
    if p.exists():
        return sp.load_npz(p).tocsr()
    fv = mesh.face_v
    nf = fv.shape[0]
    # edge -> faces
    e = np.concatenate([fv[:, [0, 1]], fv[:, [1, 2]], fv[:, [2, 0]]], axis=0)
    e = np.sort(e, axis=1)
    fid = np.tile(np.arange(nf), 3)
    order = np.lexsort((e[:, 1], e[:, 0]))
    e, fid = e[order], fid[order]
    same = np.all(e[1:] == e[:-1], axis=1)
    i = fid[:-1][same]
    j = fid[1:][same]
    A = sp.coo_matrix((np.ones(len(i)), (i, j)), shape=(nf, nf))
    A = ((A + A.T) > 0).astype(np.float64).tocsr()
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    sp.save_npz(p, A.tocoo())
    return A


def region_labels(mesh, n_regions: int = 128, seed: int = 0) -> np.ndarray:
    """Atlas-safe replacement for fishpipe.features._region_labels (which is hard-wired to
    the fishy cache). KMeans partition of atlas faces by 3-D centroid."""
    from sklearn.cluster import KMeans
    p = CACHE / "atlas" / f"region_labels_{n_regions}.npy"
    if p.exists():
        return np.load(p)
    cents = mesh.face_centroids()
    lab = KMeans(n_clusters=n_regions, n_init=4, random_state=seed).fit_predict(cents).astype(np.int32)
    np.save(p, lab)
    return lab


def principal_axis(mesh) -> np.ndarray:
    """Unit vector of the atlas's dominant (antero-posterior) axis, from face centroids."""
    c = mesh.face_centroids()
    c = c - c.mean(0)
    _, _, vt = np.linalg.svd(c, full_matrices=False)
    return vt[0]


def region_summary(fcd, mesh, n_regions: int = 128, stat: str = "mean") -> np.ndarray:
    """Atlas-safe per-region colour summary -> (N, n_regions*3) in Lab.

    stat='mean' (area-weighted) or 'maxchroma' (colour of the region's most chromatic face —
    area-independent, so a small vivid patch is not diluted).
    """
    reg = region_labels(mesh, n_regions)
    lab = rgb_to_lab(fcd.colors)
    N, Nf, Ch = lab.shape
    areas = fcd.areas
    out = np.zeros((N, n_regions, Ch))
    if stat == "mean":
        wsum = np.bincount(reg, weights=areas, minlength=n_regions)
        wsum[wsum == 0] = 1.0
        for i in range(N):
            for c in range(Ch):
                out[i, :, c] = np.bincount(reg, weights=areas * lab[i, :, c],
                                           minlength=n_regions) / wsum
    elif stat == "maxchroma":
        chroma = np.sqrt((lab[..., 1:] ** 2).sum(-1))
        idx = [np.where(reg == r)[0] for r in range(n_regions)]
        for i in range(N):
            for r, fr in enumerate(idx):
                if fr.size:
                    out[i, r] = lab[i, fr[np.argmax(chroma[i, fr])]]
    else:
        raise ValueError(stat)
    return out.reshape(N, -1)


# --------------------------------------------------------------------------- orientation
# Every rendered fish in every figure is drawn the same way: lateral view, head to the RIGHT,
# dorsal surface UP. The convention is derived from the landmarks rather than from the image,
# so it is identical for the atlas, for the donor scans and for the painted specimens.
FRONT_LM = ("FIT", "FOB", "FIL", "FIR", "FIB", "FOL", "FOR", "FOT")   # head
BACK_LM = ("TT", "TV", "TA", "TFE")                                    # caudal
TOP_LM = ("TFH", "TFT")                                                # dorsal
BOTTOM_LM = ("LBWL", "RBWR", "RBWL", "LBWR", "AH")                      # ventral


def _read_markups(path):
    import json as _json
    d = _json.loads(Path(path).read_text())
    cp = d["markups"][0]["controlPoints"]
    return np.array([c["position"] for c in cp], float), [c.get("label") for c in cp]


def landmark_labels() -> list[str]:
    """The 47 landmark labels, in file order (identical across all specimens and the atlas)."""
    p = sorted((SRC_ROOT / "landmarks").glob("*.mrk.json"))[0]
    return _read_markups(p)[1]


def specimen_landmarks(name: str) -> np.ndarray:
    return _read_markups(SRC_ROOT / "landmarks" / f"{name}.mrk.json")[0]


def atlas_landmarks() -> np.ndarray:
    """Atlas landmarks in the atlas mesh's coordinate frame (the file is LPS, the mesh RAS)."""
    P, _ = _read_markups(ATLAS_DIR / "atlasLM.mrk.json")
    return P * np.array([-1.0, -1.0, 1.0])


def orientation(mesh, lm: np.ndarray):
    """(h, v, d, flip_h, flip_v) giving a lateral view with the head right and the back up.

    The projection axes are chosen from the landmarks, not from the bounding box: the
    horizontal axis is the one along which the anterior and posterior landmark groups differ
    most, the vertical axis the one along which the dorsal and ventral groups differ most, and
    the remaining axis is depth. Bounding-box ordering is unreliable here because a cichlid's
    height and width are nearly equal, which is what made donor scans come out inconsistently
    oriented.
    """
    labels = landmark_labels()
    idx = {lab: i for i, lab in enumerate(labels)}

    def grp(names):
        k = [idx[n] for n in names if n in idx]
        return lm[k].mean(0) if k else None

    front, back = grp(FRONT_LM), grp(BACK_LM)
    top, bot = grp(TOP_LM), grp(BOTTOM_LM)
    if front is None or back is None or top is None or bot is None:
        h, v, d = atlas_axes(mesh)
        return h, v, d, False, False
    ap = np.abs(front - back)
    h = int(np.argmax(ap))
    dv = np.abs(top - bot)
    rest = [i for i in range(3) if i != h]
    v = int(rest[int(np.argmax(dv[rest]))])
    d = int([i for i in range(3) if i not in (h, v)][0])
    return h, v, d, bool(front[h] < back[h]), bool(top[v] < bot[v])


def orient_image(img: np.ndarray, flip_h: bool, flip_v: bool) -> np.ndarray:
    if flip_h:
        img = img[:, ::-1]
    if flip_v:
        img = img[::-1]
    return img


def map_point(x: float, y: float, bounds, flip_h: bool, flip_v: bool):
    """Mesh (h, v) coordinates -> the coordinates of the same surface point in a flipped
    image drawn with `extent=bounds`. Use this for annotations; do NOT flip the extent
    itself, because imshow would then undo the image flip."""
    if flip_h:
        x = bounds[0] + bounds[1] - x
    if flip_v:
        y = bounds[2] + bounds[3] - y
    return x, y


def atlas_axes(mesh):
    """(h, v, d) axis indices giving a registered lateral view of the atlas."""
    ext = mesh.vertices.max(0) - mesh.vertices.min(0)
    h, v, d = np.argsort(-ext)
    return int(h), int(v), int(d)


def thumbnails(px: int = 260, force: bool = False, which: str = "atlas"):
    """Cached RGBA lateral renders of every specimen on the shared atlas (registered frame).

    `which='atlas'` renders each specimen's atlas-space colours on the mean atlas shape (all
    frames identical in silhouette, so a scatter of them reads as one population).
    `which='native'` renders each specimen's own decimated donor mesh with its own texture —
    the pre-module view, where every silhouette differs.
    """
    import imageio.v2 as imageio
    from fishpipe.render import render_textured_rgba
    p = CACHE / f"thumbs_{which}_{px}.npz"
    if p.exists() and not force:
        z = np.load(p, allow_pickle=True)
        return list(z["names"]), z["imgs"]

    if which == "atlas":
        fcd, _, mesh = atlas_data()
        h, v, d, fh, fv = orientation(mesh, atlas_landmarks())
        lo = mesh.vertices.min(0); hi = mesh.vertices.max(0)
        pad = 0.02 * (hi - lo)
        bounds = (lo[h] - pad[h], hi[h] + pad[h], lo[v] - pad[v], hi[v] + pad[v])
        names = list(fcd.names)
        imgs = []
        for i in range(len(names)):
            im = render_textured_rgba(mesh, fcd.colors[i], px=px, h_axis=h, v_axis=v,
                                      depth_axis=d, cull=1.0, bounds=bounds, supersample=3,
                                      crop=False, align=False)
            imgs.append(orient_image(im[::3, ::3], fh, fv))
            if i % 50 == 0:
                print(f"  thumb {i}/{len(names)}")
    else:
        names = sorted(q.stem for q in SRC_IMAGES.glob("*.png"))
        imgs = []
        for i, nm in enumerate(names):
            m = load_obj(SRC_MESHES / f"{nm}.obj")
            tex = imageio.imread(SRC_IMAGES / f"{nm}.png")
            cols = sample_face_colors(m, tex)
            h, v, d, fh, fv = orientation(m, specimen_landmarks(nm))
            im = render_textured_rgba(m, cols, px=px, h_axis=h, v_axis=v, depth_axis=d,
                                      cull=1.0, supersample=3, crop=True, align=False)
            imgs.append(orient_image(im[::3, ::3], fh, fv))
            if i % 50 == 0:
                print(f"  native thumb {i}/{len(names)}")

    # pad to a common canvas so they stack into one array
    H = max(im.shape[0] for im in imgs)
    W = max(im.shape[1] for im in imgs)
    out = np.zeros((len(imgs), H, W, 4), np.uint8)
    for i, im in enumerate(imgs):
        y0 = (H - im.shape[0]) // 2
        x0 = (W - im.shape[1]) // 2
        out[i, y0:y0 + im.shape[0], x0:x0 + im.shape[1]] = im
    np.savez_compressed(p, names=np.array(names), imgs=out)
    return names, out


def balanced_cv(X, y, n_splits=5, n_repeats=4, seed=0, C=1.0):
    """Repeated stratified balanced accuracy of a standardised logistic classifier."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.model_selection import RepeatedStratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    X = np.asarray(X, float)
    y = np.asarray(y)
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    scores = []
    for tr, te in cv.split(X, y):
        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=4000, C=C, class_weight="balanced"),
        )
        clf.fit(X[tr], y[tr])
        scores.append(balanced_accuracy_score(y[te], clf.predict(X[te])))
    return float(np.mean(scores)), float(np.std(scores))
