"""Dataset loading, per-face color caching, and ground-truth labels."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from . import config
from .mesh import Mesh, load_obj, sample_face_colors


# ---------------------------------------------------------------------------
# Mesh + per-face geometry (constant across all specimens — shared mesh)
# ---------------------------------------------------------------------------
def load_mesh() -> Mesh:
    return load_obj(config.MESH_PATH)


def texture_paths() -> list[Path]:
    paths = sorted(config.DATA_DIR.glob(config.TEXTURE_GLOB))
    if not paths:
        raise FileNotFoundError(f"No textures matching {config.TEXTURE_GLOB} in {config.DATA_DIR}")
    return paths


# ---------------------------------------------------------------------------
# Per-face color tensor: (N_specimens, N_faces, 3) uint8  — cached
# ---------------------------------------------------------------------------
@dataclass
class FaceColorData:
    colors: np.ndarray       # (N, Nf, 3) uint8  RGB face-average colors
    areas: np.ndarray        # (Nf,) float64  3D face areas (shared)
    centroids: np.ndarray    # (Nf, 3) float64 3D face centroids (shared)
    names: list[str]         # specimen names (len N)


def build_face_colors(force: bool = False, verbose: bool = True) -> FaceColorData:
    """Sample every texture into per-face colors and cache to disk."""
    colors_path = config.CACHE_DIR / "face_colors_u8.npy"
    areas_path = config.CACHE_DIR / "face_areas.npy"
    cents_path = config.CACHE_DIR / "face_centroids.npy"
    names_path = config.CACHE_DIR / "specimen_names.npy"

    if not force and all(p.exists() for p in (colors_path, areas_path, cents_path, names_path)):
        return FaceColorData(
            colors=np.load(colors_path),
            areas=np.load(areas_path),
            centroids=np.load(cents_path),
            names=list(np.load(names_path)),
        )

    mesh = load_mesh()
    areas = mesh.face_areas()
    centroids = mesh.face_centroids()
    paths = texture_paths()

    n, nf = len(paths), mesh.n_faces
    colors = np.empty((n, nf, 3), dtype=np.uint8)
    names = []
    for i, p in enumerate(paths):
        tex = imageio.imread(p)
        colors[i] = sample_face_colors(mesh, tex)
        names.append(p.stem)
        if verbose and (i % 25 == 0 or i == n - 1):
            print(f"  sampled {i + 1}/{n}: {p.name}")

    np.save(colors_path, colors)
    np.save(areas_path, areas)
    np.save(cents_path, centroids)
    np.save(names_path, np.array(names))
    return FaceColorData(colors=colors, areas=areas, centroids=centroids, names=names)


# ---------------------------------------------------------------------------
# Ground truth
# ---------------------------------------------------------------------------
@dataclass
class GroundTruth:
    params: pd.DataFrame          # raw generating parameters (N rows)
    labels: dict[str, np.ndarray] # factor name -> int label array (N,)
    n_classes: dict[str, int]


def _kmeans_1d_labels(x: np.ndarray, k: int = 2) -> np.ndarray:
    km = KMeans(n_clusters=k, n_init=10, random_state=0)
    lab = km.fit_predict(x.reshape(-1, 1))
    # canonical ordering: relabel so cluster ids increase with center value
    order = np.argsort(km.cluster_centers_.ravel())
    remap = np.zeros(k, dtype=int)
    remap[order] = np.arange(k)
    return remap[lab]


def load_ground_truth(n: int | None = None) -> GroundTruth:
    df = pd.read_csv(config.GROUND_TRUTH_CSV)
    if n is not None:
        df = df.iloc[:n].reset_index(drop=True)

    labels: dict[str, np.ndarray] = {}
    # belly / tail hue: recover the 2 planted blobs via 1-D k-means on hue.
    labels["belly"] = _kmeans_1d_labels(df["belly_hue"].to_numpy(), 2)
    labels["tail"] = _kmeans_1d_labels(df["tail_hue"].to_numpy(), 2)
    # stripe count: 4 -> 0, 5 -> 1
    labels["stripe"] = (df["stripe_count"].to_numpy() == 5).astype(int)
    # rosy cheeks present: minority class
    labels["cheeks"] = df["rosy_cheeks_present"].to_numpy().astype(int)

    n_classes = {k: int(v.max() + 1) for k, v in labels.items()}
    return GroundTruth(params=df, labels=labels, n_classes=n_classes)


def joint_label(gt: GroundTruth, factors=("belly", "tail", "stripe")) -> np.ndarray:
    """Combine several binary factors into a single categorical label (for ARI)."""
    out = np.zeros(len(gt.params), dtype=int)
    for f in factors:
        out = out * gt.n_classes[f] + gt.labels[f]
    return out
