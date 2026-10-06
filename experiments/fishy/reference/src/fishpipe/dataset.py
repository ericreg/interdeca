"""Generic dataset loader: any shared-atlas-mesh + per-specimen-textures dataset.

Generalizes the fishy-specific `data.py` to arbitrary module outputs (e.g. the mussel
`colorAnalysis/` with `atlasModelUV.obj` + `atlasTextures/`). Crucially, it handles
**baking artifacts** (black patches where Blender failed to find a source) by
**population-median imputation**: because every specimen shares the atlas mesh, each face
has a distribution of colors across specimens, so a per-specimen black-bake face can be
replaced by that face's median color over the clean specimens — removing the artifact with
no special-casing downstream.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from .data import FaceColorData
from .features import rgb_to_lab
from .mesh import Mesh, load_obj, sample_face_colors


@dataclass
class Dataset:
    name: str
    mesh_path: str | Path
    texture_dir: str | Path
    texture_glob: str = "*.png"
    exclude: tuple[str, ...] = ("average",)        # substrings to skip (e.g. average_texture)
    cache_dir: Path | None = None                  # defaults to fishy_research/cache/<name>
    resampled_dir: str | Path | None = None        # optional per-specimen resampled OBJs (real shapes)

    def __post_init__(self):
        from . import config

        self.mesh_path = Path(self.mesh_path)
        self.texture_dir = Path(self.texture_dir)
        if self.cache_dir is None:
            self.cache_dir = config.CACHE_DIR / self.name
        self.cache_dir = Path(self.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def texture_paths(self) -> list[Path]:
        paths = sorted(
            p for p in self.texture_dir.glob(self.texture_glob)
            if not any(e in p.name.lower() for e in self.exclude)
        )
        if not paths:
            raise FileNotFoundError(f"No textures in {self.texture_dir} matching {self.texture_glob}")
        return paths

    def load_mesh(self) -> Mesh:
        return load_obj(self.mesh_path)


def build_face_colors(
    ds: Dataset, force: bool = False, impute_artifacts: bool = True,
    black_max: int = 12, dark_L: float = 15.0, median_L_min: float = 35.0,
    verbose: bool = True,
) -> tuple[FaceColorData, np.ndarray]:
    """Sample per-face colors for a Dataset, impute baking artifacts, cache.

    Returns (FaceColorData, artifact_mask) where artifact_mask is (N, Nf) bool of faces that
    were detected as artifacts (before imputation).

    Artifact detector (per specimen, per face), using the shared-atlas population:
      - pure near-black:  max(RGB) < black_max, OR
      - per-specimen dark hole: this face's L* < dark_L while the face's population-median
        L* > median_L_min (a face that is bright across the population but black here).
    """
    cdir = ds.cache_dir
    cpaths = {k: cdir / f"{k}.npy" for k in ("colors", "areas", "centroids", "names", "artifact")}
    if not force and all(p.exists() for p in cpaths.values()):
        fcd = FaceColorData(
            colors=np.load(cpaths["colors"]), areas=np.load(cpaths["areas"]),
            centroids=np.load(cpaths["centroids"]), names=list(np.load(cpaths["names"])),
        )
        return fcd, np.load(cpaths["artifact"])

    mesh = ds.load_mesh()
    areas = mesh.face_areas()
    centroids = mesh.face_centroids()
    paths = ds.texture_paths()
    n, nf = len(paths), mesh.n_faces

    colors = np.empty((n, nf, 3), dtype=np.uint8)
    names = []
    for i, p in enumerate(paths):
        colors[i] = sample_face_colors(mesh, imageio.imread(p))
        names.append(p.stem)
        if verbose and (i % 10 == 0 or i == n - 1):
            print(f"  sampled {i + 1}/{n}: {p.name}")

    # ---- artifact detection + population-median imputation ----
    L = rgb_to_lab(colors)[..., 0]                          # (N, Nf)
    med_L = np.median(L, axis=0)                            # (Nf,) face population median L*
    near_black = colors.max(axis=2) < black_max             # (N, Nf)
    dark_hole = (L < dark_L) & (med_L[None] > median_L_min)
    artifact = near_black | dark_hole

    if impute_artifacts:
        # per-face median color over NON-artifact specimens
        for f in np.where(artifact.any(axis=0))[0]:
            bad = artifact[:, f]
            good = ~bad
            if good.any():
                med = np.median(colors[good, f, :], axis=0)
                colors[bad, f, :] = med.astype(np.uint8)
        if verbose:
            per_spec = artifact.sum(axis=1)
            worst = np.argsort(per_spec)[::-1][:5]
            print("  artifact faces imputed per specimen (top 5):")
            for j in worst:
                print(f"    {names[j]:20s}: {per_spec[j]:6d} / {nf} ({100*per_spec[j]/nf:.1f}%)")

    fcd = FaceColorData(colors=colors, areas=areas, centroids=centroids, names=names)
    np.save(cpaths["colors"], colors); np.save(cpaths["areas"], areas)
    np.save(cpaths["centroids"], centroids); np.save(cpaths["names"], np.array(names))
    np.save(cpaths["artifact"], artifact)
    return fcd, artifact


# ---- registry of known real datasets -------------------------------------
_MUSSEL_RUN = "/mnt/data/ml_data/color-modeling-pub/mussels/out/2026_06-09_00_08_36/colorAnalysis"
MUSSELS = Dataset(
    name="mussels",
    mesh_path=f"{_MUSSEL_RUN}/atlasModelUV.obj",
    texture_dir=f"{_MUSSEL_RUN}/atlasTextures",
    resampled_dir=f"{_MUSSEL_RUN}/resampledOBJ_withUV",
)
