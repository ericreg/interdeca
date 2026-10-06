"""Extract a clean EXTERIOR (periostracum) image per specimen from a shared-atlas dataset.

On a shared atlas the exterior valve maps to a fixed UV region for every specimen, so we can
(1) auto-detect which faces are exterior (vs interior nacre) using face normals + the fact
that periostracum is darker than white nacre, then (2) crop that UV region from each texture.
Serves both the orientation-aware (Gabor) pattern descriptor and the exemplar morphospace.
"""
from __future__ import annotations

import numpy as np

from .data import FaceColorData
from .features import rgb_to_lab
from .mesh import Mesh


def face_normals(mesh: Mesh) -> np.ndarray:
    p = mesh.vertices[mesh.face_v]
    n = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
    ln = np.linalg.norm(n, axis=1, keepdims=True)
    ln[ln == 0] = 1.0
    return n / ln


def exterior_face_mask(mesh: Mesh, fcd: FaceColorData) -> np.ndarray:
    """Boolean (Nf,) marking exterior (periostracum) faces.

    Split faces by orientation along the dominant in/out normal axis, then label the
    *darker* side as exterior (periostracum is darker than the white nacre interior).
    """
    n = face_normals(mesh)
    _, _, vt = np.linalg.svd(n - n.mean(0), full_matrices=False)
    side = (n @ vt[0]) > 0
    L = rgb_to_lab(fcd.colors)[..., 0].mean(0)        # per-face mean lightness over specimens
    return side if L[side].mean() < L[~side].mean() else ~side


def exterior_uv_bbox(mesh: Mesh, ext_mask: np.ndarray, pad: float = 0.005) -> tuple:
    uv = mesh.uvs[mesh.face_vt[ext_mask]].reshape(-1, 2)
    u0, v0 = uv.min(0)
    u1, v1 = uv.max(0)
    return (max(0.0, u0 - pad), min(1.0, u1 + pad), max(0.0, v0 - pad), min(1.0, v1 + pad))


def crop_exterior(texture: np.ndarray, bbox: tuple) -> np.ndarray:
    """Crop the exterior UV bbox from a texture (RGB uint8). Handles V-flip."""
    h, w = texture.shape[:2]
    u0, u1, v0, v1 = bbox
    x0, x1 = int(u0 * (w - 1)), int(u1 * (w - 1))
    y0, y1 = int((1 - v1) * (h - 1)), int((1 - v0) * (h - 1))
    return texture[max(0, y0):y1, max(0, x0):x1]


# Fixed image-space crop of the exterior valve (fractions of the image: x0,x1,y0_top,y1_top).
# The atlas UV is shared, so one box works for every specimen. (Auto-detection via normals
# failed here because exterior/interior UV islands aren't normal-separable; this is the
# robust per-dataset alternative — for the module, derive it once from the atlas UV layout.)
MUSSEL_EXTERIOR_BOX = (0.0, 0.345, 0.30, 0.99)


def crop_box_frac(texture: np.ndarray, box=MUSSEL_EXTERIOR_BOX) -> np.ndarray:
    """Crop a fractional image-space box (x0,x1, y0_from_top, y1_from_top)."""
    h, w = texture.shape[:2]
    x0, x1, y0, y1 = box
    return texture[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)]
