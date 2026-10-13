"""Build + cache the E5 base descriptors (area_hist 24 | texton BoW K=128) for the atlas.

Separate from e5_expert_guided.py only so the (slow) descriptor build is done once and
cached; e5_expert_guided.py imports `load_feats()` from here.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

import common as C

FEAT_NPZ = C.CACHE / "e5_base_feats.npz"


def build(force: bool = False):
    if FEAT_NPZ.exists() and not force:
        d = np.load(FEAT_NPZ, allow_pickle=True)
        return d["names"], d["hist"], d["texton"]

    from fishpipe import features, textons

    t0 = time.time()
    fcd, art, mesh = C.atlas_data()
    print(f"[{time.time()-t0:.1f}s] atlas loaded: colors {fcd.colors.shape}")

    # face adjacency must exist for the texton diffusion operator
    _ = C.face_adjacency(mesh, C.ATLAS_DS.cache_dir)
    print(f"[{time.time()-t0:.1f}s] adjacency ready")

    H = features.area_hist(fcd, n_clusters=24)
    print(f"[{time.time()-t0:.1f}s] area_hist {H.shape}")

    T = textons.texton_descriptor(fcd, K=128, mode="bow",
                                  cache_dir=C.ATLAS_DS.cache_dir)
    print(f"[{time.time()-t0:.1f}s] texton BoW {T.shape}")

    names = np.array(list(fcd.names))
    np.savez_compressed(FEAT_NPZ, names=names, hist=H, texton=T)
    return names, H, T


def load_feats():
    return build(force=False)


if __name__ == "__main__":
    names, H, T = build()
    print("names", names.shape, "hist", H.shape, "texton", T.shape)
    print("hist row sums", H.sum(1)[:3], "texton row sums", T.sum(1)[:3])
