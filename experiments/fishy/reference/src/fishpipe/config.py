"""Central paths and dataset constants for the fishy research harness."""
from __future__ import annotations

import os
from pathlib import Path

# ---- Data locations (the synthetic fishy dataset) -------------------------
DATA_DIR = Path(os.environ.get("FISHY_DATA_DIR", "/mnt/data/ml_data/fishy_all"))
TEXTURE_GLOB = "fishy_*.png"
MESH_PATH = DATA_DIR / "fishy1.obj"
# Ground-truth generating parameters live next door (only first 11 textures are
# duplicated there; the full 250 textures are in fishy_all, params cover all 250).
GROUND_TRUTH_CSV = Path(
    os.environ.get("FISHY_GT_CSV", "/mnt/data/ml_data/fishy/samples.csv")
)

N_SAMPLES = 250

# ---- Repo-relative output locations ---------------------------------------
PKG_ROOT = Path(__file__).resolve().parents[2]  # fishy_research/
CACHE_DIR = PKG_ROOT / "cache"
RESULTS_DIR = PKG_ROOT / "results"

CACHE_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# ---- Ground-truth factor definitions --------------------------------------
# The four deliberately-planted cluster ("categorical") factors, per the
# generator (color_fishy.py) and FISHY_DATASET_SUMMARY.md.
CLUSTER_FACTORS = ["belly", "tail", "stripe", "cheeks"]

# Continuous nuisance parameters (normal noise) — used for diagnostics only.
CONTINUOUS_PARAMS = [
    "base_color_hue",
    "base_color_sat",
    "base_color_val",
    "stripe_spacing",
    "stripe_width",
    "stripe_longitudinal_offset",
    "belly_strength",
    "belly_translation",
    "tail_strength",
]
