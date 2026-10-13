"""Standard-library internal records shared by numerical and runner boundaries.

Pydantic lives in serde.py. These records need neither NumPy nor the core, so
paper metadata can be inspected before the scientific environment is installed.
"""

import math
import warnings
from numbers import Real
from typing import NamedTuple, Self


class Parameters(NamedTuple):
    """One sampled parameter vector, in Alek's original column order."""

    base_color_hue: float
    base_color_sat: float
    base_color_val: float
    stripe_count: int
    stripe_spacing: float
    stripe_width: float
    stripe_longitudinal_offset: float
    belly_hue: float
    belly_strength: float
    belly_translation: float
    tail_hue: float
    tail_strength: float
    rosy_cheeks_present: int
    rosy_cheeks_hue: float
    rosy_cheeks_strength: float
    rosy_cheeks_translation_y: float
    rosy_cheeks_translation_z: float

    def validated(self) -> Self:
        """Single domain definition for parameter vectors; construction stays lightweight.

        Stripe width sign is not a domain rule: the author's normal draw can be
        non-positive, so it is flagged with a warning and kept exactly as drawn.
        """
        for name, value in self._asdict().items():
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not math.isfinite(value)
            ):
                raise ValueError(f"{name} must be a finite real number.")
        if type(self.stripe_count) is not int or self.stripe_count not in (4, 5):
            raise ValueError("stripe_count must be integer 4 or 5.")
        if type(
            self.rosy_cheeks_present
        ) is not int or self.rosy_cheeks_present not in (0, 1):
            raise ValueError("rosy_cheeks_present must be integer 0 or 1.")
        for name in ("base_color_hue", "belly_hue", "tail_hue", "rosy_cheeks_hue"):
            if not 0 <= getattr(self, name) < 1:
                raise ValueError(f"{name} must lie in [0, 1).")
        for name in (
            "base_color_sat",
            "base_color_val",
            "belly_strength",
            "tail_strength",
            "rosy_cheeks_strength",
        ):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"{name} must lie in [0, 1].")
        if self.stripe_width <= 0:
            warnings.warn(
                "stripe_width is non-positive; returned as drawn.",
                RuntimeWarning,
                stacklevel=2,
            )
        return self


class EvaluationOptions(NamedTuple):
    """Validated classifier settings; fitting code cannot change its protocol."""

    palette_bins: int
    seed: int
    folds: int
    repeats: int
    logistic_c: float


class FoldScore(NamedTuple):
    """Held-out indices and the paired observed/shuffled balanced accuracies."""

    fold: int
    train_indices: tuple[int, ...]
    test_indices: tuple[int, ...]
    null_train_indices: tuple[int, ...]
    null_test_indices: tuple[int, ...]
    balanced_accuracy: float | None
    shuffled_accuracy: float | None
    null_corrected: float | None


class SchemeScore(NamedTuple):
    """One cross-validation scheme; the outer scheme-name registry is a map."""

    folds: tuple[FoldScore, ...]
    note: str


class InputFile(NamedTuple):
    """An original absolute input path with content identity and optional timestamp."""

    path: str
    bytes: int
    sha256: str
    mtime_ns: int | None = None
    present_fields: frozenset[str] = frozenset()


class StageRun(NamedTuple):
    """Legacy running/completed stage evidence, retaining absent optional fields."""

    exit_code: int | None
    status: str | None = None
    elapsed_seconds: float | None = None
    thread_environment: dict[str, str] | None = None
    present_fields: frozenset[str] = frozenset()


class PaperRun(NamedTuple):
    """Version-one resume state. Stage and installed-package registries are maps."""

    schema_version: int
    upstream_revision: str
    source: str
    module_run: str
    donor_scans: str | None
    python: str
    platform: str
    packages: dict[str, str]
    inputs: tuple[InputFile, ...]
    stages: dict[str, StageRun]
    scope: str
    analysis_python: str | None = None
    analysis_packages: dict[str, str] | None = None
    donor_scans_attached_unix_seconds: float | None = None
    present_fields: frozenset[str] = frozenset()


class BundleConfig(NamedTuple):
    """Version-one portable paths, resolved relative to the bundle root."""

    schema_version: int
    python: str
    requirements: str
    source: str
    module_run: str
    donor_scans: str
    author_arrays: str


class InventoryFile(NamedTuple):
    """Portable payload identity; family occurs only in the legacy input inventory."""

    path: str
    bytes: int
    sha256: str
    family: str | None = None


class BundleManifest(NamedTuple):
    """Version-one bundle inventory and the historical source-list digest."""

    schema_version: int
    source_tree_sha256: str
    files: tuple[InventoryFile, ...]


class Completion(NamedTuple):
    """Explicit unversioned legacy completion document."""

    upstream_revision: str
    scope: str


class BuildIdentity(NamedTuple):
    """Explicit unversioned legacy interrupted-build marker."""

    identity: str


class BundleRights(NamedTuple):
    """Original rights statements, retained as provenance rather than permissions."""

    dataset_redistribution: str
    code_license: str
    access: str


class BundleProvenance(NamedTuple):
    """Explicit unversioned legacy provenance; source origins form a genuine map."""

    bundle_scope: str
    source_head: str
    source_snapshot: str
    author_revision: str
    author_repository: str
    input_assets: int
    input_bytes: int
    input_origin: dict[str, str]
    rights: BundleRights
    historical_parity: str
    limitations: tuple[str, ...]


class Specimen(NamedTuple):
    """Explicit legacy specimen row, paired by stable sample ID."""

    sample_id: str
    donor_id: int
    donor_name: str


class ExportManifest(NamedTuple):
    """Explicit legacy export inventory tied to its original bundle manifest."""

    bundle_manifest_sha256: str
    files: tuple[InventoryFile, ...]
