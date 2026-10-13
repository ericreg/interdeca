"""Descriptive, paired donor-held-out E2 contrasts on saved descriptor arrays.

The seven donor scans are the held-out units. Scores describe this archived cohort;
they are not independent population replicates or calibrated significance tests.
"""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Literal, NamedTuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler

from experiments.fishy.serde import Documents


class Contrast(NamedTuple):
    """A fixed before/after comparison inherited from the reference follow-up."""

    post: str
    pre: str
    factor: str


CONTRASTS = (
    Contrast("post_axial", "pre_spatial_canon", "stripe"),
    Contrast("post_axial", "pre_spatial_canon", "cheeks"),
    Contrast("post_spectral", "pre_spatial_canon", "cheeks"),
    Contrast("post_axial", "pre_spatial_canon", "belly"),
    Contrast("post_axial", "pre_spatial_canon", "tail"),
    Contrast("post_hist_matched", "pre_hist", "belly"),
    Contrast("post_hist_matched", "pre_hist", "tail"),
    Contrast("post_hist_matched", "pre_hist", "stripe"),
    Contrast("post_hist_matched", "pre_hist", "cheeks"),
)


class Protocol(NamedTuple):
    """Recorded estimator choices, with no population inference."""

    feature_construction: str = (
        "saved E2 descriptors reused unchanged; some were derived from the full "
        "250-specimen cohort, so only classifier fitting and scaling hold out donors"
    )
    split: str = "LeaveOneGroupOut; all donor scans; same splits for both descriptors"
    scaling: str = "StandardScaler fitted on training specimens only"
    scaler_with_mean: bool = True
    scaler_with_std: bool = True
    classifier: str = "LogisticRegression"
    c: float = 1.0
    l1_ratio: float = 0.0
    solver: str = "lbfgs"
    tol: float = 0.0001
    fit_intercept: bool = True
    class_weight: str = "balanced"
    max_iter: int = 4000
    random_state: int = 0
    metric: str = "binary balanced accuracy on each held-out donor"
    aggregation: str = "unweighted mean of donor scores; each donor has equal weight"
    inference: str = "descriptive only; no calibrated population p-values or intervals"


class DonorScore(NamedTuple):
    """One paired fit/evaluation on precisely the same held-out donor."""

    donor_id: str
    train_donors: tuple[str, ...]
    n_train: int
    n_test: int
    train_label_counts: dict[str, int]
    test_label_counts: dict[str, int]
    train_indices_sha256: str
    test_indices_sha256: str
    status: Literal["scored", "undefined_train_class", "undefined_test_class"]
    pre_balanced_accuracy: float | None
    post_balanced_accuracy: float | None
    post_minus_pre: float | None


class ContrastResult(NamedTuple):
    """Equal-donor summary is defined only when every held-out donor is scorable."""

    post: str
    pre: str
    factor: str
    donor_scores: tuple[DonorScore, ...]
    n_scored_donors: int
    macro_pre_balanced_accuracy: float | None
    macro_post_balanced_accuracy: float | None
    macro_post_minus_pre: float | None


class InferenceResult(NamedTuple):
    """Versioned supplemental evidence bound to code, input bytes and choices."""

    schema_version: int
    source_sha256: str
    input_sha256: str
    analysis_config_sha256: str
    n_specimens: int
    donor_ids: tuple[str, ...]
    protocol: Protocol
    contrasts: tuple[ContrastResult, ...]


class Inference:
    """Validate saved E2 arrays and make paired descriptive donor comparisons."""

    @staticmethod
    def _identifier(value: object) -> str:
        """Decode fixed-width byte identifiers before checking and recording them."""
        try:
            return value.decode("utf-8") if isinstance(value, bytes) else str(value)
        except UnicodeDecodeError as error:
            raise ValueError(
                "Specimen and donor identifiers must be UTF-8 text"
            ) from error

    @staticmethod
    def digest(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    @staticmethod
    def _index_digest(indices: np.ndarray) -> str:
        # Decimal index lists make the audit digest independent of NumPy byte order.
        encoded = json.dumps(indices.tolist(), separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()

    @staticmethod
    def _binary_counts(values: np.ndarray) -> dict[str, int]:
        return {str(label): int(np.count_nonzero(values == label)) for label in (0, 1)}

    @staticmethod
    def _validated_arrays(
        arrays: object, contrasts: tuple[Contrast, ...]
    ) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray]]:
        """Reject ambiguous names, labels, groups and nonfinite features before fitting."""
        required = {"names", "donor"}
        required.update(f"label_{item.factor}" for item in contrasts)
        required.update(item.pre for item in contrasts)
        required.update(item.post for item in contrasts)
        absent = required.difference(arrays.keys())
        if absent:
            raise ValueError(f"Missing E2 arrays: {sorted(absent)}")

        names = np.asarray(arrays["names"])
        donors = np.asarray(arrays["donor"])
        if (
            names.ndim != 1
            or names.dtype.kind not in "US"
            or len(names) < 2
            or len(set(names.tolist())) != len(names)
            or any(not Inference._identifier(name).strip() for name in names)
        ):
            raise ValueError("Expected unique nonempty specimen names")
        if (
            donors.shape != names.shape
            or donors.dtype.kind not in "iuUS"
            or len(np.unique(donors)) < 2
            or any(not Inference._identifier(donor).strip() for donor in donors)
        ):
            raise ValueError("At least two identified donor groups are required")
        labels = {}
        for factor in sorted({item.factor for item in contrasts}):
            value = np.asarray(arrays[f"label_{factor}"])
            if (
                value.shape != names.shape
                or value.dtype.kind not in "iub"
                or set(np.unique(value).tolist()) != {0, 1}
            ):
                raise ValueError(f"{factor} needs paired binary 0/1 labels")
            labels[factor] = value.astype(np.int8, copy=False)
        features = {}
        for key in sorted(
            {name for item in contrasts for name in (item.pre, item.post)}
        ):
            value = np.asarray(arrays[key])
            if (
                value.ndim != 2
                or value.shape[0] != len(names)
                or value.shape[1] < 1
                or value.dtype.kind not in "iuf"
                or not np.isfinite(value).all()
            ):
                raise ValueError(f"{key} needs finite numeric specimen rows")
            features[key] = value.astype(np.float64, copy=False)
        return donors, labels, features

    @staticmethod
    def _predict(
        features: np.ndarray, labels: np.ndarray, train: np.ndarray, test: np.ndarray
    ) -> np.ndarray:
        """Fit every preprocessing parameter on the training donors only."""
        protocol = Protocol()
        scaler = StandardScaler(
            with_mean=protocol.scaler_with_mean,
            with_std=protocol.scaler_with_std,
        )
        train_scaled = scaler.fit_transform(features[train])
        test_scaled = scaler.transform(features[test])
        model = LogisticRegression(
            C=protocol.c,
            l1_ratio=protocol.l1_ratio,
            solver=protocol.solver,
            tol=protocol.tol,
            fit_intercept=protocol.fit_intercept,
            max_iter=protocol.max_iter,
            class_weight=protocol.class_weight,
            random_state=protocol.random_state,
        )
        model.fit(train_scaled, labels[train])
        return model.predict(test_scaled)

    @staticmethod
    def analyze(
        arrays: object,
        contrasts: tuple[Contrast, ...] = CONTRASTS,
        *,
        source_sha256: str = "",
        input_sha256: str = "",
    ) -> InferenceResult:
        """Score all donors, marking unsupported folds instead of dropping them."""
        if not contrasts or len(set(contrasts)) != len(contrasts):
            raise ValueError("Expected unique fixed inherited contrasts")
        donors, labels, features = Inference._validated_arrays(arrays, contrasts)
        donor_ids = tuple(Inference._identifier(value) for value in np.unique(donors))
        splits = []
        for train, test in LeaveOneGroupOut().split(
            features[contrasts[0].pre], groups=donors
        ):
            held = np.unique(donors[test])
            if len(held) != 1 or np.intersect1d(donors[train], held).size:
                raise ValueError("Held-out donor leaked into training")
            splits.append((Inference._identifier(held[0]), train, test))
        protocol = Protocol()
        config = {
            "protocol": Documents.payload(protocol),
            "contrasts": Documents.payload(contrasts),
        }
        config_digest = hashlib.sha256(
            json.dumps(
                config, sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode()
        ).hexdigest()

        results = []
        for item in contrasts:
            y = labels[item.factor]
            rows = []
            for donor_id, train, test in splits:
                train_counts = Inference._binary_counts(y[train])
                test_counts = Inference._binary_counts(y[test])
                status = "scored"
                pre_score = post_score = difference = None
                if min(train_counts.values()) == 0:
                    status = "undefined_train_class"
                elif min(test_counts.values()) == 0:
                    status = "undefined_test_class"
                else:
                    # The same train/test indices pair the two descriptors. Each
                    # classifier receives its own training-only scaler and fit.
                    before = Inference._predict(features[item.pre], y, train, test)
                    after = Inference._predict(features[item.post], y, train, test)
                    pre_score = float(balanced_accuracy_score(y[test], before))
                    post_score = float(balanced_accuracy_score(y[test], after))
                    difference = post_score - pre_score
                rows.append(
                    DonorScore(
                        donor_id,
                        tuple(
                            Inference._identifier(value)
                            for value in np.unique(donors[train])
                        ),
                        len(train),
                        len(test),
                        train_counts,
                        test_counts,
                        Inference._index_digest(train),
                        Inference._index_digest(test),
                        status,
                        pre_score,
                        post_score,
                        difference,
                    )
                )
            scored = [row for row in rows if row.status == "scored"]
            if len(scored) == len(splits):
                # A donor, rather than a specimen or repeated CV fold, is one
                # descriptive unit in these unweighted cohort means.
                macro_pre = float(
                    np.mean([row.pre_balanced_accuracy for row in scored])
                )
                macro_post = float(
                    np.mean([row.post_balanced_accuracy for row in scored])
                )
                macro_diff = float(np.mean([row.post_minus_pre for row in scored]))
            else:
                macro_pre = macro_post = macro_diff = None
            results.append(
                ContrastResult(
                    item.post,
                    item.pre,
                    item.factor,
                    tuple(rows),
                    len(scored),
                    macro_pre,
                    macro_post,
                    macro_diff,
                )
            )
        result = InferenceResult(
            1,
            source_sha256,
            input_sha256,
            config_digest,
            len(donors),
            donor_ids,
            protocol,
            tuple(results),
        )
        # Verify every output value can become strict JSON before the caller
        # allocates an output file, including any unforeseen nonfinite score.
        json.dumps(Documents.payload(result), allow_nan=False)
        return result

    @staticmethod
    def run(arrays_path: Path, output: Path) -> InferenceResult:
        """Bind the supplemental result to immutable source and NPZ input bytes."""
        arrays_path = arrays_path.expanduser().resolve()
        source = Path(__file__).resolve()
        input_digest = Inference.digest(arrays_path)
        with np.load(arrays_path, allow_pickle=False) as arrays:
            result = Inference.analyze(
                arrays,
                source_sha256=Inference.digest(source),
                input_sha256=input_digest,
            )
        if Inference.digest(arrays_path) != input_digest:
            raise ValueError("Saved E2 arrays changed while they were being read")
        # Encode before allocating anything; exclusive creation prevents a
        # supplemental run from overwriting historical or reviewed evidence.
        encoded = (
            json.dumps(Documents.payload(result), indent=2, allow_nan=False) + "\n"
        )
        output = output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x") as stream:
            stream.write(encoded)
        return result


def main() -> None:
    """Run a bounded, CPU-only supplement from existing saved E2 arrays."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arrays", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = Inference.run(args.arrays, args.output)
    print(
        f"Saved {len(result.contrasts)} descriptive contrasts across "
        f"{len(result.donor_ids)} donor-held-out folds: {args.output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
