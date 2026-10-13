"""Alek Spiridonov's cichlid generator, adapted from source commit 3db23a1.

The generator spline is intentionally distinct from Interdeca's atlas spline.
See docs/fishy/provenance.md for preserved behavior and unresolved paper differences.
"""

import colorsys
import warnings

import numpy as np
from sklearn.datasets import make_blobs

from experiments.fishy.models import (
    Brush,
    CanonicalFrame,
    GeneratorData,
    LandmarkInput,
    Parameters,
    SampledParameters,
    Spline,
)


class Generation:
    """Pure deterministic operations; Blender is confined to a separate process."""

    @staticmethod
    def sample_labeled(count: int, seed: int) -> tuple[SampledParameters, ...]:
        """Preserve original MT19937 draw order without changing global RNG state."""
        if (
            isinstance(count, bool)
            or not isinstance(count, (int, np.integer))
            or count < 2
        ):
            raise ValueError("Sample count must be an integer of at least two.")
        if (
            isinstance(seed, bool)
            or not isinstance(seed, (int, np.integer))
            or not 0 <= seed <= 4294967295
        ):
            raise ValueError("Seed must be a uint32 integer.")
        rng = np.random.RandomState(seed)
        blue, yellow, green, red = [
            colorsys.rgb_to_hsv(*(np.array(rgb) / 255))
            for rgb in ((39, 70, 144), (242, 255, 73), (21, 127, 31), (255, 102, 99))
        ]
        # Explicit RandomState in make_blobs consumes the same draws as the old
        # globally seeded implementation, while isolating other callers' randomness.
        hue = rng.normal(yellow[0], 0.015, count) % 1
        sat = np.clip(rng.normal(0, 0.05, count) + yellow[1], 0, 1)
        val = np.clip(rng.normal(0, 0.05, count) + yellow[2], 0, 1)
        stripes = rng.choice([4, 5], count, p=[0.8, 0.2])
        spacing = rng.normal(-0.08, 0.01, count)
        width = rng.normal(0.025, 0.005, count)
        # Non-positive widths are kept exactly as drawn so seeded samples match the
        # author's protocol; they are flagged here rather than resampled or clamped.
        non_positive_width = int(np.count_nonzero(width <= 0))
        if non_positive_width:
            warnings.warn(
                f"{non_positive_width} sampled stripe_width value(s) are non-positive;"
                " returned as drawn.",
                RuntimeWarning,
                stacklevel=2,
            )
        offset = rng.normal(0, 0.01, count)
        belly_blob, belly_group = make_blobs(
            count,
            n_features=1,
            centers=2,
            cluster_std=0.015,
            center_box=(-0.1, 0.1),
            random_state=rng,
        )
        belly = (blue[0] + belly_blob[:, 0]) % 1
        belly_strength = np.clip(rng.normal(0.95, 0.05, count), 0, 1)
        belly_translation = rng.normal(0, 0.05, count)
        tail_blob, tail_group = make_blobs(
            count,
            n_features=1,
            centers=2,
            cluster_std=0.015,
            center_box=(-0.1, 0.1),
            random_state=rng,
        )
        tail = (green[0] + tail_blob[:, 0]) % 1
        tail_strength = np.clip(rng.normal(0.95, 0.05, count), 0, 1)
        cheeks = rng.choice([0, 1], count, p=[0.5, 0.5])
        cheek_hue = rng.normal(red[0], 0.015, count) % 1
        cheek_strength = np.clip(rng.normal(0.95, 0.05, count), 0, 1)
        cheek_y = rng.normal(0, 0.05, count)
        cheek_z = rng.normal(0, 0.05, count)
        columns = (
            hue,
            sat,
            val,
            stripes,
            spacing,
            width,
            offset,
            belly,
            belly_strength,
            belly_translation,
            tail,
            tail_strength,
            cheeks,
            cheek_hue,
            cheek_strength,
            cheek_y,
            cheek_z,
        )
        return tuple(
            # Normalize NumPy categorical scalars only after all original draws;
            # values and RNG order remain exactly the author's protocol.
            SampledParameters(
                Parameters(
                    *(
                        int(value) if index in (3, 12) else float(value)
                        for index, value in enumerate(row)
                    )
                ).validated(),
                int(belly_group[i]),
                int(tail_group[i]),
            )
            for i, row in enumerate(zip(*columns, strict=True))
        )

    @staticmethod
    def sample(count: int, seed: int) -> tuple[Parameters, ...]:
        """Return numerical parameters alone for comparisons to the legacy source."""
        return tuple(item.parameters for item in Generation.sample_labeled(count, seed))

    @staticmethod
    def canonical(landmarks: LandmarkInput) -> tuple[CanonicalFrame, np.ndarray]:
        """Proper Kabsch rotation, unit front/back distance, posterior at zero."""
        GeneratorData.points(landmarks.points, "canonical landmarks")
        if len(landmarks.labels) != len(landmarks.points) or len(
            set(landmarks.labels)
        ) != len(landmarks.labels):
            raise ValueError("Canonical landmarks need one unique label per point.")
        groups = (
            ("TFH", "TFT"),
            ("LBWL", "RBWR", "RBWL", "LBWR", "AH"),
            ("FIT", "FOB", "FIL", "FIR", "FIB"),
            ("TT", "TV", "TA"),
        )
        # Require the documented semantic anchors rather than silently dropping
        # absent anchors and changing the canonical coordinate system.
        lookup = {label: i for i, label in enumerate(landmarks.labels)}
        missing = set(sum(groups, ())) - lookup.keys()
        if missing:
            raise ValueError(f"Missing canonical landmarks: {sorted(missing)}")
        ids = [[lookup[label] for label in group] for group in groups]
        points = landmarks.points
        top, bottom, front, back = [points[index].mean(0) for index in ids]
        pivot = np.mean([(front + back) / 2, (top + bottom) / 2], axis=0)
        axes = np.array([[0, 0, 1], [0, 0, -1], [1, 0, 0], [-1, 0, 0]])
        matrix = np.zeros((3, 3))
        for indices, axis, weight in zip(ids, axes, (1, 1, 2, 2), strict=True):
            for index in indices:
                matrix += weight * np.outer(axis, points[index] - pivot)
        u, _, vt = np.linalg.svd(matrix)
        correction = np.diag([1, 1, -1 if np.linalg.det(u @ vt) < 0 else 1])
        rotation = u @ correction @ vt
        distance = np.linalg.norm(front - back)
        if distance <= 1e-12 or np.linalg.matrix_rank(matrix) < 2:
            raise ValueError("Degenerate canonical landmark constraints.")
        rotated = (points - pivot) @ rotation.T
        scale = 1 / distance
        translation = -scale * rotated[ids[3]].mean(0)
        frame = CanonicalFrame(rotation, pivot, scale, translation).validated()
        return frame, rotated * scale + translation

    @staticmethod
    def world(points: np.ndarray, frame: CanonicalFrame) -> np.ndarray:
        """Undo canonical normalization; atlas alignment itself never scales donors."""
        GeneratorData.points(points, "world points")
        frame.validated()
        return (
            (points - frame.translation) / frame.scale
        ) @ frame.rotation + frame.pivot

    @staticmethod
    def fit(source: np.ndarray, target: np.ndarray, regularization: float) -> Spline:
        """Fit the augmented spline system with an unpenalized affine component."""
        GeneratorData.points(source, "spline source")
        GeneratorData.points(target, "spline target")
        GeneratorData.scalar(regularization, "regularization")
        if regularization < 0:
            raise ValueError("Spline regularization must be nonnegative.")
        if source.shape != target.shape or len(source) < 4:
            raise ValueError(
                "Spline needs matching landmark arrays with at least four points."
            )
        if np.linalg.svd(source - source.mean(0), compute_uv=False)[-1] < 1e-10:
            raise ValueError("Spline control points must span three dimensions.")
        radii = np.linalg.norm(source[:, None] - source[None], axis=2)
        kernel = radii**2 * np.log(radii + 1e-12)
        affine = np.column_stack([np.ones(len(source)), source])
        system = np.block(
            [
                [kernel + regularization * np.eye(len(source)), affine],
                [affine.T, np.zeros((4, 4))],
            ]
        )
        solution = np.linalg.solve(system, np.vstack([target, np.zeros((4, 3))]))
        return Spline(source.copy(), solution[:-4], solution[-4:]).validated()

    @staticmethod
    def evaluate(points: np.ndarray, spline: Spline) -> np.ndarray:
        """Evaluate the generator spline; its radial kernel is r² log(r+epsilon)."""
        GeneratorData.points(points, "spline evaluation points")
        spline.validated()
        radii = np.linalg.norm(points[:, None] - spline.controls[None], axis=2)
        return (radii**2 * np.log(radii + 1e-12)) @ spline.weights + np.column_stack(
            [np.ones(len(points)), points]
        ) @ spline.affine

    @staticmethod
    def brushes(p: Parameters) -> tuple[Brush, ...]:
        """Order matters: stripes overwrite the base and all colored patches."""
        p.validated()
        blue = colorsys.rgb_to_hsv(39 / 255, 70 / 255, 144 / 255)
        green = colorsys.rgb_to_hsv(21 / 255, 127 / 255, 31 / 255)
        red = colorsys.rgb_to_hsv(1, 102 / 255, 99 / 255)
        brushes = [
            Generation.sphere(
                (0.5, 0, 0),
                1,
                1,
                colorsys.hsv_to_rgb(
                    p.base_color_hue, p.base_color_sat, p.base_color_val
                ),
                1,
            ),
            Generation.sphere(
                (0.3 + p.belly_translation, 0, -0.55),
                0.4,
                0.05,
                colorsys.hsv_to_rgb(p.belly_hue, blue[1], blue[2]),
                p.belly_strength,
            ),
            Generation.sphere(
                (0, 0, 0),
                0.1,
                0.03,
                colorsys.hsv_to_rgb(p.tail_hue, green[1], green[2]),
                p.tail_strength,
            ),
        ]
        if p.rosy_cheeks_present:
            for side in (0.1, -0.1):
                brushes.append(
                    Generation.sphere(
                        (
                            0.8 + p.rosy_cheeks_translation_y,
                            side,
                            -0.1 + p.rosy_cheeks_translation_z,
                        ),
                        0.035,
                        0.01,
                        colorsys.hsv_to_rgb(p.rosy_cheeks_hue, red[1], red[2]),
                        p.rosy_cheeks_strength,
                    )
                )
        for index in range(p.stripe_count):
            x = 0.8 + index * p.stripe_spacing + p.stripe_longitudinal_offset
            for side in (1, -1):
                brushes.append(
                    Brush(
                        np.array([x, side * 0.02, 0.07]),
                        np.array([x, side * 0.1, -0.08]),
                        0,
                        p.stripe_width,
                        (0, 0, 0),
                        1,
                    )
                )
        return tuple(brushes)

    @staticmethod
    def sphere(
        center: tuple, radius: float, sigma: float, color: tuple, strength: float
    ) -> Brush:
        """Encode a sphere as a zero-length capsule without changing distances."""
        point = np.array(center, dtype=float)
        return Brush(point, point, radius, sigma, color, strength).validated()

    @staticmethod
    def warp(
        brushes: tuple[Brush, ...], spline: Spline, frame: CanonicalFrame
    ) -> tuple[Brush, ...]:
        """Warp endpoints and scale radii by mean local Jacobian stretch."""
        spline.validated()
        frame.validated()
        for brush in brushes:
            brush.validated()
        warped = []
        for brush in brushes:
            points = np.stack([brush.start, brush.end])
            stretches = []
            for axis in np.eye(3) * 1e-5:
                difference = Generation.evaluate(
                    points + axis, spline
                ) - Generation.evaluate(points - axis, spline)
                stretches.append(np.linalg.norm(difference, axis=1) / 2e-5)
            scale = np.mean(stretches) / frame.scale
            world = Generation.world(Generation.evaluate(points, spline), frame)
            warped.append(
                Brush(
                    world[0],
                    world[1],
                    brush.radius * scale,
                    brush.sigma * scale,
                    brush.color,
                    brush.strength,
                )
            )
        return tuple(warped)

    @staticmethod
    def paint(
        points: np.ndarray,
        brushes: tuple[Brush, ...],
        noise_amount: float,
        noise_scale: float,
    ) -> np.ndarray:
        """Original sequential alpha compositing, including its pre-brush noise.

        Noise is blended BEFORE the opaque base brush in the checked-in source.
        Preserve that behavior rather than silently implementing the draft's
        potentially different interpretation of a final 10% noise blend.
        """
        # Internal NamedTuple construction is lightweight; validate each direct
        # entry before arithmetic can turn malformed values into plausible RGB.
        GeneratorData.points(points, "paint points")
        GeneratorData.scalar(noise_amount, "noise_amount")
        GeneratorData.scalar(noise_scale, "noise_scale")
        if not 0 <= noise_amount <= 1 or noise_scale <= 0:
            raise ValueError("Noise amount/scale is outside its valid range.")
        for brush in brushes:
            brush.validated()
        field = (
            np.sin(points[:, 0] * noise_scale * 3.1)
            + np.sin(points[:, 1] * noise_scale * 2.3 + 1.7)
            + np.sin(points[:, 2] * noise_scale * 4 + 3.4)
        ) / 3
        field = (field - field.min()) / (np.ptp(field) + 1e-12)
        colors = noise_amount * np.repeat(field[:, None], 3, axis=1)
        for brush in brushes:
            segment = brush.end - brush.start
            denominator = segment @ segment
            if denominator < 1e-20:
                distance = np.linalg.norm(points - brush.start, axis=1)
            else:
                t = np.clip((points - brush.start) @ segment / denominator, 0, 1)
                distance = np.linalg.norm(
                    points - brush.start - t[:, None] * segment, axis=1
                )
            alpha = np.exp(
                -0.5
                * (np.maximum(distance - brush.radius, 0) / (brush.sigma + 1e-12)) ** 2
            )
            alpha = np.clip(alpha * brush.strength, 0, 1)[:, None]
            colors = colors * (1 - alpha) + np.asarray(brush.color) * alpha
        return np.clip(colors, 0, 1)
