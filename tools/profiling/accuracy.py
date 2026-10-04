"""Measure surface, landmark, and texture fidelity without assuming shared topology.

These diagnostics measure agreement and fidelity to the supplied inputs. The
landmarks also guide atlas construction, so their residuals are in-sample checks,
not independent validation of anatomical correspondence.
"""

import argparse
import json
from pathlib import Path
from typing import NamedTuple

import numpy as np
from PIL import Image
from skimage.color import rgb2lab
from vtkmodules.vtkCommonCore import reference
from vtkmodules.vtkCommonDataModel import vtkStaticCellLocator

from interdeca.io import LandmarkIO, MeshIO
from interdeca.models import Mesh
from tools.profiling.serde import (
    BackendAccuracy,
    ColorDifference,
    CrossBackendAccuracy,
    CrossBackendSpecimen,
    Distribution,
    MeshAccuracy,
    PopulationAccuracy,
    PopulationDataset,
    Run,
    SpecimenAccuracy,
    SurfaceDifference,
)


class Transform(NamedTuple):
    rotation: np.ndarray
    translation: np.ndarray
    scale: float


class Samples(NamedTuple):
    points: np.ndarray
    faces: np.ndarray
    weights: np.ndarray


class Projection(NamedTuple):
    distances: np.ndarray
    faces: np.ndarray
    points: np.ndarray


class Colors(NamedTuple):
    rgb: np.ndarray
    alpha: np.ndarray


def distribution(values: np.ndarray) -> Distribution:
    """Summarize finite scalar errors; RMS is sqrt(mean(error squared)).

    Empty samples retain a zero count with unavailable statistics, rather than
    implying zero error when no measurement exists.
    """
    values = np.asarray(values, dtype=float).reshape(-1)
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite measurements")
    if not len(values):
        return Distribution(count=0)
    return Distribution(
        count=len(values),
        mean=values.mean(),
        rms=np.sqrt(np.mean(values**2)),
        p50=np.quantile(values, 0.5),
        p95=np.quantile(values, 0.95),
        maximum=values.max(),
    )


def fit(source: np.ndarray, target: np.ndarray, *, scaling=False) -> Transform:
    """Fit labeled landmarks with a proper rotation and optionally one scale.

    Least squares minimizes these landmark residuals, not the distance between
    whole surfaces. A fitted scale can improve landmarks while worsening surfaces.
    """
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise ValueError("Matching N by 3 landmark arrays are required")
    a, b = source - source.mean(axis=0), target - target.mean(axis=0)
    if np.linalg.matrix_rank(a) < 2 or np.sum(a * a) == 0:
        raise ValueError("Landmarks do not determine a rigid orientation")
    # The final singular direction changes sign only to prevent a mirror fit.
    u, singular, vt = np.linalg.svd(a.T @ b)
    signs = np.ones(3)
    signs[-1] = np.linalg.det(u @ vt)
    rotation = (u * signs) @ vt
    factor = float(np.sum(singular * signs) / np.sum(a * a)) if scaling else 1.0
    translation = target.mean(axis=0) - factor * source.mean(axis=0) @ rotation
    return Transform(rotation, translation, factor)


def transformed(points: np.ndarray, transform: Transform) -> np.ndarray:
    """Apply a uniform scale, row-vector rotation, and translation in that order."""
    return transform.scale * points @ transform.rotation + transform.translation


def mesh_metrics(mesh: Mesh) -> MeshAccuracy:
    """Count scale-independent triangle collapse and sum triangle surface areas.

    The cross product gives twice the triangle area. Its length relative to the
    two edge lengths distinguishes collapsed triangles from small valid ones.
    """
    triangles = mesh.vertices[mesh.faces]
    first, second = triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
    doubled = np.linalg.norm(np.cross(first, second), axis=1)
    # This relative criterion also counts zero edges while remaining independent
    # of the coordinate units. The original face indices are never removed.
    threshold = (
        32
        * np.finfo(float).eps
        * np.linalg.norm(first, axis=1)
        * np.linalg.norm(second, axis=1)
    )
    count = int(np.count_nonzero(doubled <= threshold))
    return MeshAccuracy(
        vertices=len(mesh.vertices),
        triangles=len(mesh.faces),
        collapsed_triangles=count,
        collapsed_fraction=count / len(mesh.faces),
        area=doubled.sum() / 2,
        bbox_diagonal=np.linalg.norm(np.ptp(mesh.vertices, axis=0)),
    )


def area_samples(mesh: Mesh, count: int, seed: int) -> Samples:
    """Uniform surface-area sampling, including triangle interiors, with a fixed seed."""
    triangles = mesh.vertices[mesh.faces]
    areas = np.linalg.norm(
        np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]),
        axis=1,
    )
    if areas.sum() <= 0:
        raise ValueError("Cannot sample a zero-area mesh")
    rng = np.random.default_rng(seed)
    faces = rng.choice(len(areas), count, p=areas / areas.sum())
    # Square-root barycentric sampling avoids clustering samples at one corner.
    root, other = np.sqrt(rng.random(count)), rng.random(count)
    weights = np.column_stack((1 - root, root * (1 - other), root * other))
    return Samples(np.einsum("ni,nij->nj", weights, triangles[faces]), faces, weights)


class SurfaceIndex:
    """Reuse a closest-triangle locator for related queries on one surface."""

    def __init__(self, mesh: Mesh):
        self.mesh = mesh
        self.surface = MeshIO.to_polydata(mesh)
        self.locator = vtkStaticCellLocator()
        self.locator.SetDataSet(self.surface)
        self.locator.BuildLocator()

    def project(self, points: np.ndarray) -> Projection:
        """Find the closest point on a triangle, not merely a nearby mesh vertex."""
        distances = np.empty(len(points))
        faces = np.empty(len(points), dtype=np.int64)
        projected = np.empty_like(points)
        for index, point in enumerate(points):
            destination = [0.0, 0.0, 0.0]
            face, sub, squared = reference(0), reference(0), reference(0.0)
            self.locator.FindClosestPoint(point, destination, face, sub, squared)
            if int(face) < 0:
                raise ValueError("No closest triangle")
            distances[index], faces[index], projected[index] = (
                np.sqrt(max(float(squared), 0)),
                int(face),
                destination,
            )
        return Projection(distances, faces, projected)


def barycentric(mesh: Mesh, projection: Projection) -> np.ndarray:
    """Express each projected point as three corner weights summing to one.

    Keeping the original triangle and corner indices lets those same weights
    interpolate the corresponding specimen coordinates or per-corner UVs.
    """
    triangles = mesh.vertices[mesh.faces[projection.faces]]
    e1, e2, offset = (
        triangles[:, 1] - triangles[:, 0],
        triangles[:, 2] - triangles[:, 0],
        projection.points - triangles[:, 0],
    )
    # Cross-product coordinates avoid subtracting nearly equal squared lengths
    # for thin but valid triangles. Use the mesh contract's degeneracy threshold.
    normal = np.cross(e1, e2)
    denominator = np.sum(normal * normal, axis=1)
    a, c = np.sum(e1 * e1, axis=1), np.sum(e2 * e2, axis=1)
    if np.any(denominator <= (32 * np.finfo(float).eps) ** 2 * a * c):
        raise ValueError("Cannot interpolate a degenerate closest triangle")
    second = np.sum(np.cross(offset, e2) * normal, axis=1) / denominator
    third = np.sum(np.cross(e1, offset) * normal, axis=1) / denominator
    weights = np.clip(np.column_stack((1 - second - third, second, third)), 0, 1)
    return weights / weights.sum(axis=1, keepdims=True)


def sample_image(path: Path, uv: np.ndarray) -> Colors:
    """Use the application's pixel-center/alpha rule without decoding float images.

    Only the four pixels around each requested UV become float64. This keeps an
    8192-square source image from creating several GiB of diagnostic-only arrays.
    """
    with Image.open(path) as image:
        pixels = np.asarray(image.convert("RGBA"))
    height, width = pixels.shape[:2]
    x = np.clip(uv[:, 0] * width - 0.5, 0, width - 1)
    y = np.clip((1 - uv[:, 1]) * height - 0.5, 0, height - 1)
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    x1, y1 = np.minimum(x0 + 1, width - 1), np.minimum(y0 + 1, height - 1)
    fx, fy = x - x0, y - y0
    rgb, alpha = np.zeros((len(uv), 3)), np.zeros(len(uv))
    for ix, iy, weight in (
        (x0, y0, (1 - fx) * (1 - fy)),
        (x1, y0, fx * (1 - fy)),
        (x0, y1, (1 - fx) * fy),
        (x1, y1, fx * fy),
    ):
        selected = pixels[iy, ix].astype(float) / 255
        contribution = weight * selected[:, 3]
        alpha += contribution
        rgb += selected[:, :3] * contribution[:, None]
    rgb = np.divide(
        rgb, alpha[:, None], out=np.zeros_like(rgb), where=alpha[:, None] > 0
    )
    return Colors(np.clip(rgb, 0, 1), alpha)


def color_difference(first: Colors, second: Colors) -> ColorDifference:
    """Compare jointly opaque samples, retaining each image's coverage separately.

    RGB errors use encoded sRGB channels on a 0–255 scale. Delta E 76 is the
    Euclidean distance after converting sRGB to CIELAB with the default D65 white.
    """
    a, b = first.alpha >= 0.99, second.alpha >= 0.99
    valid = a & b
    difference = 255 * (first.rgb[valid] - second.rgb[valid])
    # Coverage is explicit; transparent samples do not become zero-color errors.
    delta = (
        np.linalg.norm(rgb2lab(first.rgb[valid]) - rgb2lab(second.rgb[valid]), axis=1)
        if valid.any()
        else np.array([])
    )
    return ColorDifference(
        samples=len(a),
        first_covered=int(a.sum()),
        second_covered=int(b.sum()),
        jointly_covered=int(valid.sum()),
        delta_e_76=distribution(delta),
        mean_absolute_channel_0_255=float(np.abs(difference).mean())
        if valid.any()
        else None,
        rms_channel_0_255=float(np.sqrt(np.mean(difference**2)))
        if valid.any()
        else None,
    )


def surface_difference(
    first: Mesh, second: Mesh, count: int, seed: int
) -> SurfaceDifference:
    """Measure sampled nearest-surface distances in both directions.

    A surface can lie close to only part of another, so a one-way distance alone
    cannot characterize agreement. Maxima here are sampled, not Hausdorff bounds.
    """
    return SurfaceDifference(
        first_to_second=distribution(
            SurfaceIndex(second)
            .project(area_samples(first, count, seed).points)
            .distances
        ),
        second_to_first=distribution(
            SurfaceIndex(first)
            .project(area_samples(second, count, seed).points)
            .distances
        ),
    )


def pca_scores(run: Path, specimen_ids: list[str]) -> np.ndarray:
    """Load the final presentation scores only after checking specimen ordering."""
    events = [
        json.loads(line) for line in (run / "cli/events.jsonl").read_text().splitlines()
    ]
    plot = [
        e["payload"] for e in events if e["type"] == "result" and e["action"] == "plot"
    ][-1]
    if plot["specimen_ids"] != specimen_ids:
        raise ValueError("PCA specimen order differs from the benchmark manifest")
    return np.load(run / "cli" / plot["scores"]["path"], allow_pickle=False)


def atlas_directory(run: Path) -> Path:
    """Resolve the capture's completed atlas result, allowing its archive to move."""
    events = [
        json.loads(line) for line in (run / "cli/events.jsonl").read_text().splitlines()
    ]
    atlas = [
        event["payload"]
        for event in events
        if event["type"] == "result" and event["action"] == "atlas"
    ][-1]
    # Resolve within the requested capture even if the archive has moved since
    # its original run; stale absolute paths must not select unrelated outputs.
    original = Run.model_validate_json((run / "run.json").read_text()).config.output
    return run / Path(atlas["directory"]).relative_to(original)


def measure_accuracy(root: Path, count: int, seed: int) -> PopulationAccuracy:
    """Measure each backend against its supplied inputs and then against the other.

    Proper landmark fits remove pose for cross-backend comparisons. Scale is
    preserved for specimen comparisons and fitted separately for the atlases.
    These diagnostics do not supply independent anatomical ground truth.
    """
    inputs = root / "inputs"
    manifest = PopulationDataset.model_validate_json(
        (root / "dataset.json").read_text()
    )
    ids = list(manifest.specimens)
    # Complete per-backend results stay in distinct coordinate systems. Match
    # labeled landmarks only for cross-backend comparisons, never by vertex index.
    runs = {
        name: root / "current" / f"{name}-baseline-1"
        for name in ("native", "morphoweave")
    }
    directories = {name: atlas_directory(run) for name, run in runs.items()}
    atlases = {
        name: MeshIO.read(path / "atlas.obj") for name, path in directories.items()
    }
    landmarks = {
        name: LandmarkIO.read(path / "atlas.mrk.json")
        for name, path in directories.items()
    }
    labels = landmarks["native"].labels
    atlas_points = {
        name: points.reordered(labels).points for name, points in landmarks.items()
    }
    input_sizes = {}
    for specimen in ids:
        points = LandmarkIO.read(inputs / "landmarks" / f"{specimen}.mrk.json").points
        input_sizes[specimen] = float(np.linalg.norm(points - points.mean(axis=0)))
    backends, anchors = {}, {}
    for name, atlas in atlases.items():
        projected = SurfaceIndex(atlas).project(atlas_points[name])
        anchors[name] = (projected.faces, barycentric(atlas, projected))
        backends[name] = BackendAccuracy(
            directory=directories[name],
            atlas=mesh_metrics(atlas),
            atlas_centroid_size=np.linalg.norm(
                atlas_points[name] - atlas_points[name].mean(axis=0)
            ),
            mean_input_centroid_size=np.mean(list(input_sizes.values())),
            atlas_landmark_to_surface=distribution(projected.distances),
        )
    rigid = fit(atlas_points["morphoweave"], atlas_points["native"])
    similarity = fit(atlas_points["morphoweave"], atlas_points["native"], scaling=True)
    morph = atlases["morphoweave"]
    rigid_mesh = Mesh(transformed(morph.vertices, rigid), morph.faces, morph.uv)
    scaled_mesh = Mesh(transformed(morph.vertices, similarity), morph.faces, morph.uv)
    # One shared atlas-space sample set locates approximate spatially matched
    # colors despite different UV layouts. This is not an anatomical identity map.
    common = area_samples(atlases["native"], count, seed)
    projected = SurfaceIndex(rigid_mesh).project(common.points)
    uv = {
        "native": np.einsum(
            "ni,nij->nj", common.weights, atlases["native"].uv[common.faces]
        ),
        "morphoweave": np.einsum(
            "ni,nij->nj", barycentric(rigid_mesh, projected), morph.uv[projected.faces]
        ),
    }
    average = {
        name: sample_image(directory / "textures/average_texture.png", uv[name])
        for name, directory in directories.items()
    }
    # Euclidean score distances remove PCA sign/axis ambiguity. Different feature
    # spaces still make these descriptive population differences, not fit accuracy.
    score_distances = {}
    for name, run in runs.items():
        scores = pca_scores(run, ids)
        matrix = np.linalg.norm(scores[:, None] - scores[None, :], axis=2)
        score_distances[name] = matrix[np.triu_indices(len(scores), 1)]
    a, b = score_distances["native"], score_distances["morphoweave"]
    cross = CrossBackendAccuracy(
        atlas_landmark_rigid_residual=distribution(
            np.linalg.norm(
                transformed(atlas_points["morphoweave"], rigid)
                - atlas_points["native"],
                axis=1,
            )
        ),
        atlas_landmark_similarity_residual=distribution(
            np.linalg.norm(
                transformed(atlas_points["morphoweave"], similarity)
                - atlas_points["native"],
                axis=1,
            )
        ),
        morphoweave_to_native_scale=similarity.scale,
        atlas_surface_rigid=surface_difference(
            atlases["native"], rigid_mesh, count, seed
        ),
        atlas_surface_similarity=surface_difference(
            atlases["native"], scaled_mesh, count, seed
        ),
        average_color_at_common_atlas_locations=color_difference(
            average["native"], average["morphoweave"]
        ),
        pca_pairwise_distance_correlation=np.corrcoef(a, b)[0, 1],
        pca_pairwise_distance_relative_rms=np.sqrt(
            np.mean((a - b) ** 2) / np.mean(a * a)
        ),
    )
    result = PopulationAccuracy(
        inputs=inputs,
        samples_per_surface=count,
        seed=seed,
        methods={
            "surface": "Uniform area samples in each direction; proper labeled-landmark rigid fits, and a separately reported fitted atlas scale. RMS/P95/max are sampled distances, not exact Hausdorff bounds. Coordinate units are input mesh units.",
            "fidelity": "Per-backend resampled vertices and triangle interiors to its aligned original mesh; reverse surface coverage distance; source-texture versus baked color at the nearest original surface location. Projection and baking are measured together.",
            "landmarks": "Project atlas landmarks to atlas triangles, carry barycentric weights to each resampled specimen, and compare with its aligned input landmarks. In-sample residuals include atlas landmark-to-surface offsets; landmarks also drive fitting.",
            "color": "Bilinear premultiplied-alpha sampling at Blender pixel centers; both alpha >= 0.99; encoded sRGB channel errors on 0–255 and Euclidean CIELAB Delta E 76. Coverage counts retained. Cross-backend matching is nearest spatial surface, not proven anatomical correspondence.",
            "pca": "Pairwise Euclidean distances between specimen scores; descriptive because atlas sampling/feature spaces differ. No cross-backend scientific ground truth is available.",
        },
        backends=backends,
        cross_backend=cross,
    )
    for index, specimen in enumerate(ids):
        resampled, aligned_lm, common_colors = {}, {}, {}
        for name, directory in directories.items():
            original = MeshIO.read(directory / "aligned" / f"{specimen}.obj")
            mesh = MeshIO.read(
                directory / "resampled" / f"{specimen}.obj", allow_degenerate=True
            )
            if not np.array_equal(
                mesh.faces, atlases[name].faces
            ) or not np.array_equal(mesh.uv, atlases[name].uv):
                raise ValueError(f"Atlas correspondence changed: {name}/{specimen}")
            resampled[name] = mesh
            points = (
                LandmarkIO.read(directory / "aligned" / f"{specimen}.mrk.json")
                .reordered(labels)
                .points
            )
            aligned_lm[name] = points
            centroid_size = np.linalg.norm(points - points.mean(axis=0))
            locator = SurfaceIndex(original)
            # Sample vertices without deduplicating coincident locations: each
            # atlas index is part of the correspondence contract and gets a chance.
            rng = np.random.default_rng(seed)
            chosen = rng.choice(
                len(mesh.vertices), min(count, len(mesh.vertices)), replace=False
            )
            vertex_distances = locator.project(mesh.vertices[chosen]).distances
            sampled = area_samples(mesh, count, seed)
            projection = locator.project(sampled.points)
            input_sample = area_samples(original, count, seed)
            reverse = SurfaceIndex(mesh).project(input_sample.points).distances
            faces, weights = anchors[name]
            transferred = np.einsum(
                "ni,nij->nj", weights, mesh.vertices[mesh.faces[faces]]
            )
            error = np.linalg.norm(transferred - points, axis=1)
            # Each backend is judged against its own original UV-preserving mesh,
            # so cross-backend UV packing never becomes a false pixel difference.
            baked_uv = np.einsum("ni,nij->nj", sampled.weights, mesh.uv[sampled.faces])
            source_uv = np.einsum(
                "ni,nij->nj",
                barycentric(original, projection),
                original.uv[projection.faces],
            )
            texture = next((inputs / "textures").glob(specimen + ".*"))
            baked = sample_image(directory / "textures" / f"{specimen}.png", baked_uv)
            source = sample_image(texture, source_uv)
            backends[name].specimens[specimen] = SpecimenAccuracy(
                aligned=mesh_metrics(original),
                resampled=mesh_metrics(mesh),
                input_centroid_size=input_sizes[specimen],
                aligned_centroid_size=centroid_size,
                size_ratio=centroid_size / input_sizes[specimen],
                vertex_to_input=distribution(vertex_distances),
                interiors_to_input=distribution(projection.distances),
                input_to_resampled=distribution(reverse),
                transferred_landmarks_to_input=distribution(error),
                transferred_landmarks_error_over_centroid_size=distribution(
                    error / centroid_size
                ),
                baked_to_source_color=color_difference(baked, source),
            )
            common_colors[name] = sample_image(
                directory / "textures" / f"{specimen}.png", uv[name]
            )
        pose = fit(aligned_lm["morphoweave"], aligned_lm["native"])
        mesh = resampled["morphoweave"]
        cross.specimens[specimen] = CrossBackendSpecimen(
            aligned_landmark_rigid_residual=distribution(
                np.linalg.norm(
                    transformed(aligned_lm["morphoweave"], pose) - aligned_lm["native"],
                    axis=1,
                )
            ),
            resampled_surface_after_specimen_rigid_fit=surface_difference(
                resampled["native"],
                Mesh(transformed(mesh.vertices, pose), mesh.faces, mesh.uv),
                count,
                seed,
            ),
            color_at_common_atlas_locations=color_difference(
                common_colors["native"], common_colors["morphoweave"]
            ),
        )
        # Persist a clearly marked checkpoint so an interrupted diagnostic can be
        # inspected; only the caller writes the final complete accuracy.json.
        (root / "accuracy.partial.json").write_text(
            result.model_dump_json(indent=2) + "\n"
        )
        print(f"Measured accuracy {index + 1}/{len(ids)}: {specimen}", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=413)
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("Samples must be positive")
    result = measure_accuracy(args.root.resolve(), args.samples, args.seed)
    (args.root / "accuracy.json").write_text(result.model_dump_json(indent=2) + "\n")


if __name__ == "__main__":
    main()
