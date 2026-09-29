"""Build staged application workflows from the stateless numerical operations."""

from collections.abc import Callable
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from typing import Any, NamedTuple
from uuid import uuid4

import numpy as np
from skimage.color import lab2rgb

from interdeca.application_models import (
    AtlasOptions,
    AtlasResult,
    ColorOptions,
    ColorResult,
    ModelOptions,
    PopulationResult,
)
from interdeca.atlas import Atlas
from interdeca.blender import Blender
from interdeca.color_analysis import ColorAnalysis
from interdeca.dataset import Dataset
from interdeca.execution import Pipeline, ResultRef, Task
from interdeca.geometry import Geometry
from interdeca.io import LandmarkIO, MeshIO, TextureIO
from interdeca.models import AtlasModel, BakedTexture, Mesh
from interdeca.population import Population


class Option(NamedTuple):
    """A named operation argument; dictionaries remain only actual lookup tables."""

    name: str
    value: Any


class TaskCall(NamedTuple):
    """Explicit positional and named inputs, with ResultRef values where required."""

    arguments: tuple = ()
    options: tuple[Option, ...] = ()


class SpecimenFiles(NamedTuple):
    """A manifest entry paired by specimen identity, not directory iteration order."""

    specimen_id: str
    mesh: Path
    landmarks: Path
    texture: Path | None


class Operations:
    """Small task adapters for I/O and combinations that have no GUI dependency."""

    @staticmethod
    def invoke(function: Callable, inputs: TaskCall) -> Any:
        """Resolve the record into a Python call at the operation boundary."""
        # Keyword dictionaries are the Python call boundary, not internal objects.
        return function(
            *inputs.arguments,
            **{option.name: option.value for option in inputs.options},
        )

    @staticmethod
    def registry() -> dict[str, Callable]:
        """Resolve public operation names without serializing executable closures."""
        functions = {
            "inspect": Dataset.inspect_specimen,
            "validate": Dataset.validate_specimens,
            "blender": Blender.check_available,
            "reference": Atlas.compute_reference,
            "contribute": Atlas.prepare_contribution,
            "mean": Atlas.assemble_mean,
            "load_atlas": Operations.load_atlas,
            "clean": Blender.clean_atlas,
            "uv": Blender.prepare_atlas_uv,
            "align": Atlas.align_specimen,
            "resample": Geometry.resample_specimen,
            "bake": Blender.bake_specimen_texture,
            "save_atlas": Operations.save_atlas,
            "save_specimen": Operations.save_specimen,
            "average": Operations.average,
            "atlas_result": Operations.atlas_result,
            "analysis": Operations.prepare_analysis,
            "texture": Operations.texture,
            "sample": ColorAnalysis.sample_specimen_colors,
            "normalization": ColorAnalysis.compute_normalization_stats,
            "process": ColorAnalysis.process_specimen_colors,
            "match": Population.align_clusters,
            "color_result": Operations.color_result,
            "features": Population.assemble_features,
            "fit": Population.fit_model,
            "population_result": PopulationResult,
            "display": Operations.display,
            "reconstruct": Operations.reconstruct,
            "select": Geometry.select_region,
            "extract": Geometry.extract_region,
        }
        return {
            name: partial(Operations.invoke, function)
            for name, function in functions.items()
        }

    @staticmethod
    def load_atlas(model: Path, landmarks: Path) -> AtlasModel:
        """Read an atlas and its anatomical landmarks in canonical coordinates."""
        return AtlasModel(MeshIO.read(model), LandmarkIO.read(landmarks))

    @staticmethod
    def save_atlas(atlas: AtlasModel, directory: Path) -> AtlasModel:
        """Publish the atlas mesh and landmarks in the new run directory."""
        # Existing files are refused by core writers; each run has its own directory.
        MeshIO.write_obj(atlas.mesh, directory / "atlas.obj")
        LandmarkIO.write_json(atlas.landmarks, directory / "atlas.mrk.json")
        return atlas

    @staticmethod
    def save_specimen(specimen: Any, directory: Path) -> Any:
        """Publish aligned or resampled geometry and retain its numerical result."""
        MeshIO.write_obj(specimen.mesh, directory / f"{specimen.specimen_id}.obj")
        LandmarkIO.write_json(
            specimen.landmarks, directory / f"{specimen.specimen_id}.mrk.json"
        )
        return specimen

    @staticmethod
    def average(textures: tuple, path: Path) -> Path | None:
        """Produce a display average only when texture transfer was requested."""
        if not textures:
            return None
        ColorAnalysis.average_textures(textures, path)
        return path

    @staticmethod
    def atlas_result(
        atlas: AtlasModel, directory: Path, textures: tuple, average: Path | None
    ) -> AtlasResult:
        """Collect the declared outputs of a successful atlas workflow."""
        return AtlasResult(atlas, directory, textures, average)

    @staticmethod
    def prepare_analysis(mesh: Mesh, count: int | None) -> Any:
        """Use every face when the requested sample budget exceeds the mesh size."""
        # Small meshes legitimately have fewer faces than the UI's default budget.
        return Geometry.prepare_analysis(
            mesh, sample_count=None if count is None else min(count, len(mesh.faces))
        )

    @staticmethod
    def texture(path: Path, topology_key: str) -> BakedTexture:
        """Inspect an existing image and associate it with the selected atlas topology."""
        pixels = TextureIO.read(path)
        height, width = pixels.rgb.shape[:2]
        return BakedTexture(path.stem, path, topology_key, (width, height))

    @staticmethod
    def color_result(
        analysis: Any, specimens: tuple, alignment: Any, options: ColorOptions
    ) -> ColorResult:
        """Retain analysis geometry, processed specimens, and any shared palette."""
        return ColorResult(
            analysis,
            specimens if alignment is None else alignment.specimens,
            alignment,
            options,
        )

    @staticmethod
    def display(colors: ColorResult, specimen_id: str) -> Any:
        """Expand one specimen into face colors using the stored shared sampling."""
        specimen = next(
            item for item in colors.specimens if item.specimen_id == specimen_id
        )
        rgb = specimen.rgb
        if colors.alignment is not None:
            # Invalid labels are -1; only covered samples can address the palette.
            rgb = np.zeros_like(rgb)
            rgb[specimen.valid] = colors.alignment.palette_rgb[
                specimen.labels[specimen.valid]
            ]
        return ColorAnalysis.expand_face_colors(colors.analysis, rgb, specimen.valid)

    @staticmethod
    def reconstruct(
        population: PopulationResult, colors: ColorResult, coordinates: np.ndarray
    ) -> Any:
        """Convert reconstructed Lab features to corresponding display face colors."""
        reconstructed = Population.reconstruct_features(population.model, coordinates)
        # Column names retain face identity even when common missing samples were dropped.
        samples = colors.analysis.sample_faces
        lookup = {int(face): index for index, face in enumerate(samples)}
        lab = np.zeros((len(samples), 3))
        valid = np.zeros(len(samples), dtype=bool)
        names = reconstructed.feature_names
        for offset in range(0, len(names), 3):
            face = int(names[offset].split(".")[0].removeprefix("face_"))
            expected = tuple(f"face_{face}.{channel}" for channel in ("L", "a", "b"))
            if names[offset : offset + 3] != expected or face not in lookup:
                raise ValueError(
                    "Reconstructed feature identity does not match the analysis mesh."
                )
            lab[lookup[face]] = reconstructed.values[0, offset : offset + 3]
            valid[lookup[face]] = True
        # Clip only display conversion; analytical reconstruction retains its original values.
        lab[:, 0] = np.clip(lab[:, 0], 0, 100)
        return ColorAnalysis.expand_face_colors(
            colors.analysis, np.clip(lab2rgb(lab), 0, 1), valid
        )


class Workflows:
    """Pure pipeline definitions plus deterministic file-manifest discovery."""

    @staticmethod
    def task(task_id: str, operation: str, *arguments: Any, **options: Any) -> Task:
        """Build an operation request without capturing application or scene state."""
        return Task(
            task_id,
            operation,
            TaskCall(tuple(arguments), tuple(Option(k, v) for k, v in options.items())),
            task_id,
        )

    @staticmethod
    def files(directory: Path, extensions: tuple[str, ...]) -> dict[str, Path]:
        """Discover one supported file per specimen and reject ambiguous identities."""
        if not directory.is_dir():
            raise ValueError(f"Directory does not exist: {directory}")
        found = {}
        for path in sorted(directory.iterdir()):
            if path.name.startswith(".") or not path.is_file():
                continue
            suffix = next(
                (suffix for suffix in extensions if path.name.lower().endswith(suffix)),
                None,
            )
            if suffix is None:
                continue
            identity = path.name[: -len(suffix)]
            if identity in found:
                raise ValueError(
                    f"Multiple files for specimen {identity!r} in {directory}."
                )
            found[identity] = path
        return found

    @staticmethod
    def manifest(options: AtlasOptions) -> tuple[SpecimenFiles, ...]:
        """Pair models, landmarks, and optional textures by exact specimen ID."""
        meshes = Workflows.files(options.models, (".obj", ".ply", ".stl", ".vtp"))
        landmarks = Workflows.files(
            options.landmarks, (".mrk.json", ".fcsv", ".json", ".csv")
        )
        textures = (
            {}
            if options.textures is None
            else Workflows.files(
                options.textures, (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")
            )
        )
        if not meshes or meshes.keys() != landmarks.keys():
            raise ValueError(
                "Models and landmarks must contain the same nonempty set of specimen IDs."
            )
        if options.textures is not None and not meshes.keys() <= textures.keys():
            raise ValueError("A matching texture is required for every specimen.")
        return tuple(
            SpecimenFiles(key, mesh, landmarks[key], textures.get(key))
            for key, mesh in meshes.items()
        )

    @staticmethod
    def atlas(options: AtlasOptions) -> Pipeline:
        """Define barriers from inspection through atlas construction and texture transfer."""
        manifest = Workflows.manifest(options)
        # A fresh directory prevents accidental replacement of a previous run's artifacts.
        stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        output = options.output / f"interdeca_{stamp}_{uuid4().hex[:8]}"
        task, ref = Workflows.task, ResultRef
        stages = [
            [task("blender", "blender", options.blender)]
            + [
                task(f"inspect_{i}", "inspect", *item)
                for i, item in enumerate(manifest)
            ]
        ]
        stages.append(
            [
                task(
                    "specimens",
                    "validate",
                    tuple(ref(f"inspect_{i}") for i in range(len(manifest))),
                )
            ]
        )
        if options.atlas_model is not None:
            stages.append(
                [
                    task(
                        "atlas",
                        "load_atlas",
                        options.atlas_model,
                        options.atlas_landmarks,
                    )
                ]
            )
        else:
            stages.append([task("reference", "reference", ref("specimens"))])
            stages.append(
                [
                    task(
                        f"contribute_{i}",
                        "contribute",
                        ref("specimens", (i,)),
                        ref("reference"),
                    )
                    for i in range(len(manifest))
                ]
            )
            stages.append(
                [
                    task(
                        "atlas",
                        "mean",
                        tuple(ref(f"contribute_{i}") for i in range(len(manifest))),
                    )
                ]
            )
        stages.append(
            [
                task(
                    "clean",
                    "clean",
                    ref("atlas"),
                    ref("blender"),
                    merge_distance=options.merge_distance,
                )
            ]
        )
        stages.append(
            [
                task(
                    "uv",
                    "uv",
                    ref("clean"),
                    ref("blender"),
                    angle_limit_degrees=options.uv_angle,
                    island_margin=options.island_margin,
                )
            ]
        )
        stages.append(
            [task("saved_atlas", "save_atlas", ref("uv"), output)]
            + [
                task(
                    f"align_{i}",
                    "align",
                    ref("specimens", (i,)),
                    ref("uv", ("landmarks",)),
                )
                for i in range(len(manifest))
            ]
        )
        stages.append(
            [
                task(
                    f"aligned_{i}",
                    "save_specimen",
                    ref(f"align_{i}", ("specimen",)),
                    output / "aligned",
                )
                for i in range(len(manifest))
            ]
        )
        stages.append(
            [
                task(
                    f"resample_{i}",
                    "resample",
                    ref(f"aligned_{i}"),
                    ref("uv"),
                    mean_landmarks=ref("uv", ("landmarks",)),
                )
                for i in range(len(manifest))
            ]
        )
        stages.append(
            [
                task(
                    f"resampled_{i}",
                    "save_specimen",
                    ref(f"resample_{i}"),
                    output / "resampled",
                )
                for i in range(len(manifest))
            ]
        )
        baked = ()
        if options.textures is not None:
            stages.append(
                [
                    task(
                        f"bake_{i}",
                        "bake",
                        ref(f"aligned_{i}"),
                        ref(f"resampled_{i}"),
                        ref("blender"),
                        output / "textures" / f"{item.specimen_id}.png",
                        resolution=(options.bake_size, options.bake_size),
                        cage_extrusion=options.extrusion,
                        margin_pixels=options.bake_margin,
                    )
                    for i, item in enumerate(manifest)
                ]
            )
            baked = tuple(ref(f"bake_{i}") for i in range(len(manifest)))
        stages.append(
            [
                task(
                    "average",
                    "average",
                    baked,
                    output / "textures" / "average_texture.png",
                )
            ]
        )
        stages.append(
            [
                task(
                    "result",
                    "atlas_result",
                    ref("saved_atlas"),
                    output,
                    baked,
                    ref("average"),
                )
            ]
        )
        return Pipeline(stages, (ref("result"),), "ATLAS and Texture Transfer")

    @staticmethod
    def colors(mesh: Mesh, options: ColorOptions) -> Pipeline:
        """Define shared sampling and population normalization before specimen processing."""
        paths = Workflows.files(
            options.directory, (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")
        )
        paths = {
            key: path
            for key, path in paths.items()
            if not key.lower().startswith(("average_texture", "mean_texture"))
        }
        if not paths:
            raise ValueError("No specimen textures found.")
        task, ref = Workflows.task, ResultRef
        stages = [
            [task("analysis", "analysis", mesh, options.sample_count)]
            + [
                task(f"texture_{i}", "texture", path, mesh.topology_key)
                for i, path in enumerate(paths.values())
            ]
        ]
        stages.append(
            [
                task(
                    f"sample_{i}",
                    "sample",
                    ref(f"texture_{i}"),
                    ref("analysis"),
                    smoothing_steps=int(options.neighbor_average),
                )
                for i in range(len(paths))
            ]
        )
        normalization = None
        if options.clustering and options.normalize:
            stages.append(
                [
                    task(
                        "normalization",
                        "normalization",
                        tuple(ref(f"sample_{i}") for i in range(len(paths))),
                    )
                ]
            )
            normalization = ref("normalization")
        stages.append(
            [
                task(
                    f"process_{i}",
                    "process",
                    ref(f"sample_{i}"),
                    normalization=normalization,
                    n_clusters=options.clusters if options.clustering else None,
                    initial_clusters=options.initial_clusters
                    if options.clustering
                    else None,
                )
                for i in range(len(paths))
            ]
        )
        specimens = tuple(ref(f"process_{i}") for i in range(len(paths)))
        alignment = None
        if options.clustering:
            stages.append([task("match", "match", specimens)])
            alignment = ref("match")
        stages.append(
            [
                task(
                    "result",
                    "color_result",
                    ref("analysis"),
                    specimens,
                    alignment,
                    options,
                )
            ]
        )
        return Pipeline(stages, (ref("result"),), "Color processing")

    @staticmethod
    def population(colors: ColorResult, options: ModelOptions) -> Pipeline:
        """Define feature assembly and fitting as separate dependent operations."""
        task, ref = Workflows.task, ResultRef
        # Preserve the legacy choice; unclustered full-resolution data still uses colors.
        representation = (
            "cluster_areas"
            if colors.options.sample_count is None and colors.alignment is not None
            else "colors"
        )
        return Pipeline(
            [
                [
                    task(
                        "features",
                        "features",
                        colors.alignment or colors.specimens,
                        colors.analysis,
                        representation=representation,
                        missing="drop",
                        area_normalization="l2",
                    )
                ],
                [
                    task(
                        "fit",
                        "fit",
                        ref("features"),
                        method=options.method,
                        n_components=options.components,
                    )
                ],
                [task("result", "population_result", ref("features"), ref("fit"))],
            ],
            (ref("result"),),
            "Population analysis",
        )

    @staticmethod
    def single(operation: str, *arguments: Any, **options: Any) -> Pipeline:
        """Wrap a small independent operation in the same execution contract."""
        return Pipeline(
            [[Workflows.task("result", operation, *arguments, **options)]],
            (ResultRef("result"),),
            operation,
        )
