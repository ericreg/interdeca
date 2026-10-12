"""Build staged application workflows from the stateless numerical operations."""

from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, NamedTuple
from uuid import uuid4

import numpy as np
from skimage.color import lab2rgb

from interdeca._morphoweave_host import REVISION, SOURCE_SHA256
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
from interdeca.execution import Option, Pipeline, ResultRef, Task, TaskCall
from interdeca.geometry import Geometry
from interdeca.io import Files, LandmarkIO, MeshIO, TextureIO
from interdeca.models import (
    AtlasModel,
    AtlasScaleConversion,
    BakedTexture,
    BlenderInstallation,
    Mesh,
    ResampledSpecimen,
    Specimen,
)
from interdeca.morphoweave import MorphoWeave
from interdeca.population import Population
from interdeca.surface import MeshProcessing, TextureBaking
from interdeca.serde import AtlasRequest, AtlasRunData, AtlasScaleConversionData


class SpecimenFiles(NamedTuple):
    """A manifest entry paired by specimen identity, not directory iteration order."""

    specimen_id: str
    mesh: Path
    landmarks: Path
    texture: Path | None


class Operations:
    """Small task adapters for I/O and combinations that have no GUI dependency."""

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
    def save_specimen(
        specimen: Specimen | ResampledSpecimen, directory: Path
    ) -> Specimen | ResampledSpecimen:
        """Publish aligned or resampled geometry and retain its numerical result."""
        # Only projected outputs may retain collapsed faces; aligned native
        # specimens still need valid source triangles for surface queries.
        MeshIO.write_obj(
            specimen.mesh,
            directory / f"{specimen.specimen_id}.obj",
            allow_degenerate=isinstance(specimen, ResampledSpecimen),
        )
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
        atlas: AtlasModel,
        directory: Path,
        textures: tuple,
        average: Path | None,
        options: AtlasOptions,
        scale_conversion: AtlasScaleConversion | None,
        prepared_input_directory: Path | None = None,
        surface_installation: BlenderInstallation | None = None,
    ) -> AtlasResult:
        """Collect the declared outputs of a successful atlas workflow."""
        # Preserve backend identity and coordinate conventions for side-by-side
        # analysis; matching hosts alone does not establish numerical equivalence.
        upstream = options.backend == "morphoweave"
        manifest = AtlasRunData(
            options=AtlasRequest.from_internal(options),
            prepared_input_directory=prepared_input_directory,
            texture_average_method="encoded_rgb_mean" if textures else None,
            blender_executable=(
                None if surface_installation is None else surface_installation.executable
            ),
            blender_version=(
                None if surface_installation is None else surface_installation.version
            ),
            upstream_revision=REVISION if upstream else None,
            upstream_source_sha256=SOURCE_SHA256 if upstream else None,
            atlas_scale_conversion=(
                None
                if scale_conversion is None
                else AtlasScaleConversionData.from_internal(scale_conversion)
            ),
            reference_normalizes_scale=(
                None
                if options.atlas_model is not None
                else upstream or options.normalize_scale
            ),
            correspondence_method="morphoweave_tps_projection"
            if upstream
            else "barycentric_common_space",
            projection_distance_space="aligned_specimen"
            if upstream
            else "common_warped",
        )
        Files.write(
            directory / "atlas_run.json",
            (manifest.model_dump_json(indent=2) + "\n").encode(),
        )
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
    def task(
        task_id: str, operation: Callable[..., Any], *arguments: Any, **options: Any
    ) -> Task:
        """Keep the callable separate from inputs so queued values can be snapshotted."""
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
        options = AtlasRequest.from_internal(options).to_internal()
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
        # Root inputs require exact pairing: extra texture IDs are errors too.
        # Legacy explicit texture directories may also contain average previews.
        if options.input_dir is not None and meshes.keys() != textures.keys():
            missing = sorted(meshes.keys() - textures.keys())
            extra = sorted(textures.keys() - meshes.keys())
            raise ValueError(
                "Texture specimen IDs must exactly match models and landmarks; "
                f"missing={missing}, extra={extra}."
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
        options = AtlasRequest.from_internal(options).to_internal()
        manifest = Workflows.manifest(options)
        # A fresh directory prevents accidental replacement of a previous run's artifacts.
        stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        run_name = f"interdeca_{stamp}_{uuid4().hex[:8]}"
        output = options.output / run_name
        # Each request snapshots freshly normalized inputs rather than reusing
        # stale conversions after source edits or an interrupted preparation.
        prepared = (
            None
            if options.input_dir is None
            else options.input_dir / "_interdeca" / "prepared" / run_name
        )
        task, ref = Workflows.task, ResultRef
        # Atlas construction and surface processing are independent choices.
        # Rust needs no installation; Blender is discovered once per workflow.
        use_blender = options.surface_backend == "blender"
        surface_arguments = (ref("blender_installation"),) if use_blender else ()
        surface_options = {"workspace_parent": prepared} if use_blender else {}
        stages = [
            [
                task(
                    f"inspect_{i}", Dataset.prepare_specimen, *item, prepared,
                    mesh_coordinate_system=options.input_coordinates,
                )
                if prepared is not None
                else task(
                    f"inspect_{i}", Dataset.inspect_specimen, *item,
                    mesh_coordinate_system=options.input_coordinates,
                )
                for i, item in enumerate(manifest)
            ]
        ]
        if options.overwrite:
            # This stage is a barrier: no parallel preparation can start until
            # the previous generated workspace has been removed in full.
            protected = tuple(
                path for item in manifest for path in item[1:] if path is not None
            ) + tuple(
                path for path in (options.atlas_model, options.atlas_landmarks)
                if path is not None
            )
            stages.insert(0, [task(
                "reset_workspace", Dataset.reset_workspace, options.input_dir, protected,
            )])
        if use_blender:
            # Reject an unavailable Blender before an overwrite removes old runs.
            stages.insert(0, [task(
                "blender_installation", Blender.check_available, options.blender,
            )])
        stages.append(
            [
                task(
                    "specimens",
                    Dataset.validate_specimens,
                    tuple(ref(f"inspect_{i}") for i in range(len(manifest))),
                )
            ]
        )
        if options.backend == "morphoweave":
            # Final alignment returns owned arrays and external texture paths.
            # Its staging files/cache have no consumers during later baking.
            # Failures before that boundary still close the resource normally.
            stages.append(
                [
                    task(
                        "morphoweave_inputs", MorphoWeave.stage_inputs, ref("specimens"),
                        workspace_parent=prepared,
                    )._replace(
                        manage_resource=True, release_after="morphoweave_aligned"
                    )
                ]
            )
        if options.atlas_model is not None:
            stages.append(
                [
                    task(
                        "atlas",
                        Operations.load_atlas,
                        options.atlas_model,
                        options.atlas_landmarks,
                    )
                ]
            )
        elif options.backend == "morphoweave":
            stages.append(
                [
                    task(
                        "atlas",
                        MorphoWeave.build_atlas,
                        ref("specimens"),
                        normalize_scale=options.normalize_scale,
                        workspace=ref("morphoweave_inputs"),
                    )
                ]
            )
        else:
            stages.append(
                [
                    task(
                        "reference",
                        Atlas.compute_reference,
                        ref("specimens"),
                        allow_scaling=options.normalize_scale,
                    )
                ]
            )
            stages.append(
                [
                    task(
                        f"contribute_{i}",
                        Atlas.prepare_contribution,
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
                        Atlas.assemble_mean,
                        tuple(ref(f"contribute_{i}") for i in range(len(manifest))),
                    )
                ]
            )
        # Convert generated MorphoWeave means before any distance-based cleanup.
        # Supplied atlases already define their coordinate convention and size.
        atlas_source = ref("atlas")
        scale_conversion = None
        if options.backend == "morphoweave" and options.atlas_model is None:
            stages.append(
                [
                    task(
                        "scaled_atlas",
                        MorphoWeave.restore_input_scale,
                        atlas_source,
                        ref("specimens"),
                    )
                ]
            )
            atlas_source = ref("scaled_atlas", ("atlas",))
            scale_conversion = ref("scaled_atlas", ("conversion",))
        stages.append(
            [
                task(
                    "clean",
                    Blender.clean_atlas if use_blender else MeshProcessing.clean_atlas,
                    atlas_source,
                    *surface_arguments,
                    merge_distance=options.merge_distance,
                    **surface_options,
                )
            ]
        )
        stages.append(
            [
                task(
                    "uv",
                    Blender.prepare_atlas_uv
                    if use_blender
                    else MeshProcessing.prepare_atlas_uv,
                    ref("clean"),
                    *surface_arguments,
                    angle_limit_degrees=options.uv_angle,
                    island_margin=options.island_margin,
                    **surface_options,
                )
            ]
        )
        # MorphoWeave's high-level runAlign owns the whole alignment batch. The
        # native backend keeps its original per-specimen implementation intact.
        if options.backend == "morphoweave":
            stages.append(
                [
                    task(
                        "morphoweave_aligned",
                        MorphoWeave.align_specimens,
                        ref("specimens"),
                        ref("uv"),
                        normalize_scale=options.normalize_scale,
                        workspace=ref("morphoweave_inputs"),
                    )
                ]
            )
            aligned = tuple(
                ref("morphoweave_aligned", (i,)) for i in range(len(manifest))
            )
        else:
            stages.append(
                [
                    task(
                        f"align_{i}",
                        Atlas.align_specimen,
                        ref("specimens", (i,)),
                        ref("uv", ("landmarks",)),
                        allow_scaling=options.normalize_scale,
                    )
                    for i in range(len(manifest))
                ]
            )
            aligned = tuple(
                ref(f"align_{i}", ("specimen",)) for i in range(len(manifest))
            )
        stages.append([task("saved_atlas", Operations.save_atlas, ref("uv"), output)])
        stages.append(
            [
                task(
                    f"aligned_{i}",
                    Operations.save_specimen,
                    aligned[i],
                    output / "aligned",
                )
                for i in range(len(manifest))
            ]
        )
        stages.append(
            [
                task(
                    f"resample_{i}",
                    MorphoWeave.resample_specimen
                    if options.backend == "morphoweave"
                    else Geometry.resample_specimen,
                    ref(f"aligned_{i}"),
                    ref("uv"),
                    **(
                        {}
                        if options.backend == "morphoweave"
                        else {"mean_landmarks": ref("uv", ("landmarks",))}
                    ),
                )
                for i in range(len(manifest))
            ]
        )
        stages.append(
            [
                task(
                    f"resampled_{i}",
                    Operations.save_specimen,
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
                        Blender.bake_specimen_texture
                        if use_blender
                        else TextureBaking.bake_specimen_texture,
                        ref(f"aligned_{i}"),
                        ref(f"resampled_{i}"),
                        *surface_arguments,
                        output / "textures" / f"{item.specimen_id}.png",
                        resolution=(options.bake_size, options.bake_size),
                        cage_extrusion=options.extrusion,
                        margin_pixels=options.bake_margin,
                        **surface_options,
                    )
                    for i, item in enumerate(manifest)
                ]
            )
            baked = tuple(ref(f"bake_{i}") for i in range(len(manifest)))
        stages.append(
            [
                task(
                    "average",
                    Operations.average,
                    baked,
                    output / "textures" / "average_texture.png",
                )
            ]
        )
        stages.append(
            [
                task(
                    "result",
                    Operations.atlas_result,
                    ref("saved_atlas"),
                    output,
                    baked,
                    ref("average"),
                    options,
                    scale_conversion,
                    prepared_input_directory=prepared,
                    surface_installation=ref("blender_installation") if use_blender else None,
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
            [task("analysis", Operations.prepare_analysis, mesh, options.sample_count)]
            + [
                task(f"texture_{i}", Operations.texture, path, mesh.topology_key)
                for i, path in enumerate(paths.values())
            ]
        ]
        stages.append(
            [
                task(
                    f"sample_{i}",
                    ColorAnalysis.sample_specimen_colors,
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
                        ColorAnalysis.compute_normalization_stats,
                        tuple(ref(f"sample_{i}") for i in range(len(paths))),
                    )
                ]
            )
            normalization = ref("normalization")
        stages.append(
            [
                task(
                    f"process_{i}",
                    ColorAnalysis.process_specimen_colors,
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
            stages.append([task("match", Population.align_clusters, specimens)])
            alignment = ref("match")
        stages.append(
            [
                task(
                    "result",
                    Operations.color_result,
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
                        Population.assemble_features,
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
                        Population.fit_model,
                        ref("features"),
                        method=options.method,
                        n_components=options.components,
                    )
                ],
                [task("result", PopulationResult, ref("features"), ref("fit"))],
            ],
            (ref("result"),),
            "Population analysis",
        )

    @staticmethod
    def single(
        operation: Callable[..., Any], *arguments: Any, **options: Any
    ) -> Pipeline:
        """Wrap a small independent operation in the same execution contract."""
        task = Workflows.task("result", operation, *arguments, **options)
        return Pipeline([[task]], (ResultRef("result"),), task.operation_name)
