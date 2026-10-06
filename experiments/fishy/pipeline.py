"""Reproduce generation and run the existing Interdeca numerical pipeline."""

import csv
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from experiments.fishy.artifacts import Artifacts
from experiments.fishy.generation import Generation
from experiments.fishy.serde import Config, Manifest, ParameterData, Sample
from interdeca import Atlas, Blender, Dataset, Geometry, LandmarkIO, MeshIO, TextureIO


class Pipeline:
    """Small research orchestration; the production core never imports this code."""

    @staticmethod
    def donors(config: Config) -> tuple:
        """Check all donor geometry and semantic landmarks before writing outputs."""
        specimens = Dataset.validate_specimens(
            [
                Dataset.inspect_specimen(
                    d.donor_id,
                    d.mesh.path,
                    d.landmarks.path,
                    mesh_coordinate_system=d.mesh_coordinates,
                    landmark_coordinate_system=d.landmark_coordinates,
                )
                for d in config.donors
            ]
        )
        for specimen in specimens:
            if len(specimen.landmarks.labels) != config.generation.expected_landmarks:
                raise ValueError(f"{specimen.specimen_id}: unexpected landmark count.")
            if specimen.mesh.uv is None or np.any(
                (specimen.mesh.uv < 0) | (specimen.mesh.uv > 1)
            ):
                raise ValueError(
                    f"{specimen.specimen_id}: generation requires native UVs in one tile."
                )
            Generation.canonical(specimen.landmarks)
        return specimens

    @staticmethod
    def generate(config_path: Path, output: Path, executable: str | None) -> None:
        """Generate paired inputs and a separate parameter table with stable IDs."""
        config = Artifacts.config(config_path)
        donors = Pipeline.donors(config)
        settings = config.generation
        sampled = Generation.sample_labeled(settings.samples, settings.seed)
        parameters = [
            ParameterData.model_validate(p.parameters._asdict()).to_internal()
            for p in sampled
        ]
        installation = Blender.check_available(executable)
        frames = [Generation.canonical(d.landmarks) for d in donors]
        reference = next(
            i for i, d in enumerate(donors) if d.specimen_id == config.reference_id
        )
        splines = [
            Generation.fit(frames[reference][1], points, settings.tps_lambda)
            for _, points in frames
        ]
        # Preserve the original independent donor-assignment RNG and donor ordering.
        assigned = np.random.RandomState(settings.seed).randint(
            0, len(donors), settings.samples
        )
        Artifacts.start(output, "generate", [config_path])
        Artifacts.write(output / "config.json", config.model_dump(mode="json"))
        Artifacts.write(
            output / "blender.json",
            {
                "executable": str(installation.executable),
                "version": str(installation.version),
                "device": "CPU",
                "attribute": "BYTE_COLOR",
                "color_assignment": "linear color property, as in source generator",
                "emission_margin": settings.emission_margin,
            },
        )
        for name in ("meshes", "landmarks", "images", "logs", "ground_truth"):
            (output / name).mkdir()
        samples = []
        # Write parameter rows as samples finish; a failed bake leaves a useful log
        # and partial table, but never a completed manifest or success marker.
        with (output / "ground_truth/samples.csv").open("x", newline="") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=[
                    "sample_id",
                    "donor_id",
                    "belly_group",
                    "tail_group",
                    *ParameterData.model_fields,
                ],
            )
            writer.writeheader()
            for index, (parameters_i, donor_index) in enumerate(
                zip(parameters, assigned, strict=True)
            ):
                donor = donors[donor_index]
                sample_id = f"{donor.specimen_id}_fishy_{index:06d}"
                print(
                    f"Generate {index + 1}/{len(parameters)}: {sample_id}", flush=True
                )
                brushes = Generation.warp(
                    Generation.brushes(parameters_i),
                    splines[donor_index],
                    frames[donor_index][0],
                )
                rgb = Generation.paint(
                    donor.mesh.vertices,
                    brushes,
                    settings.noise_amount,
                    settings.noise_scale,
                )
                texture = output / "images" / f"{sample_id}.png"
                with tempfile.TemporaryDirectory(prefix="fishy-emission-") as temporary:
                    payload = Path(temporary) / "input.npz"
                    np.savez(
                        payload,
                        vertices=donor.mesh.vertices,
                        faces=donor.mesh.faces,
                        uv=donor.mesh.uv,
                        rgb=rgb,
                    )
                    worker = Path(__file__).with_name("_blender_worker.py")
                    command = [
                        str(installation.executable),
                        "--background",
                        "--factory-startup",
                        "--python-exit-code",
                        "1",
                        "--python",
                        str(worker),
                        "--",
                        str(payload),
                        str(texture),
                        str(settings.resolution),
                        str(settings.emission_margin),
                    ]
                    with (output / "logs" / f"{sample_id}.log").open("x") as log:
                        try:
                            subprocess.run(
                                command,
                                stdout=log,
                                stderr=subprocess.STDOUT,
                                timeout=config.transfer.timeout,
                                check=True,
                            )
                        except (
                            subprocess.CalledProcessError,
                            subprocess.TimeoutExpired,
                        ) as error:
                            raise RuntimeError(
                                f"Emission bake failed for {sample_id}; inspect {log.name}. "
                                "The run is incomplete and must not be used for analysis."
                            ) from error
                pixels = TextureIO.read(texture)
                if pixels.rgb.shape != (settings.resolution, settings.resolution, 3):
                    raise ValueError(
                        f"Unexpected baked image dimensions for {sample_id}"
                    )
                mesh_path = output / "meshes" / f"{sample_id}.obj"
                landmarks_path = output / "landmarks" / f"{sample_id}.mrk.json"
                # Writers publish normalized RAS coordinates with untouched native UVs.
                MeshIO.write_obj(donor.mesh, mesh_path)
                LandmarkIO.write_json(donor.landmarks, landmarks_path)
                samples.append(
                    Sample(
                        sample_id=sample_id,
                        donor_id=donor.specimen_id,
                        mesh=Artifacts.asset(mesh_path, output),
                        landmarks=Artifacts.asset(landmarks_path, output),
                        texture=Artifacts.asset(texture, output),
                    )
                )
                writer.writerow(
                    {
                        "sample_id": sample_id,
                        "donor_id": donor.specimen_id,
                        "belly_group": sampled[index].belly_group,
                        "tail_group": sampled[index].tail_group,
                        **parameters_i._asdict(),
                    }
                )
                stream.flush()
        manifest = Manifest(kind="generated", backend="generator", samples=samples)
        Artifacts.write(output / "manifest.json", manifest.model_dump(mode="json"))
        Artifacts.complete(output)

    @staticmethod
    def transfer(
        config_path: Path, manifest_path: Path, output: Path, executable: str | None
    ) -> None:
        """Run Eric's current atlas core; this is explicitly an Interdeca rerun."""
        config = Config.model_validate_json(config_path.read_text())
        manifest = Artifacts.manifest(manifest_path)
        if manifest.kind != "generated":
            raise ValueError("Transfer expects a generated native-space manifest.")
        if len(manifest.samples) != config.generation.samples:
            raise ValueError(
                "Manifest sample count differs from the generation configuration."
            )
        donor_ids = {donor.donor_id for donor in config.donors}
        if any(sample.donor_id not in donor_ids for sample in manifest.samples):
            raise ValueError("Manifest contains a donor absent from the configuration.")
        specimens = Dataset.validate_specimens(
            [
                Dataset.inspect_specimen(
                    s.sample_id, s.mesh.path, s.landmarks.path, s.texture.path
                )
                for s in manifest.samples
            ],
            require_textures=True,
        )
        installation = Blender.check_available(executable)
        options = config.transfer
        Artifacts.start(output, "transfer-interdeca", [config_path, manifest_path])
        Artifacts.write(
            output / "input-manifest.json", manifest.model_dump(mode="json")
        )
        Artifacts.write(output / "config.json", config.model_dump(mode="json"))
        Artifacts.write(
            output / "blender.json",
            {
                "executable": str(installation.executable),
                "version": str(installation.version),
            },
        )
        # Every generated specimen contributes, including repeated donor shapes.
        # Deduplicating donors here would change the population's mean atlas.
        reference = Atlas.compute_reference(specimens, allow_scaling=False)
        atlas = Atlas.assemble_mean(
            [Atlas.prepare_contribution(s, reference) for s in specimens]
        )
        atlas = Blender.clean_atlas(
            atlas,
            installation,
            merge_distance=options.merge_distance,
            timeout=options.timeout,
        )
        atlas = Blender.prepare_atlas_uv(
            atlas,
            installation,
            angle_limit_degrees=options.angle_degrees,
            island_margin=options.island_margin,
            timeout=options.timeout,
        )
        MeshIO.write_obj(atlas.mesh, output / "atlas.obj")
        LandmarkIO.write_json(atlas.landmarks, output / "atlas.mrk.json")
        # Fingerprint the serialized mesh as readers will see it, not an in-memory
        # float array that may round differently during a text OBJ round trip.
        atlas_saved = MeshIO.read(output / "atlas.obj")
        Artifacts.write(
            output / "atlas-info.json",
            {
                "backend": "interdeca",
                "topology_key": atlas_saved.topology_key,
                "reference_sample": reference.reference_id,
                "vertices": len(atlas.mesh.vertices),
                "faces": len(atlas.mesh.faces),
                "warning": "Not certified equivalent to the paper's legacy ATLAS run.",
            },
        )
        result = []
        for sample, specimen in zip(manifest.samples, specimens, strict=True):
            print(f"Transfer: {sample.sample_id}", flush=True)
            aligned = Atlas.align_specimen(specimen, atlas.landmarks).specimen
            target = Geometry.resample_specimen(
                aligned, atlas, mean_landmarks=atlas.landmarks
            )
            MeshIO.write_obj(
                aligned.mesh, output / "aligned" / f"{sample.sample_id}.obj"
            )
            LandmarkIO.write_json(
                aligned.landmarks, output / "aligned" / f"{sample.sample_id}.mrk.json"
            )
            mesh_path = output / "resampled" / f"{sample.sample_id}.obj"
            landmarks_path = output / "resampled" / f"{sample.sample_id}.mrk.json"
            MeshIO.write_obj(target.mesh, mesh_path)
            LandmarkIO.write_json(target.landmarks, landmarks_path)
            np.save(
                output / "resampled" / f"{sample.sample_id}.projection.npy",
                target.projection_distances,
            )
            texture = Blender.bake_specimen_texture(
                aligned,
                target,
                installation,
                output / "textures" / f"{sample.sample_id}.png",
                resolution=(options.resolution, options.resolution),
                cage_extrusion=options.cage_extrusion,
                margin_pixels=options.margin,
                timeout=options.timeout,
            )
            result.append(
                Sample(
                    sample_id=sample.sample_id,
                    donor_id=sample.donor_id,
                    mesh=Artifacts.asset(mesh_path, output),
                    landmarks=Artifacts.asset(landmarks_path, output),
                    texture=Artifacts.asset(texture.path, output),
                )
            )
        result_manifest = Manifest(
            kind="atlas",
            backend="interdeca",
            samples=result,
            atlas_mesh=Artifacts.asset(output / "atlas.obj", output),
            atlas_landmarks=Artifacts.asset(output / "atlas.mrk.json", output),
        )
        Artifacts.write(
            output / "manifest.json", result_manifest.model_dump(mode="json")
        )
        Artifacts.complete(output)
