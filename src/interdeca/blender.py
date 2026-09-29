"""Blender subprocess boundary for atlas UV generation and texture transfer.

Every invocation owns its temporary directory and Blender process. Scheduling,
process pools, retries, and UI progress belong to future execution managers.
"""

import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from interdeca._blender_protocol import BlenderBakeRequest, BlenderUVRequest
from interdeca.io import Files, TextureIO
from interdeca.models import (
    AtlasModel,
    BakedTexture,
    BlenderInstallation,
    Mesh,
    ResampledSpecimen,
    Specimen,
)
from interdeca.serde import BlenderBakeRequestData, BlenderUVRequestData, MeshData


class BlenderError(RuntimeError):
    """Blender failed or timed out; the exception includes captured diagnostics."""


class Blender:
    """Namespace for operations requiring an external Blender installation."""

    @staticmethod
    def check_available(
        executable: Path | str | None = None, *, timeout: float = 15
    ) -> BlenderInstallation:
        """Find Blender and read its version without installing or modifying it.

        Resolution order is an explicit executable, BLENDER_EXECUTABLE, PATH,
        then conventional macOS/Windows installation locations. An explicitly
        supplied executable is authoritative; failures do not fall back silently.
        """
        # Discovery should fail promptly rather than hang a workflow on an
        # invalid executable or an unbounded version query.
        if not np.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be finite and positive.")
        # Honor an explicit choice before searching so a broken requested
        # installation is not silently replaced by a different Blender version.
        requested = (
            str(executable)
            if executable is not None
            else os.environ.get("BLENDER_EXECUTABLE")
        )
        if requested:
            candidate = Path(shutil.which(requested) or requested).expanduser()
        else:
            # Common install locations provide a fallback when Blender has not
            # been added to the shell's executable search path.
            discovered = shutil.which("blender")
            candidates = [Path(discovered)] if discovered else []
            candidates.extend(
                (
                    Path("/Applications/Blender.app/Contents/MacOS/Blender"),
                    Path.home() / "Applications/Blender.app/Contents/MacOS/Blender",
                )
            )
            if os.name == "nt":
                root = (
                    Path(os.environ.get("PROGRAMFILES", "C:/Program Files"))
                    / "Blender Foundation"
                )
                candidates.extend(
                    sorted(root.glob("Blender */blender.exe"), reverse=True)
                )
            candidate = next(
                (path for path in candidates if path.is_file()),
                Path("/nonexistent-blender"),
            )
        if not candidate.is_file():
            raise FileNotFoundError(
                "Blender executable was not found. Pass its path or set BLENDER_EXECUTABLE."
            )
        # Ask the executable itself for its version rather than trusting the
        # filename or installation directory, which users can rename.
        try:
            completed = subprocess.run(
                [str(candidate.resolve()), "--version"],
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            raise BlenderError(
                f"Could not query Blender at {candidate}: {error}"
            ) from error
        # A recognizable supported version is needed before relying on the
        # Blender operators used by the worker scripts.
        match = re.search(r"Blender\s+(\d+)\.(\d+)\.(\d+)", completed.stdout)
        if completed.returncode != 0 or match is None:
            raise BlenderError(
                f"Blender version query failed: {completed.stdout}\n{completed.stderr}"
            )
        version = tuple(int(part) for part in match.groups())
        if version < (3, 6, 0):
            raise BlenderError(f"Blender 3.6 or newer is required; found {version}.")
        return BlenderInstallation(
            candidate.resolve(), version, completed.stdout.splitlines()[0]
        )

    @staticmethod
    def prepare_atlas_uv(
        atlas: AtlasModel,
        installation: BlenderInstallation,
        *,
        angle_limit_degrees: float = 66,
        island_margin: float = 0.02,
        timeout: float = 600,
        threads: int = 1,
    ) -> AtlasModel:
        """Create a shared Smart UV map while preserving all vertices and faces.

        The input is already a valid triangular mesh. This operation performs no
        merging, remeshing, or decimation: those would destroy correspondence.
        Angle limits are public degrees and converted to Blender's radians in
        the worker. The returned atlas owns its new per-corner UV array.
        """
        # Validate the public settings before starting Blender so configuration
        # mistakes are reported directly to the caller.
        if not np.isfinite(angle_limit_degrees) or not 0 < angle_limit_degrees <= 89:
            raise ValueError("angle_limit_degrees must lie in (0, 89].")
        if not np.isfinite(island_margin) or not 0 <= island_margin < 1:
            raise ValueError("island_margin must lie in [0, 1).")
        # Each invocation needs its own files so concurrent atlas preparations
        # cannot overwrite one another's mesh, request, or UV output.
        with tempfile.TemporaryDirectory(prefix="interdeca-uv-") as directory:
            root = Path(directory)
            Blender._save_mesh(atlas.mesh, root / "atlas.npz")
            config = BlenderUVRequest(
                mesh=str(root / "atlas.npz"),
                output=str(root / "uv.npy"),
                angle_limit_degrees=angle_limit_degrees,
                island_margin=island_margin,
            )
            Blender._run(installation, root, config, timeout=timeout, threads=threads)
            uv = np.load(root / "uv.npy", allow_pickle=False)
        # Smart UV packing should produce one tile; tolerate only float roundoff.
        if not np.isfinite(uv).all() or np.any((uv < -1e-6) | (uv > 1 + 1e-6)):
            raise BlenderError("Blender returned UVs outside the atlas texture tile.")
        # Accept only the new UVs from Blender, retaining the host's original
        # coordinates and connectivity so correspondence and precision are preserved.
        return AtlasModel(
            MeshData(
                vertices=atlas.mesh.vertices,
                faces=atlas.mesh.faces,
                uv=np.clip(uv, 0, 1),
            ).to_internal(),
            atlas.landmarks,
        )

    @staticmethod
    def bake_specimen_texture(
        specimen: Specimen,
        target: ResampledSpecimen,
        installation: BlenderInstallation,
        output_path: Path | str,
        *,
        resolution: tuple[int, int] = (2048, 2048),
        cage_extrusion: float = 0.01,
        max_ray_distance: float = 0,
        margin_pixels: int = 8,
        timeout: float = 1800,
        threads: int = 1,
        overwrite: bool = False,
    ) -> BakedTexture:
        """Bake native sRGB texture colors onto the specimen's shared atlas UVs.

        Source and target must be in the same aligned coordinate frame. Baking
        uses CPU Cycles, selected-to-active projection, and diffuse color only
        (no illumination). Distances use the mesh's length unit; zero maximum
        ray distance means Blender's unlimited setting. Tune ray distances to
        specimen scale and inspect the result for projection misses or occlusion.

        Blender runs in a private workspace and the PNG is published only after
        successful completion. Existing output files are refused by default.
        """
        # Matching identity and landmark positions prevent texture transfer
        # between different specimens or incompatible alignment frames.
        if specimen.specimen_id != target.specimen_id:
            raise ValueError("Source specimen and resampled target IDs must agree.")
        source_landmarks = specimen.landmarks.reordered(target.landmarks.labels)
        if not np.allclose(
            source_landmarks.points, target.landmarks.points, rtol=1e-8, atol=1e-10
        ):
            raise ValueError(
                "Source and resampled target landmarks must be in the same aligned frame."
            )
        # The source UVs locate colors in its image; the target UVs determine
        # where those colors are written in the shared atlas texture.
        if (
            specimen.texture_path is None
            or specimen.mesh.uv is None
            or target.mesh.uv is None
        ):
            raise ValueError(
                "Baking requires source texture, source UVs, and target atlas UVs."
            )
        if np.any((target.mesh.uv < 0) | (target.mesh.uv > 1)):
            raise ValueError("Target UVs must lie within one [0, 1] texture tile.")
        # Check image and ray settings on the host before paying the cost of
        # launching Blender or allocating a potentially large bake image.
        if len(resolution) != 2 or any(
            not isinstance(size, int) or size < 1 for size in resolution
        ):
            raise ValueError(
                "resolution must contain positive integer width and height."
            )
        if not isinstance(margin_pixels, int) or margin_pixels < 0:
            raise ValueError("margin_pixels must be a nonnegative integer.")
        if any(
            not np.isfinite(value) or value < 0
            for value in (cage_extrusion, max_ray_distance)
        ):
            raise ValueError("Baking distances must be finite and nonnegative.")
        # Protect the source image and refuse accidental replacement of existing
        # artifacts before starting work that would later be unable to save.
        destination = Path(output_path).expanduser().resolve()
        if destination.suffix.lower() != ".png":
            raise ValueError("Baked texture output must use the .png extension.")
        if destination == specimen.texture_path:
            raise ValueError("A bake output cannot replace its source texture.")
        if destination.exists() and not overwrite:
            raise FileExistsError(destination)
        # Decode the source once here so corrupt input is reported at the file
        # boundary instead of appearing as an opaque Blender subprocess failure.
        TextureIO.read(specimen.texture_path)
        # Keep intermediate files private until the worker has completed and its
        # output dimensions have been checked against the requested resolution.
        with tempfile.TemporaryDirectory(prefix="interdeca-bake-") as directory:
            root = Path(directory)
            Blender._save_mesh(specimen.mesh, root / "source.npz")
            Blender._save_mesh(target.mesh, root / "target.npz")
            config = BlenderBakeRequest(
                source=str(root / "source.npz"),
                target=str(root / "target.npz"),
                texture=str(specimen.texture_path),
                output=str(root / "baked.png"),
                resolution=tuple(resolution),
                cage_extrusion=cage_extrusion,
                max_ray_distance=max_ray_distance,
                margin_pixels=margin_pixels,
            )
            Blender._run(installation, root, config, timeout=timeout, threads=threads)
            rgb, _alpha = TextureIO.read(root / "baked.png")
            if rgb.shape[:2] != (resolution[1], resolution[0]):
                raise BlenderError(
                    "Blender produced an image with unexpected dimensions."
                )
            # Publish the checked image only after success so a worker failure
            # cannot leave its intermediate image at the requested output path.
            Files.write(
                destination, (root / "baked.png").read_bytes(), overwrite=overwrite
            )
        # Attach the atlas identity so downstream sampling can reject an image
        # baked with a different topology or UV layout.
        return BakedTexture(
            specimen.specimen_id,
            destination,
            target.mesh.topology_key,
            tuple(resolution),
        )

    @staticmethod
    def _save_mesh(mesh: Mesh, path: Path) -> None:
        """Pass plain numeric arrays to Blender without Python object pickling."""
        # Validate before crossing processes, and omit absent UVs entirely so the
        # archive contains only numeric arrays rather than a pickled None object.
        mesh = MeshData.from_internal(mesh).to_internal()
        if mesh.uv is None:
            np.savez(path, vertices=mesh.vertices, faces=mesh.faces)
        else:
            np.savez(path, vertices=mesh.vertices, faces=mesh.faces, uv=mesh.uv)

    @staticmethod
    def _run(
        installation: BlenderInstallation,
        root: Path,
        config: BlenderUVRequest | BlenderBakeRequest,
        *,
        timeout: float,
        threads: int,
    ) -> None:
        """Run the packaged worker with a JSON argument file and captured errors."""
        # Explicit time and thread limits let an execution manager bound the
        # resources used by each independently launched Blender process.
        if (
            not np.isfinite(timeout)
            or timeout <= 0
            or not isinstance(threads, int)
            or threads < 1
        ):
            raise ValueError(
                "timeout and threads must be positive; threads must be an integer."
            )
        config_path = root / "config.json"
        # Validate and serialize with the shared schema so the separate Blender
        # interpreter receives only fields understood by this worker version.
        data = (
            BlenderUVRequestData.from_internal(config)
            if isinstance(config, BlenderUVRequest)
            else BlenderBakeRequestData.from_internal(config)
        )
        config_path.write_text(data.model_dump_json(), encoding="utf-8")
        worker = Path(__file__).with_name("_blender_worker.py")
        environment = os.environ.copy()
        # Make the small shared protocol importable without exposing compiled
        # host dependencies that may be incompatible with Blender's Python.
        environment["PYTHONPATH"] = str(worker.parent)
        # Use a clean background scene and a nonzero exit code for script errors
        # so worker failures cannot be mistaken for successful Blender exits.
        command = [
            str(installation.executable),
            "--background",
            "--factory-startup",
            "--python-use-system-env",
            "--threads",
            str(threads),
            "--python-exit-code",
            "1",
            "--python",
            str(worker),
            "--",
            str(config_path),
        ]
        try:
            # Argument arrays avoid shell interpolation of user paths and specimen IDs.
            result = subprocess.run(
                command,
                cwd=root,
                env=environment,
                capture_output=True,
                text=True,
                errors="replace",
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as error:
            raise BlenderError(
                f"Blender {config.operation} exceeded {timeout} seconds."
            ) from error
        except OSError as error:
            raise BlenderError(f"Unable to start Blender: {error}") from error
        # Both a successful exit and an artifact are required; retain the end of
        # the worker log because that usually contains the actionable error.
        if result.returncode != 0 or not Path(config.output).is_file():
            diagnostics = (result.stdout + "\n" + result.stderr)[-12000:]
            raise BlenderError(
                f"Blender {config.operation} failed (exit {result.returncode}):\n{diagnostics}"
            )
