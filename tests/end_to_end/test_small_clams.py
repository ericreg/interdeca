"""Compare independent CLI and real Slicer runs, including their presentation boundary."""

import json
import os
import shutil
import signal
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from interdeca_client.artifacts import ArtifactStore
from PIL import Image

from interdeca.blender import Blender
from interdeca.io import LandmarkIO, MeshIO
from interdeca.serde import (
    AtlasRunData,
    ExportPresentation,
    FaceColorsPresentation,
    MeshPresentation,
    MorphControlsData,
    PlotPresentation,
    PopulationPresentationData,
    SelectionPresentation,
)

REPOSITORY = Path(__file__).resolve().parents[2]
SPECIMENS = ("UF_IZ_507781", "UF_IZ_507782", "UF_IZ_507879")
TEXTURES = (".png", ".tiff", ".png")


def executable(name, environment, conventional):
    """Honor an explicit tool choice and fail rather than silently skip integration."""
    requested = os.environ.get(environment)
    candidate = requested or shutil.which(name) or conventional
    path = Path(candidate).expanduser()
    assert path.is_file(), (
        f"Set {environment} to the {name} executable; not found: {path}"
    )
    return path


def run_process(command, log, environment, worker_pid=None):
    """Retain full diagnostics and bound external processes, including on interruption."""
    with log.open("w", encoding="utf-8") as stream:
        process = subprocess.Popen(
            [str(value) for value in command],
            cwd=REPOSITORY,
            env=environment,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            code = process.wait(timeout=2400)
        finally:
            if process.poll() is None:
                # The Slicer worker owns another process group; terminate only this run's IDs.
                groups = [process.pid]
                if worker_pid is not None and worker_pid.exists():
                    groups.append(int(worker_pid.read_text()))
                for pid in groups:
                    try:
                        os.killpg(pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                process.wait()
    assert code == 0, (
        f"Process exited {code}; full log: {log}\n"
        + log.read_text(encoding="utf-8", errors="replace")[-20000:]
    )


def prepare_inputs(root):
    """Convert only the three raw clam meshes from LPS to RAS in private scratch space."""
    source = (
        Path(
            os.environ.get(
                "INTERDECA_SMALL_CLAMS",
                REPOSITORY.parent / "3d_color_package/tests/data/small_clams",
            )
        )
        .expanduser()
        .resolve()
    )
    assert source.is_dir(), (
        f"Set INTERDECA_SMALL_CLAMS to the small_clams fixture: {source}"
    )
    for folder in ("models", "landmarks", "textures"):
        (root / folder).mkdir(parents=True)
    low, high = np.full(3, np.inf), np.full(3, -np.inf)
    for specimen, suffix in zip(SPECIMENS, TEXTURES, strict=True):
        mesh = MeshIO.read(
            source / "models" / f"{specimen}.obj", coordinate_system="LPS"
        )
        landmarks = LandmarkIO.read(source / "landmarks" / f"{specimen}.mrk.json")
        MeshIO.write_obj(mesh, root / "models" / f"{specimen}.obj")
        LandmarkIO.write_json(landmarks, root / "landmarks" / f"{specimen}.mrk.json")
        shutil.copy2(
            source / "textures" / f"{specimen}{suffix}",
            root / "textures" / f"{specimen}{suffix}",
        )
        low = np.minimum(low, landmarks.points.min(axis=0))
        high = np.maximum(high, landmarks.points.max(axis=0))
    # A rectangle in the two longest anatomical directions selects part of the shell.
    # Both hosts receive exactly these world RAS control points, independently of outputs.
    axes = np.argsort(high - low)[-2:]
    curve = np.tile((low + high) / 2, (4, 1))
    first, second = axes
    curve[:, first] = [
        low[first],
        (low[first] + high[first]) / 2,
        (low[first] + high[first]) / 2,
        low[first],
    ]
    curve[:, second] = [low[second], low[second], high[second], high[second]]
    np.save(root / "curve.npy", curve, allow_pickle=False)


def assert_mesh_equal(left, right, label):
    """Array comparison preserves correspondence; sorting vertices or faces would hide bugs."""
    np.testing.assert_allclose(
        left.vertices, right.vertices, rtol=1e-12, atol=1e-9, err_msg=label
    )
    np.testing.assert_array_equal(left.faces, right.faces, err_msg=label)
    assert (left.uv is None) == (right.uv is None), label
    if left.uv is not None:
        np.testing.assert_array_equal(left.uv, right.uv, err_msg=label + " corner UVs")


def compare_atlases(left, right):
    """Check the published atlas contract, allowing host-specific paths and discovery."""
    expected = {
        "atlas.obj",
        "atlas.mrk.json",
        "atlas_run.json",
        "textures/average_texture.png",
    }
    for specimen in SPECIMENS:
        expected.add(f"textures/{specimen}.png")
        for directory in ("aligned", "resampled"):
            expected.update(
                {f"{directory}/{specimen}.obj", f"{directory}/{specimen}.mrk.json"}
            )
    for root in (left, right):
        assert {
            path.relative_to(root).as_posix()
            for path in root.rglob("*")
            if path.is_file()
        } == expected
    atlas = MeshIO.read(left / "atlas.obj")
    assert len(atlas.faces) > 1000, (
        "The test must use the dense clam geometry, not a synthetic replacement"
    )
    for relative in sorted(expected):
        a, b = left / relative, right / relative
        if a.suffix == ".obj":
            correspondence = relative.startswith("resampled/")
            first, second = (
                MeshIO.read(path, allow_degenerate=correspondence) for path in (a, b)
            )
            assert_mesh_equal(first, second, relative)
            if relative.startswith("resampled/"):
                np.testing.assert_array_equal(first.faces, atlas.faces)
                np.testing.assert_array_equal(first.uv, atlas.uv)
        elif relative == "atlas_run.json":
            # Scientific settings and the upstream revision must match independently
            # across both runs; destination paths and tool discovery belong to the host.
            first, second = (
                AtlasRunData.model_validate_json(path.read_text()).model_dump(
                    mode="json"
                )
                for path in (a, b)
            )
            first["options"].pop("output")
            second["options"].pop("output")
            # The CLI discovers the executable from the same environment where
            # the Slicer form supplies its explicit path; both were checked above.
            first["options"].pop("blender")
            second["options"].pop("blender")
            assert first == second
        elif a.suffix == ".json":
            first, second = LandmarkIO.read(a), LandmarkIO.read(b)
            assert first.labels == second.labels
            np.testing.assert_allclose(
                first.points, second.points, rtol=1e-12, atol=1e-9, err_msg=relative
            )
        else:
            with Image.open(a) as first, Image.open(b) as second:
                assert first.size == second.size == (512, 512)
                # Decode pixels rather than compare PNG compression or metadata bytes.
                np.testing.assert_array_equal(
                    np.asarray(first), np.asarray(second), err_msg=relative
                )


def compare_presentations(cli_root, slicer_root):
    """Validate protocol data, then compare values and the actual MRML rendering outputs."""
    events = [
        json.loads(line)
        for line in (cli_root / "events.jsonl").read_text().splitlines()
    ]
    assert not any(event["type"] in ("failed", "cancelled") for event in events)
    cli = {
        event["action"]: event["payload"]
        for event in events
        if event["type"] == "result"
    }
    slicer = json.loads((slicer_root / "presentations.json").read_text())
    required = {
        "atlas",
        "colors",
        "population",
        "plot",
        "morph_controls",
        "morphospace",
        "display",
        "selection",
        "export",
        "clear_selection",
    }
    assert required <= cli.keys() and required <= slicer.keys()
    stores = (ArtifactStore(cli_root), ArtifactStore(slicer_root))

    def arrays(action, field):
        return tuple(
            store.read(payload[action][field], "outputs")
            for store, payload in zip(stores, (cli, slicer), strict=True)
        )

    # Fitted model IDs and artifact paths belong to individual sessions; data must agree.
    for schema, action in (
        (PopulationPresentationData, "population"),
        (PlotPresentation, "plot"),
        (MorphControlsData, "morph_controls"),
        (FaceColorsPresentation, "display"),
        (FaceColorsPresentation, "morphospace"),
        (SelectionPresentation, "selection"),
        (ExportPresentation, "export"),
        (MeshPresentation, "atlas"),
    ):
        for payload in (cli, slicer):
            schema.model_validate_json(json.dumps(payload[action]))
    assert (
        cli["population"]["specimen_ids"]
        == slicer["population"]["specimen_ids"]
        == list(SPECIMENS)
    )
    for field in ("method", "components", "representation", "can_reconstruct"):
        assert cli["population"][field] == slicer["population"][field]
    assert cli["population"]["can_reconstruct"]
    scores, slicer_scores = arrays("plot", "scores")
    assert scores.shape == (3, 2) and np.isfinite(scores).all()
    np.testing.assert_allclose(scores, slicer_scores, rtol=1e-10, atol=1e-10)
    assert cli["plot"]["x_axis"] == slicer["plot"]["x_axis"] == 1
    assert cli["plot"]["y_axis"] == slicer["plot"]["y_axis"] == 0
    with np.load(slicer_root / "scene_plot.npz", allow_pickle=False) as scene:
        np.testing.assert_allclose(scene["x"], scores[:, 1], rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(scene["y"], scores[:, 0], rtol=1e-10, atol=1e-10)
        np.testing.assert_array_equal(scene["labels"], SPECIMENS)
    for field in ("positions", "coordinates"):
        np.testing.assert_allclose(
            cli["morph_controls"][field],
            slicer["morph_controls"][field],
            rtol=1e-10,
            atol=1e-10,
        )
        with np.load(
            slicer_root / "scene_morph_controls.npz", allow_pickle=False
        ) as scene:
            np.testing.assert_allclose(
                scene[field], cli["morph_controls"][field], rtol=1e-10, atol=1e-10
            )
    for action in ("display", "morphospace"):
        rgb, slicer_rgb = arrays(action, "rgb")
        valid, slicer_valid = arrays(action, "valid")
        assert valid.any() and np.isfinite(rgb).all()
        np.testing.assert_allclose(rgb, slicer_rgb, rtol=1e-10, atol=1e-10)
        np.testing.assert_array_equal(valid, slicer_valid)
        expected = rgb.copy()
        expected[~valid] = 0.45
        np.testing.assert_array_equal(
            np.load(slicer_root / f"scene_{action}.npy", allow_pickle=False),
            np.rint(expected * 255).astype(np.uint8),
        )
    for field in ("vertices", "faces"):
        np.testing.assert_array_equal(*arrays("selection", field))
    selected_vertices, _ = arrays("selection", "vertices")
    selected_faces, _ = arrays("selection", "faces")
    assert 0 < len(selected_faces) < len(arrays("display", "valid")[0])
    with np.load(slicer_root / "scene_selection.npz", allow_pickle=False) as scene:
        np.testing.assert_array_equal(
            scene["selected"], np.isin(scene["indices"], selected_vertices)
        )
    for action in ("atlas", "export"):
        with np.load(slicer_root / f"scene_{action}.npz", allow_pickle=False) as scene:
            for field in ("vertices", "faces", "uv"):
                first = stores[0].read(cli[action]["mesh"][field], "outputs")
                second = stores[1].read(slicer[action]["mesh"][field], "outputs")
                if field == "vertices":
                    np.testing.assert_allclose(first, second, rtol=1e-12, atol=1e-9)
                    np.testing.assert_allclose(
                        first, scene[field], rtol=1e-12, atol=1e-9
                    )
                else:
                    np.testing.assert_array_equal(first, second)
                    np.testing.assert_array_equal(first, scene[field])
    for field in ("original_vertices", "original_faces"):
        np.testing.assert_array_equal(*arrays("export", field))
    assert not list((cli_root / "inputs").iterdir()), "CLI left exchange inputs behind"
    return Path(cli["atlas"]["directory"]), Path(slicer["atlas"]["directory"])


@pytest.mark.end_to_end
@pytest.mark.parametrize("backend", ["native", "morphoweave"])
def test_small_clams_slicer_cli_parity(tmp_path, backend):
    """Build both atlases from raw clams and compare the same downstream user actions."""
    slicer = executable(
        "Slicer", "SLICER_EXECUTABLE", "/Applications/Slicer.app/Contents/MacOS/Slicer"
    )
    blender = Blender.check_available()
    inputs = tmp_path / "inputs"
    print(f"Small-clams artifacts and logs: {tmp_path}", flush=True)
    prepare_inputs(inputs)
    environment = os.environ.copy()
    environment["BLENDER_EXECUTABLE"] = str(blender.executable)

    # Launch a real CLI process in the same uv interpreter as pytest, with its full parser.
    cli_root, slicer_root = tmp_path / "cli", tmp_path / "slicer"
    run_process(
        [
            sys.executable,
            "-m",
            "interdeca.cli",
            "--output",
            cli_root,
            "atlas",
            "--backend",
            backend,
            "--models",
            inputs / "models",
            "--landmarks",
            inputs / "landmarks",
            "--textures",
            inputs / "textures",
            "--output",
            tmp_path / "cli-atlas",
            "--bake-size",
            "512",
            "colors",
            "--sample-only",
            "--samples",
            "1000",
            "--neighbor-average",
            "fit",
            "--method",
            "pca",
            "--components",
            "2",
            "axes",
            "--x",
            "2",
            "--y",
            "1",
            "reconstruct",
            "--x",
            "0.7",
            "--y",
            "0.3",
            SPECIMENS[0],
            "show",
            SPECIMENS[1],
            "select",
            "--curve",
            inputs / "curve.npy",
            "--both-sides",
            "export",
            "clear-selection",
        ],
        tmp_path / "cli.log",
        environment,
    )

    # Slicer's ABI must never see the uv site-packages. Only the thin host sources cross.
    for key in (
        "PYTHONHOME",
        "VIRTUAL_ENV",
        "QT_PLUGIN_PATH",
        "QT_QPA_PLATFORM_PLUGIN_PATH",
    ):
        environment.pop(key, None)
    environment.update(
        PYTHONPATH=os.pathsep.join(
            [str(REPOSITORY / "client/src"), str(REPOSITORY / "slicer/InterDeCANext")]
        ),
        INTERDECA_WORKER_PYTHON=sys.executable,
        INTERDECA_E2E_OUTPUT=str(slicer_root),
        INTERDECA_E2E_INPUTS=str(inputs),
        INTERDECA_E2E_BACKEND=backend,
    )
    run_process(
        [
            slicer,
            "--testing",
            "--no-splash",
            "--no-main-window",
            "--additional-module-path",
            REPOSITORY / "slicer/InterDeCANext",
            "--python-script",
            Path(__file__).with_name("slicer_driver.py"),
        ],
        tmp_path / "slicer.log",
        environment,
        slicer_root / "worker.pid",
    )
    outcome = json.loads((slicer_root / "outcome.json").read_text())
    assert outcome["status"] == "passed"
    compare_atlases(*compare_presentations(cli_root, slicer_root))
    print(
        f"Matched Slicer {outcome['slicer_version']} and CLI outputs; Blender {blender.version}",
        flush=True,
    )
