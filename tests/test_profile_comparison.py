"""Prove the comparison gate rejects changed results, not just unequal timings."""

import json
import shutil

import numpy as np
import pytest
from PIL import Image

from interdeca.io import MeshIO
from interdeca.models import Mesh
from tools.profiling.compare import compare, compare_outputs, digest, outputs
from tools.profiling.serde import Config, Run, Span


def write_json(path, data):
    path.write_text(json.dumps(data) + "\n")


def make_suite(root):
    """Small synthetic artifacts exercise the gate without running numerical work."""
    root.mkdir()
    (root / "inputs").mkdir()
    (root / "inputs/source.txt").write_text("unchanged fixture")
    write_json(
        root / "environment.json",
        {
            "platform": "test",
            "python": "test",
            "logical_cpus": 1,
            "physical_cpus": 1,
            "ram_bytes": 1024,
            "versions": {"numpy": "test"},
            "thread_environment": {},
            "git_head": "test-revision",
            "lock_sha256": "lock",
            "inputs_sha256": {"source.txt": digest(root / "inputs/source.txt")},
            "blender": {
                "version": [5, 2, 1],
                "description": "test",
                "executable": "/test/blender",
            },
        },
    )
    for backend in ("native", "morphoweave"):
        directory = root / f"{backend}-baseline-1"
        atlas = directory / "atlas/generated"
        atlas.mkdir(parents=True)
        arrays = directory / "cli/outputs"
        arrays.mkdir(parents=True)
        mesh = Mesh(
            np.array(
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]]
            ),
            np.array([[0, 1, 2], [0, 2, 3]]),
            np.array(
                [
                    [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]],
                    [[0.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
                ]
            ),
        )
        MeshIO.write_obj(mesh, atlas / "atlas.obj")
        Image.fromarray(np.full((3, 3, 4), 127, dtype=np.uint8)).save(
            atlas / "texture.png"
        )
        metadata = {
            "options": {
                "backend": backend,
                "models": str(root / "inputs/models"),
                "landmarks": str(root / "inputs/landmarks"),
                "textures": str(root / "inputs/textures"),
                "output": str(directory / "atlas"),
                "blender": None,
                "bake_size": 512,
            },
            "upstream_revision": "pinned",
            "atlas_scale_conversion": {"factor": 0.088},
        }
        write_json(atlas / "atlas_run.json", metadata)
        payload = {
            "directory": str(atlas),
            "average": str(atlas / "texture.png"),
            "mesh": {},
        }
        files = {
            "atlas/atlas.obj": atlas / "atlas.obj",
            "atlas/texture.png": atlas / "texture.png",
        }
        for name, value in zip(("vertices", "faces", "uv"), mesh, strict=True):
            path = arrays / f"{name}.npy"
            np.save(path, value, allow_pickle=False)
            payload["mesh"][name] = {
                "path": f"outputs/{name}.npy",
                "shape": list(value.shape),
                "dtype": value.dtype.str,
            }
            files[f"presentations/atlas/mesh/{name}"] = path
        actions = {
            "atlas": payload,
            "colors": {"dataset_id": "dataset"},
            "population": {"specimen_ids": ["a", "b", "c"], "model_id": "model"},
            "plot": {"x_axis": 1, "y_axis": 0},
            "morph_controls": {},
            "morphospace": {},
            "display": {},
            "selection": {},
            "export": {},
            "clear_selection": {},
        }
        events = []
        for index, (action, result) in enumerate(actions.items()):
            base = {
                "protocol_version": 1,
                "request_id": f"r{index}",
                "run_id": f"u{index}",
                "input_revision": 0,
            }
            events += [
                {**base, "type": "accepted", "action": action},
                {**base, "type": "result", "action": action, "payload": result},
                {**base, "type": "completed", "message": action, "details": ""},
            ]
        (directory / "cli/events.jsonl").write_text(
            "".join(json.dumps(e) + "\n" for e in events)
        )
        run = Run(
            config=Config(
                backend=backend,
                mode="baseline",
                inputs=root / "inputs",
                output=directory,
            ),
            exit_code=0,
            wall_seconds=2,
            sample_interval_seconds=0.1,
            parent_peak_rss_bytes=100,
            tree_peak_rss_bytes=150,
            parent_peak_uss_bytes=None,
            tree_peak_uss_bytes=None,
            sampled_processes=2,
            output_hashes={key: digest(path) for key, path in files.items()},
        )
        (directory / "run.json").write_text(run.model_dump_json())
        span = Span(
            kind="session",
            name="CLI session",
            start=0,
            end=1,
            wall_seconds=1,
            process_cpu_seconds=0.8,
            thread_cpu_seconds=0.8,
            children_cpu_seconds=0.1,
            exclusive_wall_seconds=1,
            exclusive_thread_cpu_seconds=0.8,
        )
        (directory / "spans.jsonl").write_text(span.model_dump_json() + "\n")
    return root


@pytest.fixture
def suites(tmp_path):
    return make_suite(tmp_path / "reference"), make_suite(tmp_path / "candidate")


def test_self_and_relocated_copy(suites, tmp_path):
    reference, candidate = suites
    assert not compare(reference, reference).issues
    assert not compare(reference, candidate).issues
    destination = tmp_path / "archive"
    shutil.copytree(candidate, destination)
    shutil.rmtree(candidate)
    assert not compare(reference, destination).issues


@pytest.mark.parametrize(
    "mutation,fragment",
    [
        ("vertices", "max_abs="),
        ("faces", "faces"),
        ("uv", "uv"),
        ("texture", "decoded pixels"),
        ("scale", "atlas_scale_conversion/factor"),
        ("settings", "bake_size"),
        ("upstream", "upstream_revision"),
        ("specimens", "specimen_ids"),
        ("axes", "x_axis"),
    ],
)
def test_changed_numerical_outputs_fail(suites, mutation, fragment):
    reference, candidate = suites
    directory = candidate / "native-baseline-1"
    original = outputs(directory)
    if mutation in ("vertices", "faces", "uv"):
        path = directory / "cli/outputs" / f"{mutation}.npy"
        value = np.load(path, allow_pickle=False)
        if mutation == "faces":
            value = value[::-1].copy()
        else:
            value.flat[0] += 0.001
        np.save(path, value, allow_pickle=False)
    elif mutation == "texture":
        path = directory / "atlas/generated/texture.png"
        with Image.open(path) as image:
            pixels = np.array(image)
        pixels[0, 0, 0] += 1
        Image.fromarray(pixels).save(path)
    elif mutation in ("scale", "settings", "upstream"):
        path = directory / "atlas/generated/atlas_run.json"
        metadata = json.loads(path.read_text())
        if mutation == "scale":
            metadata["atlas_scale_conversion"]["factor"] *= 2
        elif mutation == "settings":
            metadata["options"]["bake_size"] = 256
        else:
            metadata["upstream_revision"] = "changed"
        write_json(path, metadata)
    else:
        path = directory / "cli/events.jsonl"
        text = (
            path.read_text().replace('["a", "b", "c"]', '["b", "a", "c"]')
            if mutation == "specimens"
            else path.read_text().replace('"x_axis": 1', '"x_axis": 0')
        )
        path.write_text(text)
    # A candidate normally records its own new hashes; the reference remains fixed.
    run = original.run.model_copy(
        update={
            "output_hashes": {key: digest(path) for key, path in original.files.items()}
        }
    )
    (directory / "run.json").write_text(run.model_dump_json())
    result = compare(reference, candidate)
    assert not result.identical_outputs
    assert any(fragment in issue for issue in result.issues), result.issues


@pytest.mark.parametrize(
    "mutation", ["input", "failed", "missing", "completion", "hash"]
)
def test_invalid_capture_rejected(suites, mutation):
    reference, candidate = suites
    directory = candidate / "native-baseline-1"
    if mutation == "input":
        (candidate / "inputs/source.txt").write_text("changed")
    elif mutation == "failed":
        run = json.loads((directory / "run.json").read_text())
        run["exit_code"] = 1
        write_json(directory / "run.json", run)
    elif mutation == "missing":
        (directory / "cli/outputs/vertices.npy").unlink()
    elif mutation == "completion":
        path = directory / "cli/events.jsonl"
        lines = path.read_text().splitlines()
        path.write_text("\n".join(lines[:-1]) + "\n")
    else:
        (directory / "atlas/generated/texture.png").write_bytes(b"bad")
    with pytest.raises((ValueError, FileNotFoundError)):
        compare(reference, candidate)


def test_environment_change_and_identity_normalization(suites):
    reference, candidate = suites
    directory = candidate / "native-baseline-1"
    path = directory / "cli/events.jsonl"
    path.write_text(
        path.read_text()
        .replace('"r0"', '"different-request"')
        .replace('"u0"', '"different-run"')
    )
    assert not compare(reference, candidate).issues
    path = candidate / "environment.json"
    env = json.loads(path.read_text())
    env["versions"]["numpy"] = "different"
    write_json(path, env)
    result = compare(reference, candidate)
    assert not result.compatible_environment
    assert any("numpy" in issue for issue in result.issues)


def test_compares_two_successful_runs(suites):
    reference, candidate = suites
    assert not compare_outputs(
        outputs(reference / "native-baseline-1"),
        outputs(candidate / "native-baseline-1"),
    )


def test_resource_deltas_do_not_replace_correctness(suites):
    reference, candidate = suites
    directory = candidate / "native-baseline-1"
    path = directory / "spans.jsonl"
    span = Span.model_validate_json(path.read_text())
    path.write_text(
        span.model_copy(update={"wall_seconds": 0.5}).model_dump_json() + "\n"
    )
    run_path = directory / "run.json"
    run = Run.model_validate_json(run_path.read_text())
    run_path.write_text(
        run.model_copy(update={"parent_peak_rss_bytes": 200}).model_dump_json()
    )
    result = compare(reference, candidate)
    assert result.identical_outputs
    assert result.metrics["native"]["session_wall_s"].reduction_percent == 50
    assert any(
        "native/peak_rss_bytes" in alert for alert in result.investigation_alerts
    )


def test_incomplete_capture_is_not_ignored(suites):
    reference, candidate = suites
    incomplete = candidate / "native-memory-1"
    incomplete.mkdir()
    write_json(incomplete / "config.json", {})
    with pytest.raises(ValueError, match="Incomplete capture"):
        compare(reference, candidate)
