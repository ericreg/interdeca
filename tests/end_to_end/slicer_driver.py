"""Run the real Slicer widgets and save their results for the uv pytest parent.

Only the independent client and Slicer presentation package enter this process.
The parent supplies their source paths, never the uv environment's site-packages.
"""

import json
import os
import shutil
import sys
import traceback
from pathlib import Path
from time import monotonic, sleep

import numpy as np
import vtk
from interdeca_client.artifacts import ArtifactStore
from interdeca_slicer.scene import SceneBridge
from vtk.util.numpy_support import vtk_to_numpy

import slicer


class SlicerRun:
    """Observe completed presentations without replacing any application operations."""

    def __init__(self):
        self.root = Path(os.environ["INTERDECA_E2E_OUTPUT"])
        self.inputs = Path(os.environ["INTERDECA_E2E_INPUTS"])
        self.root.mkdir()
        self.store = ArtifactStore(self.root)
        self.presentations = {}
        self.errors = []
        self.application = None

    def encode(self, value):
        """Archive plain presentation records before their exchange files disappear."""
        if isinstance(value, np.ndarray):
            return self.store.write(value, "outputs")
        if isinstance(value, Path):
            return str(value)
        if hasattr(value, "_asdict"):
            return {name: self.encode(item) for name, item in value._asdict().items()}
        if isinstance(value, tuple):
            return [self.encode(item) for item in value]
        return value

    def error(self, message, details=""):
        # Modal dialogs would stall a noninteractive test; keep the same error fatal.
        self.errors.append(message + "\n" + details)
        print(self.errors[-1], flush=True)

    def wait(self, predicate, label, timeout=1200):
        """Pump Qt so worker callbacks and real MRML updates execute on the UI thread."""
        deadline = monotonic() + timeout
        print(f"[Slicer E2E] {label}", flush=True)
        while monotonic() < deadline:
            slicer.app.processEvents()
            if self.errors:
                raise AssertionError("\n".join(self.errors))
            if predicate():
                return
            sleep(0.01)
        raise TimeoutError(f"Timed out: {label}; state={self.application.client.state}")

    def idle(self, label):
        # Include transport acknowledgements: an empty run ID alone can precede acceptance.
        self.wait(lambda: not self.application.client.busy, label)

    def click(self, button, label):
        self.idle("ready for " + label)
        assert button.enabled, f"Disabled action: {label}"
        button.click()
        self.idle(label)

    def mesh(self, action, node):
        snapshot = SceneBridge.snapshot(node)
        np.savez(self.root / f"scene_{action}.npz", **snapshot.mesh._asdict())

    def present(self, action, result):
        # First let the production UI render; a copied worker result alone is not parity.
        self.original_present(action, result)
        self.presentations[action] = self.encode(result)
        (self.root / "presentations.json").write_text(
            json.dumps(self.presentations, indent=2), encoding="utf-8"
        )
        app = self.application
        if action == "atlas":
            self.mesh(action, app.colors.model.currentNode())
        elif action == "plot":
            table = app.plot.table.GetTable()
            np.savez(
                self.root / "scene_plot.npz",
                x=vtk_to_numpy(table.GetColumnByName("X")),
                y=vtk_to_numpy(table.GetColumnByName("Y")),
                labels=np.array(
                    [
                        table.GetValue(i, 2).ToString()
                        for i in range(table.GetNumberOfRows())
                    ]
                ),
            )
        elif action in ("display", "morphospace"):
            node = app.colors.model.currentNode()
            assert node.GetDisplayNode().GetScalarVisibility()
            np.save(
                self.root / f"scene_{action}.npy",
                vtk_to_numpy(
                    node.GetPolyData().GetCellData().GetArray("InterDeCA_Colors")
                ),
                allow_pickle=False,
            )
        elif action == "morph_controls":
            table = app.plot.point_table
            np.savez(
                self.root / "scene_morph_controls.npz",
                positions=np.array([app.colors.x.value, app.colors.y.value]) / 1000,
                coordinates=np.array(
                    [vtk_to_numpy(table.GetTable().GetColumn(i))[0] for i in range(2)]
                ),
            )
        elif action == "selection":
            data = app.selection.model.currentNode().GetPolyData().GetPointData()
            np.savez(
                self.root / "scene_selection.npz",
                indices=vtk_to_numpy(data.GetArray(SceneBridge.index_name)),
                selected=vtk_to_numpy(data.GetArray("InterDeCA_Selection")),
            )
        elif action == "export":
            self.mesh(action, slicer.util.getNode("E2E SelectedRegion"))

    def run(self):
        # Loading the actual module exercises its startup, widgets, and independent worker.
        # Hidden Slicer has no module-selector toolbar; requesting the representation
        # still constructs the registered module's production widget and startup panel.
        widget = slicer.modules.interdecanext.widgetRepresentation().self()
        app = self.application = widget.application
        app.show_error = self.error
        self.original_present = app.present
        app.present = self.present
        self.wait(lambda: app.client.connected, "worker startup", timeout=90)
        (self.root / "worker.pid").write_text(str(app.client.process.pid))
        assert app.client.methods == ("pca", "ica", "umap")

        # These form values match the CLI flags, including explicit low-resolution bakes.
        for control, value in (
            (app.atlas.models, self.inputs / "models"),
            (app.atlas.landmarks, self.inputs / "landmarks"),
            (app.atlas.textures, self.inputs / "textures"),
            (app.atlas.output, self.root / "atlas"),
            (app.atlas.blender, os.environ["BLENDER_EXECUTABLE"]),
        ):
            control.setCurrentPath(str(value))
        app.atlas.size.setValue(512)
        app.atlas.backend.setCurrentIndex(
            1 if os.environ["INTERDECA_E2E_BACKEND"] == "morphoweave" else 0
        )
        self.click(app.atlas.run, "atlas, alignment, correspondence, and baking")

        app.colors.sample_only.setChecked(True)
        app.colors.samples.setValue(1000)
        app.colors.neighbors.setChecked(True)
        self.click(app.colors.process, "color sampling")
        app.colors.method.setCurrentText("PCA")
        self.click(app.colors.compare, "population PCA")

        # Change both widget axes together to avoid submitting a transient equal-axis pair.
        for combo, index in ((app.colors.x_axis, 1), (app.colors.y_axis, 0)):
            combo.blockSignals(True)
            combo.setCurrentIndex(index)
            combo.blockSignals(False)
        app.colors.axes_changed()
        self.idle("swapped plot axes")
        app.colors.start.setCurrentText("UF_IZ_507781")
        self.click(app.colors.visualize, "initial reconstruction")
        app.colors.x.setValue(700)
        app.colors.y.setValue(300)
        self.wait(
            lambda: not app.colors._timer.isActive() and not app.client.busy,
            "slider reconstruction",
        )
        assert np.allclose(
            self.presentations["morph_controls"]["positions"], [0.7, 0.3]
        )
        app.colors.texture.setCurrentText("UF_IZ_507782.png")
        self.click(app.colors.apply, "processed specimen display")

        # A shared RAS curve selects a proper subset; its MRML snapshot crosses the socket.
        curve = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLMarkupsClosedCurveNode", "E2E curve"
        )
        for point in np.load(self.inputs / "curve.npy", allow_pickle=False):
            curve.AddControlPointWorld(vtk.vtkVector3d(*point))
        app.selection.curve.setCurrentNode(curve)
        app.selection.side.setChecked(False)
        app.selection.name.setText("E2E SelectedRegion")
        self.click(app.selection.apply, "region selection")
        self.click(app.selection.export, "region export")
        self.click(app.selection.clear, "clear selection")
        assert app.client.state.selection is None
        assert (
            app.selection.model.currentNode()
            .GetPolyData()
            .GetPointData()
            .GetArray("InterDeCA_Selection")
            is None
        )

        # The host must stay thin even after exercising every numerical workflow above.
        assert not any(
            name == "interdeca" or name.startswith("interdeca.") for name in sys.modules
        )
        assert not any(
            name == "umap" or name.startswith("umap.") for name in sys.modules
        )

    def close(self):
        if self.application is None:
            return
        client = self.application.client
        self.application.close()
        if client._thread is not None:
            client._thread.join(15)
        if client.root is not None:
            # Retain early subprocess diagnostics as well as the structured worker log.
            for name in ("worker.log", "stdout.log", "stderr.log"):
                source = client.root / name
                if source.exists():
                    shutil.copy2(source, self.root / name)
        assert client.process is None or client.process.poll() == 0, (
            "Worker failed to exit cleanly"
        )
        # Once reaped, this PID must not be reused by the parent's timeout cleanup.
        (self.root / "worker.pid").unlink(missing_ok=True)


def main():
    """Always emit an explicit outcome; process exit by itself is insufficient evidence."""
    run = SlicerRun()
    code = 0
    try:
        run.run()
    except Exception:  # noqa: BLE001 -- every driver failure must reach the pytest parent
        code = 1
        traceback.print_exc()
    finally:
        try:
            run.close()
        except Exception:  # noqa: BLE001 -- cleanup failures also make the run fail
            code = 1
            traceback.print_exc()
        (run.root / "outcome.json").write_text(
            json.dumps(
                {
                    "status": "passed" if code == 0 else "failed",
                    "slicer_version": slicer.app.applicationVersion,
                }
            ),
            encoding="utf-8",
        )
        print(f"[Slicer E2E] exit {code}", flush=True)
        slicer.util.exit(code)


# --testing exits after the startup script returns, so finish the workflow here.
main()
