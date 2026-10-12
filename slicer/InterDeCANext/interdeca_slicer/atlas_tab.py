"""ATLAS widgets that send user inputs through the application client."""

from pathlib import Path

import qt

from interdeca_slicer.widgets import Widgets


class AtlasTab:
    """Collect form values; the worker validates requests and constructs pipelines."""

    def __init__(self, app):
        self.app = app
        self.widget = qt.QWidget()
        layout = qt.QFormLayout(self.widget)
        inputs = Widgets.section(layout, "Input Directories")
        self.input_dir = Widgets.path(inputs, "Input directory:")
        self.coordinates = Widgets.combo(inputs, "Mesh coordinates:")
        self.coordinates.addItems(["LPS", "RAS"])
        self.coordinates.setCurrentIndex(
            1 if app.settings.value("atlas/input_coordinates", "LPS") == "RAS" else 0
        )
        self.output = Widgets.path(inputs, "Output directory (optional):")
        # Each new form starts unchecked because replacement deletes saved runs.
        self.overwrite = Widgets.check(inputs, "Overwrite _interdeca:", False)
        self.overwrite.toolTip = "Delete this dataset's _interdeca folder, including previous prepared inputs and results, before running. Raw models, landmarks, and textures remain unchanged."
        self.validation = qt.QLabel("Select a dataset containing models/, landmarks/, and textures/.")
        self.validation.wordWrap = True
        inputs.addRow(self.validation)
        self.backend = Widgets.combo(inputs, "Atlas implementation:")
        self.backend.addItems(["InterDeCA (original)", "MorphoWeave (comparison)"])
        self.backend.setCurrentIndex(
            1 if app.settings.value("atlas/backend", "native") == "morphoweave" else 0
        )
        self.normalize_scale = Widgets.check(inputs, "Normalize specimen scale:", False)
        self.normalize_scale.toolTip = "Allow scaling during specimen alignment. Generated MorphoWeave atlases are restored to mean input size in either mode."
        override = Widgets.section(layout, "Atlas Override (Optional)", True)
        self.atlas_model = Widgets.path(override, "Atlas Model:", True)
        self.atlas_landmarks = Widgets.path(override, "Atlas Landmarks:", True)
        surface = Widgets.section(layout, "Surface processing (cleanup, UV, bake)", True)
        # Atlas construction remains independent of the tool used to clean,
        # unwrap, and bake its final shared surface.
        self.surface_backend = Widgets.combo(surface, "Surface backend:")
        self.surface_backend.addItems(["Rust", "Blender"])
        self.surface_backend.setCurrentIndex(
            1 if app.settings.value("atlas/surface_backend", "rust") == "blender" else 0
        )
        self.blender = Widgets.path(surface, "Blender executable (optional):", True)
        self.blender.toolTip = "Used only with Blender surface processing. Leave empty for automatic discovery."
        self.merge = Widgets.spin(surface, "Merge by distance:", 0, 1000, 0.0001, 6)
        self.merge.toolTip = "Merge atlas vertices before final UV mapping and correspondence. Zero disables cleanup."
        self.angle = Widgets.spin(surface, "Smart UV angle (deg):", 1, 89, 66, 1)
        self.margin = Widgets.spin(surface, "Island margin (UV):", 0, 0.05, 0.002, 4)
        self.size = Widgets.spin(surface, "Bake size (px):", 128, 8192, 2048)
        self.extrusion = Widgets.spin(surface, "Bake extrusion:", 0, 10, 0.001, 6)
        self.bake_margin = Widgets.spin(surface, "Bake margin (px):", 0, 64, 2)
        self.run = Widgets.button(
            layout, "Run ATLAS and Texture Transfer", lambda: app.guard(self.execute)
        )
        self.paths = (
            self.input_dir,
            self.output,
            self.atlas_model,
            self.atlas_landmarks,
            self.blender,
        )
        self.settings_keys = (
            "input_dir",
            "output",
            "atlas_model",
            "atlas_landmarks",
            "blender",
        )
        for key, widget in zip(self.settings_keys, self.paths):
            saved = app.settings.value("atlas/" + key, "")
            if saved:
                widget.setCurrentPath(str(saved))
            widget.connect("currentPathChanged(QString)", self.changed)
        for spin in (
            self.merge,
            self.angle,
            self.margin,
            self.size,
            self.extrusion,
            self.bake_margin,
        ):
            signal = (
                "valueChanged(int)"
                if isinstance(spin, qt.QSpinBox)
                else "valueChanged(double)"
            )
            spin.connect(signal, self.changed)
        self.coordinates.connect("currentIndexChanged(int)", self.changed)
        self.overwrite.connect("toggled(bool)", self.changed)
        self.backend.connect("currentIndexChanged(int)", self.changed)
        self.surface_backend.connect("currentIndexChanged(int)", self.changed)
        self.normalize_scale.connect("toggled(bool)", self.changed)

    def changed(self, *args):
        """Persist input paths and invalidate results derived from older settings."""
        for key, widget in zip(self.settings_keys, self.paths):
            self.app.settings.setValue("atlas/" + key, widget.currentPath)
        self.app.settings.setValue(
            "atlas/backend", ("native", "morphoweave")[int(self.backend.currentIndex)]
        )
        self.app.client.invalidate("all")
        self.app.settings.setValue(
            "atlas/surface_backend", ("rust", "blender")[int(self.surface_backend.currentIndex)]
        )
        self.app.settings.setValue("atlas/input_coordinates", self.coordinates.currentText)
        root = Path(self.input_dir.currentPath) if self.input_dir.currentPath else None
        valid = root is not None and all(
            (root / name).is_dir() for name in ("models", "landmarks", "textures")
        )
        self.validation.text = (
            "Input folders found. Matching file names are validated on Run; inputs are prepared under _interdeca/."
            if valid else
            "Select a dataset containing models/, landmarks/, and textures/."
        )

    def execute(self):
        """Send atlas form values to the worker through the application client."""
        # Convert empty optional paths to None rather than the current directory.
        optional = lambda widget: (
            str(widget.currentPath) if widget.currentPath else None
        )
        self.app.client.run_atlas(
            input_dir=str(self.input_dir.currentPath),
            input_coordinates=self.coordinates.currentText,
            output=optional(self.output),
            overwrite=bool(self.overwrite.checked),
            atlas_model=optional(self.atlas_model),
            atlas_landmarks=optional(self.atlas_landmarks),
            merge_distance=float(self.merge.value),
            uv_angle=float(self.angle.value),
            island_margin=float(self.margin.value),
            bake_size=int(self.size.value),
            extrusion=float(self.extrusion.value),
            bake_margin=int(self.bake_margin.value),
            backend=("native", "morphoweave")[int(self.backend.currentIndex)],
            normalize_scale=bool(self.normalize_scale.checked),
            surface_backend=("rust", "blender")[int(self.surface_backend.currentIndex)],
            blender=optional(self.blender),
        )

    def render(self, busy):
        """Lock pipeline inputs during execution and enable Run only with required paths."""
        self.widget.enabled = not busy
        self.blender.enabled = not busy and int(self.surface_backend.currentIndex) == 1
        root = Path(self.input_dir.currentPath) if self.input_dir.currentPath else None
        self.run.enabled = not busy and root is not None and all(
            (root / name).is_dir() for name in ("models", "landmarks", "textures")
        )
