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
        self.models = Widgets.path(inputs, "Models:")
        self.landmarks = Widgets.path(inputs, "Landmarks:")
        self.textures = Widgets.path(inputs, "Textures:")
        self.output = Widgets.path(inputs, "Output directory:")
        self.validation = qt.QLabel("Select model, landmark, and output directories.")
        self.validation.wordWrap = True
        inputs.addRow(self.validation)
        override = Widgets.section(layout, "Atlas Override (Optional)", True)
        self.atlas_model = Widgets.path(override, "Atlas Model:", True)
        self.atlas_landmarks = Widgets.path(override, "Atlas Landmarks:", True)
        blender = Widgets.section(layout, "Blender (cleanup, UV, bake)", True)
        self.blender = Widgets.path(blender, "Blender executable:", True)
        self.merge = Widgets.spin(blender, "Merge by distance:", 0, 1000, 0.0001, 6)
        self.merge.toolTip = "Merge atlas vertices before final UV mapping and correspondence. Zero disables cleanup."
        self.angle = Widgets.spin(blender, "Smart UV angle (deg):", 1, 89, 66, 1)
        self.margin = Widgets.spin(blender, "Island margin (UV):", 0, 0.05, 0.002, 4)
        self.size = Widgets.spin(blender, "Bake size (px):", 128, 8192, 2048)
        self.extrusion = Widgets.spin(blender, "Bake extrusion:", 0, 10, 0.001, 6)
        self.bake_margin = Widgets.spin(blender, "Bake margin (px):", 0, 64, 2)
        self.run = Widgets.button(
            layout, "Run ATLAS and Texture Transfer", lambda: app.guard(self.execute)
        )
        self.paths = (
            self.models,
            self.landmarks,
            self.textures,
            self.output,
            self.atlas_model,
            self.atlas_landmarks,
            self.blender,
        )
        self.settings_keys = (
            "models",
            "landmarks",
            "textures",
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

    def changed(self, *args):
        """Persist input paths and invalidate results derived from older settings."""
        for key, widget in zip(self.settings_keys, self.paths):
            self.app.settings.setValue("atlas/" + key, widget.currentPath)
        self.app.client.invalidate("all")
        valid = all(
            widget.currentPath and Path(widget.currentPath).is_dir()
            for widget in (self.models, self.landmarks)
        )
        self.validation.text = (
            "Input directories found. Specimen pairing and geometry are validated on Run."
            if valid
            else "Select existing model and landmark directories."
        )

    def execute(self):
        """Send atlas form values to the worker through the application client."""
        # Convert empty optional paths to None rather than the current directory.
        optional = lambda widget: (
            str(widget.currentPath) if widget.currentPath else None
        )
        self.app.client.run_atlas(
            models=str(self.models.currentPath),
            landmarks=str(self.landmarks.currentPath),
            output=str(self.output.currentPath),
            textures=optional(self.textures),
            atlas_model=optional(self.atlas_model),
            atlas_landmarks=optional(self.atlas_landmarks),
            blender=self.blender.currentPath or None,
            merge_distance=float(self.merge.value),
            uv_angle=float(self.angle.value),
            island_margin=float(self.margin.value),
            bake_size=int(self.size.value),
            extrusion=float(self.extrusion.value),
            bake_margin=int(self.bake_margin.value),
        )

    def render(self, busy):
        """Lock pipeline inputs during execution and enable Run only with required paths."""
        self.widget.enabled = not busy
        self.run.enabled = not busy and all(
            widget.currentPath for widget in (self.models, self.landmarks, self.output)
        )
