"""Color processing, individual display, population plots, and morphospace controls."""

from pathlib import Path

import qt

from interdeca_slicer.scene import SceneBridge
from interdeca_slicer.widgets import Widgets


class ColorTab:
    """Collect user actions and render worker-provided population controls."""

    def __init__(self, app):
        self.app = app
        self.widget = qt.QWidget()
        layout = qt.QFormLayout(self.widget)
        first = Widgets.section(layout, "Step 1: Multi-texture Processing")
        self.model = Widgets.node(first, "Model:")
        self.directory = Widgets.path(first, "Texture Directory:")
        modes = qt.QWidget()
        mode_layout = qt.QHBoxLayout(modes)
        self.clustering = qt.QRadioButton("Clustering", modes)
        self.sample_only = qt.QRadioButton("Subsample and Average Only", modes)
        self.clustering.checked = True
        mode_layout.addWidget(self.clustering)
        mode_layout.addWidget(self.sample_only)
        first.addRow("Mode:", modes)
        self.normalize = Widgets.check(first, "Normalize Luminosity:")
        self.initial = Widgets.spin(first, "Initial Clusters:", 2, 64, 24)
        self.clusters = Widgets.spin(first, "Consolidated Clusters:", 2, 64, 8)
        self.samples = Widgets.spin(
            first, "Number of Faces (Subsampling):", 100, 100000, 10000
        )
        self.all_faces = Widgets.check(first, "Disable Subsampling (Use All Faces):")
        self.neighbors = Widgets.check(first, "Neighbor Average:")
        self.process = Widgets.button(
            first, "Cluster", lambda: app.guard(self.process_colors)
        )
        second = Widgets.section(layout, "Step 2: Individual Visualization")
        self.texture = Widgets.combo(second, "Select Texture:")
        self.raw = Widgets.check(second, "Raw Texture:")
        self.apply = Widgets.button(
            second, "Apply Texture", lambda: app.guard(self.show_texture)
        )
        third = Widgets.section(layout, "Step 3: Population Analysis")
        self.method = Widgets.combo(third, "Dimensionality Reduction:")
        self.method.addItems(["PCA", "ICA", "UMAP"])
        self.components = Widgets.spin(third, "Number of Components:", 2, 20, 2)
        self.x_axis = Widgets.combo(third, "X-axis:")
        self.y_axis = Widgets.combo(third, "Y-axis:")
        self.compare = Widgets.button(
            third, "Compare Textures", lambda: app.guard(self.fit)
        )
        fourth = Widgets.section(layout, "Step 4: Morphospace", True)
        self.start = Widgets.combo(fourth, "Starting Texture:")
        self.x = qt.QSlider(qt.Qt.Horizontal)
        self.y = qt.QSlider(qt.Qt.Horizontal)
        self.x_label, self.y_label = qt.QLabel("X: 0"), qt.QLabel("Y: 0")
        for slider, label, name in (
            (self.x, self.x_label, "X-axis Position:"),
            (self.y, self.y_label, "Y-axis Position:"),
        ):
            slider.setRange(0, 1000)
            fourth.addRow(name, slider)
            fourth.addRow(label)
        self.visualize = Widgets.button(
            fourth, "Visualize Morphospace", lambda: app.guard(self.begin_morph)
        )
        self.morph_hint = qt.QLabel(
            "Fit a population model with spatial color features first."
        )
        self.morph_hint.wordWrap = True
        fourth.addRow(self.morph_hint)
        # Debouncing prevents a slider drag from building a queue of obsolete computations.
        self._timer = qt.QTimer()
        self._timer.setSingleShot(True)
        self._timer.setInterval(120)
        self._timer.connect("timeout()", lambda: app.guard(self.explore))
        self.x.connect("valueChanged(int)", self.slider_changed)
        self.y.connect("valueChanged(int)", self.slider_changed)
        self.start.connect("currentIndexChanged(int)", self.start_changed)
        self.x_axis.connect("currentIndexChanged(int)", self.axes_changed)
        self.y_axis.connect("currentIndexChanged(int)", self.axes_changed)
        saved = app.settings.value("colors/directory", "")
        if saved:
            self.directory.setCurrentPath(str(saved))
        self.model.connect("currentNodeChanged(vtkMRMLNode*)", self.model_changed)
        self.directory.connect("currentPathChanged(QString)", self.directory_changed)
        self.texture.connect(
            "currentIndexChanged(int)", lambda *_: app.render(app.client.state)
        )
        for check in (
            self.clustering,
            self.sample_only,
            self.normalize,
            self.all_faces,
            self.neighbors,
        ):
            check.connect("toggled(bool)", self.settings_changed)
        for spin in (self.initial, self.clusters, self.samples):
            spin.connect("valueChanged(int)", self.settings_changed)
        self.method.connect("currentIndexChanged(int)", self.fit_settings_changed)
        self.components.connect("valueChanged(int)", self.fit_settings_changed)
        self.inputs = (
            self.model,
            self.directory,
            modes,
            self.normalize,
            self.initial,
            self.clusters,
            self.samples,
            self.all_faces,
            self.neighbors,
        )
        self.refresh_textures()

    def model_changed(self, *args):
        """Invalidate color results when the analysis surface changes."""
        self.settings_changed()
        self.app.watch_sources()

    def directory_changed(self, *args):
        """Persist the texture directory and invalidate its former population."""
        self.app.settings.setValue("colors/directory", self.directory.currentPath)
        self.refresh_textures()
        self.settings_changed()

    def refresh_textures(self):
        """Offer supported images, including averages that remain preview-only."""
        directory = (
            Path(self.directory.currentPath) if self.directory.currentPath else None
        )
        self.texture.clear()
        if directory is not None and directory.is_dir():
            self.texture.addItems(
                [
                    path.name
                    for path in sorted(directory.iterdir())
                    if path.suffix.lower()
                    in (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")
                ]
            )

    def settings_changed(self, *args):
        """Invalidate processing results after sampling or clustering choices change."""
        if self.app._presenting:
            return
        if self.clusters.value > self.initial.value:
            self.clusters.setValue(self.initial.value)
        self._timer.stop()
        self.app.client.invalidate("colors")

    def fit_settings_changed(self, *args):
        """Invalidate only the population fit when estimator settings change."""
        if self.app._presenting:
            return
        self._timer.stop()
        self.app.client.invalidate("population")

    def process_colors(self):
        """Submit validated processing settings with a canonical mesh snapshot."""
        snapshot = SceneBridge.snapshot(self.model.currentNode())
        self.app.remember_source(snapshot)
        self.app.client.process_colors(
            snapshot.mesh,
            directory=str(self.directory.currentPath),
            clustering=bool(self.clustering.checked),
            normalize=bool(self.normalize.checked),
            initial_clusters=int(self.initial.value),
            clusters=int(self.clusters.value),
            sample_count=None if self.all_faces.checked else int(self.samples.value),
            neighbor_average=bool(self.neighbors.checked),
        )

    def show_texture(self):
        """Display an existing raw texture or request stored processed colors."""
        path = Path(self.directory.currentPath) / self.texture.currentText
        if self.raw.checked or path.stem.lower().startswith(
            ("average_texture", "mean_texture")
        ):
            SceneBridge.texture(self.model.currentNode(), path)
            SceneBridge.focus(self.model.currentNode())
        else:
            self.app.remember_source(SceneBridge.snapshot(self.model.currentNode()))
            self.app.client.show_specimen(path.stem)

    def fit(self):
        """Fit the chosen estimator to the current processed population."""
        self.app.remember_source(SceneBridge.snapshot(self.model.currentNode()))
        self.app.client.fit_population(
            method=self.method.currentText.lower(),
            components=int(self.components.value),
        )

    def population_ready(self, result):
        """Populate component and specimen selectors from the completed model."""
        # Block signals while changing both axes so no temporary invalid pair is plotted.
        for combo in (self.x_axis, self.y_axis, self.start):
            combo.blockSignals(True)
            combo.clear()
        prefix = (
            "PC"
            if result.method == "pca"
            else "IC"
            if result.method == "ica"
            else "Component"
        )
        labels = [f"{prefix}{i + 1}" for i in range(result.components)]
        self.x_axis.addItems(labels)
        self.y_axis.addItems(labels)
        self.y_axis.setCurrentIndex(min(1, len(labels) - 1))
        self.start.addItems(list(result.specimen_ids))
        for combo in (self.x_axis, self.y_axis, self.start):
            combo.blockSignals(False)

    def axes_changed(self, *args):
        """Redraw the existing scores when the selected plot columns change."""
        state = self.app.client.state
        if (
            state.population is None
            or min(self.x_axis.currentIndex, self.y_axis.currentIndex) < 0
        ):
            return
        if state.morph_active:
            self.app.guard(self.begin_morph)
        else:
            self.app.guard(
                self.app.client.set_axes,
                self.x_axis.currentIndex,
                self.y_axis.currentIndex,
            )

    def start_changed(self, *args):
        """Move the morphospace starting point to the selected specimen."""
        if self.app.client.state.morph_active and self.start.currentIndex >= 0:
            self.app.guard(self.begin_morph)

    def begin_morph(self):
        """Ask the worker to initialize morphospace from the selected specimen."""
        if self.app.client.state.population is None or self.start.currentIndex < 0:
            return
        self._timer.stop()
        self.app.remember_source(SceneBridge.snapshot(self.model.currentNode()))
        self.app.client.set_axes(self.x_axis.currentIndex, self.y_axis.currentIndex)
        self.app.client.reconstruct(self.start.currentText, None)

    def slider_changed(self, *args):
        """Debounce widget edits; the worker resolves coordinates and coalesces execution."""
        if self.app.client.state.morph_active:
            self._timer.start()

    def explore(self):
        """Send normalized widget positions without interpreting population scores."""
        if not self.app.client.state.morph_active:
            return
        self.app.remember_source(SceneBridge.snapshot(self.model.currentNode()))
        self.app.client.reconstruct(
            self.start.currentText, (self.x.value / 1000, self.y.value / 1000)
        )

    def morph_controls(self, controls):
        """Apply worker-provided positions and labels without submitting another action."""
        for slider, position in zip((self.x, self.y), controls.positions):
            slider.blockSignals(True)
            slider.setValue(round(position * 1000))
            slider.blockSignals(False)
        self.x_label.text = f"X: {controls.coordinates[0]:.3f}"
        self.y_label.text = f"Y: {controls.coordinates[1]:.3f}"
        self.app.plot.point(*controls.coordinates)

    def render(self, state):
        """Enable supported actions while protecting inputs used by active tasks."""
        busy = self.app.client.busy
        if not state.morph_active:
            self._timer.stop()
        for widget in self.inputs:
            widget.enabled = not busy
        clustering = bool(self.clustering.checked)
        for widget in (self.initial, self.clusters, self.normalize):
            widget.enabled = not busy and clustering
        self.samples.enabled = not busy and not self.all_faces.checked
        self.process.text = "Cluster" if clustering else "Subsample and Average"
        self.process.enabled = (
            not busy
            and self.model.currentNode() is not None
            and bool(self.directory.currentPath)
        )
        self.apply.enabled = (
            not busy
            and self.model.currentNode() is not None
            and self.texture.currentIndex >= 0
        )
        self.compare.enabled = not busy and state.colors is not None
        self.method.enabled = self.components.enabled = not busy
        self.x_axis.enabled = self.y_axis.enabled = (
            state.population is not None and not busy
        )
        capable = state.population is not None and state.population.can_reconstruct
        self.start.enabled = self.visualize.enabled = capable and (
            not busy or state.action == "morphospace"
        )
        self.x.enabled = self.y.enabled = (
            capable
            and self.app.client.state.morph_active
            and (not busy or state.action == "morphospace")
        )
        self.morph_hint.text = (
            "UMAP reconstruction is approximate."
            if capable and state.population.method == "umap"
            else "Explore spatial color variation."
            if capable
            else "Morphospace requires a fitted model with spatial color features and an inverse transform."
        )

    def close(self):
        """Stop pending slider callbacks before the view is destroyed."""
        self._timer.stop()
