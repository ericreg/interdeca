# Manually test InterDeCA Next with the small clams dataset

Use this walkthrough to check the Slicer controls and inspect their results.
The fixture is the same three-specimen population used by the earlier project:

| Specimen | Source mesh | Landmarks | Source texture |
| --- | --- | --- | --- |
| `UF_IZ_507781` | 544,440 triangles | 48 | PNG |
| `UF_IZ_507782` | 335,214 triangles | 48 | TIFF |
| `UF_IZ_507879` | 265,122 triangles | 48 | PNG |

The source dataset is at:

```text
/Users/eric/code/3d_color_package/tests/data/small_clams
```

“Small” describes the number of specimens; the meshes are still dense. Start
with 1,000 color samples and a 512-pixel bake. These are manual-check settings,
not a claim about the resolution needed for scientific analysis.

There are two starting points. **Route A** reuses the previous atlas and baked
textures to get straight to the interactive controls. **Route B** also tests the
new atlas construction and native texture pipeline. Both lead into the same color,
population, morphospace, and selection checks below.

## Before starting

1. Restart Slicer after the latest source changes. Open **Shape Analysis →
   InterDeCA Next** and confirm all three tabs appear.
2. Run `uv sync` to install all core dependencies, including UMAP, and confirm
   the independent worker starts in Slicer. Use PCA for this tutorial: the core
   requires at least four specimens for UMAP, so this
   three-specimen fixture is not a UMAP test.
3. Save any scene you want to retain, then work in a fresh scene so old models
   and plots do not obscure the results.
4. Route B runs cleanup, UV projection, and baking in the native worker;
   no Blender installation is required.

See [Slicer setup](../slicer.md) if the module or required packages are missing.
Keep original input files and earlier baselines unchanged. This tutorial puts
new inputs and outputs under `/Users/eric/interdeca-runs/small_clams/`.

## Route A: start from the earlier small-clams atlas

This checks the new interactive analysis using already-baked data. It does not
test the new atlas construction or native baking stages.

Open Slicer's **Python Console** and paste:

```python
from pathlib import Path

from interdeca.io import MeshIO
from interdeca_slicer.scene import SceneBridge

baseline = Path(
    "/Users/eric/code/3d_color_package/tests/output/"
    "slicer_small_clams/slicer-5.12.4_blender-5.2.1/colorAnalysis"
)

# The core reader keeps RAS coordinates and canonical vertex IDs across UV seams.
atlas_node = SceneBridge.create_model(
    MeshIO.read(baseline / "atlasModelUV.obj"), "SmallClams Atlas"
)
SceneBridge.focus(atlas_node)
```

Using this reader preserves the mesh/UV representation expected by the new
analysis. The atlas should appear as a shell; a plain gray initial display is
normal because no texture has been applied yet.

In **MultiRecolor**, select **SmallClams Atlas** as **Model**, and set
**Texture Directory** to:

```text
/Users/eric/code/3d_color_package/tests/output/slicer_small_clams/slicer-5.12.4_blender-5.2.1/colorAnalysis/atlasTextures
```

Continue at [Inspect individual textures and process colors](#inspect-individual-textures-and-process-colors).

## Route B: construct a new atlas and bake textures

### Prepare a copy in a common coordinate frame

The original OBJ positions match the LPS positions in the landmark files.
The new core reads OBJ positions as RAS and converts landmark positions using
their JSON metadata. Using the original files directly would therefore put
mesh and landmarks in different frames.

Run the following once in **Terminal**. It writes RAS meshes and RAS landmark
files into a separate working directory and copies the original texture images.
Changing LPS to RAS reverses both X and Y; it preserves distances, triangle
winding, connectivity, and per-corner UVs. Do not flip the texture images.

```bash
cd /Users/eric/code/interdeca
uv run python - <<'PY'
from pathlib import Path
from shutil import copyfile

from interdeca.io import LandmarkIO, MeshIO

source = Path("/Users/eric/code/3d_color_package/tests/data/small_clams")
work = Path.home() / "interdeca-runs" / "small_clams"
destination = work / "inputs_ras"
specimens = (
    ("UF_IZ_507781", ".png"),
    ("UF_IZ_507782", ".tiff"),
    ("UF_IZ_507879", ".png"),
)

# Refuse an existing copy so rerunning cannot replace previously prepared inputs.
destination.mkdir(parents=True, exist_ok=False)
for folder in ("models", "landmarks", "textures"):
    (destination / folder).mkdir()

for specimen_id, texture_suffix in specimens:
    # Both readers produce RAS positions; the landmark reader honors LPS metadata.
    mesh = MeshIO.read(
        source / "models" / f"{specimen_id}.obj", coordinate_system="LPS"
    )
    landmarks = LandmarkIO.read(
        source / "landmarks" / f"{specimen_id}.mrk.json"
    )

    # Save the new coordinates while retaining face identities and texture layout.
    MeshIO.write_obj(mesh, destination / "models" / f"{specimen_id}.obj")
    LandmarkIO.write_json(
        landmarks, destination / "landmarks" / f"{specimen_id}.mrk.json"
    )
    texture_name = specimen_id + texture_suffix
    copyfile(
        source / "textures" / texture_name,
        destination / "textures" / texture_name,
    )
    print(f"Prepared {specimen_id}", flush=True)

# The atlas workflow creates a separate timestamped directory inside this folder.
(work / "runs").mkdir(exist_ok=True)
print(f"Inputs: {destination}")
print(f"Output parent: {work / 'runs'}")
PY
```

Expect three `Prepared ...` messages. On subsequent runs, reuse this prepared
copy and skip the command. If preparation stopped partway through, inspect the
error and choose a fresh working directory before retrying. The conversion is
specific to these raw clams files; do not repeat it on an already-RAS atlas.

### Set the ATLAS controls

In **ATLAS**, enter:

| Control | Value |
| --- | --- |
| Models | `/Users/eric/interdeca-runs/small_clams/inputs_ras/models` |
| Landmarks | `/Users/eric/interdeca-runs/small_clams/inputs_ras/landmarks` |
| Textures | `/Users/eric/interdeca-runs/small_clams/inputs_ras/textures` |
| Output directory | `/Users/eric/interdeca-runs/small_clams/runs` |
| Atlas Model / Atlas Landmarks override | Leave both empty |
| Merge by distance | `0.0001` |
| Smart UV angle (deg) | `66` |
| Island margin (UV) | `0.002` |
| Bake size (px) | `512` for the first check |
| Bake extrusion | `0.001` |
| Bake margin (px) | `2` |

Expand **Surface processing (cleanup, UV, bake)** to see the numerical settings. They match
the earlier workflow defaults except for the smaller first-pass bake. After a
successful check, use `2048` for a higher-resolution visual comparison.

Click **Run ATLAS and Texture Transfer**. Expect progress through inspection,
reference/contribution/mean construction, cleanup, UVs, alignment, resampling,
baking, and averaging. Some tasks are much slower than others; the task counter
is not an elapsed-time estimate. Input controls lock while the run is active.

**Completion check:** the log reports `completed: ATLAS and Texture Transfer`
and a saved-results path. The module switches to **MultiRecolor**, selects the
new **ATLAS Model**, and fills its texture directory. The average texture is
applied for preview.

The saved directory has this structure:

```text
runs/interdeca_<timestamp>_<id>/
  atlas.obj
  atlas.mrk.json
  aligned/       # Three OBJ files and three landmark files
  resampled/     # Three OBJ files and three landmark files
  textures/
    UF_IZ_507781.png
    UF_IZ_507782.png
    UF_IZ_507879.png
    average_texture.png
```

All three baked specimen images are PNG, including the specimen whose source
image is TIFF. Inspect the individual bakes as well as their average: an average
can hide a poor correspondence or a failed bake for one specimen.

## Inspect individual textures and process colors

Both routes should now have one atlas selected in **MultiRecolor** and the
matching baked-texture directory selected. Use the atlas and textures from the
same run; the original specimen textures have different UV maps.

1. In **Step 2: Individual Visualization**, choose `UF_IZ_507781.png`.
   Check **Raw Texture** and click **Apply Texture**. Rotate the shell to inspect
   both sides. Repeat for the other two specimens.
2. Choose `average_texture.png` and apply it. It is a preview-only image and
   must not become a fourth specimen in the population.
3. In **Step 1: Multi-texture Processing**, choose **Subsample and Average Only**,
   set **Number of Faces (Subsampling)** to `1000`, leave **Disable Subsampling
   (Use All Faces)** unchecked, and leave **Neighbor Average** unchecked.
4. Click **Subsample and Average** and wait for `completed: Color processing`.
5. Choose an individual specimen again, uncheck **Raw Texture**, and click
   **Apply Texture**. Repeat for all three specimens.

**Check:** raw displays preserve the baked image detail; sampled displays look
coarser because each measured color fills its surrounding face region. Gray
regions indicate missing valid color coverage. Large unexpected gray regions,
an almost uniform result, or colors on the wrong part of the shell deserve
investigation before population analysis.

To check neighbor averaging, enable **Neighbor Average** and process again.
Expect local smoothing. A setting change invalidates the previous processing
and population fit, so rerun processing before using downstream controls.

## Plot PCA and explore morphospace

1. In **Step 3: Population Analysis**, choose **PCA**, set **Number of Components**
   to `2`, and click **Compare Textures**.
2. Confirm a plot appears alongside the shell. The specimen series should contain
   the three clam IDs, excluding `average_texture`. Points may overlap, so inspect
   the **InterDeCA population** table in Slicer's Tables module if needed.
3. Set **X-axis** to `PC2` and **Y-axis** to `PC1`. The existing plot should redraw
   without another population-fitting run. Restore `PC1` and `PC2` afterward.
4. Expand **Step 4: Morphospace**, choose `UF_IZ_507781` under **Starting Texture**,
   and click **Visualize Morphospace**.
5. Move one slider slowly, then the other. Check that the red plot marker moves
   and the surface colors update. Try a short rapid drag and release: the final
   surface should settle on the latest slider position.
6. Change **Starting Texture** to another specimen. Its coordinates should become
   the new starting point.

Three centered specimens support at most two independent components; requesting
more is an expected validation error. PCA coordinates may have different signs
from an earlier implementation. Inspect specimen identity and behavior rather
than expecting identical screen positions or image hashes.

Morphospace reconstructs color features on the fixed atlas. The shell geometry
should stay fixed. With sampled data, the starting reconstruction should resemble
the processed display; it need not reproduce the full raw image detail.

## Check clustering and its shared palette

1. Return to **Step 1**, choose **Clustering**, and use **Initial Clusters = 24**
   and **Consolidated Clusters = 8**. Keep `1000` samples, subsampling enabled,
   **Normalize Luminosity** unchecked, and **Neighbor Average** unchecked.
2. Click **Cluster**. Then apply each individual specimen with **Raw Texture**
   unchecked. Expect a limited shared palette across the specimens.
3. Fit PCA again. Keeping subsampling enabled retains spatial color features,
   so morphospace should remain available after this new fit.
4. Optionally enable **Normalize Luminosity**, cluster again, and compare the
   processed colors. Toggle one setting at a time so its effect is interpretable.

For an additional feature-mode check, enable **Disable Subsampling (Use All
Faces)** while clustering, then cluster and fit PCA again. This uses cluster-area
features. **Morphospace should be disabled**, because a cluster's total area does
not say where its colors belong on the surface. This branch processes every face
and can use more memory. Return to sampled processing and refit to restore spatial
morphospace.

## Select, accumulate, clear, and export a region

1. Apply a raw specimen texture for a recognizable background, then open
   **Mesh Selection**. Select the same atlas as **Target Mesh**.
2. Set **Selection Markup** to **None**, click **Draw Curve**, and place at least
   four well-separated points around a modest patch on one visible side of the
   shell. Press **Escape** to stop placing points.
3. Leave **Select only one side** checked and click **Apply Selection**. Confirm
   highlighting appears and the vertex/face counts are nonzero.
4. To add another patch, set **Selection Markup** to **None** again before
   **Draw Curve**. This creates a new curve rather than adding points to the old
   one. Draw, press **Escape**, and apply. The accumulated count should not decrease.
5. Enter **Export Name = SmallClamsPatch** and click **Export Selected Region as
   Model**. Find the new node in Slicer's Data module. Hide the full atlas there
   to inspect the exported patch, then show the atlas again.
6. Click **Clear Selection**. The counts and overlay should clear, the atlas's
   previous appearance should return, and the exported node should remain.

The export button creates a scene model. To keep it on disk, use Slicer's **Save**
dialog; VTP retains its UV and numerical point/cell arrays. The export is the
accumulated applied selection, not merely the latest drawn curve.

For a side-filter comparison, clear the selection, apply one curve with the
filter enabled, record the count, then clear and apply that same curve with the
filter disabled. Clearing between comparisons avoids mixing the two results.
Selection uses a projected convex region and a side heuristic; it is not an exact
concave or geodesic cut along the drawn curve.

## Cancellation and atlas override checks

Do these after inspecting your successful results.

- **Cancellation:** start color processing and click **Cancel Operation** while it
  is active. Expect a cancelling state until the current task finishes, followed
  by cancellation and unlocked controls. Later tasks and obsolete display results
  should not be applied. If a short run finishes before the click, try another run.
- **Atlas override:** in **ATLAS**, choose the `atlas.obj` and `atlas.mrk.json` from
  a successful Route B run as the paired override. Keep the prepared input
  directories and textures. Run again. Construction of the mean atlas is skipped,
  while cleanup, UV generation, resampling, and baking produce a new run directory.
- **Merge disabled:** repeat that override check with **Merge by distance = 0**.
  This disables merging, not UV generation. Use this run's own baked textures.
- **No-texture branch:** clear **Textures** and run the override again. Expect
  geometry/landmark outputs and an empty downstream texture-directory selector.
  Do not pair the new atlas with textures left over from another UV map.

Changing ATLAS settings invalidates downstream results. If you only want to
analyze an already-baked atlas, use Route A's loading approach with that atlas's
own OBJ and texture directory instead of running an override.

## Record the outcome

Keep this checklist with the chosen run directory and screenshots:

| Check | Expected outcome | Pass / fail / notes |
| --- | --- | --- |
| Atlas construction, if using Route B | Completion, atlas, three resampled specimens, four texture images | |
| Individual raw preview | Three distinct specimen displays and an average preview | |
| Sampled processing | Three processed specimens; coarse spatial colors | |
| PCA | Three specimens and two components; axes change without refitting | |
| Morphospace | Marker and colors respond; final slider position wins | |
| Clustering | Shared palette; reprocessing invalidates the former fit | |
| Cluster-area mode | Spatial morphospace unavailable | |
| Selection | Nonzero counts, accumulation, clear, independent exported node | |
| Cancellation | Stops subsequent tasks; controls become usable again | |
| Override / zero merge / no textures, if exercised | New isolated outputs and appropriate selector handoff | |

For a failure, record the route, last successful step, exact input/output paths,
changed settings, last task name, and complete error text from Slicer's error log
or Python Console. Keep partial output directories for inspection. A saved scene
can preserve visual artifacts, but the current application does not restore its
live fitted estimator/session state after a restart; rerun processing and fitting.

If an error says more samples are needed to cover disconnected components,
increase the sampling count to at least the reported minimum. If landmark IDs
are rejected as integers, restart Slicer to load the compatibility fix. If PCA
reports insufficient feature rank, inspect the raw and sampled textures before
trying to fit again; do not treat a failed fit as a completed result.

## What has been checked while preparing this tutorial

The original three meshes, textures, and 48-landmark sets were read with the new
core. Numeric landmark IDs are now accepted alongside strings, while invalid
ID types remain rejected. A nearest-vertex check confirmed the coordinate-frame
mismatch in the unconverted inputs and supported the LPS-to-RAS preparation above.
The previous atlas was also read with its UVs intact. Using its baked textures,
the core pipeline successfully processed all three specimens at 1,000 samples,
fitted two-component PCA, and reconstructed face colors. That is a computation
check of Route A's data, not a completed manual UI inspection.

These checks do not establish full numerical equivalence with the legacy
application. The full Route B atlas/baking run on this real dataset has not been
executed while preparing the tutorial; it remains part of the manual checklist.
