# Frozen I-JEPA surface-appearance experiment

This experiment lives inside InterDeCA. It implements the frozen inference
protocol documented by Eric Regina in
[`learned-color-space` at `b30c495`](https://github.com/bshi42/3d_color_package/tree/b30c495c701093e456231ec8a035012266ccef74/learned_color_space).
That commit directly descends from `demo` at `464edfb`.

**Reproduction boundary:** the upstream commit includes rendering and sampling
helpers and a technical description, but does not include the referenced
`ijepa_texture_embeddings/ijepa_pipeline_2.py`, active configuration, exact
32-specimen cohort or analysis outputs. This is an independently executable
reconstruction of the documented method, not recovered original experiment
code or a certified reproduction of its published figures. The missing inputs
remain necessary for that claim. See the [execution record](../../docs/learned-color/reproduction.md).

## Method

1. Pair native OBJ meshes and original textures by exact filename stem. An
   explicit expected count prevents quietly substituting another cohort.
2. Render a 2048² orthographic exterior view using Eric's scene helpers,
   smooth shading and fixed material coefficients. Group-PCA signs follow the
   reference specimen's standardized third-moment signature.
3. Crop alpha-defined surface content with 12-pixel padding. The example uses
   the documented compatibility mode that stretches the crop into a square.
   This distorts projected shape; it is an explicit protocol choice.
4. Sample 64 distinct, unaugmented 256² crops per specimen, entirely within the
   surface mask. Use the documented specimen-specific BLAKE2b seed and greedy
   maximum-IoU 0.6 rule. Reject shortages rather than duplicating crops.
5. Load the frozen `facebook/ijepa_vith14_1k` encoder and image processor at the
   same immutable revision. The processor resizes to 224². Average all 256
   spatial tokens into each 1280-dimensional raw crop vector.
6. Average the 64 raw crop vectors, then L2-normalize once per specimen.
7. Fit standardized PCA and cosine UMAP to the complete cohort and report
   cosine neighbors. These are descriptive fits, not supervised or held-out
   performance estimates. PCA loadings and scaling parameters are saved.

The model is original **I-JEPA**, not V-JEPA 2. No encoder fine-tuning, predictor,
EMA target encoder or JEPA training loss is used. The representation retains
geometry, shading and acquisition artifacts and discards crop coordinates. It
does not replace the atlas's anatomical correspondence or inverse mapping.

The integration currently implements `group_pca`, matching the documented active
profile. It does not silently choose landmark-group alignment or implement the
alternative fine-tuning plans. Nearly symmetric shapes and principal-axis
permutations remain limitations of this orientation method.

## Environment

Use the checked-in frozen environment independently of Slicer and the numerical core:

```sh
uv python install 3.12.8
uv venv --python 3.12.8 data/learned-color/env
uv pip install --python data/learned-color/env/bin/python \
  -r experiments/learned_color/requirements.txt
```

The first embedding run downloads the approximately 2.5 GB pretrained
SafeTensors checkpoint. It is pinned to
`f157467ea509bc356ff9f61fd3c0d840eec5e04e`; the downloaded weights are not committed.
Inference uses float32 and eager attention. Choose `cuda`, `mps`, or `cpu`
explicitly. A requested unavailable accelerator fails. CPU execution requires
`allow_cpu: true`; there is no silent CPU fallback. MPS is an explicit addition
for the local Mac validation run, not a claim of bitwise equality to Eric's GPU.

## Configure and run

Copy `configs/mussels-paper.example.json` to a local JSON file and edit the
input/output paths and device. Paths resolve relative to the configuration file.
Use Eric's exact 32 specimen IDs in `specimen_ids`, or supply a directory
containing exactly his cohort. The example's expected count is 32. If you run the
47 locally available mussels, set a different profile and expected count 47 and
describe it as a new method rerun.

From the repository root:

```sh
data/learned-color/env/bin/python -m experiments.learned_color \
  --config /path/to/config.json --stage all
```

For a clean population export with exact specimen identities in the accompanying
CSV rather than overlapping point labels:

```sh
MPLBACKEND=Agg data/learned-color/env/bin/python -m experiments.learned_color.plot \
  --run /path/to/completed-run --output /path/to/new-export-directory
```

Stages are also selectable individually: `prepare`, `render`, `sample`, `embed`,
and `analyze`. Preparation refuses an existing output directory. Source/config
changes and input checksum changes are rejected before downstream work. Render,
patch and embedding hashes establish the stage handoff. Failed stages retain
their partial outputs; start a new run when changing a protocol or recovering
an incomplete stage rather than mixing artifacts from different settings.

## Outputs

- `config.json`, `run.json`: resolved protocol, input hashes, source hashes,
  platform, package versions and an explicit original-reproduction limitation.
- `renders/`, `renders.json`: RGBA views, camera frames, sign diagnostics and
  image SHA256s.
- `patches/`, `patch_manifest.csv` and JSON: exact ordered crops, coordinates,
  background fractions and source-render/patch hashes.
- `crop_embeddings.npz`, `specimen_embeddings.npz`, `embedding_metadata.json`:
  raw crop vectors, unit specimen vectors, checkpoint revision, full processor
  and model configurations, pooling/aggregation semantics and content hashes.
- `analysis/`: coordinate and neighbor CSVs, PCA transform NPZ, population
  PNG/PDF and descriptive summary. `completed.json` appears only after analysis
  succeeds. It certifies this configured run, not the missing original run.

For populations of two or three specimens, the documented PCA-coordinate fallback
is labeled explicitly instead of presenting it as a fitted UMAP. The paper
profile uses 32 specimens.

`reference/upstream.json` records source hashes and the two import/typing
adaptations to Eric's numerical helpers. The original technical description is
retained as attributed reference material; its links to unavailable upstream
files are historical, not links to implemented local modules.

The encoder interface follows the
[Transformers I-JEPA documentation](https://huggingface.co/docs/transformers/v4.56.2/model_doc/ijepa)
and the [immutable model snapshot](https://huggingface.co/facebook/ijepa_vith14_1k/tree/f157467ea509bc356ff9f61fd3c0d840eec5e04e).
