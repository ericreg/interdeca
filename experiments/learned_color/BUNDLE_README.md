# Reproduce the frozen I-JEPA mussel analysis

This local research bundle contains actual native data for the recovered 32-mussel
cohort, its current source snapshot, the pinned I-JEPA checkpoint, and preserved
numerical reference arrays. It implements paper paragraphs 137–142. It is an
independent reconstruction of the documented method. Original author embedding
arrays remain unavailable, and the original draft's following figure depicts six
fish rather than this cohort. See `figure-map.json` and `ACCESS.json`.

Copy this entire directory to a new location. Do not copy an old virtualenv.
Commands below run from the bundle directory. `reproduce.py` also resolves its
bundle location when launched from another working directory.

## Install and verify

Prerequisites: Python to launch the bootstrap and the `uv` package manager.
Installation needs package-index access. It creates a new Python 3.12.8 environment
in `.venv`, installs the exact versions in `requirements.txt`, and writes the
actual platform dependency closure to `.runtime/resolved-requirements.txt`.
The 62 version pins in `requirements.txt` come from the original Mac run. Linux
installs the independently resolved 77-package CUDA closure from
`environment/requirements-linux-cuda.txt`, which retains the original versions
and pins the additional Torch platform dependencies. These are platform-specific
version locks, not a single cross-platform wheel/hash lock.

```sh
python reproduce.py verify
python reproduce.py install
python reproduce.py preflight --device cuda
```

For a separately created clean environment, place `--python /absolute/path/to/python`
before the subcommand, for example `python reproduce.py --python /tmp/env/bin/python
preflight --device cuda`. Installation creates the default `.venv`, or installs
into an explicitly supplied already-created clean interpreter when `--python`
is provided. The supported checked platforms are macOS arm64 and Linux x86_64;
other platform wheel availability has not been established.

Select `cuda`, `mps`, or `cpu` explicitly. No accelerator silently falls back to CPU.
Linux requires a working VTK headless OpenGL/EGL backend. `preflight` performs a tiny
offscreen render and a device allocation, recording the actual render-window class.
If EGL is unavailable, provision an appropriate OpenGL/Xvfb environment before
running; this bundle does not silently alter rendering settings.

## Checkpoint and offline execution

The checkpoint is `facebook/ijepa_vith14_1k` at immutable revision
`f157467ea509bc356ff9f61fd3c0d840eec5e04e`. `model-contract.json` lists SHA256s for
the weights, model config, and processor config. All three regular files are
included under `model/hub`; they have no dependency on another machine's cache.
Run commands force Hugging Face offline mode and use only that cache by default.
To test an independent download into a new isolated cache:

```sh
python reproduce.py fetch-model --cache .runtime/fresh-model/hub
```

The downloaded files must match the bundled contract. Add
`--model-cache .runtime/fresh-model/hub` to `run` to use that verified download.
The original model card at the exact revision is preserved as `model/model-card.md`
and declares `cc-by-nc-4.0`; its noncommercial terms and attribution requirements
are separate from native-data and source access terms.

## Real-data smoke, full run, checks, and figures

The smoke uses four actual specimens including camera reference UF_IZ_438751,
with all 64 crops per specimen. It executes rendering, sampling, model inference,
aggregation, PCA, and a genuine UMAP fit. It does not stand in for the 32-specimen
study. Each command refuses to overwrite an existing run/output directory.

```sh
python reproduce.py run --profile smoke --device cuda --batch-size 2
python reproduce.py validate --run runs/smoke
python reproduce.py figures --run runs/smoke --output figures/smoke

python reproduce.py run --profile full --device cuda --batch-size 2
python reproduce.py validate --run runs/full
python reproduce.py figures --run runs/full --output figures/full
```

Batch size 2 reduces memory pressure on 10 GB GPUs. The original profile used 8;
an override is explicit in the new run's config. It does not change crop selection
or averaging definitions, but exact cross-device floating-point equality is not
assumed. CPU inference is supported but can be slow; MPS is available on supported
Apple hardware. Full input data occupy about 1.6 GB, weights about 2.5 GB, with
additional disk/RAM needed for dependencies and outputs. Only one full inference
job should run on a device at a time.

`validate` requires the exact shipped smoke/full scientific profile, complete
source inventory, and native input hashes bound by specimen ID to the bundled
provenance. Only device, its CPU opt-in, batch size, and input/output locations
may differ from the shipped profile. It also checks current completion/config/
output hashes, all saved crop pixels/masks/duplicates/overlaps, raw crop averaging and
single normalization, saved PCA transforms, and independent cosine rankings.
`validation.json` separately reports differences from the included October 6 MPS
vectors. Internal consistency can pass while cross-device historical agreement
differs; neither certifies missing original author arrays. A four-specimen fit
has different PCA/UMAP coordinates from a 32-specimen fit by design.

## Bundle contents and evidence boundaries

- `source/`, `source-identity.json`, `working-tree.patch`: exact current source,
  including uncommitted fixes and this tooling. `.git` is not required to run.
- `inputs/`: 32 native meshes, textures, and supplied landmarks. Landmarks are
  preserved provenance; this active group-PCA method does not consume them.
- `input-provenance.json`, `ACCESS.json`: specimen/checksum inventory, research
  handoff provenance, and unresolved public redistribution rights. This is a
  private local reproduction bundle, not a public dataset release.
- `configs/`: relocatable full/smoke profiles; actual resolved configs go to
  `.runtime` and each new run records its exact config/source identity.
- `reference/historical-*`: unchanged historical numerical arrays, coordinates,
  processor/model metadata, figures, and receipts. Any old absolute paths there
  describe history; runtime commands never use them as current input paths.
- `manifest.json` and `manifest.sha256`: every shipped source/data/model/reference
  file. Verification rejects modified bytes and unexpected Python source.
  Runtime envs, caches, runs, and exports are deliberately outside that snapshot.

Package integrity is relative to the supplied manifest. Retain its digest through
an independent trusted channel when transferring. It is not a signature or a
claim that the original investigator's outputs have been reproduced.

The figures are descriptive fits to the complete configured cohort. They do not
establish taxonomic validity, held-out performance, superiority to ColorAtlas,
texture-only invariance, or anatomical correspondence. Square stretching,
single-view occlusion, shading, and acquisition artifacts remain part of the
estimator.
