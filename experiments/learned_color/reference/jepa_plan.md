# JEPA-Style Above-Surface Texture-Patch Embeddings

## Summary

Build a new `ijepa_texture_embeddings/` experiment that renders **one high-resolution “above” view per clam**, samples **non-white texture patches** from that view, and produces per-specimen embeddings from a **pretrained ViT encoder**, with an optional **cross-patch JEPA fine-tuning** stage layered on top.

Data model (revised): instead of many full-frame views per specimen, each specimen contributes a single large render that is **tiled into many overlapping, non-duplicate, all-surface patches**. This directly serves the *texture* goal — patches contain only shell surface (no silhouette, no background), so the representation can't cheat on outline/pose — and it yields far more genuine diversity than near-duplicate full views.

We have only **32 specimens** (confirmed: 32 OBJ↔texture pairs in `../data/All_Clams`). That is the real `n` for *per-specimen* generalization, so training a ViT from random init is still off the table — patches raise the sample count but not the number of independent objects. The sibling baseline shows the right pattern at this scale: `dinov2_texture_embedding/` uses a **frozen pretrained** `facebook/dinov2-small` ([texture_dino_space.py:292-295](dinov2_texture_embedding/texture_dino_space.py#L292-L295)). This plan mirrors that and only *adds* a JEPA component on top of pretrained weights.

Default decisions:
- **Encoder is pretrained, never random-init.** Primary path: a frozen pretrained ViT/I-JEPA checkpoint as a feature extractor (apples-to-apples with the DINOv2 baseline). Optional second path: light fine-tuning of that checkpoint.
- **Patch size is configurable**, defaulting large enough to be a valid pretrained-encoder input (a 16×16px patch is a single ViT token and cannot be fed to a foundation model without hallucinatory upsampling). Patches are resized to the encoder's expected input size before encoding.
- **Renders are high resolution** so that many distinct, non-overlapping-enough patches fit on the shell surface.
- **If we fine-tune, use a cross-patch objective.** Patches drawn from the same specimen are natural positives — predict one patch's pooled latent from another patch of the *same* specimen. (Alternatively, strict intra-patch I-JEPA masking now works too, since a large patch still tokenizes into a grid.)
- Train/fine-tune on ROCm/PyTorch GPU only; no silent CPU fallback.

## Dependencies

None of these are currently declared in `pyproject.toml` (it has `click`, `imagecodecs`, `imageio`, `pyqt5`, `qt-py`, `vtk` only). Add before implementation:
- `torch` (ROCm build — must match the host's already-installed ROCm version; do not let `uv`/`pip` silently resolve a CUDA-only wheel).
- `pydantic` for the config schema.
- `scikit-learn` (PCA) and `umap-learn` (UMAP) for the visualization stage — same as used informally in `dinov2_texture_embedding/` and `beta_vae_texture_embeddings/`.
- A source for `pretrained_checkpoint`: `transformers` (HF) for a ViT/I-JEPA from the Hub like the DINOv2 baseline does, or `timm`. Pin this — the encoder is never trained from scratch.
- Confirm the ROCm PyTorch wheel actually reports `torch.cuda.is_available() == True` on this host before relying on the "fail loudly" check in Test Plan; that check can't distinguish "no GPU" from "wrong wheel installed."

## Key Changes

- Add a Click + Pydantic config pipeline in `ijepa_texture_embeddings/ijepa_pipeline.py` with commands:
  - `write-config`
  - `render-above`   (one high-res above render per specimen)
  - `sample-patches` (extract non-white patches from each render)
  - `train`          (finetune mode only)
  - `embed`
  - `run-all`
- Add `ijepa_texture_embeddings/ijepa_config.yaml` with documented defaults:
  - `mode: frozen` — `frozen` (extract embeddings from the pretrained encoder, no training) or `finetune` (cross-patch JEPA on top of pretrained weights). `frozen` is the default and the required first milestone.
  - `pretrained_checkpoint` — pretrained ViT/I-JEPA weights to load. **Random initialization is not an option.**
  - `encoder_input_size` — size patches are resized to before encoding (e.g. 224 for DINOv2/ViT). Patches smaller than this are upsampled; larger ones downsampled.
  - Render keys:
    - `render_image_size: 2048` — large above render so many distinct patches fit. Tune to mesh detail / texture resolution.
    - `transparent: true` — render with an alpha background so "non-white" is unambiguous (reuse the alpha path in [render_clam.py:202-206](render_clam.py#L202-L206)). For opaque renders, fall back to `opaque_content_mask` with `background_tolerance`.
    - `render_seed` — even the single above render should be reproducible (orientation tie-breaks, sampling).
  - Patch-sampling keys:
    - `patch_size_px: 128` — configurable; the sampling window in render pixels.
    - `patches_per_specimen: 256` — target count; sampling stops early if the surface can't yield this many valid non-duplicate patches.
    - `require_full_content: true` — reject any patch containing a background/white pixel (strict, per the request). A softer `max_background_fraction` knob can relax this if too few patches pass.
    - Overlap/dedup policy: patches **may overlap but must not be duplicates** — enforce a minimum center-to-center distance (`min_patch_center_distance_px`) and/or a maximum pairwise overlap (`max_patch_overlap_iou`), and reject pixel-identical crops.
    - `patch_sample_seed` — deterministic patch coordinates; two runs must produce the same patch set.
  - Jitter keys (applied to sampled patches):
    - Photometric: `brightness_jitter`, `contrast_jitter`, `saturation_jitter`, `hue_jitter` (mild).
    - Geometric: `random_flip`, `max_rotation_degrees` (small) — applied after sampling, before encoding.
  - Fine-tune-only keys (ignored in `frozen` mode):
    - `epochs`, `batch_size`, `learning_rate`, `weight_decay`, `warmup_epochs`, `optimizer` (AdamW). JEPA is sensitive to the LR/warmup schedule; do not leave these as undocumented code defaults.
    - `ema_momentum_start` / `ema_momentum_end` for the EMA target encoder — the decay schedule directly affects target stability.
    - Cross-patch objective: predict the pooled target-encoder latent of one patch from the online-encoder latent of another patch of the *same* specimen. (For strict intra-patch I-JEPA instead, also specify `target_block_scale_range`, `target_block_aspect_ratio_range`, `num_target_blocks`, `context_scale_range`.)
  - Evaluation keys:
    - `holdout_specimen_ids` / `holdout_fraction` — specimens reserved from any fine-tuning, used for the comparison metric. With n=32 and no split the evaluation is circular; mandatory whenever `mode: finetune`.
    - `taxonomy_labels_path` (optional) — external labels (species/genus) to ground evaluation against, instead of only agreeing with another unsupervised method.
- Reuse `render_clam.py` orientation/rendering and content-mask helpers so the above render inherits the corrected “above” shell orientation and the patch sampler reuses the existing background detection.
- Render the explicitly-paired OBJ/texture specimens (the 32 obj∩texture pairs in `../data/All_Clams`), **not** a broad glob. Pin `models_dir`/`textures_dir` and assert the resolved pair count, mirroring [render_clam.py:167-184](render_clam.py#L167-L184) — `data/` also contains `Cichlid_Full_Dataset` and `archive` (~177 OBJs total), so an unpinned glob would silently render the wrong set.

## Data Flow

- Render stage (`render-above`):
  - For each specimen, compute the exterior “above” direction using the existing geometry orientation logic and render a single high-res view.
  - Save to `ijepa_texture_embeddings/rendered_above/{specimen_id}_above.png` (RGBA when `transparent: true`).
  - Save a content mask per render (alpha-derived, or `opaque_content_mask`).

- Patch-sampling stage (`sample-patches`):
  - For each render, sample `patches_per_specimen` windows of `patch_size_px`, keeping only patches that are fully on the shell surface (`require_full_content`).
  - Enforce overlap/dedup: patches may overlap but must differ (min center distance / max IoU; reject identical crops).
  - Save patches to `ijepa_texture_embeddings/patches/{specimen_id}/{specimen_id}_p{index:04d}.png`.
  - Save metadata to `ijepa_texture_embeddings/outputs/tables/patch_manifest.csv`.
  - Manifest columns: specimen id, patch index, patch path, source render path, patch top-left x/y, patch size, background fraction, applied-jitter params, model path, texture path.

- Training stage (`mode: finetune` only — skipped in `frozen` mode):
  - Initialize both online and target encoders from `pretrained_checkpoint`. Never from random init.
  - Train on the fine-tune split only; `holdout_specimen_ids` are excluded so the next stage's evaluation is not circular.
  - Cross-patch JEPA objective (default):
    - Sample two patches A, B from the same specimen.
    - Online ViT encodes A; EMA target ViT encodes B without gradients.
    - Predictor maps A's tokens to B's pooled target latent.
    - Loss is MSE between normalized predicted and normalized EMA target embeddings; no pixel reconstruction.
  - **Collapse monitoring (required):** EMA-target latent prediction collapses readily on low-diversity data, and a collapsed model still shows a low, smooth, decreasing, finite loss. Log per-epoch: std of normalized embeddings, effective rank / participation ratio of the embedding covariance, and a KNN-probe metric on the holdout. "Loss is finite" is not a health check.

- Embedding stage:
  - In `frozen` mode, embed each patch directly from the pretrained encoder. In `finetune` mode, use the EMA target encoder (average-pooled — the canonical JEPA eval choice).
  - Export per-patch embeddings and per-specimen embeddings.
  - Per-specimen embedding is the normalized mean of its normalized patch embeddings.

- Visualization/evaluation:
  - Write PCA and UMAP plots with shell thumbnails.
  - Write nearest-neighbor contact sheets from the embeddings.
  - Compare against the existing frozen-DINOv2 baseline in `dinov2_texture_embedding/dinov2_texture_embeddings.npz` (already in the repo — run unconditionally). Apples-to-apples only in `frozen` mode (pretrained vs pretrained); a fine-tuned model that has seen the data must be compared **on the holdout**.
  - **External validity:** neighbor-overlap with DINOv2 only measures agreement with another unsupervised method — high overlap can mean "learned nothing new," low overlap can mean "learned something better" *or* "garbage." When `taxonomy_labels_path` is available, also report an external metric (KNN accuracy or silhouette score against species/genus labels). If no labels exist, state this as a limitation rather than treating overlap as a quality score.

## Outputs

Expected output structure:

```text
ijepa_texture_embeddings/
  ijepa_config.yaml
  ijepa_pipeline.py
  rendered_above/
  patches/
  outputs/
    checkpoints/            # finetune mode only; absent in frozen mode
      ijepa_last.pt
      ijepa_ema_encoder.pt
    embeddings/
      ijepa_per_patch_embeddings.npz
      ijepa_specimen_embeddings.npz
    tables/
      patch_manifest.csv
      ijepa_specimen_coordinates.csv
      ijepa_neighbor_comparison.csv
      training_metrics.csv
    figures/
      ijepa_pca_shells.png
      ijepa_umap_shells.png
      nearest_neighbors/
```

## Test Plan

- Config/CLI checks:
  - `python ijepa_texture_embeddings/ijepa_pipeline.py --help`
  - `python ijepa_texture_embeddings/ijepa_pipeline.py write-config --output ijepa_texture_embeddings/ijepa_config.yaml`
- Small smoke run:
  - Two specimens, `patches_per_specimen: 8`, small `render_image_size`, `mode: frozen`.
  - Verify above renders, sampled patch PNGs, manifest rows, embeddings, PCA/UMAP outputs.
  - Repeat once with `mode: finetune`, `epochs: 1` to exercise the training path and checkpoint write.
- Rendering validation:
  - Assert the above render content mask is non-empty.
  - Assert two runs with the same `render_seed` produce identical renders.
- Patch-sampling validation:
  - Assert every saved patch has **zero** background pixels when `require_full_content: true` (no white in view).
  - Assert no two patches from a specimen are identical (dedup), while overlap is permitted.
  - Assert patch count per specimen ≤ `patches_per_specimen`, and warn if a specimen yields far fewer (surface too small / patch too large).
  - Assert two runs with the same `patch_sample_seed` produce byte-identical `patch_manifest.csv`.
- Frozen-mode validation:
  - Confirm `pretrained_checkpoint` loads and produces finite, non-constant embeddings (std of normalized embeddings above a floor).
- Fine-tune validation:
  - Fail loudly if `torch.cuda.is_available()` is false; confirm logs say `cuda`.
  - Confirm loss is finite **and** collapse metrics stay healthy (embedding std and effective rank do not trend to zero; holdout KNN-probe does not degrade).
  - Confirm holdout specimens were excluded from training.
- Full acceptance:
  - Full run renders one above view per paired clam and samples patches for all 32.
  - Exports one final embedding per specimen.
  - Produces PCA, UMAP, nearest-neighbor sheets, and the DINOv2 comparison table; reports the external-validity metric when labels are available.

## Assumptions

- Each specimen contributes one high-res above render; the training/embedding unit is a **non-white surface patch** sampled from that render (overlap allowed, duplicates rejected), with per-patch jitter.
- "Non-white / no full white pixels" is defined via the existing render content mask (alpha when `transparent: true`, else `opaque_content_mask` with tolerance), not a naive RGB==255 test.
- The encoder is always pretrained; the experiment adds a JEPA component (frozen extraction first, optional cross-patch fine-tuning second) rather than reproducing Meta I-JEPA's from-scratch training, which is infeasible at n=32.
- Rendered shell views are the input data; UV texture maps are not used directly (patches come from the rendered surface, which already carries the texture).
- Existing ROCm PyTorch setup should be preserved: PyTorch exposes AMD GPUs through `cuda`.
