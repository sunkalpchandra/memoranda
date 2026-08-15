# ConceptLens — project plan

Status legend: `[ ]` todo · `[~]` in progress · `[x]` done

## Phase 0 — Scaffold
- [x] repo, venv, .gitignore, README
- [ ] pyproject + package skeleton
- [ ] configs directory, logging helper
- [ ] pytest scaffold + CI-free local test runner
- [ ] GitHub remote

## Phase 1 — Data access (DANDI 000469, streamed)
- [ ] `conceptlens/dandi.py`: list assets (41 files), map to subject/session/task, cache JSON manifest
- [ ] `conceptlens/nwb.py`: open remote NWB via remfile+h5py; helpers for stimulus templates,
      presentation order, trials table, units (spike times, electrode location), subject metadata
- [ ] `scripts/01_extract_images.py`: dump every image (screening + Sternberg) to
      `data/images/sub-XX/ses-Y/<name>.png` + `data/manifests/images.csv`
- [ ] orientation check (NWB stores 400×300 — MATLAB transpose?) and fix
- [ ] perceptual-hash dedup across subjects → `image_uid`, `data/manifests/image_dedup.csv`
- [ ] `scripts/02_extract_trials.py`: trial tables for both tasks → parquet/csv
- [ ] `scripts/03_extract_units.py`: spike times + waveform metrics + brain area per unit
- [ ] dataset summary doc (n images/subject, categories, overlap between subjects)

## Phase 2 — Neural characterization (replicate Kyzar / Kamiński)
- [ ] per-image firing rate in 200–1000 ms after onset (screening: 6 reps; Sternberg encoding)
- [ ] concept-cell test: permuted 1-way ANOVA + post-hoc max-vs-rest permutation t-test
- [ ] depth-of-selectivity index, picture selectivity index
- [ ] per-image "neural response profile": #selective cells, max z-score, area breakdown
- [ ] population RDM per screening session (units × images → images × images)
- [ ] verify: MTL concept-cell % ≈ 32.8 (screening), MFC ≈ 5.4 (Kyzar Table / text)
- [ ] which screening images became Sternberg images (rank of selectivity)

## Phase 3 — Deep vision model features
- [ ] `conceptlens/models/registry.py`: uniform loader → (model, preprocess, layer hooks)
- [ ] CNNs: AlexNet, VGG16, ResNet-18/50, ConvNeXt-T (torchvision, ImageNet weights)
- [ ] Transformers: ViT-B/16, DINOv2-S/B, CLIP ViT-B/32 (open_clip)
- [ ] layer-wise activations (early/mid/late/penultimate) → pooled features cached as .npy
- [ ] CLIP zero-shot labeling: category taxonomy (person/face, animal, place/landscape,
      object, food, vehicle, text/logo, scene w/ people, cartoon)
- [ ] CLIP zero-shot attributes: famous person?, indoor/outdoor, natural/man-made, #faces
- [ ] low-level image statistics: luminance, RMS contrast, colorfulness, entropy,
      spatial-frequency slope, edge density, saliency proxy
- [ ] ImageNet top-5 labels per image (for eyeballing)
- [ ] `scripts/10_extract_features.py` (all models, all images, resumable)

## Phase 4 — Analyses
- [ ] A1 stimulus set characterization: category counts, per-subject composition, dedup stats
- [ ] A2 Sternberg-5 vs rest: category enrichment (faces?), DNN typicality/outlierness,
      distance-to-centroid, nearest-neighbour distance, CLIP semantic distinctiveness
- [ ] A3 RSA: neural RDM (per session, per area) vs DNN layer RDMs; layer-depth curve;
      noise ceiling via split-half; MTL vs MFC contrast
- [ ] A4 encoding models: ridge from DNN features → per-unit responses, CV R²; which
      layer/model best predicts amygdala vs hippocampus concept cells
- [ ] A5 decoding: predict "is concept-cell-preferred image" from DNN features (LOSO CV)
- [ ] A6 behaviour: semantic/perceptual similarity among the 5 memoranda vs RT/accuracy on
      IN/OUT probes; per-trial probe–memoranda max-similarity as regressor
- [ ] A7 model comparison table: which model family aligns best with human MTL selectivity
- [ ] statistics: permutation tests, FDR, bootstrap CIs; all seeds fixed

## Phase 5 — Reporting
- [ ] figures under `results/figures/` (one script per figure)
- [ ] `docs/methods.md`, `docs/results.md`, `docs/decisions.md`
- [ ] HTML report / artifact dashboard
- [ ] final README results section

## Working rules
- small commits, one logical change each; push often
- every script resumable and idempotent; caches keyed by (model, layer, image_uid)
- nothing >5 MB committed; raw images and features stay in `data/` (gitignored)
- reproducibility: `configs/*.yaml` + fixed seeds; `make all` runs the whole pipeline
