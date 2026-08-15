# Memoranda — project plan

Status legend: `[ ]` todo · `[~]` in progress · `[x]` done

## Phase 0 — Scaffold
- [x] repo, venv, .gitignore, README
- [x] pyproject + package skeleton
- [x] configs directory, logging helper
- [x] pytest scaffold + CI-free local test runner
- [x] GitHub remote

## Phase 1 — Data access (DANDI 000469, streamed)
- [x] `memoranda/dandi.py`: list assets (41 files), map to subject/session/task, cache JSON manifest
- [x] `memoranda/nwb.py`: open remote NWB via remfile+h5py; helpers for stimulus templates,
      presentation order, trials table, units (spike times, electrode location), subject metadata
- [x] `scripts/01_extract_images.py`: dump every image (screening + Sternberg) to
      `data/images/sub-XX/ses-Y/<name>.png` + `data/manifests/images.csv`
- [x] orientation check (NWB stores 400×300 — MATLAB transpose?) and fix
- [x] perceptual-hash dedup across subjects → `image_uid`, `data/manifests/image_dedup.csv`
- [x] `scripts/02_extract_trials.py`: trial tables for both tasks → parquet/csv
- [x] `scripts/03_extract_units.py`: spike times + waveform metrics + brain area per unit
- [x] dataset summary doc (n images/subject, categories, overlap between subjects)

## Phase 2 — Neural characterization (replicate Kyzar / Kamiński)
- [x] per-image firing rate in 200–1000 ms after onset (screening: 6 reps; Sternberg encoding)
- [x] concept-cell test: permuted 1-way ANOVA + post-hoc max-vs-rest permutation t-test
- [x] depth-of-selectivity index, picture selectivity index
- [x] per-image "neural response profile": #selective cells, max z-score, area breakdown
- [x] population RDM per screening session (units × images → images × images)
- [x] verify: MTL concept-cell % ≈ 32.8 (screening), MFC ≈ 5.4 (Kyzar Table / text)
- [x] which screening images became Sternberg images (rank of selectivity)

## Phase 3 — Deep vision model features
- [x] `memoranda/models/registry.py`: uniform loader → (model, preprocess, layer hooks)
- [x] CNNs: AlexNet, VGG16, ResNet-18/50, ConvNeXt-T (torchvision, ImageNet weights)
- [x] Transformers: ViT-B/16, DINOv2-S/B, CLIP ViT-B/32 (open_clip)
- [x] layer-wise activations (early/mid/late/penultimate) → pooled features cached as .npy
- [x] CLIP zero-shot labeling: category taxonomy (person/face, animal, place/landscape,
      object, food, vehicle, text/logo, scene w/ people, cartoon)
- [x] CLIP zero-shot attributes: famous person?, indoor/outdoor, natural/man-made, #faces
- [x] low-level image statistics: luminance, RMS contrast, colorfulness, entropy,
      spatial-frequency slope, edge density, saliency proxy
- [x] ImageNet top-5 labels per image (script 14)
- [x] `scripts/10_extract_features.py` (all models, all images, resumable)

## Phase 4 — Analyses
- [x] A1 stimulus set characterization: category counts, per-subject composition, dedup stats
- [x] A2 Sternberg-5 vs rest: category enrichment (faces?), DNN typicality/outlierness,
      distance-to-centroid, nearest-neighbour distance, CLIP semantic distinctiveness
- [x] A3 RSA: neural RDM (per session, per area) vs DNN layer RDMs; layer-depth curve;
      noise ceiling via split-half; MTL vs MFC contrast
- [x] A4 encoding models: ridge from DNN features → per-unit responses, CV R²; which
      layer/model best predicts amygdala vs hippocampus concept cells
- [x] A5 concept-cell preferences, per-image neural score, animacy × area, image drive (scripts 31–34)
- [x] A6 behaviour: semantic/perceptual similarity among the 5 memoranda vs RT/accuracy on
      IN/OUT probes; per-trial probe–memoranda max-similarity as regressor
- [x] A7 model comparison table: which model family aligns best with human MTL selectivity
- [x] statistics: permutation tests, FDR, bootstrap CIs; all seeds fixed

## Phase 5 — Reporting
- [x] figures under `results/figures/` (one script per figure)
- [x] `docs/methods.md`, `docs/results.md`, `docs/decisions.md`
- [x] HTML report / artifact dashboard (script 90)
- [x] README results section (living)

## Working rules
- small commits, one logical change each; push often
- every script resumable and idempotent; caches keyed by (model, layer, image_uid)
- nothing >5 MB committed; raw images and features stay in `data/` (gitignored)
- reproducibility: `configs/*.yaml` + fixed seeds; `make all` runs the whole pipeline

## Phase 6 — extensions (added 2026-08-15)
- [x] A0 maintenance persistent-activity replication (script 21)
- [x] within/between-category RSA (53), commonality analysis (52), time-resolved RSA (46)
- [x] similarity tuning per cell (47), by area/depth (50), time-resolved (54)
- [x] encoding label-shuffle null (51)
- [x] face-trained network (VGGFace2 InceptionResnetV1) + MTCNN faces (15, 55)
- [ ] time-resolved encoding models
- [x] hand-checked category labels for the 342 pictures (16)
- [x] Sternberg probe-period (22) and memoranda geometry (24)
- [x] CI green (ruff + pytest on committed CSVs)
- [x] group-level permutation (58), category decoding (23), cross-patient tuning (36), LOSO preference (37), rp robustness (56), reliability (57)
- [x] time-resolved encoding models (68)
- [x] hierarchical (mixed-effects) versions of the group tests (59, 67)
- [~] noise-corrected encoding across layers with the shuffle null (51 core/depth done; wide set running)
- [x] hand "famous" labels for all 342 pictures (66)
- [x] model scale contrast (69) and same-architecture objective contrast (63)
- [x] project website (site/, GitHub Pages) + on-site full report

## Phase 8 — review and second experiment round (2026-08-15)
- [x] adversarial statistical review (docs/review_2026-08-15.md) and response
- [x] selection-corrected post-hoc test; strict criterion columns; strict variants of A5, RSA
- [x] patient-level / cluster-robust tests (73); family-wide FDR; wording revised everywhere
- [x] second-rater famous labels (κ 0.93); identity invariance (untestable); semantic vs visual (71); within-category tuning (72); reweighted RSA (74)
- [~] shuffle null over 24 more layers (51 --set wide) — running
- [ ] a mixed model with crossed cell/model random effects for the layer sweeps
- [ ] more bilateral patients would be needed to de-confound laterality (dataset limit)
