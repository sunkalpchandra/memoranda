# Methods

## Data
DANDI 000469 (Kyzar et al. 2024), 41 NWB files, streamed over HTTP with `remfile` + `h5py`
(`memoranda/nwb.py`). We read: `stimulus/templates` (images), `stimulus/presentation`
(onsets), `intervals/trials`, `units` (spike times, quality metrics), `general/…/electrodes`.
Templates are stored rotated; `images.fix_orientation` restores 300×400 landscape.
Screening presentation indices are `image_number − 1`; Sternberg indices are positional
(`nwb.presentation_index_mode`).

Every template gets a dataset-wide `image_uid` (`memoranda/dedup.py`: sha1 exact match,
then dhash ≤ 6/64 bits union–find). 342 unique images.

## Neural responses (`memoranda/neural.py`, script 20)
Firing rate in 200–1000 ms after onset (paper window). Concept cell = permuted one-way
ANOVA over image identity (1000 label permutations, p < 0.05) **and** permutation Welch-t of
the max-response image vs all others (p < 0.05); units with < 50 spikes excluded.
Screening: 54–63 images × 6 repetitions. Sternberg: 5 images, all encoding presentations
(enc1–3), probes excluded. Extras per unit: depth of selectivity (Rainer/Miller), Treves–Rolls
sparseness, preferred/non-preferred rate, baseline rate (−200–0 ms).

## Vision models (`memoranda/models/`, script 10)
AlexNet, VGG-16, ResNet-18/50, ConvNeXt-T, ViT-B/16 (torchvision ImageNet-1k weights),
DINOv2-S/14 (timm, self-supervised), CLIP ViT-B/32 (open_clip, OpenAI weights). Forward
hooks on 5–8 layers per model; per layer we keep `gap` (spatial/token mean), `cls`
(first token, transformers) and `rp` (fixed sparse random projection of the spatially
capped activation to 2048 dims). Features stored float16 in `data/features/<model>.npz`.

CLIP zero-shot (`models/zeroshot.py`, `configs/taxonomy.yaml`): 9 mutually exclusive
categories (prompt-ensembled), 10 binary attribute contrasts (probability of the first
prompt), 56 fine labels.

Low-level statistics (`memoranda/imstats.py`): luminance, RMS & Michelson contrast,
colourfulness (Hasler–Süsstrunk), saturation, hue & grey entropy, edge density,
spectral slope, high-frequency energy, edge centre of mass.

## Analyses
* **A1/A2 (script 30)** composition; memoranda vs rest of each subject's screening set:
  Fisher exact for categories, Mann–Whitney AUC (pooled and mean within-subject) for
  continuous features, BH-FDR within block. DNN geometry (`analysis/geometry.py`):
  cosine distance to centroid, nearest-neighbour distance, leave-one-out category
  typicality — dataset-wide and within subject.
* **A5 (scripts 31–33)** preferred-image category of concept cells vs the composition
  actually shown to those cells (Fisher, FDR); attribute AUCs; per-image concept-cell
  rate; animacy × area; right/left amygdala animal preference (Fisher).
* **A3 (scripts 40, 43–46)** RSA (`analysis/rsa.py`): neural RDM = correlation distance
  between images over z-scored unit responses; model RDM = correlation distance over
  features; Spearman ρ on upper triangles. Sessions as random effects (t-test on
  Fisher-z). Split-half reliability (10 random 3/3 splits, Spearman–Brown) as ceiling.
  Partial Spearman controlling category/low-level RDMs; rank RDM regression. Pooled RDM:
  per-session rank-normalised RDMs averaged over co-shown pairs; permutation of image
  identity for p. Time-resolved: 200-ms windows, 25-ms step.
* **A4 (script 41)** encoding models (`analysis/encoding.py`): 6-fold CV ridge, in-fold
  z-scoring + PCA(20), penalty by LOO-GCV; Pearson r between held-out predictions and
  per-image mean rate; ceiling = split-half tuning reliability. Baselines: category
  one-hot, low-level statistics.
* **A6 (script 42)** behaviour (`analysis/behavior.py`): per-trial cosine similarity between
  probe and encoded set (max), and within-set mean; per-subject OLS slopes with load as
  covariate → group t-test / Wilcoxon.

## Statistics conventions
Two-sided unless stated; permutation tests report (k+1)/(n+1); FDR = Benjamini–Hochberg
within each family; effect sizes as AUC (0.5 = null) or odds ratios; sessions/subjects are
the unit of inference for group tests.
