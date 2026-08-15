# Decisions & data quirks

## D1 — image orientation
NWB templates are (400, 300, 3); the upright picture is `np.rot90(arr, k=-1)`
(300×400 landscape). Verified visually (faces upright, text readable).

## D2 — StimulusPresentation indexing differs by task
- Sternberg files (`ses-2`): `data` is a positional index into `order_of_images` (0–5, 5 = null image_999).
- Screening files (`ses-1`): `data` is `image_number − 1` (values 0–68 while only 54–63
  templates exist; numbers 10, 20, 30, … are unused). `nwb.presentation_index_mode`
  detects this automatically.

## D3 — image identity across sessions
Only 342 unique pictures exist across all 41 files (sha1 exact match; dhash ≤ 6 bits
catches 3 near-duplicates). One picture is shared by up to 16 subjects. `image_uid`
is the dataset-wide key; every Sternberg image of subjects with a screening session
maps back to that subject's own screening set (100/100; sub-19 had no screening).

## D4 — category labels
CLIP zero-shot (`configs/taxonomy.yaml`) is the first pass; every per-category contact
sheet (`results/figures/labels_<category>.png`) was reviewed by eye and 40 pictures were
re-assigned in `configs/label_overrides.csv` (e.g. Yoda/Spider-Man/Dory → cartoon_art, a
macaque and a horse's eye → animal, ships and spacecraft → vehicle). `image_labels.csv`
keeps CLIP's decision in `category_clip`; `category` is the corrected label used everywhere.
Re-running the category-dependent analyses did not change any conclusion.

## D5 — faces
"Face present" is taken from MTCNN detections (`data/manifests/faces.csv`, p ≥ 0.9), not
from the CLIP attribute; 41 % of pictures contain a face. The VGGFace2 InceptionResnetV1
embedding of the largest face (whole picture if none) is stored as model `facenet_vggface2`.

## D6 — encoding-model null
6-fold CV Pearson r is negatively biased for signal-free targets with ~60 samples; script 51
computes a per-cell label-shuffle null and reports debiased r. Non-selective cells sit at
−0.05…−0.16 raw.
