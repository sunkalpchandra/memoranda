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
