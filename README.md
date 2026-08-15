# memoranda

**Project page:** https://sunkalpchandra.github.io/memoranda/ · full report on the same site.

*memoranda* (n., pl.) — the things to be remembered. Here: the pictures that
epilepsy patients held in working memory while single neurons were recorded.

**What is it about these pictures that makes human concept cells fire?**

This project takes the images that were shown to epilepsy patients in the
Sternberg working-memory dataset of Kyzar et al. (2024, *Scientific Data*;
DANDI 000469) and asks a simple question: *what is it about these pictures
that makes single neurons in the human medial temporal lobe fire?*

The dataset provides 1809 single neurons (amygdala, hippocampus, dACC,
pre-SMA, vmPFC) recorded while 21 patients viewed 54–64 images in a screening
task and then held the 5 most "neuron-selective" of those images in working
memory (Kamiński et al. 2017, *Nature Neuroscience*). Every image is stored in
the NWB files. We pull them out, push them through a zoo of convolutional and
transformer vision models (AlexNet → ResNet → ConvNeXt, ViT, CLIP, DINOv2),
and relate the resulting representations back to the neural selectivity that
the original authors measured.

## Questions

1. **What kind of images were shown?** Category, semantic content, low-level
   statistics (luminance, contrast, spatial frequency, colorfulness) — for the
   whole screening set and for the 5-image Sternberg subsets.
2. **Are the "winning" Sternberg images special?** Do the images that drove
   concept cells sit somewhere particular in DNN feature space (e.g. faces,
   prototypical exemplars, high-salience, semantically distinct)?
3. **Which model layers look like the MTL?** Representational similarity
   analysis between neural population RDMs (per session, over the screening
   images) and layer-wise DNN RDMs.
4. **Can DNN features predict single-neuron responses?** Cross-validated
   encoding models from DNN embeddings to per-image firing rates.
5. **Does image similarity predict working-memory behaviour?** Semantic /
   perceptual confusability of the 5 memoranda vs. reaction time and accuracy
   on probe trials.

## Layout

```
memoranda/        python package (data access, models, analysis)
scripts/            CLI entry points, one per pipeline stage
configs/            YAML configs for models and analyses
tests/              pytest
data/manifests/     small, committed CSV/JSON manifests
results/            figures + tables (committed)
docs/               notes, methods, decisions
```

## Findings so far

Full numbers in [docs/results.md](docs/results.md) (revised after an adversarial review,
[docs/review_2026-08-15.md](docs/review_2026-08-15.md)); methods in [docs/methods.md](docs/methods.md).

**Replication.** All 1809 units, behaviour (88.7 % correct) and concept-cell fractions
(MTL 31.8 %, MFC 6.0 %) reproduce Kyzar et al.; MTL concept cells stay active during
WM maintenance when their image is held (p = 7e-9), MFC cells do not. The paper-style post-hoc
test is permissive (MFC prevalence equals the false-positive rate); a selection-corrected strict
criterion gives MTL 15.9 % / MFC 2.1 %, and the single-cell results below hold for both.

**The stimuli.** Only 342 unique pictures underlie the 41 files (37 % faces / people — nearly
all famous, 14 % animals, 11 % landmarks, 10 % vehicles, the rest objects, scenes, nature, food).
No picture-level property predicts a patient's five Sternberg memoranda (52 contrasts, family-wide
q ≥ 0.29); a detected face is the only nominal hit (62 % vs 49 %) and is expected from the concept
cells' face bias. Other patients' preferences carry no information (LOSO AUC 0.44–0.49): being a
concept-cell image is a *patient × image* property.

**What concept cells like.** Strict amygdala concept cells prefer pictures with a face
(patient-level p = 0.03); per shown image, animals (0.19 cells/patient) > faces (0.14) > vehicles >
places > food > objects (0.01). Right amygdala concept cells prefer animals 4× more often than left
(21 % vs 5 %) — Mormann et al. 2011 rediscovered — and generalise along DNN similarity far more than
left ones (patient-level p = 0.008; hemisphere partly confounded with patient).

**Neural ↔ DNN geometry (RSA).** MTL population RDMs correlate with every model's late layers
(ρ ≈ 0.06 per session, 0.13 pooled across patients; ≈ 0.24 of the ceiling), rising from early to
mid/late layers (monotonic for ResNet-50 and CLIP, plateauing for AlexNet/VGG/DINOv2); MFC is weak
and without a depth gradient (MTL > MFC decisive as a fixed effect, marginal across patients);
amygdala ≫ hippocampus. Late layers and the category RDM are largely redundant, with a small unique
late-layer component; ViT-family late layers keep a within-category correspondence. The
correspondence emerges ~200 ms after onset and peaks at 350 ms.

**Single neurons generalise along DNN similarity.** A concept cell's response to the other
54–62 images correlates with their late-layer similarity to its preferred image (cell-pooled
ρ ≈ 0.17, patient-mean 0.10, p = 0.016 over patients), rising with depth, emerging at ~225 ms,
surviving within category (0.11) and beyond text-space similarity. Architecture, training objective
and scale barely matter — a CLIP-trained ResNet-50 and the ImageNet ResNet-50 are indistinguishable
layer by layer, and CLIP-L/14 / DINOv2-B are no better than their small versions.

| ![RSA layer curves](results/figures/A3_rsa_layer_curves.png) |
|:--:|
| *MTL–model similarity climbs with layer depth in every architecture; MFC stays flat.* |

| ![time-resolved](results/figures/A3_time_resolved.png) | ![similarity tuning](results/figures/A4_similarity_tuning.png) |
|:--:|:--:|
| *Sliding-window RSA* | *Per-neuron generalisation along DNN similarity* |

## Data

DANDI 000469 — https://dandiarchive.org/dandiset/000469 (9.8 GB, 41 NWB files).
We do **not** download the archive; NWB groups are streamed over HTTP with
`remfile` + `h5py`, and only images / trials / spike times are cached locally.

## References

- Kyzar M, Kamiński J, Brzezicka A, Reed CM, Chung JM, Mamelak AN, Rutishauser U.
  *Dataset of human single-neuron activity during a Sternberg working memory task.*
  Sci Data 11:89 (2024). https://doi.org/10.1038/s41597-024-02943-8
- Kamiński J, Sullivan S, Chung JM, Ross IB, Mamelak AN, Rutishauser U.
  *Persistently active neurons in human medial frontal and medial temporal lobe
  support working memory.* Nat Neurosci 20:590–601 (2017).
  https://doi.org/10.1038/nn.4509
