# ConceptLens

**Characterizing the visual stimuli behind human concept cells with deep vision models.**

ConceptLens takes the images that were shown to epilepsy patients in the
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
conceptlens/        python package (data access, models, analysis)
scripts/            CLI entry points, one per pipeline stage
configs/            YAML configs for models and analyses
tests/              pytest
data/manifests/     small, committed CSV/JSON manifests
results/            figures + tables (committed)
docs/               notes, methods, decisions
```

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
