# Handoff — how to pick this up

## Environment
```
python3 -m venv --system-site-packages .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/pip install --no-deps facenet-pytorch      # only for scripts/15 and 55
make test                                             # 40 tests, ~4 s
```
Apple-silicon: MPS is used automatically; a CPU-only box works (feature extraction ~10 min).

## Data
Nothing is downloaded up-front. `scripts/01–03` stream from DANDI (~10 min total) and leave
`data/images/*.png`, `data/neural/*.npz` and the committed manifests. Everything under
`data/manifests/` and `results/` is committed and sufficient to re-run all analyses ≥ 30
without touching DANDI (`data/features/*.npz` and `data/neural/*.npz` are needed for 40+ and
are rebuilt by 10/03).

## Long-running stages
| script | wall time (M2, 8 GB) | notes |
|---|---|---|
| 10 features | ~8 min | downloads ~1.7 GB of weights on first run |
| 41 encoding sweep | ~40 min | 6-fold ridge × 70 predictors × 907 units |
| 51 encoding null (core) | ~90 min | 100 shuffles × 6 predictors × 163 concept cells |
| 51 --set depth | ~60 min | 40 shuffles × 12 predictors |
| 40 RSA sweep | ~3 min | |
| 58 group permutation | ~10 min | |
Everything else runs in seconds to a couple of minutes.

## Where the conclusions live
`docs/results.md` (numbers, chronological, with amendments), `README.md` (summary),
`results/report.html` (`scripts/90_build_report.py --embed` for a self-contained page),
`docs/figure_index.md` (what produces what).

## Known caveats to carry forward
- Neural RDM reliability is low (~0.06); ceiling-normalise or pool across patients.
- CV encoding r is negatively biased; quote the shuffle-null-debiased values (A4_encoding_null*).
- Category labels: CLIP + 40 hand overrides; `category_clip` keeps the raw decision.
- Hemisphere effects are partly confounded with patient.
- Sternberg concept-cell selection uses enc1–3 in most scripts (25 shows enc1-only ≈ paper).

## Ideas not yet done
time-resolved encoding; a hierarchical model of RSA ρ with layer × region; a face-trained
CNN with more capacity than InceptionResnetV1; hand labels for "famous person yes/no";
recomputing everything with a 100–1000 ms window robustness sweep.
