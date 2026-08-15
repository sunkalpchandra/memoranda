#!/usr/bin/env bash
# Reproduce the whole pipeline from a clean checkout (needs network for DANDI + weights).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
$PY scripts/01_extract_images.py
$PY scripts/02_extract_trials.py
$PY scripts/03_extract_units.py
$PY scripts/04_montage.py
$PY scripts/10_extract_features.py
$PY scripts/11_image_stats.py
$PY scripts/12_zeroshot_labels.py
$PY scripts/13_label_sheet.py
$PY scripts/14_imagenet_labels.py
$PY scripts/15_faces.py
$PY scripts/16_apply_label_overrides.py
$PY scripts/20_neural_selectivity.py
$PY scripts/21_maintenance_replication.py
$PY scripts/30_stimset_analysis.py
$PY scripts/31_concept_cell_preferences.py
$PY scripts/32_top_images.py
$PY scripts/33_animacy.py
$PY scripts/40_rsa_sweep.py
$PY scripts/41_encoding_sweep.py
$PY scripts/42_behavior.py
$PY scripts/43_rsa_figures.py
$PY scripts/44_rsa_partial.py
$PY scripts/45_pooled_rdm.py
$PY scripts/46_time_resolved_rsa.py
$PY scripts/47_similarity_tuning.py
$PY scripts/48_encoding_figures.py
$PY scripts/49_model_comparison.py
$PY scripts/50_area_layer_profiles.py
$PY scripts/51_encoding_null.py
$PY scripts/52_variance_partition.py
$PY scripts/53_within_category_rsa.py
$PY scripts/54_time_resolved_similarity.py
$PY scripts/55_face_space.py
$PY scripts/22_probe_responses.py
$PY scripts/35_memoranda_gallery.py
$PY scripts/90_build_report.py
