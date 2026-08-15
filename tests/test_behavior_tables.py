"""Integration checks that need the committed manifests (skipped if absent)."""

import numpy as np
import pandas as pd
import pytest

from memoranda.paths import MANIFESTS

needs = [MANIFESTS / n for n in ("trials_sternberg.csv", "images.csv", "unit_selectivity.csv", "image_labels.csv", "image_stats.csv")]
skip = pytest.mark.skipif(not all(p.exists() for p in needs), reason="manifests not built")


@skip
def test_trial_uid_mapping_consistent_with_probe_flag():
    from memoranda.analysis.behavior import sternberg_trials_with_uids

    t = sternberg_trials_with_uids()
    assert t.enc1.notna().all() and t.probe.notna().all()
    inset = np.array([r.probe in (r.enc1, r.enc2, r.enc3) for r in t.itertuples()])
    assert (inset == t.probe_in.to_numpy().astype(bool)).all()
    # load equals number of encoded images
    n_enc = t[["enc1", "enc2", "enc3"]].notna().sum(1)
    assert (n_enc == t.loads).all()


@skip
def test_subject_image_table_shape():
    from memoranda.analysis.tables import subject_image_table

    st = subject_image_table()
    assert st.in_sternberg.sum() == 105
    assert (st.groupby("subject").in_sternberg.sum() == 5).all()
    # LOSO reuse never counts the subject itself
    assert (st.n_sternberg_other <= st.n_subjects_sternberg).all()


@skip
def test_selectivity_replication_bounds():
    sel = pd.read_csv(MANIFESTS / "unit_selectivity.csv")
    scr = sel[sel.task == "screening"]
    frac_mtl = scr[scr.region == "MTL"].concept_cell.mean()
    frac_mfc = scr[scr.region == "MFC"].concept_cell.mean()
    assert 0.25 < frac_mtl < 0.40  # paper: 32.8 %
    assert frac_mfc < 0.10  # paper: 5.4 %
