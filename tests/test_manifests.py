"""Sanity checks on committed manifests against numbers published in Kyzar et al. 2024."""

import pandas as pd
import pytest

from conceptlens.paths import MANIFESTS

units_csv = MANIFESTS / "units.csv"
images_csv = MANIFESTS / "images.csv"


@pytest.mark.skipif(not units_csv.exists(), reason="run scripts/03 first")
def test_unit_counts_match_paper():
    u = pd.read_csv(units_csv)
    assert len(u) == 1809
    sb = u[u.task == "sternberg"].groupby("area").size()
    assert sb["hippocampus"] == 190
    assert sb["amygdala"] == 259
    assert sb["dACC"] == 171
    assert sb["preSMA"] == 250
    assert sb["vmPFC"] == 32
    sc = u[u.task == "screening"]
    assert len(sc) == 907
    assert len(u[u.task == "sternberg"]) == 902


@pytest.mark.skipif(not images_csv.exists(), reason="run scripts/01 first")
def test_image_manifest_shape():
    d = pd.read_csv(images_csv)
    assert d.subject.nunique() == 21
    sb = d[(d.task == "sternberg") & (~d.is_null)]
    assert (sb.groupby("subject").size() == 5).all()
    sc = d[d.task == "screening"]
    assert sc.groupby("subject").size().between(54, 64).all()
    assert d.image_uid.nunique() >= 300
