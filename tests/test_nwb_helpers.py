from conceptlens import nwb


def test_area_mapping():
    assert nwb.area_from_location("amygdala_right") == "amygdala"
    assert nwb.area_from_location("hippocampus_left") == "hippocampus"
    assert nwb.area_from_location("dorsal_anterior_cingulate_cortex_left") == "dACC"
    assert nwb.area_from_location("pre_supplementary_motor_area_right") == "preSMA"
    assert nwb.area_from_location("ventral_medial_prefrontal_cortex_right") == "vmPFC"
    assert nwb.area_from_location("something_else") == "other"


def test_hemisphere():
    assert nwb.hemisphere_from_location("amygdala_right") == "R"
    assert nwb.hemisphere_from_location("amygdala_left") == "L"


def test_region():
    assert nwb.region_of("amygdala") == "MTL"
    assert nwb.region_of("hippocampus") == "MTL"
    assert nwb.region_of("dACC") == "MFC"
    assert nwb.region_of("preSMA") == "MFC"
    assert nwb.region_of("vmPFC") == "MFC"
