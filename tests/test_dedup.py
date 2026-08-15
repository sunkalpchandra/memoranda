import pandas as pd

from conceptlens.dedup import assign_uids, unique_images


def _row(sub, ses, idx, sha, dh, null=False):
    return dict(subject=sub, session=ses, stim_index=idx, sha1=sha, dhash=dh, is_null=null)


def test_exact_and_near_grouping():
    z = "0" * 64
    near = "0" * 60 + "1111"  # hamming 4
    far = "1" * 64
    df = pd.DataFrame(
        [
            _row(1, 1, 0, "a", z),
            _row(1, 1, 1, "b", far),
            _row(1, 2, 0, "a", z),  # exact dup of first
            _row(2, 1, 0, "c", near),  # near dup of first
            _row(2, 1, 1, "d", "1" * 32 + "0" * 32),
            _row(2, 2, 5, "n", "0" * 64, null=True),
        ]
    )
    out = assign_uids(df)
    uid = dict(zip(out["sha1"], out["image_uid"]))
    assert uid["a"] == uid["c"]
    assert uid["a"] != uid["b"]
    assert uid["n"] == "img_null"
    kinds = dict(zip(out["sha1"], out["dup_kind"]))
    assert kinds["c"] == "near"
    assert out.loc[(out.subject == 1) & (out.session == 2), "dup_kind"].item() == "exact"
    u = unique_images(out)
    assert len(u) == 3
