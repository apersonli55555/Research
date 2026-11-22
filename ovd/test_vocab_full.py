"""Checks for ovd/vocab_full.py and ovd/full_analysis.py (python -m pytest ovd -q)."""
import os, sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import full_analysis as fa  # noqa: E402
import vocab_full as vf  # noqa: E402
import vocab_pilot as vp  # noqa: E402


def test_clean_lvis_names_unique():
    raw = ["bow_(weapon)", "bow_(decorative_ribbons)", "aerosol_can", "bat_(animal)"]
    out = vf.clean_lvis(raw)
    assert out == ["bow (weapon)", "bow (decorative ribbons)", "aerosol can", "bat"] and len(set(out)) == len(out)


def test_nms_keeps_highest_and_distinct():
    b = np.array([[0, 0, 10, 10], [0, 0, 10, 11], [50, 50, 60, 60]], float)
    keep = vf.nms(b, np.array([0.5, 0.9, 0.3]))
    assert list(keep) == [1, 2]


def test_weighted_ap_equals_pilot_ap_and_duplication():
    rng = np.random.default_rng(1)
    s = rng.random(30); y = (rng.random(30) < 0.5).astype(float); npos = int(y.sum()) + 2
    assert abs(fa.wap(s, y, np.ones(30), npos) - vp.ap_from(list(zip(s, y.astype(bool))), npos)) < 1e-9
    w = rng.integers(0, 3, 30).astype(float)                       # weight k == the record repeated k times
    s2 = np.repeat(s, w.astype(int)); y2 = np.repeat(y, w.astype(int)); n2 = npos * 1.0
    assert abs(fa.wap(s, y, w, n2) - fa.wap(s2, y2, np.ones(len(s2)), n2)) < 1e-9


def test_group_names():
    assert [fa.group_of(c) for c in ("k1", "rand20_3", "near5", "vis80", "far20", "all_4", "FIX")] == \
        ["k1", "rand20", "near5", "vis80", "far20", "all", "FIX"]


def test_target_record_flags_label_competition():
    boxes = np.array([[0, 0, 10, 10], [20, 20, 30, 30]], float)
    S = np.array([[0.6, 0.9], [0.5, 0.1]])                          # box 0: target 0 loses to class 1
    r = vf.target_record(boxes, S, 0, [[0, 0, 10, 10]], 0.35, False)
    assert r["dets"][0][5] == 0 and r["dets"][1][5] == 1
    assert r["instances"][0]["raw"] == 0.6 and r["instances"][0]["deployed"] == 0.0


def test_gdino_chunks_fit_token_limit():
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("bert-base-uncased")
    except Exception:                                               # offline: skip
        return
    g = vf.GDino.__new__(vf.GDino); g.tok = tok; g.ntok = {}
    names = [f"word{i} thing" for i in range(400)]
    for ch in g.chunks(names):
        assert len(tok(" . ".join(ch) + " .")["input_ids"]) <= 256
    assert sum(len(c) for c in g.chunks(names)) == 400
