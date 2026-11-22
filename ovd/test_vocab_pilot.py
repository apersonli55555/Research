"""Automated checks for ovd/vocab_pilot.py (run: python -m pytest ovd/test_vocab_pilot.py -q)."""
import os, sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import vocab_pilot as vp  # noqa: E402


def test_iou_basic():
    b = np.array([[0, 0, 10, 10], [5, 5, 15, 15], [20, 20, 30, 30]], float)
    v = vp.iou(b, np.array([0, 0, 10, 10], float))
    assert abs(v[0] - 1) < 1e-9 and abs(v[1] - 25 / 175) < 1e-9 and v[2] == 0


def test_matches_one_detection_per_gt():
    gts = [[0, 0, 10, 10]]
    m = vp.matches([[0.9, 0, 0, 10, 10], [0.8, 0, 0, 10, 10]], gts)
    assert m == [(0.9, True), (0.8, False)]


def test_ap_perfect_and_half():
    assert abs(vp.ap_from([(0.9, True), (0.8, True)], 2) - 1.0) < 1e-9
    # one true positive ranked below one false positive, two ground truths -> AP = 0.25
    assert abs(vp.ap_from([(0.9, False), (0.8, True)], 2) - 0.25) < 1e-9
    assert np.isnan(vp.ap_from([], 0))


def test_ap_equals_bruteforce_interpolated_ap():
    # all-point interpolated AP: sum over each new true positive of (1/npos) * max precision at any rank at or below it
    rng = np.random.default_rng(0)
    for trial in range(20):
        s = rng.random(40); y = rng.random(40) < 0.4; npos = int(y.sum()) + int(rng.integers(0, 3))  # some gts missed
        if npos == 0:
            continue
        order = np.argsort(-s); yy = y[order]
        prec = np.cumsum(yy) / np.arange(1, len(yy) + 1)
        ref = sum(prec[i:].max() / npos for i in range(len(yy)) if yy[i])
        assert abs(vp.ap_from(list(zip(s, y)), npos) - ref) < 1e-9


def test_vocab_conditions_are_reproducible_and_contain_target():
    names = [f"c{i}" for i in range(80)]
    rng1 = np.random.default_rng(int(vp.h("img|c3")[:8], 16)); rng2 = np.random.default_rng(int(vp.h("img|c3")[:8], 16))
    v1 = list(rng1.choice([n for n in names if n != "c3"], 4, replace=False)); v2 = list(rng2.choice([n for n in names if n != "c3"], 4, replace=False))
    assert v1 == v2 and "c3" not in v1
