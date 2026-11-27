"""Tests for bank/yolo_calib.py helpers. Run: python -m pytest -q ovd/bank/test_yolo_calib.py"""
import os, sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from yolo_calib import hard_nms_rows  # noqa: E402


def test_hard_nms_rows():
    S = torch.tensor([[0.9, 0.8, 0.5, -1.0], [0.4, -1.0, -1.0, -1.0], [-1.0] * 4])
    b = torch.tensor([[0, 0, 10, 10], [1, 1, 10, 10], [50, 50, 60, 60], [0, 0, 0, 0]], dtype=torch.float)
    B = b[None].repeat(3, 1, 1)
    out, _ = hard_nms_rows(S, B)
    assert out[0].tolist() == [0.8999999761581421, -1.0, 0.5, -1.0]       # overlapping 0.8 suppressed, disjoint kept
    assert out[1].tolist() == [0.4000000059604645, -1.0, -1.0, -1.0]
    assert (out[2] == -1).all()                                              # empty class stays empty


def test_thr_restored():
    import full_image
    assert full_image.THR == 0.02
