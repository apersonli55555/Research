"""python -m pytest ovd/bank/test_full_image.py -q  -- batched GPU Soft-NMS == the verified per-list Soft-NMS."""
import glob, json, os, sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from full_image import soft_nms_batched, top_per_image  # noqa: E402
from followup_analysis import soft_nms  # noqa: E402


def test_batched_soft_nms_matches_reference():
    lists = []
    for p in sorted(glob.glob(os.path.join(HERE, "..", "results", "bank", "fixed_bank_soft_confirm_v1", "per_image", "*.json")))[:40]:
        lists += [r["dets"] for r in json.load(open(p))["rows"] if r["cond"] in ("bank", "fixed", "stock5")]
    lists = [d for d in lists if d][:300]
    S = torch.full((len(lists), 50), -1.0, dtype=torch.float64); B = torch.zeros((len(lists), 50, 4), dtype=torch.float64)
    for i, d in enumerate(lists):
        S[i, :len(d)] = torch.tensor([x[0] for x in d], dtype=torch.float64); B[i, :len(d)] = torch.tensor([x[1:5] for x in d], dtype=torch.float64)
    os_, ob = soft_nms_batched(S, B)
    for i, d in enumerate(lists):
        ref = soft_nms(d); got = [(float(s), ob[i, k].tolist()) for k, s in enumerate(os_[i].tolist()) if s >= 0]
        assert len(ref) == len(got), (i, len(ref), len(got))
        for r, (s, b) in zip(ref, got):
            assert abs(r[0] - s) < 1e-9 and max(abs(u - v) for u, v in zip(r[1:], b)) < 1e-9


def test_top_per_image_keeps_300_best():
    S = torch.rand(1203, 50); S[S < 0.5] = -1; B = torch.rand(1203, 50, 4)
    out = top_per_image(S, B); assert len(out) == 300
    assert all(out[k][1] >= out[k + 1][1] for k in range(299)) and out[-1][1] >= 0.5
