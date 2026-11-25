"""Tests for the class-wise score map of Amendment I (bank/bank_calib.py). Run: python -m pytest -q ovd/bank/test_calib.py"""
import os, sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bank_calib import apply_map, top_list  # noqa: E402

torch.manual_seed(0)
C, K = 40, 8


def stats():
    mu = torch.randn(C) - 4; sd = torch.rand(C) + 0.5
    return dict(mu=mu, sd=sd, mu_g=float(mu.mean()), sd_g=float(sd.mean()))


def scores():
    s = torch.rand(C, K) * 0.9 + 0.02
    s[torch.rand(C, K) < 0.3] = -1                       # empty slots
    return s


def test_empty_slots_stay_empty():
    s, st = scores(), stats()
    for m in ("none", "shift", "z"):
        out = apply_map(m, s, st)
        assert torch.equal(out[s <= 0], s[s <= 0])
        assert (out[s > 0] > 0).all() and (out[s > 0] < 1).all()


def test_within_class_order_preserved():
    """A per-class increasing map leaves each class's ranking (hence its per-class AP without a cap) unchanged."""
    s, st = scores(), stats()
    for m in ("shift", "z"):
        out = apply_map(m, s, st)
        for c in range(C):
            v = s[c] > 0
            assert torch.equal(torch.argsort(s[c][v], descending=True), torch.argsort(out[c][v], descending=True))


def test_request_invariance():
    """Mapping then selecting a requested subset of classes equals selecting then mapping with the same constants:
    a class's mapped scores do not depend on which other classes are requested."""
    s, st = scores(), stats()
    for m in ("shift", "z"):
        full = apply_map(m, s, st)
        for _ in range(5):
            req = torch.randperm(C)[: torch.randint(1, C, (1,)).item()]
            sub_st = dict(mu=st["mu"][req], sd=st["sd"][req], mu_g=st["mu_g"], sd_g=st["sd_g"])
            assert torch.allclose(apply_map(m, s[req], sub_st), full[req])


def test_shift_equalizes_reference_level():
    """A score at a class's reference mean logit maps to the same value for every class."""
    st = stats(); s = torch.sigmoid(st["mu"])[:, None].repeat(1, 2)
    out = apply_map("shift", s, st)
    assert torch.allclose(out, torch.full_like(out, torch.sigmoid(torch.tensor(st["mu_g"])).item()), atol=1e-5)


def test_top_list_cap_and_order():
    s = scores(); b = torch.rand(C, K, 4)
    lst = top_list(s, b, 10)
    assert len(lst) == 10 and all(lst[i][1] >= lst[i + 1][1] for i in range(9))
    assert len(top_list(s, b, None)) == int((s > 0).sum())
