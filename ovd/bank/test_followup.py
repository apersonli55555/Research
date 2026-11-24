"""Tests for the follow-up scripts (CPU, saved files only):  python -m pytest ovd/bank/test_followup.py -q"""
import glob, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from followup import rand_request, load_inputs  # noqa: E402
from followup_analysis import soft_nms  # noqa: E402

RUN = os.path.join(HERE, "..", "results", "bank", "fixed_bank_soft_confirm_v1", "per_image")


def test_soft_nms_reproduces_saved_run3h():
    n = 0
    for p in sorted(glob.glob(os.path.join(RUN, "*.json")))[:60]:
        R = {(r["target"], r["cond"]): r["dets"] for r in json.load(open(p))["rows"]}
        for (t, c), dets in R.items():
            if c.endswith("_soft"):
                continue
            a, b = soft_nms(dets), R[(t, c + "_soft")]
            assert len(a) == len(b)
            for x, y in zip(a, b):
                assert abs(x[0] - y[0]) < 1e-6 and max(abs(u - v) for u, v in zip(x[1:], y[1:])) < 1e-6
            n += 1
    assert n > 500


def test_image_independent_requests():
    images, names, targets, request, urls, saved = load_inputs()
    for f in images[:50]:
        for c in targets[f]:
            r = rand_request(names, c, "indep-request-A")
            assert r == rand_request(names, c, "indep-request-A")          # fixed across machines and runs
            assert c in r and len(set(r)) == 5 and set(r) <= set(names)
            assert r != rand_request(names, c, "indep-request-B")
    # the four other names do not depend on the image: same target -> same request on every image
    assert rand_request(names, names[0], "indep-request-A") == rand_request(names, names[0], "indep-request-A")
