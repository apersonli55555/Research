#!/usr/bin/env python3
"""Fingerprint each part of a plan file so two machines' plans can be compared piece by piece.

  python ovd/plan_fingerprint.py OUT/plan_lvis.json
"""
import hashlib, json, sys

P = json.load(open(sys.argv[1]))
h = lambda x: hashlib.md5(json.dumps(x, sort_keys=True).encode()).hexdigest()[:10]
print("names     ", h(P["names"]))
print("images    ", h(sorted(P["info"])))
print("gt        ", h({f: d["gt"] for f, d in P["info"].items()}))
print("pairs     ", h(sorted((p["image"], p["target"], p["present"]) for p in P["plan"])))
fams = {}
for p in P["plan"]:
    for c, v in p["conds"].items():
        fam = "all" if c.startswith("all_") else c.split("_")[0].rstrip("0123456789")
        fams.setdefault(fam, []).append((p["image"], p["target"], c, v))
for fam in sorted(fams):
    print(f"{fam:10s}", h(sorted(fams[fam])))
