#!/usr/bin/env python3
"""Convert labels exported from the Fall Label Desk page into <run>/labels.csv.

Input: one or more JSON files, each a list of {"id": "<clip id>", "data": {"label": ...}}
documents (or {"documents": [...]}) as read from the page's database collections
(test, pool0, pool1, ...). Clip ids are mapped back to OOPS! video names through
<run>/label_ids.json. The resulting labels.csv is what experiment.py reads.

    python vision/import_labels.py --run RUN_DIR export_test.json export_pool0.json ...
"""
import argparse, csv, json, os


def docs_of(obj):
    if isinstance(obj, dict):
        for k in ("documents", "docs", "items", "results"):
            if k in obj:
                return obj[k]
        return [obj]
    return obj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("exports", nargs="+")
    a = ap.parse_args()
    by_id = {v: k for k, v in json.load(open(os.path.join(a.run, "label_ids.json"))).items()}
    labels = {}
    for path in a.exports:
        for d in docs_of(json.load(open(path))):
            cid = d.get("id") or d.get("doc_id") or str(d.get("path", "")).split("/")[-1]
            data = d.get("data", d)
            lab = data.get("label")
            if cid in by_id and lab in ("fall", "not_fall", "unsure"):
                labels[by_id[cid]] = lab
    with open(os.path.join(a.run, "labels.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["name", "label", "labeler"])
        for name, lab in sorted(labels.items()):
            w.writerow([name, lab, "web"])
    counts = {k: sum(v == k for v in labels.values()) for k in ("fall", "not_fall", "unsure")}
    print(f"wrote {len(labels)} labels to {a.run}/labels.csv: {counts}")


if __name__ == "__main__":
    main()
