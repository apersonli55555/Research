#!/usr/bin/env python3
"""Assemble the Fall Label Desk web page from oops_mine.py output.

Gives every clip a short, stable id (t0001... for the random test sample, p0001... for
the training pool), copies clips to <page>/clips/<id>.mp4 and writes <page>/index.html
with the clip list embedded. Ids are stored in <run>/label_ids.json and never change, so
the page can be republished as more clips arrive without losing labels.

Pool clips are only added with --include-pool, after mining has finished (until then the
top-K pool still changes).

    python vision/build_label_page.py --run RUN_DIR --page PAGE_DIR [--include-pool]
Prints the clip files that are new since the last build (for publishing).
"""
import argparse, csv, hashlib, json, os, shutil

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--page", required=True)
    ap.add_argument("--include-pool", action="store_true")
    a = ap.parse_args()

    ids_path = os.path.join(a.run, "label_ids.json")
    ids = json.load(open(ids_path)) if os.path.exists(ids_path) else {}
    with open(os.path.join(a.run, "scores.csv")) as fh:
        rows = [r for r in csv.DictReader(fh) if r["kept"] == "1"]
    src = lambda n: os.path.join(a.run, "clips", n.replace("/", "_") + ".mp4")
    rows = [r for r in rows if os.path.exists(src(r["name"]))]
    order = lambda r: hashlib.md5(r["name"].encode()).hexdigest()   # model-independent order

    n_t = sum(1 for v in ids.values() if v.startswith("t"))
    n_p = sum(1 for v in ids.values() if v.startswith("p"))
    for r in sorted((r for r in rows if r["random_sample"] == "1"), key=order):
        if r["name"] not in ids:
            n_t += 1; ids[r["name"]] = f"t{n_t:04d}"
    if a.include_pool:
        for r in sorted((r for r in rows if r["random_sample"] == "0"), key=order):
            if r["name"] not in ids:
                n_p += 1; ids[r["name"]] = f"p{n_p:04d}"
    json.dump(ids, open(ids_path + ".tmp", "w"), indent=0)
    os.replace(ids_path + ".tmp", ids_path)

    os.makedirs(os.path.join(a.page, "clips"), exist_ok=True)
    present = {r["name"] for r in rows}
    manifest, new = [], []
    for name, cid in sorted(ids.items(), key=lambda kv: kv[1]):
        if name not in present:
            continue
        dst = os.path.join(a.page, "clips", cid + ".mp4")
        if not os.path.exists(dst):
            shutil.copyfile(src(name), dst)
            new.append(f"clips/{cid}.mp4")
        manifest.append({"id": cid})
    tpl = open(os.path.join(HERE, "label_page", "template.html")).read()
    with open(os.path.join(a.page, "index.html"), "w") as fh:
        fh.write(tpl.replace("__MANIFEST__", json.dumps(manifest, separators=(",", ":"))))
    print(json.dumps({"test": sum(m["id"][0] == "t" for m in manifest),
                      "pool": sum(m["id"][0] == "p" for m in manifest), "new_files": new}))


if __name__ == "__main__":
    main()
