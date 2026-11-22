#!/usr/bin/env python3
"""Vocabulary-set interference, Stage 2 + 3 (OVD runs 2-4): full characterisation on held-out data, mechanism
probes, the fixed-context corrections and latency. Rules: ovd/PLAN.md (fixed before results).

Datasets (public annotations only; no image used here is in the pilot's development set)
  coco   COCO val2017, the images ranked 501.. by MD5 of the file name (the first 500 are the pilot's dev split)
  lvis   LVIS v1.0 val (federated labels), images ranked by MD5 of the file name, excluding every dev image;
         present targets = up to 3 annotated classes per image, absent targets = 2 *verified-absent* classes
         (neg_category_ids); detections of a class the image is "not exhaustively" labelled for are ignored in AP

Conditions per (image, target c); the target string is always the same name; seeds = MD5(dataset|image|c)
  k1                 {c}                                          (single-class inference)
  rand{k}_{d}        c + k-1 random other classes, random order   COCO k in 5, 20 (5 draws); LVIS k in 5..500 (3 draws)
  near{k}            c + its k-1 nearest classes by CLIP text embedding         (semantically similar)
  vis{k}             c + the k-1 classes CLIP ranks highest for the target's own image crop (visually confusable;
                     whole image for absent targets) - uses no label beyond the box every condition already uses
  far{k}             c + k-1 random classes from the half of the list least similar to c (unrelated)
  all_{d}            the full list (80 or 1203), 5 random orders
  prune              baseline in the style of VocAda (Liu et al., ICIP 2025) without its captioner: the K classes CLIP
                     ranks highest for the whole image (K = 10 COCO, 50 LVIS, fixed before results); c may be absent
Grounding DINO's 256-token limit: a vocabulary that does not fit is packed, in its order, into consecutive chunks
that fit (standard practice); only the chunk holding c can score c, so only that chunk is run (its size is stored).

Per run we store, for the target only: every box with target score >= 0.02 (top 50; YOLO-World after per-class
NMS 0.7) as [score, x1, y1, x2, y2, top] (top = 1 if c is that box's top class within the vocabulary, i.e. the box
keeps label c under standard single-label post-processing), the best box, and per ground-truth instance the raw score
and the deployed score (top & >= the model's threshold).

Corrections (training-free), computed once per image because by construction they do not depend on the vocabulary
  gdino  fixctx   every class is scored inside a FIXED context: the dataset's full class list in canonical id order,
                  packed into token-limit chunks; the deployment vocabulary only selects which classes are reported
  yolo   fixguide the text features that steer the image features (C2fAttn guide) are the fixed full list; the
                  deployment vocabulary is used only in the classification head
  all    indep    scores read per class, no label competition (each box may report every class above threshold)
Mechanism probe (gdino, model "gdino_qs"): query selection restricted to the target's tokens (hook on the encoder's
class embedding, which is used only to pick the 900 decoder queries) on k1, rand20_0, all_0.

  python ovd/vocab_full.py data    --dataset coco|lvis --out OUT
  python ovd/vocab_full.py run     --dataset coco|lvis --out OUT --model gdino|owlv2|yoloworld|gdino_qs
  python ovd/vocab_full.py bench   --dataset coco|lvis --out OUT --model gdino|owlv2|yoloworld
  python ovd/vocab_full.py analyze --out OUT          (see ovd/full_analysis.py)
"""
import argparse, hashlib, io, json, os, re, sys, time, urllib.request, zipfile

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from vocab_pilot import THRESH, iou  # noqa: E402

h = lambda s: hashlib.md5(s.encode()).hexdigest()
COCO_ANN = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
LVIS_ANN = "https://dl.fbaipublicfiles.com/LVIS/lvis_v1_val.json.zip"
N_DEV = 500                                   # pilot development images (never used here)
CFG = {"coco": dict(prune=10, n=1000, sizes=(5, 20), draws=5, fam_sizes=(5, 20), present_cap=None),
       "lvis": dict(prune=50, n=600, sizes=(5, 20, 80, 200, 500), draws=3, fam_sizes=(5, 20, 80), present_cap=3)}
ORDERS, N_ABSENT, MAX_DETS, MIN_SCORE = 5, 2, 50, 0.02


def clean_lvis(names):
    base = [re.sub(r"_\(.*\)$", "", n).replace("_", " ") for n in names]
    dup = {b for b in base if base.count(b) > 1}
    return [n.replace("_(", " (").replace("_", " ") if b in dup else b for n, b in zip(names, base)]


def fetch_json(url, member, path):
    if not os.path.exists(path):
        print("downloading", url, flush=True)
        with zipfile.ZipFile(io.BytesIO(urllib.request.urlopen(url).read())) as z:
            open(path, "wb").write(z.read(member))
    return json.load(open(path))


def dev_files(out):
    C = fetch_json(COCO_ANN, "annotations/instances_val2017.json", os.path.join(out, "instances_val2017.json"))
    has = {x["image_id"] for x in C["annotations"] if not x.get("iscrowd", 0)}
    return C, [im["file_name"] for im in sorted((im for im in C["images"] if im["id"] in has), key=lambda im: h(im["file_name"]))]


# ----------------------------------------------------------------------------- data
def data(a):
    cfg = CFG[a.dataset]; n_img = a.n_images or cfg["n"]
    os.makedirs(os.path.join(a.out, "images"), exist_ok=True)
    C, ranked = dev_files(a.out)
    dev = set(ranked[:N_DEV])
    if a.dataset == "coco":
        cats = {c["id"]: c["name"] for c in C["categories"]}
        ids = sorted(cats); names = [cats[k] for k in ids]
        imgs = {im["file_name"]: im for im in C["images"]}
        files = ranked[N_DEV:N_DEV + n_img]
        anns = {}
        for x in C["annotations"]:
            if not x.get("iscrowd", 0):
                anns.setdefault(x["image_id"], []).append(x)
        info = {}
        for f in files:
            im = imgs[f]; gts = anns[im["id"]]
            present = sorted({cats[x["category_id"]] for x in gts})
            absent = sorted((n for n in names if n not in present), key=lambda n: h(f + n))[:N_ABSENT]
            info[f] = dict(url=im["coco_url"], present=present, absent=absent, not_exh=[],
                           gt=[dict(cls=cats[x["category_id"]], box=xyxy(x["bbox"])) for x in gts])
    else:
        L = fetch_json(LVIS_ANN, "lvis_v1_val.json", os.path.join(a.out, "lvis_v1_val.json"))
        ids = sorted(c["id"] for c in L["categories"]); raw = {c["id"]: c["name"] for c in L["categories"]}
        names = clean_lvis([raw[k] for k in ids]); cats = dict(zip(ids, names))
        anns = {}
        for x in L["annotations"]:
            anns.setdefault(x["image_id"], []).append(x)
        fname = lambda im: im["coco_url"].split("/")[-1]
        cand = sorted((im for im in L["images"] if im["id"] in anns and fname(im) not in dev and im.get("neg_category_ids")),
                      key=lambda im: h(fname(im)))[:n_img]
        info = {}
        for im in cand:
            f = fname(im); gts = anns[im["id"]]
            pres_all = sorted({cats[x["category_id"]] for x in gts})
            present = sorted(pres_all, key=lambda n: h(f + "|p|" + n))[: cfg["present_cap"]]
            absent = sorted((cats[k] for k in im["neg_category_ids"]), key=lambda n: h(f + n))[:N_ABSENT]
            info[f] = dict(url=im["coco_url"], present=present, absent=absent,
                           not_exh=[cats[k] for k in im.get("not_exhaustive_category_ids", [])],
                           gt=[dict(cls=cats[x["category_id"]], box=xyxy(x["bbox"])) for x in gts])
    for f, d in info.items():
        p = os.path.join(a.out, "images", f)
        if not os.path.exists(p):
            for t in range(4):
                try:
                    urllib.request.urlretrieve(d["url"], p); break
                except Exception as e:
                    print("retry", f, e); time.sleep(2 ** t)
    # CLIP text similarities between classes and, for the visual family, image-crop similarity to class names
    import open_clip, torch
    from PIL import Image
    model, _, prep = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")
    tok = open_clip.get_tokenizer("ViT-B-32"); model.eval()
    with torch.no_grad():
        T = torch.cat([model.encode_text(tok([f"a photo of a {n}" for n in names[i:i + 256]])) for i in range(0, len(names), 256)])
        T = torch.nn.functional.normalize(T.float(), dim=-1).numpy()
    sim = T @ T.T
    plan = []
    for f, d in info.items():
        img = Image.open(os.path.join(a.out, "images", f)).convert("RGB")
        with torch.no_grad():                            # VocAda-style pruning baseline (CLIP only, no captioner)
            ew = torch.nn.functional.normalize(model.encode_image(prep(img)[None]).float(), dim=-1).numpy()[0]
        pruned = [names[j] for j in np.argsort(-(T @ ew))[: cfg["prune"]]]
        for c in d["present"] + d["absent"]:
            ci = names.index(c); others = [n for n in names if n != c]
            rng = np.random.default_rng(int(h(f"{a.dataset}|{f}|{c}")[:8], 16))
            boxes = [g["box"] for g in d["gt"] if g["cls"] == c]
            crop = img.crop(tuple(max(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1])))) if boxes else img
            with torch.no_grad():
                e = torch.nn.functional.normalize(model.encode_image(prep(crop)[None]).float(), dim=-1).numpy()[0]
            vis = [names[j] for j in np.argsort(-(T @ e)) if j != ci]
            near = [names[j] for j in np.argsort(-sim[ci]) if j != ci]
            far = near[len(near) // 2:]
            conds = {"k1": [c]}
            for k in cfg["sizes"]:
                for dd in range(cfg["draws"]):
                    v = [c] + list(rng.choice(others, k - 1, replace=False)); rng.shuffle(v); conds[f"rand{k}_{dd}"] = v
            for k in cfg["fam_sizes"]:
                for fam, pool in (("near", near[: k - 1]), ("vis", vis[: k - 1]), ("far", list(rng.choice(far, k - 1, replace=False)))):
                    v = [c] + list(pool); rng.shuffle(v); conds[f"{fam}{k}"] = v
            for dd in range(ORDERS):
                v = list(names); rng.shuffle(v); conds[f"all_{dd}"] = v
            conds["prune"] = list(pruned)                # may not contain c: then c is not detected at all
            plan.append(dict(image=f, target=c, present=c in d["present"], conds=conds))
    json.dump(dict(dataset=a.dataset, names=names, plan=plan, info=info), open(os.path.join(a.out, f"plan_{a.dataset}.json"), "w"))
    print(f"{a.dataset}: {len(info)} images, {len(plan)} (image, target) pairs, "
          f"{sum(len(p['conds']) for p in plan)} detector runs per model", flush=True)


def xyxy(b):
    return [b[0], b[1], b[0] + b[2], b[1] + b[3]]


# ----------------------------------------------------------------------------- detectors
class GDino:
    """Grounding DINO base. score(box, class) = max sigmoid over the class phrase's tokens. The image backbone output
    is cached per image (identical numbers; verified in tests) so many vocabularies per image cost less."""
    LIMIT = 256

    def __init__(self, dev, qs_target=False):
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        mid = "IDEA-Research/grounding-dino-base"
        self.torch, self.dev = torch, dev
        self.proc = AutoProcessor.from_pretrained(mid)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(mid).to(dev).eval()
        self.tok = self.proc.tokenizer
        self.ntok = {}
        bb = self.model.model.backbone; orig = bb.forward; self._bb = {}

        def cached(pixel_values, pixel_mask):
            k = (tuple(pixel_values.shape), float(pixel_values.double().sum()), float(pixel_values[..., ::7, ::5].double().sum()))
            if k not in self._bb:
                self._bb.clear(); self._bb[k] = orig(pixel_values, pixel_mask)
            return self._bb[k]
        bb.forward = cached
        self.qs_span = None
        if qs_target:                                     # mechanism probe: query selection sees only the target
            def hook(mod, inp, out):
                if self.qs_span is None:
                    return out
                keep = torch.full_like(out, float("-inf")); keep[..., self.qs_span] = out[..., self.qs_span]
                return keep
            self.model.model.encoder_output_class_embed.register_forward_hook(hook)

    def tokens(self, n):
        if n not in self.ntok:
            self.ntok[n] = self.tok(n, add_special_tokens=False)["input_ids"]
        return self.ntok[n]

    def chunks(self, vocab):
        """Pack the vocabulary, in order, into prompts of at most 256 tokens ([CLS] ... [SEP], ' .' after each)."""
        out, cur, used = [], [], 2
        for n in vocab:
            t = len(self.tokens(n)) + 1
            if cur and used + t > self.LIMIT:
                out.append(cur); cur, used = [], 2
            cur.append(n); used += t
        return out + [cur]

    def __call__(self, img, vocab, target=None):
        torch = self.torch
        text = " . ".join(vocab) + " ."
        x = self.proc(images=img, text=text, return_tensors="pt").to(self.dev)
        ids = x["input_ids"][0].tolist()
        spans, pos = [], 1
        for n in vocab:
            t = self.tokens(n)
            while pos < len(ids) and ids[pos:pos + len(t)] != t:
                pos += 1
            spans.append(list(range(pos, min(pos + len(t), len(ids)))) if pos < len(ids) else [])
            pos += len(t)
        self.qs_span = spans[vocab.index(target)] if (target is not None and spans[vocab.index(target)]) else None
        with torch.no_grad():
            o = self.model(**x)
        p = o.logits[0].sigmoid().float().cpu().numpy()
        S = np.stack([p[:, s].max(1) if s else np.zeros(len(p)) for s in spans], 1)
        W, H = img.size
        b = o.pred_boxes[0].float().cpu().numpy()
        boxes = np.stack([(b[:, 0] - b[:, 2] / 2) * W, (b[:, 1] - b[:, 3] / 2) * H, (b[:, 0] + b[:, 2] / 2) * W, (b[:, 1] + b[:, 3] / 2) * H], 1)
        return boxes, S


class Owl:
    """OWLv2: queries are scored independently, so one pass with the full list gives every vocabulary exactly."""
    def __init__(self, dev):
        import torch
        from transformers import Owlv2ForObjectDetection, Owlv2Processor
        mid = "google/owlv2-base-patch16-ensemble"
        self.torch, self.dev = torch, dev
        self.proc = Owlv2Processor.from_pretrained(mid)
        self.model = Owlv2ForObjectDetection.from_pretrained(mid).to(dev).eval()

    def __call__(self, img, vocab):
        torch = self.torch
        x = self.proc(text=[[f"a photo of a {n}" for n in vocab]], images=img, return_tensors="pt", truncation=True).to(self.dev)
        with torch.no_grad():
            o = self.model(**x)
        S = o.logits[0].sigmoid().float().cpu().numpy()
        s = max(img.size); b = o.pred_boxes[0].float().cpu().numpy()
        boxes = np.stack([(b[:, 0] - b[:, 2] / 2) * s, (b[:, 1] - b[:, 3] / 2) * s, (b[:, 0] + b[:, 2] / 2) * s, (b[:, 1] + b[:, 3] / 2) * s], 1)
        return boxes, S


class YoloWorld:
    """YOLO-World v2 s with a hand-written forward so that the guide text (C2fAttn) and the head text can differ.
    Text embeddings are cached per class name (the stock set_classes reloads CLIP every call)."""
    def __init__(self, dev, names):
        import torch
        from ultralytics import YOLOWorld
        from ultralytics.nn.modules import C2fAttn, WorldDetect
        self.torch, self.dev, self.C2fAttn, self.WorldDetect = torch, dev, C2fAttn, WorldDetect
        self.net = YOLOWorld("yolov8s-worldv2.pt").model.to(dev).eval()
        E = torch.cat([self.net.get_text_pe(names[i:i + 80], cache_clip_model=True)[0] for i in range(0, len(names), 80)])
        self.emb = {n: E[i] for i, n in enumerate(names)}

    def text(self, vocab):
        return self.torch.stack([self.emb[n] for n in vocab])[None].to(self.dev)

    def prep(self, img):
        torch = self.torch
        W, H = img.size; r = 640 / max(W, H); w, hh = round(W * r), round(H * r)
        im = np.full((640, 640, 3), 114, np.uint8); dx, dy = (640 - w) // 2, (640 - hh) // 2
        im[dy:dy + hh, dx:dx + w] = np.asarray(img.resize((w, hh), 2))
        return torch.from_numpy(im).permute(2, 0, 1)[None].float().div(255).to(self.dev), r, dx, dy

    def __call__(self, img, vocab, guide=None):
        torch = self.torch
        x, r, dx, dy = self.prep(img)
        head = self.text(vocab); g = self.text(guide) if guide is not None else head
        net = self.net; net.model[-1].nc = len(vocab); y = []
        with torch.no_grad():
            for m in net.model:
                if m.f != -1:
                    x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
                if isinstance(m, self.C2fAttn):
                    x = m(x, g)
                elif isinstance(m, self.WorldDetect):
                    x = m(x, head)
                else:
                    x = m(x)
                y.append(x if m.i in net.save else None)
        p = (x[0] if isinstance(x, tuple) else x)[0].float().cpu().numpy()        # (4 + nc, anchors), xywh in px
        b = p[:4].T; S = p[4:].T
        boxes = np.stack([b[:, 0] - b[:, 2] / 2 - dx, b[:, 1] - b[:, 3] / 2 - dy, b[:, 0] + b[:, 2] / 2 - dx, b[:, 1] + b[:, 3] / 2 - dy], 1) / r
        return boxes, S


def nms(boxes, scores, thr=0.7):
    order = np.argsort(-scores); keep = []
    while len(order):
        i = order[0]; keep.append(i)
        if len(order) == 1:
            break
        order = order[1:][iou(boxes[order[1:]], boxes[i]) < thr]
    return np.array(keep, int)


def target_record(boxes, S, c, gts, th, use_nms):
    """The stored output for target column c (see the module docstring)."""
    s = S[:, c]; top = S.argmax(1) == c
    cand = np.where(s >= MIN_SCORE)[0]
    if use_nms and len(cand):
        cand = cand[nms(boxes[cand], s[cand])]
    cand = cand[np.argsort(-s[cand])][:MAX_DETS]
    dets = [[round(float(s[i]), 5)] + [round(float(v), 1) for v in boxes[i]] + [int(top[i])] for i in cand]
    bi = int(np.argmax(s))
    inst = []
    for g in gts:
        m = iou(boxes, np.array(g)) >= 0.5
        dep = s * (top & (s >= th))
        inst.append(dict(raw=float(s[m].max()) if m.any() else 0.0, deployed=float(dep[m].max()) if m.any() else 0.0,
                         best_iou=float(iou(boxes[bi:bi + 1], np.array(g))[0])))
    return dict(dets=dets, best_box=[round(float(v), 1) for v in boxes[bi]], instances=inst,
                img_max_raw=float(s.max()), img_max_deployed=float((s * (top & (s >= th))).max()))


# ----------------------------------------------------------------------------- run
def run(a):
    import torch
    from PIL import Image
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    P = json.load(open(os.path.join(a.out, f"plan_{a.dataset}.json")))
    names, plan = P["names"], P["plan"][: a.limit or None]
    out = os.path.join(a.out, f"full_{a.dataset}_{a.model}.jsonl")
    done = set()
    if os.path.exists(out):
        for line in open(out):
            try:
                r = json.loads(line); done.add((r["image"], r["target"], r["cond"]))
            except json.JSONDecodeError:
                pass                                                     # a line cut off by a disconnect
    base = a.model.replace("_qs", "")
    th = THRESH[base]
    if base == "gdino":
        det = GDino(dev, qs_target=a.model == "gdino_qs")
    elif base == "owlv2":
        det = Owl(dev)
    else:
        det = YoloWorld(dev, names)
    use_nms = base == "yoloworld"
    by_img = {}
    for p in plan:
        by_img.setdefault(p["image"], []).append(p)
    t0, n = time.time(), 0
    with open(out, "a") as fh:
        for f, pairs in by_img.items():
            todo = [(p, cn, v) for p in pairs for cn, v in p["conds"].items()
                    if (a.model != "gdino_qs" or cn in ("k1", "rand20_0", "all_0")) and (f, p["target"], cn) not in done]
            fix_todo = [p for p in pairs if (f, p["target"], "FIX") not in done and a.model != "gdino_qs"]
            if not todo and not fix_todo:
                continue
            img = Image.open(os.path.join(a.out, "images", f)).convert("RGB")
            gt = P["info"][f]["gt"]
            full = det(img, names) if base == "owlv2" else None              # exact for every vocabulary
            for p, cn, vocab in todo:
                c = p["target"]; gts = [g["box"] for g in gt if g["cls"] == c]
                t1 = time.time(); extra = {}
                if c not in vocab:                                           # pruned away: no output for c
                    rec = dict(image=f, target=c, cond=cn, present=p["present"], k=len(vocab), sec=0.0, pruned_out=True, dets=[],
                               best_box=[0, 0, 0, 0], instances=[dict(raw=0.0, deployed=0.0, best_iou=0.0) for _ in gts],
                               img_max_raw=0.0, img_max_deployed=0.0)
                    fh.write(json.dumps(rec) + "\n"); n += 1; continue
                if base == "owlv2":
                    cols = [names.index(v) for v in vocab]; boxes, S = full[0], full[1][:, cols]
                elif base == "gdino":
                    ch = det.chunks(vocab); mine = next(x for x in ch if c in x)
                    boxes, S = det(img, mine, target=c if a.model == "gdino_qs" else None)
                    vocab = mine; extra = dict(n_chunks=len(ch), chunk_k=len(mine))
                else:
                    boxes, S = det(img, vocab)
                rec = dict(image=f, target=c, cond=cn, present=p["present"], k=len(p["conds"][cn]), sec=time.time() - t1, **extra)
                rec.update(target_record(boxes, S, vocab.index(c), gts, th, use_nms))
                fh.write(json.dumps(rec) + "\n"); n += 1
            # vocabulary-independent corrections: one record per target, the same whatever the deployment vocabulary
            if fix_todo:
                if base == "gdino":
                    ctx = det.chunks(names)                                   # canonical order, fixed chunks
                    for p in fix_todo:
                        c = p["target"]; mine = next(x for x in ctx if c in x)
                        boxes, S = det(img, mine)
                        gts = [g["box"] for g in gt if g["cls"] == c]
                        rec = dict(image=f, target=c, cond="FIX", present=p["present"], k=len(mine), n_chunks=len(ctx), chunk_k=len(mine))
                        rec.update(target_record(boxes, S, mine.index(c), gts, th, False))
                        fh.write(json.dumps(rec) + "\n")
                elif base == "yoloworld":
                    boxes, S = det(img, names, guide=names)                    # head over all names: indep per class
                    for p in fix_todo:
                        c = p["target"]; gts = [g["box"] for g in gt if g["cls"] == c]
                        rec = dict(image=f, target=c, cond="FIX", present=p["present"], k=len(names))
                        rec.update(target_record(boxes, S, names.index(c), gts, th, True))
                        fh.write(json.dumps(rec) + "\n")
                else:                                                          # OWLv2: independent scores = invariant
                    for p in fix_todo:
                        c = p["target"]; gts = [g["box"] for g in gt if g["cls"] == c]
                        rec = dict(image=f, target=c, cond="FIX", present=p["present"], k=len(names))
                        rec.update(target_record(full[0], full[1], names.index(c), gts, th, False))
                        fh.write(json.dumps(rec) + "\n")
            fh.flush()
            if n and (n // 200) != ((n - len(todo)) // 200):
                print(f"  {n} runs, {(time.time() - t0) / n:.3f} s/run (incl. corrections)", flush=True)
    print(f"{a.model} on {a.dataset}: {n} new runs written to {out}", flush=True)


# ----------------------------------------------------------------------------- latency
def bench(a):
    """Wall-clock per image (CUDA synchronised, 3 warm-up images) for each way of answering a deployment vocabulary."""
    import torch
    from PIL import Image
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    P = json.load(open(os.path.join(a.out, f"plan_{a.dataset}.json")))
    names = P["names"]; files = list(P["info"])[: a.bench_images + 3]
    sync = (lambda: torch.cuda.synchronize()) if dev == "cuda" else (lambda: None)
    rng = np.random.default_rng(0)
    vocabs = {"full": list(names)}
    for k in (5, 20) + ((80, 200) if len(names) > 80 else ()):
        vocabs[f"rand{k}"] = list(rng.choice(names, k, replace=False))
    res = {}
    if a.model == "gdino":
        det = GDino(dev)
        ctx = det.chunks(names)
        def orig(img, v):
            for ch in det.chunks(v):
                det(img, ch)
        def fixctx(img, v):
            for ch in ctx:
                if any(n in ch for n in v):
                    det(img, ch)
        def single(img, v):
            for n in v:
                det(img, [n])
        methods = dict(orig=orig, fixctx=fixctx, single=single)
    elif a.model == "owlv2":
        det = Owl(dev)
        methods = dict(orig=lambda img, v: det(img, v), single=lambda img, v: [det(img, [n]) for n in v])
    else:
        det = YoloWorld(dev, names)
        methods = dict(orig=lambda img, v: det(img, v), fixguide=lambda img, v: det(img, v, guide=names),
                       single=lambda img, v: [det(img, [n]) for n in v])
    imgs = [Image.open(os.path.join(a.out, "images", f)).convert("RGB") for f in files]
    warm = min(3, len(imgs) - 1)
    for mname, fn in methods.items():
        for vname, v in vocabs.items():
            if mname == "single" and len(v) > 80:
                continue
            ts = []
            for i, img in enumerate(imgs):
                if a.model == "gdino":
                    det._bb.clear()                                   # every image starts with an empty cache
                sync(); t = time.time(); fn(img, v); sync()
                if i >= warm:
                    ts.append(time.time() - t)
            res[f"{mname}/{vname}"] = dict(mean_s=float(np.mean(ts)), sd_s=float(np.std(ts)), n=len(ts))
            print(f"{a.model} {a.dataset} {mname:9s} {vname:8s} {np.mean(ts):.3f} s/image", flush=True)
    path = os.path.join(a.out, f"bench_{a.dataset}.json")
    allr = json.load(open(path)) if os.path.exists(path) else {}
    allr[a.model] = dict(res, device=torch.cuda.get_device_name(0) if dev == "cuda" else "cpu")
    json.dump(allr, open(path, "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["data", "run", "bench"])
    ap.add_argument("--dataset", default="coco", choices=list(CFG))
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="gdino", choices=["gdino", "owlv2", "yoloworld", "gdino_qs"])
    ap.add_argument("--n-images", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="testing only: number of (image, target) pairs")
    ap.add_argument("--bench-images", type=int, default=30)
    a = ap.parse_args()
    {"data": data, "run": run, "bench": bench}[a.stage](a)


if __name__ == "__main__":
    main()
