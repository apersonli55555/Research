#!/usr/bin/env python3
"""Follow-up experiments for the fixed class bank (expert review, improvements #3-#7 and #9). GPU, resumable.

  python ovd/bank/followup.py --out /content/drive/MyDrive/ovd/followup_v1 --images /content/drive/MyDrive/ovd/images

Uses the 600 images and 2,669 (image, target) pairs of the confirmation run (run 3h,
ovd/results/bank/fixed_bank_soft_confirm_v1). The bank, its injection, the standard and fixed-chunk read-outs and the
export (top 50 boxes with score >= 0.02, pixel xyxy) are copied from the expert's notebook cells
(ovd/bank/validate_fixed_bank_cells.py). Only raw detections are saved; class-wise Gaussian Soft-NMS is applied afterwards
on CPU (it reproduces the saved Soft-NMS outputs of run 3h exactly from the saved raw outputs: 8,007/8,007 lists).

Parts (each writes one JSON per image into OUT/<part>/, so an interrupted run resumes where it stopped):
  A  reproduction check: standard five-class prompt and canonical bank on 20 images vs the saved run-3h outputs
  B  #3 image-independent requests: every pair scored with the target alone (stock1) and inside two requests of the
     target plus four names drawn at random from the 1,203 LVIS names without looking at the image (stock5_indA/B)
  C  #4 bank dictionary and #6 construction ablations, all 600 images: (#4) class vectors rebuilt from two random name
     orderings and from each name alone (compared vector by vector), and banks holding a fixed random half of the classes
     (dict_half_a/b); (#6) first / last wordpiece instead of the mean (pool_first, pool_last); full instead of identity
     text mask (mask_full)
  E  #5 cost crossover: wall time and peak GPU memory of standard (chunked) prompts, fixed chunks and the bank for
     vocabularies of 1-1,203 names on 30 images, plus the one-time bank set-up cost
  F  #7 second checkpoint: grounding-dino-tiny, bank / standard five-class / fixed chunks on all 600 images, timed
  G  #9 thumbnails of the 600 images (longest side 512) for the qualitative figure
"""
import argparse, json, os, random, sys, time, zipfile
from contextlib import ExitStack, nullcontext
from pathlib import Path
from unittest.mock import patch
import urllib.request

import numpy as np

HERE = Path(__file__).resolve().parent
OVD = HERE.parent
sys.path.insert(0, str(OVD))
RUN = OVD / "results" / "bank" / "fixed_bank_soft_confirm_v1"


def load_inputs():
    M = json.loads((RUN / "manifest.json").read_text())
    names = json.loads((HERE / "lvis_names.json").read_text())
    urls = json.loads((HERE / "confirm_images.json").read_text())
    targets = {}
    for t in M["targets"]:
        targets.setdefault(t["image"], []).append(t["target"])
    request = {}
    for p in sorted((RUN / "per_image").glob("*.json")):
        d = json.loads(p.read_text())
        request[d["rows"][0]["image"]] = d["request"]
    saved = {}
    for p in sorted((RUN / "per_image").glob("*.json")):
        for r in json.loads(p.read_text())["rows"]:
            saved[(r["image"], r["target"], r["cond"])] = r["dets"]
    images = sorted(M["images"])
    assert len(images) == 600 and len(names) == 1203 and set(request) == set(images) == set(targets) == set(urls)
    return images, names, targets, request, urls, saved


class Images:
    def __init__(self, drive_dir, cache="/content/lvis_images" if os.path.isdir("/content") else "/tmp/lvis_images"):
        self.drive, self.cache = Path(drive_dir) if drive_dir else None, Path(cache)
        self.cache.mkdir(parents=True, exist_ok=True)

    def path(self, f, url):
        if self.drive and (self.drive / f).is_file():
            return self.drive / f
        p = self.cache / f
        if not p.is_file():
            for attempt in range(4):
                try:
                    data = urllib.request.urlopen(url, timeout=60).read(); break
                except Exception:
                    if attempt == 3:
                        raise
                    time.sleep(5 * (attempt + 1))
            tmp = p.with_suffix(".part"); tmp.write_bytes(data); tmp.rename(p)
        return p

    def load(self, f, url):
        from PIL import Image
        with Image.open(self.path(f, url)) as im:
            return im.convert("RGB")


class Engine:
    """The expert's fixed-bank functions (validate_fixed_bank_cells.py, cells 2-4), wrapped for one checkpoint."""

    def __init__(self, dev, names, mid="IDEA-Research/grounding-dino-base"):
        import torch
        from transformers.modeling_outputs import BaseModelOutput
        from transformers.models.grounding_dino import modeling_grounding_dino as gd
        from vocab_full import GDino
        self.torch, self.gd, self.BaseModelOutput = torch, gd, BaseModelOutput
        self.g = GDino(dev, mid=mid); self.model = self.g.model; self.dev = dev; self.mid = mid
        self.names = names; self.index = {c: i for i, c in enumerate(names)}; self.n = len(names)
        self.gm = gd.generate_masks_with_special_tokens_and_transfer_map
        self.chunks = self.g.chunks(names)
        self.chunk_of = {c: i for i, ch in enumerate(self.chunks) for c in ch}
        n, t = self.n, torch
        self.ids = t.full((1, n), self.g.tok.convert_tokens_to_ids("."), device=dev, dtype=t.long)
        self.identity = t.eye(n, dtype=t.bool, device=dev)[None]
        self.fullmask = t.ones((1, n, n), dtype=t.bool, device=dev)
        self.positions = t.zeros_like(self.ids); self.ones = t.ones_like(self.ids); self.zeros = t.zeros_like(self.ids)

    @staticmethod
    def caption(cs):
        return " . ".join(cs) + " ."

    def inputs(self, im, cs):
        return self.g.proc(images=im, text=self.caption(cs), return_tensors="pt").to(self.dev)

    def cached_text(self, ids):
        with self.torch.inference_mode():
            mask, pos = self.gm(ids)
            bm = mask[:, None] if mask.ndim == 3 else mask
            return self.model.model.text_backbone(ids, bm, self.torch.zeros_like(ids), pos, return_dict=True).last_hidden_state

    def injected(self, hidden, mask, pos):
        ctx = ExitStack()
        try:
            ctx.enter_context(patch.object(self.gd, "generate_masks_with_special_tokens_and_transfer_map", return_value=(mask, pos)))
            ctx.enter_context(patch.object(self.model.model.text_backbone, "forward",
                                           return_value=self.BaseModelOutput(last_hidden_state=hidden)))
            length = max(256, hidden.shape[1])
            owners = [self.model.config, self.model.model.config, *self.model.modules()]
            visited = set()
            for obj in owners:
                if id(obj) not in visited and isinstance(getattr(obj, "max_text_len", None), int):
                    ctx.enter_context(patch.object(obj, "max_text_len", length)); visited.add(id(obj))
            return ctx
        except BaseException:
            ctx.close(); raise

    def parity(self, im):
        """Cached-text control: injecting the native text features must reproduce the native outputs (expert cell 3)."""
        torch = self.torch
        x = self.inputs(im, self.names[:5])
        with torch.inference_mode():
            stock = self.model(**x)
            h = self.cached_text(x.input_ids); mask, pos = self.gm(x.input_ids)
            with self.injected(h, mask, pos):
                ctrl = self.model(**x)
        torch.testing.assert_close(stock.pred_boxes, ctrl.pred_boxes, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(stock.logits, ctrl.logits, atol=1e-5, rtol=1e-5)

    def build_bank(self, order=None, pool="mean", solo=False):
        """One vector per class from contextualized BERT wordpieces. Expert default: canonical order, mean, chunked.
        order: the name order used to form the 256-token chunks (vectors are returned in canonical order)."""
        torch = self.torch; vec = {}
        groups = [[c] for c in self.names] if solo else self.g.chunks(order or self.names)
        for chunk in groups:
            ids = self.g.tok(self.caption(chunk), return_tensors="pt")["input_ids"].to(self.dev)
            assert ids.shape[1] <= 256
            tokens = self.cached_text(ids); at = 1
            for c in chunk:
                sub = self.g.tokens(c)
                assert ids[0, at:at + len(sub)].tolist() == sub
                span = tokens[:, at:at + len(sub)]
                vec[c] = {"mean": span.mean(dim=1, keepdim=True), "first": span[:, :1], "last": span[:, -1:]}[pool]
                at += len(sub) + 1
            assert at == ids.shape[1] - 1
        return torch.cat([vec[c] for c in self.names], dim=1)

    def bank_call(self, im, bank, mask=None):
        """One pass with the bank as the text input (expert cell 3). A bank of n' < 1,203 classes (dictionary test)
        gets id / mask / position tensors of its own size."""
        torch = self.torch; n = bank.shape[1]
        if n == self.n:
            ids, ident, pos, ones, zeros = self.ids, self.identity, self.positions, self.ones, self.zeros
        else:
            ids = torch.full((1, n), self.g.tok.convert_tokens_to_ids("."), device=self.dev, dtype=torch.long)
            ident = torch.eye(n, dtype=torch.bool, device=self.dev)[None]
            pos, ones, zeros = torch.zeros_like(ids), torch.ones_like(ids), torch.zeros_like(ids)
        with torch.inference_mode(), self.injected(bank, ident if mask is None else mask, pos):
            v = self.g.proc.image_processor(images=im, return_tensors="pt").to(self.dev)
            o = self.model(**v, input_ids=ids, token_type_ids=zeros, attention_mask=ones)
            assert o.logits.shape[-1] >= n and torch.isfinite(o.logits[..., :n]).all()
            return o.pred_boxes[0], o.logits[0, :, :n].sigmoid()

    def decode_prompt(self, im, cs):
        with self.torch.inference_mode():
            o = self.model(**self.inputs(im, cs))
            logits = o.logits[0]; offset = 1; scores = []
            for c in cs:
                width = len(self.g.tokens(c))
                scores.append(logits[:, offset:offset + width].amax(1)); offset += width + 1
            return o.pred_boxes[0], self.torch.stack(scores, dim=1).sigmoid()

    def compute(self, method, im, request, bank=None):
        """Scores for every name in `request`: dict name -> (boxes, scores). method: bank | stock | fixed.
        stock = the request as the prompt, split into the standard 256-token chunks when it is longer."""
        if method == "bank":
            boxes, scores = self.bank_call(im, bank)
            return {c: (boxes, scores[:, self.index[c]]) for c in request}
        found = {}
        if method == "stock":
            parts = [(ch, ch) for ch in self.g.chunks(request)]
        else:
            parts = [(self.chunks[i], [c for c in request if self.chunk_of[c] == i])
                     for i in sorted({self.chunk_of[c] for c in request})]
        for prompt, wanted in parts:
            boxes, scores = self.decode_prompt(im, prompt)
            for c in wanted:
                found[c] = (boxes, scores[:, prompt.index(c)])
        return found

    @staticmethod
    def export(f, im, result, wanted, cond, extra=None):
        W, H = im.size; rows = []
        for c in wanted:
            box, s = result[c]
            xy, wh = box[:, :2], box[:, 2:]
            b = (__import__("torch").cat([xy - wh / 2, xy + wh / 2], 1) * box.new_tensor([W, H, W, H])).float().cpu().numpy()
            s = s.float().cpu().numpy()
            dets = [[round(float(s[i]), 7), *[round(v, 3) for v in b[i].tolist()]] for i in np.argsort(-s)[:50] if s[i] >= .02]
            rows.append(dict(image=f, target=c, cond=cond, dets=dets, **(extra or {})))
        return rows

    def measure(self, fn):
        torch = self.torch
        self.g._bb.clear()                               # each method pays for its own image backbone
        cuda = self.dev.startswith("cuda")
        if cuda:
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); base = torch.cuda.memory_allocated()
        start = time.perf_counter(); result = fn()
        if cuda:
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        mem = (torch.cuda.max_memory_allocated() - base) if cuda else 0
        return result, elapsed, int(mem)


def save(path, obj):
    tmp = path.with_suffix(".tmp"); tmp.write_text(json.dumps(obj, allow_nan=False)); tmp.rename(path)


def rand_request(names, target, seed):
    rng = random.Random(f"{seed}/{target}")
    cs = [target] + rng.sample([c for c in names if c != target], 4)
    rng.shuffle(cs)
    return cs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--images", default=None, help="folder that may already hold the LVIS images (else downloaded)")
    ap.add_argument("--parts", default="A,B,C,E,F,G")
    ap.add_argument("--limit", type=int, default=0, help="only the first N images (smoke test)")
    ap.add_argument("--speed-images", type=int, default=30)
    ap.add_argument("--sizes", default="1,2,5,10,20,50,100,200,400,800,1203")
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    import torch, transformers
    dev = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    images, names, targets, request, urls, saved = load_inputs()
    if a.limit:
        images = images[:a.limit]
    imgs = Images(a.images)
    parts = a.parts.split(",")
    env = dict(gpu=torch.cuda.get_device_name() if dev.startswith("cuda") else "cpu", torch=torch.__version__,
               transformers=transformers.__version__, device=dev, parts=parts, limit=a.limit, started=time.time())
    (out / f"env_{int(time.time())}.json").write_text(json.dumps(env, indent=1))
    print(json.dumps(env), flush=True)

    def todo(part, files):
        d = out / part; d.mkdir(exist_ok=True)
        return d, [f for f in files if not (d / (f + ".json")).exists()]

    E = None
    if set(parts) & {"A", "B", "C", "E"}:
        E = Engine(dev, names)
        E.parity(imgs.load(images[0], urls[images[0]])); print("parity check passed (base)", flush=True)
        t0 = time.perf_counter(); canon = E.build_bank()
        if dev.startswith("cuda"):
            torch.cuda.synchronize()
        bank_setup_s = time.perf_counter() - t0

    if "A" in parts:
        d, left = todo("A", images[:20])
        for j, f in enumerate(left):
            im = imgs.load(f, urls[f]); rows = []
            rows += E.export(f, im, E.compute("stock", im, request[f]), targets[f], "stock5")
            rows += E.export(f, im, E.compute("bank", im, request[f], canon), targets[f], "bank")
            diff = []
            for r in rows:
                old = saved[(f, r["target"], r["cond"])]
                if old and r["dets"]:
                    diff.append(dict(cond=r["cond"], target=r["target"], score=abs(old[0][0] - r["dets"][0][0]),
                                     box=max(abs(x - y) for x, y in zip(old[0][1:], r["dets"][0][1:]))))
                else:
                    diff.append(dict(cond=r["cond"], target=r["target"], empty_old=not old, empty_new=not r["dets"]))
            save(d / (f + ".json"), dict(rows=rows, diff=diff))
            print(f"A {j + 1}/{len(left)} max top-score diff {max([x.get('score', 0) for x in diff]):.2e}", flush=True)

    if "B" in parts:
        d, left = todo("B", images)
        for j, f in enumerate(left):
            im = imgs.load(f, urls[f]); rows = []
            for c in targets[f]:
                rows += E.export(f, im, E.compute("stock", im, [c]), [c], "stock1")
                for tag in ("A", "B"):
                    req = rand_request(names, c, "indep-request-" + tag)
                    rows += E.export(f, im, E.compute("stock", im, req), [c], "stock5_ind" + tag, dict(request=req))
            save(d / (f + ".json"), dict(rows=rows))
            if j % 25 == 0 or j == len(left) - 1:
                print(f"B {j + 1}/{len(left)}", flush=True)

    if "C" in parts:
        # #4 dictionary: (a) does a class vector depend on the other names it was encoded with? Banks built from two
        # random name orderings and from each name alone are compared with the canonical bank vector by vector (the
        # text encoder masks every phrase separately, so they should be identical); (b) does detection depend on which
        # other classes are in the bank? Two banks holding a fixed random half of the classes (chosen without looking
        # at any image), scored on the pairs whose target they contain.
        meta = {}
        for tag, kw in [("perm1", dict(order=random.Random("bank-perm-1").sample(names, len(names)))),
                        ("perm2", dict(order=random.Random("bank-perm-2").sample(names, len(names)))),
                        ("solo", dict(solo=True)), ("pool_first", dict(pool="first")), ("pool_last", dict(pool="last"))]:
            b = E.build_bank(**kw)
            cos = torch.nn.functional.cosine_similarity(b[0].float(), canon[0].float(), dim=-1)
            meta[tag] = dict(max_abs_diff_to_canon=float((b - canon).abs().max()), cos_mean=float(cos.mean()), cos_min=float(cos.min()))
            if tag.startswith("pool"):
                meta[tag]["bank"] = b
        variants = {"canon": (canon, None, names), "pool_first": (meta["pool_first"].pop("bank"), None, names),
                    "pool_last": (meta["pool_last"].pop("bank"), None, names), "mask_full": (canon, E.fullmask, names)}
        for tag in ("dict_half_a", "dict_half_b"):
            keep = sorted(random.Random("bank-" + tag).sample(range(len(names)), len(names) // 2))
            variants[tag] = (canon[:, keep], None, [names[i] for i in keep])
            meta[tag] = dict(classes=len(keep))
        (out / "C_meta.json").write_text(json.dumps(meta, indent=1)); print("C banks:", json.dumps(meta), flush=True)
        d, left = todo("C", images)
        for j, f in enumerate(left):
            im = imgs.load(f, urls[f]); rows = []
            for tag, (b, m, vocab) in variants.items():
                boxes, scores = E.bank_call(im, b, m); col = {c: i for i, c in enumerate(vocab)}
                want = [c for c in targets[f] if c in col]
                rows += E.export(f, im, {c: (boxes, scores[:, col[c]]) for c in want}, want, tag)
            save(d / (f + ".json"), dict(rows=rows))
            if j % 25 == 0 or j == len(left) - 1:
                print(f"C {j + 1}/{len(left)}", flush=True)

    if "E" in parts:
        sizes = [int(k) for k in a.sizes.split(",")]
        perm = random.Random("speed-vocab").sample(names, len(names))
        setup = dict(bank_build_s=bank_setup_s, bank_bytes=int(canon.numel() * canon.element_size()),
                     model_params=sum(p.numel() for p in E.model.parameters()), sizes=sizes)
        (out / "E_setup.json").write_text(json.dumps(setup, indent=1))
        warm = imgs.load(images[0], urls[images[0]])
        for k in sizes:                                   # warm-up, not recorded
            for m in ("stock", "fixed", "bank"):
                E.measure(lambda: E.compute(m, warm, perm[:k], canon))
        d, left = todo("E", images[:a.speed_images])
        orders = [("stock", "fixed", "bank"), ("bank", "stock", "fixed"), ("fixed", "bank", "stock")]
        for j, f in enumerate(left):
            im = imgs.load(f, urls[f]); res = []
            for k in sizes:
                req = perm[:k]
                for m in orders[(j + k) % 3]:
                    _, s, mem = E.measure(lambda: E.compute(m, im, req, canon))
                    passes = 1 if m == "bank" else (len(E.g.chunks(req)) if m == "stock" else len({E.chunk_of[c] for c in req}))
                    res.append(dict(k=k, method=m, seconds=s, peak_bytes=mem, passes=passes))
            save(d / (f + ".json"), dict(image=f, timing=res))
            print(f"E {j + 1}/{len(left)}", flush=True)

    if "F" in parts:
        if E is not None:
            del E.g, E.model, E
            if dev.startswith("cuda"):
                torch.cuda.empty_cache()
        T = Engine(dev, names, mid="IDEA-Research/grounding-dino-tiny")
        T.parity(imgs.load(images[0], urls[images[0]])); print("parity check passed (tiny)", flush=True)
        t0 = time.perf_counter(); tb = T.build_bank(); (out / "F_setup.json").write_text(json.dumps(dict(bank_build_s=time.perf_counter() - t0)))
        warm = imgs.load(images[0], urls[images[0]])
        for m in ("bank", "stock", "fixed"):
            T.measure(lambda: T.compute(m, warm, request[images[0]], tb))
        import itertools
        orders = list(itertools.permutations(("bank", "stock", "fixed")))
        d, left = todo("F", images)
        for j, f in enumerate(left):
            im = imgs.load(f, urls[f]); rows = []; timing = {}
            for m in orders[j % 6]:
                result, s, mem = T.measure(lambda: T.compute(m, im, request[f], tb))
                cond = {"bank": "bank", "stock": "stock5", "fixed": "fixed"}[m]
                timing[cond] = s
                rows += T.export(f, im, result, targets[f], cond)
            save(d / (f + ".json"), dict(rows=rows, timing=timing, request=request[f]))
            if j % 25 == 0 or j == len(left) - 1:
                print(f"F {j + 1}/{len(left)}  bank {timing['bank']:.3f}s stock5 {timing['stock5']:.3f}s fixed {timing['fixed']:.3f}s", flush=True)

    if "G" in parts:
        d = out / "G"; d.mkdir(exist_ok=True); sizes = {}
        for f in images:
            im = imgs.load(f, urls[f]); sizes[f] = im.size
            if not (d / f).exists():
                t = im.copy(); t.thumbnail((512, 512)); t.save(d / f, quality=85)
        (out / "G_sizes.json").write_text(json.dumps(sizes))
        print("G thumbnails:", len(sizes), flush=True)

    zpath = out.parent / (out.name + ".zip")
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(out.rglob("*")):
            if p.is_file() and p.suffix != ".tmp":
                z.write(p, p.relative_to(out.parent))
    print("ALL DONE. Results zip:", zpath, f"({zpath.stat().st_size / 1e6:.1f} MB)", flush=True)


if __name__ == "__main__":
    main()
