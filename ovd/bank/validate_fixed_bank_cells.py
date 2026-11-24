# Fixed class bank for Grounding DINO — FRESH-IMAGE VALIDATION (100 non-pilot LVIS images, 467 pairs, NVIDIA L4).
# Notebook cells as provided by the owner's expert (validate_fixed_bank.ipynb), copied from the owner's message of
# 24 Nov 2026. Colab export: '%' lines are IPython magics. Results: ovd/results/bank/bank_validation_1795488385837158854/.
# Cell 1 (markdown): Run all cells on an L4. Uses your existing Drive plan and ZIP. The first 100 non-pilot images in a
# fixed hash ordering form the validation set (or all available if fewer). All planned targets are kept. Times exclude
# file I/O, detection export, and one-time bank setup. Results include the original independent target-score metric
# for all three methods.

# ---- Cell 2 ----
from google.colab import drive
drive.mount('/content/drive')
%pip -q install ultralytics open_clip_torch ijson
from pathlib import Path
from collections import defaultdict
from contextlib import ExitStack
from unittest.mock import patch
import sys, tempfile, zipfile, json, time, hashlib
import numpy as np
import torch
from PIL import Image
from transformers.modeling_outputs import BaseModelOutput
from transformers.models.grounding_dino import modeling_grounding_dino as gd

root = Path('/content/drive/MyDrive/ovd')
archive = root / 'bundle_20261124_012119.zip'
tmp = Path(tempfile.mkdtemp())
with zipfile.ZipFile(archive) as z:
    for name in z.namelist():
        if name.startswith('bundle/code/') and name.endswith('.py'):
            path = (tmp / name).resolve()
            assert path.is_relative_to(tmp)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(z.read(name))
    plan = json.loads(z.read('bundle/plan_100pairs.json'))
    fixed = [json.loads(line) for line in z.read('bundle/fixed_chunk_dets_100pairs.jsonl').splitlines()]
sys.path.insert(0, str(tmp / 'bundle/code'))
from vocab_full import GDino
from full_analysis import APTable
assert torch.cuda.is_available() and 'L4' in torch.cuda.get_device_name(), 'Select L4 under Runtime → Change runtime type.'
g = GDino('cuda')
model = g.model
names = plan['names']
index = {c:i for i,c in enumerate(names)}
assert len(index) == len(names) == 1203
import ijson
import itertools
import random
import transformers

pilot_images = set(plan['info'])
full_path = root/'plan_lvis.json'
with full_path.open('rb') as fh:
    assert list(ijson.items(fh,'names.item')) == names
eligible = defaultdict(list)
with full_path.open('rb') as fh:
    for p in ijson.items(fh,'plan.item'):
        if p['image'] not in pilot_images:
            eligible[p['image']].append({k:p[k] for k in ('image','target','present')})
ordered = sorted(eligible, key=lambda f: hashlib.sha256(('bank-validation-v1/'+f).encode()).digest())
chosen = ordered[:100]
assert chosen, 'The original plan has no non-pilot images. Supply a fresh LVIS plan.'
with full_path.open('rb') as fh:
    info = {f:v for f,v in ijson.kvitems(fh,'info',use_float=True) if f in set(chosen)}
plan = dict(names=names,info=info,plan=[p for f in chosen for p in eligible[f]])
pairs = {f:[p['target'] for p in eligible[f]] for f in chosen}
assert set(info)==set(chosen) and not set(chosen)&pilot_images
assert all(0<len(cs)<=5 and len(set(cs))==len(cs) for cs in pairs.values())
assert all((root/'images'/f).is_file() for f in chosen), 'Missing images in MyDrive/ovd/images'
run = root/('bank_validation_'+str(time.time_ns()))
run.mkdir()
(run/'manifest.json').write_text(json.dumps(dict(images=chosen,plan=plan['plan'],pilot_excluded=sorted(pilot_images),
    gpu=torch.cuda.get_device_name(),torch=str(torch.__version__),transformers=transformers.__version__,
    checkpoint=getattr(g.model.config,'_commit_hash',None), bundle_sha256=hashlib.sha256(archive.read_bytes()).hexdigest())))
print(len(chosen),'fresh images,',len(plan['plan']),'pairs. Output:',run)
def image(f):
    with Image.open(root/'images'/f) as im:
        return im.convert('RGB')
def caption(cs):
    return ' . '.join(cs) + ' .'
def inputs(im,cs):
    return g.proc(images=im,text=caption(cs),return_tensors='pt').to(g.dev)
gm = gd.generate_masks_with_special_tokens_and_transfer_map

# ---- Cell 3 ----
@torch.inference_mode()
def cached_text(ids):
    mask,pos=gm(ids)
    bm=mask[:,None] if mask.ndim==3 else mask
    return model.model.text_backbone(ids,bm,torch.zeros_like(ids),pos,
                                     return_dict=True).last_hidden_state

def injected(hidden,mask,pos):
    # All modifications are scoped to one inference and restored on exit.
    ctx=ExitStack()
    try:
        ctx.enter_context(patch.object(gd,'generate_masks_with_special_tokens_and_transfer_map',
                                       return_value=(mask,pos)))
        ctx.enter_context(patch.object(model.model.text_backbone,'forward',
                                       return_value=BaseModelOutput(last_hidden_state=hidden)))
        length=max(256,hidden.shape[1])
        owners=[model.config,model.model.config,*model.modules()]
        visited=set()
        for obj in owners:
            if id(obj) not in visited and isinstance(getattr(obj,'max_text_len',None),int):
                ctx.enter_context(patch.object(obj,'max_text_len',length))
                visited.add(id(obj))
        return ctx
    except BaseException:
        ctx.close()
        raise

small=names[:5]
first=image(sorted(pairs)[0])
x=inputs(first,small)
with torch.inference_mode():
    stock=model(**x)
    h=cached_text(x.input_ids)
    mask,pos=gm(x.input_ids)
    with injected(h,mask,pos):
        ctrl=model(**x)
    torch.testing.assert_close(stock.pred_boxes,ctrl.pred_boxes,atol=1e-5,rtol=1e-5)
    torch.testing.assert_close(stock.logits,ctrl.logits,atol=1e-5,rtol=1e-5)
print('Passed: cached-text control reproduces native boxes and token scores.')

@torch.inference_mode()
def build_bank():
    pieces=[]
    for chunk in g.chunks(names):
        ids=g.tok(caption(chunk),return_tensors='pt')['input_ids'].to(g.dev)
        assert ids.shape[1]<=256
        tokens=cached_text(ids)
        at=1
        for c in chunk:
            sub=g.tokens(c)
            assert ids[0,at:at+len(sub)].tolist()==sub
            pieces.append(tokens[:,at:at+len(sub)].mean(dim=1,keepdim=True))
            at+=len(sub)+1
        assert at==ids.shape[1]-1
    return torch.cat(pieces,dim=1)
bank=build_bank()
n=len(names)
assert bank.shape[1]==n
ids=torch.full((1,n),g.tok.convert_tokens_to_ids('.'),device=g.dev,dtype=torch.long)
identity=torch.eye(n,dtype=torch.bool,device=g.dev)[None]
positions=torch.zeros_like(ids)
ones=torch.ones_like(ids)
zeros=torch.zeros_like(ids)
def bank_call(im):
    v=g.proc.image_processor(images=im,return_tensors='pt').to(g.dev)
    o=model(**v,input_ids=ids,token_type_ids=zeros,attention_mask=ones)
    assert o.logits.shape[-1]>=n and torch.isfinite(o.logits[...,:n]).all()
    return o.pred_boxes[0],o.logits[0,:,:n].sigmoid()

def stock_call(im):
    o=model(**inputs(im,small))
    return o.pred_boxes[0],o.logits[0].sigmoid()

def measure(fn,im):
    g._bb.clear()  # each method pays for its own backbone on each image
    torch.cuda.synchronize()
    start=time.perf_counter()
    result=fn(im)
    torch.cuda.synchronize()
    return result,time.perf_counter()-start

# ---- Cell 4 ----
chunks=g.chunks(names)
chunk_of={c:i for i,ch in enumerate(chunks) for c in ch}
def request5(f):
    cs=pairs[f].copy()
    rng=random.Random('fixed-bank-five/'+f)
    cs+=rng.sample([n for n in names if n not in cs],5-len(cs))
    rng.shuffle(cs)
    return cs

def decode_prompt(im,cs):
    o=model(**inputs(im,cs))
    logits=o.logits[0]
    offset=1
    scores=[]
    for c in cs:
        width=len(g.tokens(c))
        scores.append(logits[:,offset:offset+width].amax(1))
        offset+=width+1
    return o.pred_boxes[0],torch.stack(scores,dim=1).sigmoid()

def report(boxes,scores,cs,request):
    return {c:(boxes,scores[:,cs.index(c)]) for c in request}
def compute(method,im,request):
    if method=='bank':
        boxes,scores=bank_call(im)
        return {c:(boxes,scores[:,index[c]]) for c in request}
    if method=='stock5':
        return report(*decode_prompt(im,request),request,request)
    found={}
    for i in sorted({chunk_of[c] for c in request}):
        ch=chunks[i]
        found.update(report(*decode_prompt(im,ch),ch,[c for c in request if chunk_of[c]==i]))
    return found

def export(f,im,result,method):
    W,H=im.size
    rows=[]
    for c in pairs[f]:
        box,s=result[c]
        xy,wh=box[:,:2],box[:,2:]
        b=(torch.cat([xy-wh/2,xy+wh/2],1)*box.new_tensor([W,H,W,H])).float().cpu().numpy()
        s=s.float().cpu().numpy()
        dets=[[float(s[i]),*b[i].tolist()] for i in np.argsort(-s)[:50] if s[i]>=.02]
        rows.append(dict(image=f,target=c,cond=method,dets=dets))
    return rows

methods=('bank','stock5','fixed')
orders=list(itertools.permutations(methods))
records=[]
times=[]
from contextlib import nullcontext
with torch.inference_mode():
    # Compare export columns on the actual request path, including reordering.
    for method in methods:
        with (injected(bank,identity,positions) if method=='bank' else nullcontext()):
            g._bb.clear()
            compute(method,first,small)
    with injected(bank,identity,positions):
        probe=names[:200]
        a=compute('bank',first,[small[0]])[small[0]]
        b=compute('bank',first,list(reversed(probe)))[small[0]]
        for u,v in zip(a,b):
            torch.testing.assert_close(u,v,atol=1e-5,rtol=1e-5)
    for j,f in enumerate(chosen):
        im=image(f)
        req=request5(f)
        t=dict(image=f,chunks=len({chunk_of[c] for c in req}),request=req)
        rows=[]
        for method in orders[j%6]:
            with (injected(bank,identity,positions) if method=='bank' else nullcontext()):
                result,elapsed=measure(lambda img:compute(method,img,req),im)
            t[method]=elapsed
            rows+=export(f,im,result,method)
        with (run/'per_image.jsonl').open('a') as fh:
            fh.write(json.dumps(dict(timing=t,rows=rows),allow_nan=False)+'\n')
        records+=rows
        times.append(t)
        print(f'{j+1}/{len(chosen)}  bank {t["bank"]:.3f}s  stock {t["stock5"]:.3f}s  chunks {t["fixed"]:.3f}s',flush=True)

table=APTable(records,plan,'indep')
ap={m:table.map(m) for m in methods}
avg={m:float(np.mean([t[m] for t in times])) for m in methods}
delta=ap['bank']-ap['fixed']
rng=np.random.default_rng(20261124)
diff=[]
for _ in range(1000):
    w=np.bincount(rng.integers(len(chosen),size=len(chosen)),minlength=len(chosen))
    diff.append(table.map('bank',w)-table.map('fixed',w))
ci=np.nanpercentile(diff,[2.5,97.5]).tolist()
summary=dict(images=len(chosen),pairs=len(plan['plan']),gpu=torch.cuda.get_device_name(),
             mAP50=ap,bank_minus_fixed=delta,paired_bootstrap_95pct=ci,
             within_one_point=delta>=-.01,mean_seconds=avg,
             bank_over_stock5=avg['bank']/avg['stock5'],fixed_over_bank=avg['fixed']/avg['bank'])
(run/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
print(json.dumps(summary,indent=2))
print('Saved',run/'summary.json')
