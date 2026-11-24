# Fixed class bank for Grounding DINO — DEVELOPMENT PILOT (24 images / 100 pairs).
# Notebook cells as provided by the owner's expert (fixed_bank_l4.ipynb), copied verbatim from the owner's message
# of 27 Nov 2026. Colab notebook export: lines starting with '%' are IPython magics. Result file:
# ovd/results/bank/fixed_bank_1795484911264138629.json (run on a Tesla T4).
#
# Cell 1 (markdown): One fixed visual encoder pass over 1,203 class embeddings. Use the same L4 and Drive as the previous
# run. Each LVIS class gets one vector, obtained by averaging its contextualized BERT wordpieces. All vectors enter
# bidirectional fusion together. The detector uses the same fixed bank for every image and all user lists. This changes
# how phrases and separators are represented. Accuracy and latency are experimental. The 100 pairs have already been
# used for method development. Run all cells. A short-prompt parity check precedes the experiment. The final cell
# prints matched AP50, cold-image latency and a latency ratio versus the stock five-class call. Timings cover
# preprocessing through score tensors and exclude disk I/O and evaluation exports.

# ---- Cell 2 ----
from google.colab import drive
drive.mount('/content/drive')
%pip -q install ultralytics open_clip_torch
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
assert torch.cuda.is_available(), 'Select an L4 GPU runtime.'
g = GDino('cuda')
model = g.model
names = plan['names']
index = {c:i for i,c in enumerate(names)}
assert len(index) == len(names) == 1203
pairs = defaultdict(list)
for row in fixed:
    pairs[row['image']].append(row['target'])
assert {(f,c) for f,cs in pairs.items() for c in cs} == {(p['image'],p['target']) for p in plan['plan']}
assert len(fixed)==100 and len(pairs)==24
def image(f):
    with Image.open(root/'images'/f) as im:
        return im.convert('RGB')
def caption(cs):
    return ' . '.join(cs) + ' .'
def inputs(im,cs):
    return g.proc(images=im,text=caption(cs),return_tensors='pt').to(g.dev)
gm = gd.generate_masks_with_special_tokens_and_transfer_map
print(torch.cuda.get_device_name(), 'fixed AP50', APTable(fixed,plan,'indep').map('FIX'))

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
records=[]
times=[]
with torch.inference_mode():
    measure(stock_call,first)
    with injected(bank,identity,positions):
        measure(bank_call,first)
    for j,f in enumerate(sorted(pairs)):
        im=image(f)
        # Alternate execution order. The context installs a session-constant bank.
        def wide():
            with injected(bank,identity,positions):
                return measure(bank_call,im)
        if j%2:
            (boxes,sc),tw=wide()
            _,ts=measure(stock_call,im)
        else:
            _,ts=measure(stock_call,im)
            (boxes,sc),tw=wide()
        W,H=im.size
        xy,wh=boxes[:,:2],boxes[:,2:]
        b=torch.cat([xy-wh/2,xy+wh/2],dim=-1)
        b=(b*b.new_tensor([W,H,W,H])).float().cpu().numpy()
        for c in pairs[f]:
            s=sc[:,index[c]].float().cpu().numpy()
            top=np.argsort(-s)[:50]
            dets=[[float(s[i]),*b[i].tolist()] for i in top if s[i]>=.02]
            records.append(dict(image=f,target=c,cond='FIX',dets=dets))
        times.append(dict(image=f,stock5_s=ts,bank_s=tw))
        print(j+1,'/ 24',f,'stock',round(ts,3),'bank',round(tw,3),flush=True)

score=APTable(records,plan,'indep').map('FIX')
baseline=APTable(fixed,plan,'indep').map('FIX')
wide_s=float(np.mean([x['bank_s'] for x in times]))
stock_s=float(np.mean([x['stock5_s'] for x in times]))
summary=dict(bank_mAP50=score,fixed_mAP50=baseline,accuracy_threshold=baseline-.01,
             within_one_point=score>=baseline-.01,bank_s=wide_s,stock5_s=stock_s,
             latency_ratio=wide_s/stock_s,images=24,pairs=100,
             bundle_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
             gpu=torch.cuda.get_device_name(),torch=str(torch.__version__))
print(json.dumps(summary,indent=2))
dest=root/('fixed_bank_'+str(time.time_ns())+'.json')
dest.write_text(json.dumps(dict(summary=summary,times=times,records=records),allow_nan=False))
print('Saved',dest)
