"""Usage: python request_audit.py supplement_anonymous.zip [output.json]"""
import io,json,sys,zipfile
from pathlib import Path
import numpy as np

def analyse(archive):
    root = 'supplement/results/bank/'
    with zipfile.ZipFile(archive) as z:
        manifest=json.loads(z.read(root+'fixed_bank_soft_confirm_v1/manifest.json'))
        nested=z.read(root+'followup_t4_v1/followup_v1_data.zip')
    labels={(r['image'],r['target']):r['present'] for r in manifest['targets']}
    scores={}
    with zipfile.ZipFile(io.BytesIO(nested)) as follow:
        for name in follow.namelist():
            if '/B/' not in name or not name.endswith('.json'):continue
            for r in json.loads(follow.read(name))['rows']:
                if r['cond'] in ('stock5_indA','stock5_indB'):
                    key=r['image'],r['target']
                    pair=scores.setdefault(key,{})
                    assert r['cond'] not in pair
                    pair[r['cond']]=max((d[0] for d in r['dets']),default=0.)
    assert scores.keys()==labels.keys() and all(len(p)==2 for p in scores.values())
    images=sorted(manifest['images'])
    image_id={f:i for i,f in enumerate(images)}
    rng=np.random.default_rng(248)
    draws=rng.integers(len(images),size=(3000,len(images)))
    result={}
    for present in (True,False):
        keys=[k for k in labels if labels[k]==present]
        ids=np.array([image_id[k[0]] for k in keys])
        top=np.array([[scores[k][c] for c in ('stock5_indA','stock5_indB')] for k in keys])
        den=np.bincount(ids,minlength=len(images))
        rows=[]
        for t in (.20,.25,.30,.35,.40,.45,.50):
            a,b=top.T>=t
            diff=a^b
            num=np.bincount(ids[diff],minlength=len(images))
            pct=100*num[draws].sum(axis=1)/den[draws].sum(axis=1)
            rows.append(dict(threshold=t,pairs=len(keys),a_only=int((a & ~b).sum()),b_only=int((b & ~a).sum()),flips=int(diff.sum()),percent=100*diff.mean(),ci95=np.percentile(pct,[2.5,97.5]).tolist()))
        result['present' if present else 'absent']=rows
    assert result['present'][3]['flips']==132
    return result
if __name__=='__main__':
    out=json.dumps(analyse(sys.argv[1]),indent=2)
    if len(sys.argv)>2:Path(sys.argv[2]).write_text(out)
    else:print(out)
