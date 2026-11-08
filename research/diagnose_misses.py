#!/usr/bin/env python3
"""Diagnostics behind Section 3.2 of MANUSCRIPT.md (first repetition of each fall). Usage: python3 research/diagnose_misses.py /path/to/SisFall_dataset"""
import sys, numpy as np
sys.path.insert(0, __import__('os').path.dirname(__file__))
from benchmark_sisfall import *
root=sys.argv[1]; fs=50; L=75
mins=[]; times=[]
for act,subj,rep,path in iter_dataset(root):
    if act[0]!='F' or rep!=1: continue
    mag,rot=load_trial(path,fs); base=mag[:100].mean()
    F=window_features(mag,rot,L,base)
    ok=F[:,0]>24
    mins.append(F[ok,2].min() if ok.any() else np.nan)
    gi=np.argmax(mag); post=np.abs(mag[gi:]-base)
    settle=np.argmax(post[5:]<=1.5)+5 if (post[5:]<=1.5).any() else -1
    times.append(settle/fs)
mins=np.array(mins); print('min stillness among windows w/ peak>24: pct',np.nanpercentile(mins,[10,25,50,75,90]))
t=np.array(times); print('time from peak to first |a-g|<=1.5 (s): pct',np.percentile(t[t>=0],[10,25,50,75,90]), 'never',np.mean(t<0))
