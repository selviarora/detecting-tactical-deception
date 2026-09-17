"""Score cached completion-mean activations using the exported exploratory detector.

Example:
  python score_detector.py --acts /path/acts_L22.npz --out /path/scores.jsonl
The cache must contain X with shape (n, 8192), from Llama 3.3 70B hidden_states[22].
"""
import argparse
import json
from pathlib import Path
import numpy as np

def score(X, model):
    X=np.asarray(X,dtype=np.float64)
    if X.ndim!=2 or X.shape[1]!=len(model['w']) or not np.isfinite(X).all():
        raise ValueError('Expected finite completion-mean activations with shape (n, 8192)')
    return ((X-model['mu'])/model['sc'])@model['w']+float(model['intercept'])

def main():
    here=Path(__file__).resolve().parent
    ap=argparse.ArgumentParser();ap.add_argument('--acts',required=True,type=Path)
    ap.add_argument('--out',required=True,type=Path)
    ap.add_argument('--threshold',choices=['mixed_5pct','mixed_1pct','alpaca_1pct'],default='mixed_5pct')
    args=ap.parse_args()
    model=np.load(here/'selected_detector.npz',allow_pickle=False)
    cfg=json.loads((here/'selected_detector_config.json').read_text())
    z=np.load(args.acts,allow_pickle=False);s=score(z['X'],model)
    ids=z['keys'] if 'keys' in z else z['ids'] if 'ids' in z else np.arange(len(s))
    if len(ids)!=len(s):raise ValueError('Cache IDs and rows have different lengths')
    threshold=cfg['thresholds'][args.threshold]
    args.out.parent.mkdir(parents=True,exist_ok=True)
    with args.out.open('w') as f:
        for key,value in zip(ids,s):
            f.write(json.dumps({'id':str(key),'score':float(value),'threshold':threshold,
                'flagged':bool(value>threshold),'operating_point':args.threshold,
                'status':'exploratory; flags are not ground-truth deception labels'})+'\n')
    print(f'Scored {len(s)} rows; {int((s>threshold).sum())} flags at {args.threshold}.')

if __name__=='__main__':main()
