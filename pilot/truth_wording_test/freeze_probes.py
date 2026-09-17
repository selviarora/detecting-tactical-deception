"""Export the four already-frozen probes and historical thresholds, without fitting."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
import numpy as np

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--repo',type=Path,required=True); a=ap.parse_args()
 out=Path(__file__).resolve().parent; sys.path.insert(0,str(a.repo/'bench'))
 from eval_probe import load_probe
 ref=[json.loads(s) for s in (a.repo/'bench/data/refusals.jsonl').read_text().splitlines()]
 ai=np.array([i for i,r in enumerate(ref) if r['category']=='alpaca'])
 rng=np.random.default_rng(20260917); rng.shuffle(ai); cal=ai[:150]
 X=np.load(a.repo/'bench/cache_llama70b/refusals_completion_L22.npz')['X'][cal].astype(np.float32)
 paths={'apollo':a.repo/'apollo_data/example_results/instructed_pairs/detector.pt',
        'corrected':a.repo/'bench/results/fixed_v2_answer_k1.pt',
        'V1':a.repo/'bench/probes/llama70b_V1_L22.npz',
        'retrained':a.repo/'pilot/detector_revision_v1/selected_detector.npz'}
 probe_dir=out/'probes'; probe_dir.mkdir(exist_ok=True)
 manifest={}
 for name,path in paths.items():
  w,mu,sc,layer=load_probe(path); assert layer==22
  b=0.0
  if name=='retrained':
   z=np.load(path); b=float(z['intercept'])
   threshold=json.loads((path.parent/'selected_detector_config.json').read_text())['thresholds']['alpaca_1pct']
  else: threshold=float(np.percentile(((X-mu)/sc)@w,99))
  cal_scores=((X-mu)/sc)@w+b
  dest=probe_dir/f'{name}.npz'
  np.savez_compressed(dest,w=w,mu=mu,sc=sc,intercept=b,layer=layer,
                      threshold=threshold,calibration_sd=float(cal_scores.std()))
  exported=np.load(dest)
  check=((X-exported['mu'])/exported['sc'])@exported['w']+float(exported['intercept'])
  assert np.allclose(check,cal_scores,rtol=0,atol=1e-6)
  manifest[name]={'file':str(dest.relative_to(out)), 'export_sha256':hashlib.sha256(dest.read_bytes()).hexdigest(),
    'source':str(path),'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
    'threshold':threshold,'calibration_sd':float(cal_scores.std()),'layer':layer,
    'status':'frozen before this diagnostic; threshold is historical nominal Alpaca 1%, not v5 calibration'}
 (out/'probes.json').write_text(json.dumps(manifest,indent=2)+'\n')
 print('Exported and verified',len(manifest),'frozen probes; no fitting.')
if __name__=='__main__':main()
