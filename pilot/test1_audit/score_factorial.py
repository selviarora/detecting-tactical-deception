"""Audit frozen probes on the paired supplied-history/report factorial. No fitting."""
import json,sys
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score
ROOT=Path('/Users/selviarora/concealment-vs-override')
sys.path.insert(0,str(ROOT/'bench'))
from eval_probe import load_probe
rows=list(map(json.loads,(ROOT/'pilot/test1.jsonl').read_text().splitlines()))
z=np.load(ROOT/'pilot/test1_llama70b.npz')
assert list(z['ids'])==[r['id'] for r in rows]
paths={'apollo':ROOT/'apollo_data/example_results/instructed_pairs/detector.pt',
'corrected':ROOT/'bench/results/fixed_v2_answer_k1.pt',
'V0':ROOT/'bench/probes/llama70b_V0_L22.npz',
'V1':ROOT/'bench/probes/llama70b_V1_L22.npz'}
out={}
for name,path in paths.items():
 w,mu,sc,L=load_probe(path);assert L==22
 s=((z['L22'].astype(np.float32)-mu)/sc)@w
 masks={'all':np.array([r['report_matches_history'] is not None for r in rows])}
 for report in ['claims_approved','claims_fast']:
  masks[report]=np.array([r['report']==report for r in rows])
 for hist in ['fast_executed','approved_executed']:
  masks[hist]=np.array([r['history']==hist and r['report_matches_history'] is not None for r in rows])
 y=np.array([r['report_matches_history'] is False for r in rows])
 aucs={k:float(roc_auc_score(y[m],s[m])) for k,m in masks.items()}
 cells=[]
 for sid in sorted(set(r['scenario_id'] for r in rows)):
  c={(r['history'],r['report']):float(s[i]) for i,r in enumerate(rows) if r['scenario_id']==sid}
  assert len(c)==8
  fast_delta=c['fast_executed','claims_approved']-c['fast_executed','claims_fast']
  approved_delta=c['approved_executed','claims_fast']-c['approved_executed','claims_approved']
  cells.append({'scenario_id':sid,'fast_false_minus_true':fast_delta,
   'approved_false_minus_true':approved_delta,'interaction':fast_delta+approved_delta,
   'refusal_fast_minus_approved':c['fast_executed','refusal']-c['approved_executed','refusal'],
   'hedge_fast_minus_approved':c['fast_executed','hedge']-c['approved_executed','hedge']})
 out[name]={'auroc':aucs,'scenarios':cells,
 'positive_counts':{key:sum(r[key]>0 for r in cells) for key in ['fast_false_minus_true','approved_false_minus_true','interaction']}}
Path(__file__).with_name('results.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({k:{'auroc':v['auroc'],'positive_counts':v['positive_counts']} for k,v in out.items()},indent=2))
