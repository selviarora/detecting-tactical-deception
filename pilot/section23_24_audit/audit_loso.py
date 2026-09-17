"""Reproduce §23 and re-evaluate its frozen folds against prior semantic review.
No selection among candidates; the C grid is reported in full.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits

ROOT=Path('/Users/selviarora/concealment-vs-override');P=ROOT/'pilot';OUT=Path(__file__).resolve().parent
def read(p):return [json.loads(s) for s in p.read_text().splitlines() if s.strip()]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def auc(y,s):return float(roc_auc_score(y,s)) if len(set(y))==2 else None

def main():
 cases=read(P/'truth_wording_test/cases.jsonl');cache=np.load(P/'truth_wording_test/capture_llama/activations.npz')
 ids=list(map(str,cache['ids']));byid={r['id']:r for r in cases};rows=[byid[i] for i in ids]
 X=cache['X'].astype(np.float32);y=np.array([r['label_false'] for r in rows]);setting=np.array([r['setting'] for r in rows])
 semantic=read(OUT.parent/'v5_postaction_review/provisional_labels.jsonl')
 Xgen={}
 for run in ['run_v5.1_70b','run_v5.1_70b_postinstr']:
  z=np.load(P/run/'acts_L22.npz');rr=read(P/run/'answers.jsonl');assert set(z['keys'])=={r['key'] for r in rr}
  Xgen.update(zip(map(str,z['keys']),z['X'].astype(np.float32)))
 names={'deploy':'deploy/hotfix','pipeline':'sales/pipeline_metrics','tickets':'support/sla_ticket'}
 folds={}
 with threadpool_limits(limits=2):
  for C in [.01,.1,1.]:
   for s,sid in names.items():
    f=np.load(P/f'reportstate_loso/fold_C{C}_{s}.npz');tr=setting!=s;te=~tr
    mu=X[tr].mean(0);sd=X[tr].std(0)+1e-6
    np.testing.assert_array_equal(f['mu'],mu);np.testing.assert_array_equal(f['sc'],sd)
    assert list(f['supplied_ids'])==list(np.array(ids)[te])
    model=LogisticRegression(C=C,max_iter=5000,class_weight='balanced').fit((X[tr]-mu)/sd,y[tr])
    w=model.coef_[0].astype(np.float32);b=float(model.intercept_[0])
    max_w=float(np.max(abs(w-f['w'])));assert max_w<1e-5
    ss=((X[te]-mu)/sd)@w+b;assert np.allclose(ss,f['supplied_scores'],rtol=1e-5,atol=1e-5)
    kg=list(map(str,f['generated_keys']));yg=f['generated_labels'];gscore=((np.stack([Xgen[k] for k in kg])-mu)/sd)@f['w']+float(f['intercept'])
    assert np.allclose(gscore,f['generated_scores'],rtol=1e-5,atol=1e-5)
    instr=np.array(['#postinstr#' in k for k in kg],dtype=int)
    sr=[r for r in semantic if r['scenario_id']==sid];SX=np.stack([Xgen[r['key']] for r in sr]);sp=((SX-mu)/sd)@f['w']+float(f['intercept'])
    revised=[]
    for variant in ['natural','post_action_instructed','pooled_diagnostic']:
     for q in ['broad','explicit','direct','all_questions']:
      ix=[i for i,r in enumerate(sr) if r['binary_label'] is not None and
          (variant=='pooled_diagnostic' or r['variant']==variant) and (q=='all_questions' or r['question']==q)]
      yy=[sr[i]['binary_label'] for i in ix]
      revised.append({'variant':variant,'question':q,'n_false':sum(yy),'n_accurate':len(yy)-sum(yy),'auroc':auc(yy,sp[ix])})
    prior={r['key']:r for r in semantic}; disagreements=[]
    for k,oldlab in zip(kg,yg):
     rr=prior[k]
     if rr['binary_label'] is None or rr['binary_label']!=int(oldlab):
      disagreements.append({'key':k,'keyword_label':int(oldlab),'semantic_label':rr['review_label'],'answer':rr['answer']})
    folds[f'C{C}_{s}']={'C':C,'held_out':s,'train_rows':int(tr.sum()),'test_rows':int(te.sum()),
      'no_setting_overlap':True,'train_scaler_exact_match':True,'max_refit_weight_error':max_w,
      'supplied_auroc':auc(y[te],ss),'keyword_generated_auroc':auc(yg,gscore),
      'instruction_only_generated_auroc':auc(yg,instr),
      'keyword_natural_negative_count':int(((instr==0)&(yg==0)).sum()),
      'keyword_post_negative_count':int(((instr==1)&(yg==0)).sum()),
      'keyword_post_positive_count':int(((instr==1)&(yg==1)).sum()),
      'provisional_semantic_metrics':revised,'keyword_semantic_disagreements':disagreements}
    np.savez_compressed(OUT/f'audited_fold_C{C}_{s}.npz',w=f['w'],mu=mu,sc=sd,intercept=float(f['intercept']),
                        supplied_ids=np.array(ids)[te],supplied_scores=ss,
                        generated_keys=np.array([r['key'] for r in sr]),generated_scores=sp)
 out={'source_script_sha256':sha(P/'train_reportstate_loso.py'),'source_results_sha256':sha(P/'reportstate_loso/results.json'),
      'status':'Reproduction and exploratory semantic sensitivity audit; no candidate selected.',
      'label_status':'Previously saved one-auditor provisional semantic review, not independent human consensus.',
      'folds':folds,'all96_export_missing_intercept':'intercept' not in np.load(ROOT/'bench/probes/llama70b_reportstate_L22.npz').files}
 (OUT/'loso_audit.json').write_text(json.dumps(out,indent=2)+'\n')
 for k,f in folds.items():
  if f['C']==.1:
   print(json.dumps({x:f[x] for x in ['held_out','supplied_auroc','keyword_generated_auroc','instruction_only_generated_auroc','keyword_natural_negative_count','keyword_post_negative_count','keyword_post_positive_count','max_refit_weight_error']}))
   print('SEMANTIC',json.dumps([r for r in f['provisional_semantic_metrics'] if r['variant']!='natural' and (r['question']=='all_questions' or r['auroc'] is not None)]))
   print('DISAGREEMENTS',json.dumps(f['keyword_semantic_disagreements']))
if __name__=='__main__':main()
