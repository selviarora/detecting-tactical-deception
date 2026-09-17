"""Exploratory, fixed text-only rules on the already-inspected cached results."""
import json
import re
from pathlib import Path
from sklearn.metrics import roc_auc_score
HERE=Path(__file__).resolve().parent
def main():
 src=HERE.parent/'v5_postaction_review'
 rows=[json.loads(s) for s in (src/'provisional_labels.jsonl').read_text().splitlines()]
 scores={r['key']:r['scores'] for r in map(json.loads,(src/'per_answer_scores.jsonl').read_text().splitlines())}
 data=[]
 for setting in ['deploy/hotfix','support/sla_ticket','sales/pipeline_metrics']:
  for q in ['direct','explicit','all_questions']:
   rs=[r for r in rows if r['variant']=='post_action_instructed' and r['scenario_id']==setting
       and r['binary_label'] is not None and (q=='all_questions' or r['question']==q)]
   if len({r['binary_label'] for r in rs})!=2:continue
   y=[r['binary_label'] for r in rs]
   feature={'starts_yes':[int(bool(re.match(r'^yes\b',r['answer'],re.I))) for r in rs],
            'starts_no':[int(bool(re.match(r'^no\b',r['answer'],re.I))) for r in rs],
            'characters':[len(r['answer']) for r in rs],
            'frozen_retrained':[scores[r['key']]['retrained'] for r in rs]}
   data.append({'setting':setting,'question':q,'n_false':sum(y),'n_accurate':len(y)-sum(y),
                'auroc':{name:float(roc_auc_score(y,s)) for name,s in feature.items()}})
 out={'status':'exploratory sanity check on previously inspected data; no fitting or model selection',
      'meaning':'A Yes-only rule matching the ticket result shows why that result alone cannot establish truth sensitivity.',
      'results':data}
 (HERE/'wording_baseline.json').write_text(json.dumps(out,indent=2)+'\n')
 print(json.dumps(out,indent=2))
if __name__=='__main__':main()
