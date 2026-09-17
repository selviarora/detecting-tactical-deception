"""CPU-only checks of balancing, cancellation and exact Llama readout indexing."""
import itertools
import hashlib
import json
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score
from build_cases import build
from score import analyze
from capture import capture_vectors, checkpoint

def main():
 rows=build();assert len(rows)==96 and len({r['id'] for r in rows})==96
 y=np.array([r['label_false'] for r in rows]); assert y.sum()==48
 cells=defaultdict(list)
 for r in rows:cells[(r['setting'],r['note'],r['question_polarity'],r['answer_form'])].append(r)
 assert len(cells)==24
 for rs in cells.values():
  assert len(rs)==4
  for yes in [False,True]:
   rr=[r for r in rs if r['answer_yes']==yes]
   assert len({r['assistant'] for r in rr})==1 and {r['label_false'] for r in rr}=={0,1}
  for world in [False,True]:
   rr=[r for r in rs if r['world_compliant']==world]
   assert rr[0]['messages']==rr[1]['messages']
 # Because each report text occurs equally in true and false worlds, any fixed
 # answer-only mapping has balanced AUROC; length and Yes are simple examples.
 assert roc_auc_score(y,[int(r['answer_yes']) for r in rows])==0.5
 assert roc_auc_score(y,[len(r['assistant']) for r in rows])==0.5
 assert roc_auc_score(y,[int(r['world_compliant']) for r in rows])==0.5
 additive=np.array([3*int(r['answer_yes'])+7*int(r['world_compliant'])+2*int(r['note']=='neutral') for r in rows])
 p={'dummy':{'threshold':0.5,'calibration_sd':1.0}}
 result=analyze(rows,{'dummy':additive},p)['dummy']
 assert all(abs(r['mean_false_minus_true'])<1e-12 for r in result['factorial_blocks'])
 truth=analyze(rows,{'dummy':y.astype(float)},p)['dummy']
 assert all(r['mean_false_minus_true']==1 for r in truth['factorial_blocks'])
 # Verify the hook corresponds exactly to hidden_states[layer], with locally
 # instantiated random weights. No model download and no 70B inference.
 import torch
 from transformers import LlamaConfig,LlamaForCausalLM
 torch.manual_seed(7)
 config=LlamaConfig(vocab_size=40,hidden_size=32,intermediate_size=48,num_hidden_layers=3,
                    num_attention_heads=4,num_key_value_heads=2,max_position_embeddings=64)
 config._attn_implementation='eager'
 model=LlamaForCausalLM(config).eval()
 enc={'input_ids':torch.tensor([[2,5,7,3,11,17]]),'attention_mask':torch.ones((1,6),dtype=torch.long)}
 with torch.inference_mode(): expected=model(**enc,output_hidden_states=True,use_cache=False).hidden_states[2][0]
 av,pv=capture_vectors(model,enc,4,2)
 np.testing.assert_allclose(av,expected[4:].float().mean(0).numpy(),rtol=0,atol=1e-7)
 np.testing.assert_allclose(pv,expected[3].float().numpy(),rtol=0,atol=1e-7)
 with tempfile.TemporaryDirectory() as t:
  d=Path(t)
  first=checkpoint(d,['a'],[av],[pv],[2],{'test':True},{'test':True},2)
  assert first['status']=='partial'
  second=checkpoint(d,['a','b'],[av,av],[pv,pv],[2,2],{'test':True},{'test':True},2)
  z=np.load(d/'activations.npz');assert list(z['ids'])==['a','b']
  assert second['status']=='complete' and json.loads(str(z['metadata']))['n_done']==2
 # Run the real scoring CLI on a disposable synthetic cache, then verify that
 # it refuses the same cache marked incomplete. These are harness tests only.
 here=Path(__file__).resolve().parent
 with tempfile.TemporaryDirectory() as t:
  d=Path(t)
  sig={'case_sha256':hashlib.sha256((here/'cases.jsonl').read_bytes()).hexdigest(),
       'model':'meta-llama/Llama-3.3-70B-Instruct','layer':22,
       'probes_manifest_sha256':hashlib.sha256((here/'probes.json').read_bytes()).hexdigest()}
  meta={'signature':sig,'status':'complete','expected_rows':96,'n_done':96,'runtime':{'synthetic_test':True}}
  def fixture():
   np.savez_compressed(d/'fixture.npz',ids=np.array([r['id'] for r in rows]),X=np.zeros((96,8192),np.float32),
                       X_prompt=np.zeros((96,8192),np.float32),metadata=np.array(json.dumps(meta)))
  fixture()
  cmd=[sys.executable,str(here/'score.py'),'--capture',str(d/'fixture.npz'),'--out',str(d/'scored')]
  done=subprocess.run(cmd,capture_output=True,text=True)
  assert done.returncode==0,done.stderr
  assert len(json.loads((d/'scored/results.json').read_text())['analyses'])==4
  meta['status']='partial';fixture()
  rejected=subprocess.run(cmd,capture_output=True,text=True)
  assert rejected.returncode!=0, 'Partial cache should not be scored'
 report={'case_balance':'pass','same_answer_different_truth':'pass','same_prefix_answer_alternatives':'pass',
         'answer_only_baseline_auroc':0.5,'state_only_baseline_auroc':0.5,'additive_confound_cancellation':'pass',
         'truth_signal_positive_control':'pass','llama_hook_matches_hidden_states':'pass',
         'partial_complete_atomic_checkpoint':'pass','scoring_cli_disposable_fixture':'pass',
         'partial_cache_rejected':'pass','gpu_used':False,'model_downloaded':False,
         'note':'Readout checked on a tiny random local Llama, not the 70B model.'}
 Path(__file__).with_name('verification.json').write_text(json.dumps(report,indent=2)+'\n')
 print(json.dumps(report,indent=2))
if __name__=='__main__':main()
