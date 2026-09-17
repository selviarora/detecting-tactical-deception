"""Forward-only supplied-answer capture; resumable, one model load, no generation.

python capture.py --out capture_llama --dry-run
python capture.py --out smoke --max-rows 4
python capture.py --out capture_llama
"""
import argparse
import hashlib
import json
import os
import platform
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read_cases(path): return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]

def encode_answer(tok, row):
 """Reject boundary merges; never silently include prompt tokens in the readout."""
 prefix=tok.apply_chat_template(row['messages'],tokenize=False,add_generation_prompt=True)
 prefix_ids=tok(prefix,add_special_tokens=False)['input_ids']
 enc=tok(prefix+row['assistant'],add_special_tokens=False,return_tensors='pt',return_offsets_mapping=True)
 n=len(prefix_ids); ids=enc['input_ids'][0].tolist(); offsets=enc.pop('offset_mapping')[0].tolist()
 if ids[:n]!=prefix_ids or len(ids)<=n:
  raise ValueError(f"Prompt/answer token boundary changed for {row['id']}; stop and fix the readout explicitly.")
 if any(end>len(prefix) for start,end in offsets[:n]) or any(start<len(prefix) for start,end in offsets[n:]):
  raise ValueError(f"Token crosses the answer boundary for {row['id']}")
 return enc,n

def capture_vectors(model, enc, prefix_length, layer):
 """Llama hidden_states[layer] is output of decoder block layer-1 for internal layers."""
 import torch
 if not 1<=layer<len(model.model.layers):
  raise ValueError('Only internal layers supported; final-layer normalization needs a different readout.')
 saved={}
 def hook(module, inputs, output):
  h=output[0] if isinstance(output,tuple) else output
  answer=h[0,prefix_length:,:]
  if not len(answer): raise ValueError('Empty answer span')
  saved['answer']=answer.float().mean(0).cpu().numpy()
  saved['prompt']=h[0,prefix_length-1,:].float().cpu().numpy()
 handle=model.model.layers[layer-1].register_forward_hook(hook)
 try:
  with torch.inference_mode():
   model.model(**enc,use_cache=False,return_dict=True)
 finally: handle.remove()
 if set(saved)!={'answer','prompt'}:raise RuntimeError('Capture hook did not run')
 return saved['answer'],saved['prompt']

def checkpoint(out, ids, answer, prompt, lengths, signature, runtime, expected):
 complete=len(ids)==expected
 meta={'signature':signature,'runtime':runtime,'n_done':len(ids),'expected_rows':expected,
       'status':'complete' if complete else 'partial','activation_dtype':'float32',
       'readout':'mean over supplied answer tokens; no prompt tokens or EOS; hidden_states[22]'}
 tmp=out/'activations.tmp.npz'
 with tmp.open('wb') as f:
  np.savez_compressed(f,ids=np.array(ids),X=np.stack(answer),X_prompt=np.stack(prompt),
                      answer_tokens=np.array(lengths),metadata=np.array(json.dumps(meta,sort_keys=True)))
 os.replace(tmp,out/'activations.npz')
 tmp_json=out/'capture.tmp.json'; tmp_json.write_text(json.dumps(meta,indent=2)+'\n')
 os.replace(tmp_json,out/'capture.json')
 return meta

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--cases',type=Path,default=HERE/'cases.jsonl')
 ap.add_argument('--model',default='meta-llama/Llama-3.3-70B-Instruct');ap.add_argument('--revision',default='main')
 ap.add_argument('--out',type=Path,required=True); ap.add_argument('--resume',action='store_true')
 ap.add_argument('--max-rows',type=int);ap.add_argument('--checkpoint-every',type=int,default=8)
 ap.add_argument('--dry-run',action='store_true');a=ap.parse_args()
 rows=read_cases(a.cases); manifest=json.loads((HERE/'manifest.json').read_text())
 assert digest(a.cases)==manifest['cases_sha256'] and len(rows)==manifest['rows']==96
 assert len({r['id'] for r in rows})==len(rows)
 if a.model!=manifest['model']:raise ValueError('This protocol freezes Llama-3.3-70B-Instruct; do not change the base model.')
 if a.max_rows is not None and not 1<=a.max_rows<=len(rows):raise ValueError('Invalid --max-rows')
 if a.checkpoint_every<1:raise ValueError('Invalid checkpoint interval')
 if a.dry_run:
  print(json.dumps({'mode':'dry_run','model_loaded':False,'rows':len(rows),'selected_rows':a.max_rows or len(rows),
    'case_hash_verified':True,'output':str(a.out),'requires_gpu_for_capture':True},indent=2));return
 signature={'case_sha256':digest(a.cases),'model':a.model,'requested_revision':a.revision,
            'layer':22,'script_sha256':digest(__file__),'probes_manifest_sha256':digest(HERE/'probes.json')}
 ids=[];answer=[];prompt=[];lengths=[];old_runtime=None
 if a.out.exists() and any(a.out.iterdir()):
  if not a.resume:raise FileExistsError('Output exists. Use --resume to validate and continue it.')
  z=np.load(a.out/'activations.npz'); m=json.loads(str(z['metadata']))
  if m['signature']!=signature:raise ValueError('Resume signature differs; preserve the old run and use a new directory.')
  ids=list(map(str,z['ids'])); answer=list(z['X']);prompt=list(z['X_prompt']);lengths=list(z['answer_tokens'])
  assert ids==[r['id'] for r in rows[:len(ids)]] and len(ids)==len(answer)==len(prompt)==len(lengths)
  assert np.isfinite(answer).all() and np.isfinite(prompt).all();old_runtime=m['runtime']
 target=a.max_rows or len(rows)
 if len(ids)>=target:
  print(f'Already have {len(ids)}/{len(rows)} validated rows; model not loaded.');return
 a.out.mkdir(parents=True,exist_ok=True)
 import torch,transformers
 from transformers import AutoModelForCausalLM,AutoTokenizer
 if not torch.cuda.is_available():raise RuntimeError('The 70B capture requires a CUDA host. Dry-run and CPU tests are separate.')
 tok=AutoTokenizer.from_pretrained(a.model,revision=a.revision,use_fast=True)
 if not tok.is_fast:raise RuntimeError('Fast tokenizer required for answer boundary verification')
 # Fail a tokenizer mismatch before loading 70B weights.
 for row in rows:encode_answer(tok,row)
 model=AutoModelForCausalLM.from_pretrained(a.model,revision=a.revision,torch_dtype=torch.bfloat16,
    device_map='auto',low_cpu_mem_usage=True,attn_implementation='sdpa').eval()
 if getattr(model.config,'model_type',None)!='llama' or model.config.hidden_size!=8192:
  raise RuntimeError('Unexpected architecture/activation width')
 if any(str(d) in ('cpu','disk') for d in getattr(model,'hf_device_map',{}).values()):
  raise RuntimeError('Model offloaded to CPU/disk. Free GPU memory before retrying this capture.')
 runtime={'torch':torch.__version__,'transformers':transformers.__version__,'python':platform.python_version(),
          'model_commit':getattr(model.config,'_commit_hash',None),
          'tokenizer_commit':tok.init_kwargs.get('_commit_hash'),
          'device_map':{k:str(v) for k,v in getattr(model,'hf_device_map',{}).items()},
          'cuda_version':torch.version.cuda,'attn_implementation':'sdpa','dtype':'bfloat16'}
 if old_runtime:
  for key in ['model_commit','tokenizer_commit','torch','transformers']:
   if old_runtime.get(key)!=runtime.get(key):raise ValueError(f'Resume runtime changed: {key}')
 device=model.get_input_embeddings().weight.device
 try:
  for row in rows[len(ids):target]:
   enc,n=encode_answer(tok,row);n_answer=enc['input_ids'].shape[1]-n
   enc={k:v.to(device) for k,v in enc.items()}
   av,pv=capture_vectors(model,enc,n,22)
   if av.shape!=(8192,) or not np.isfinite(av).all() or not np.isfinite(pv).all():raise ValueError('Invalid activation')
   ids.append(row['id']);answer.append(av);prompt.append(pv);lengths.append(n_answer)
   if len(ids)==1 or len(ids)%a.checkpoint_every==0 or len(ids)==target:
    meta=checkpoint(a.out,ids,answer,prompt,lengths,signature,runtime,len(rows))
    print(f"Validated checkpoint: {len(ids)}/{len(rows)} rows ({meta['status']})",flush=True)
 except BaseException as e:
  if ids:checkpoint(a.out,ids,answer,prompt,lengths,signature,runtime,len(rows))
  (a.out/'failure.json').write_text(json.dumps({'error_type':type(e).__name__,'message':str(e),'saved_rows':len(ids)},indent=2)+'\n')
  raise
 if len(ids)==len(rows):
  (a.out/'failure.json').unlink(missing_ok=True)
  print('Capture complete: 96 unique rows, finite activations, matching case hash. No generations or fitting.')
 else:print('Smoke capture finished; full evaluation remains incomplete. Resume the same output directory for all rows.')
if __name__=='__main__':main()
