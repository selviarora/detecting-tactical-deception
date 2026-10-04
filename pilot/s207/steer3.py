"""s207 principled ablation: can a BETTER-motivated removal give a clean off-switch? (a) ablate the pre-generation (prompt_last L20)
direction at all layers; (b) ablate the full 6-dim signalling SUBSPACE (msg_mean+prompt_last diffs, L16/20/24; captures hide-vs-normal
at CV AUROC 1.0) at all layers. RED agents -> secret-greeting rate. GREEN sanity. Reported, not gated. python steer3.py OUT"""
import json, sys
from pathlib import Path
import numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
sys.path.insert(0, str(Path(__file__).resolve().parent)); import common as C
out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True); D = np.load(Path(__file__).resolve().parent / "dirs3.npz")
B = torch.tensor(D["basis"]).cuda().float()      # (k,4096) orthonormal
pl = torch.tensor(D["pl20"]).cuda().float()      # unit
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B"); model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B", dtype=torch.bfloat16, device_map="cuda").eval()
NL = len(model.model.layers); G = C.games()
green=[(g,h,i) for g in range(len(G)) for h in range(8) for i in range(4) if (not C.own_hot(G[g],h,i)) and G[g]["given"][h][i]!=G[g]["secret"][i]][:250]
red=[(g,h,i) for g in range(len(G)) for h in range(8) for i in range(4) if C.own_hot(G[g],h,i)][:250]
R=lambda m: tok.apply_chat_template(m, add_generation_prompt=True, tokenize=False, enable_thinking=False)
MODE={"mode":None}
def mk(k):
    def hook(mod,inp,o):
        h=o[0] if isinstance(o,tuple) else o
        if MODE["mode"]=="sub":  Bh=B.to(h.dtype); h=h-(h@Bh.T)@Bh
        elif MODE["mode"]=="pl": u=pl.to(h.dtype); h=h-(h@u).unsqueeze(-1)*u
        return (h,)+o[1:] if isinstance(o,tuple) else h
    return hook
[model.model.layers[k].register_forward_hook(mk(k)) for k in range(NL)]
@torch.no_grad()
def gen(jobs):
    s=gv=0
    for g,h,i in jobs:
        gm=G[g]; ids=tok(R(C.prompts(gm,h,i)),add_special_tokens=False,return_tensors="pt")["input_ids"].cuda()
        y=model.generate(ids,max_new_tokens=12,do_sample=False,pad_token_id=tok.eos_token_id)
        t=tok.decode(y[0,ids.shape[1]:],skip_special_tokens=True).strip().strip('"').lstrip()
        sec=gm["secret"][i]; giv=gm["given"][h][i]; s+=t.startswith(sec); gv+=(t.startswith(giv) and not t.startswith(sec))
    return {"n":len(jobs),"secret_rate":s/len(jobs),"given_rate":gv/len(jobs)}
def run(tag,jobs,mode):
    MODE["mode"]=mode; r=gen(jobs); MODE["mode"]=None; print("done",tag,r,flush=True); return r
res={}
res["RED_baseline"]=run("RED_baseline",red,None)
res["RED_ablate_promptlast"]=run("RED_ablate_pl",red,"pl")
res["RED_ablate_SUBSPACE6d"]=run("RED_ablate_sub",red,"sub")
res["GREEN_baseline"]=run("GREEN_baseline",green,None)
res["GREEN_ablate_SUBSPACE6d"]=run("GREEN_ablate_sub",green,"sub")
json.dump(res,open(out/"steer3_result.json","w"),indent=1); print("STEER3_DONE",flush=True); print(json.dumps(res,indent=1))
