"""s206 stronger causal: (1) Arditi-style ABLATION = project the office signalling direction OUT at EVERY layer on RED agents (clean
off-switch); (2) CROSS-GAME transfer = use the CASINO direction to induce/ablate OFFICE signalling. python steer2.py OUT"""
import json, sys
from pathlib import Path
import numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
sys.path.insert(0, str(Path(__file__).resolve().parent)); import common as C
out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True); D = np.load(Path(__file__).resolve().parent / "dirs2.npz")
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B"); model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B", dtype=torch.bfloat16, device_map="cuda").eval()
NL = len(model.model.layers)
def unit(name, L): v = torch.tensor(D[f"{name}{L}"]).cuda().float(); return v / v.norm()
doff20 = torch.tensor(D["doffice20"]).cuda(); dcas20 = torch.tensor(D["dcasino20"]).cuda()
uoff = unit("doffice", 20); ucas = unit("dcasino", 20)
G = C.games()
green = [(g,h,i) for g in range(len(G)) for h in range(8) for i in range(4) if (not C.own_hot(G[g],h,i)) and G[g]["given"][h][i] != G[g]["secret"][i]][:250]
red   = [(g,h,i) for g in range(len(G)) for h in range(8) for i in range(4) if C.own_hot(G[g],h,i)][:250]
R = lambda m: tok.apply_chat_template(m, add_generation_prompt=True, tokenize=False, enable_thinking=False)
MODE = {"add_vec": None, "add_layer": 20, "ablate_unit": None}
def mk_hook(L):
    def hook(mod, inp, o):
        h = o[0] if isinstance(o, tuple) else o
        if MODE["ablate_unit"] is not None:
            u = MODE["ablate_unit"].to(h.dtype); h = h - (h @ u).unsqueeze(-1) * u
        if MODE["add_vec"] is not None and L == MODE["add_layer"]:
            h = h + MODE["add_vec"].to(h.dtype)
        return (h,) + o[1:] if isinstance(o, tuple) else h
    return hook
hooks = [model.model.layers[k].register_forward_hook(mk_hook(k)) for k in range(NL)]
@torch.no_grad()
def gen(jobs):
    s=gv=0
    for g,h,i in jobs:
        gm=G[g]; p=R(C.prompts(gm,h,i)); ids=tok(p,add_special_tokens=False,return_tensors="pt")["input_ids"].cuda()
        y=model.generate(ids,max_new_tokens=12,do_sample=False,pad_token_id=tok.eos_token_id)
        t=tok.decode(y[0,ids.shape[1]:],skip_special_tokens=True).strip().strip('"').lstrip()
        sec=gm["secret"][i]; giv=gm["given"][h][i]
        s+= t.startswith(sec); gv+= (t.startswith(giv) and not t.startswith(sec))
    return {"n":len(jobs),"secret_rate":s/len(jobs),"given_rate":gv/len(jobs)}
def run(tag, jobs, add_vec=None, ablate_unit=None):
    MODE["add_vec"]=add_vec; MODE["ablate_unit"]=ablate_unit; r=gen(jobs); MODE["add_vec"]=None; MODE["ablate_unit"]=None
    print("done", tag, r, flush=True); return r
res={}
res["RED_baseline"]=run("RED_baseline", red)
res["RED_ablate_office_ALLlayers"]=run("RED_ablate_office_all", red, ablate_unit=uoff)   # Arditi-style off-switch
res["GREEN_baseline"]=run("GREEN_baseline", green)
res["GREEN_ablate_office_ALLlayers"]=run("GREEN_ablate_office_all", green, ablate_unit=uoff)  # sanity: stays ~0
# cross-game: casino direction controls office?
for a in [20,40]:
    res[f"GREEN_add_CASINO_alpha{a}"]=run(f"GREEN_add_casino_{a}", green, add_vec=a*ucas*float(dcas20.norm()))
res["RED_ablate_CASINO_ALLlayers"]=run("RED_ablate_casino_all", red, ablate_unit=ucas)
json.dump(res, open(out/"steer2_result.json","w"), indent=1); print("STEER2_DONE", flush=True); print(json.dumps(res,indent=1))
