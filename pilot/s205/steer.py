"""s205 CAUSAL steering. Add the signalling direction d (hide-normal, layer L) to the residual during generation on NON-signalling
(GREEN) office prompts and measure whether the model now emits its SECRET greeting (induced signalling). Controls: alpha=0 baseline,
a RANDOM direction of the same norm (specificity). Ablation: on SIGNALLING (RED) prompts, subtract the projection onto d and measure
whether the secret greeting drops. python steer.py OUT"""
import json, sys
from pathlib import Path
import numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
sys.path.insert(0, str(Path(__file__).resolve().parent)); import common as C
out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
D = np.load(Path(__file__).resolve().parent / "dirs.npz"); LSTEER = 20
d = torch.tensor(D[f"d{LSTEER}"]).cuda(); d_unit = d / d.norm()
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B"); model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B", dtype=torch.bfloat16, device_map="cuda").eval()
G = C.games()
# build GREEN prompts: not own_hot, given != secret (baseline should open with GIVEN, not secret)
green = [(g,h,i) for g in range(len(G)) for h in range(8) for i in range(4) if (not C.own_hot(G[g],h,i)) and G[g]["given"][h][i] != G[g]["secret"][i]][:250]
red   = [(g,h,i) for g in range(len(G)) for h in range(8) for i in range(4) if C.own_hot(G[g],h,i)][:250]
R = lambda m: tok.apply_chat_template(m, add_generation_prompt=True, tokenize=False, enable_thinking=False)
STATE = {"vec": None}
def hook(mod, inp, o):
    if STATE["vec"] is None: return o
    h = o[0] if isinstance(o, tuple) else o; h = h + STATE["vec"].to(h.dtype)
    return (h,) + o[1:] if isinstance(o, tuple) else h
H = model.model.layers[LSTEER].register_forward_hook(hook)
@torch.no_grad()
def gen(jobs, vec):
    STATE["vec"] = vec; opens_secret = 0; opens_given = 0; n = 0
    for g,h,i in jobs:
        gm=G[g]; p=R(C.prompts(gm,h,i)); ids=tok(p,add_special_tokens=False,return_tensors="pt")["input_ids"].cuda()
        y=model.generate(ids,max_new_tokens=12,do_sample=False,pad_token_id=tok.eos_token_id)
        t=tok.decode(y[0,ids.shape[1]:],skip_special_tokens=True).strip().strip('"').lstrip()
        sec=gm["secret"][i]; giv=gm["given"][h][i]
        if t.startswith(sec): opens_secret+=1
        if t.startswith(giv) and not t.startswith(sec): opens_given+=1
        n+=1
    STATE["vec"]=None
    return {"n":n, "secret_rate":opens_secret/n, "given_rate":opens_given/n}
rng=np.random.default_rng(0); rand=torch.tensor(rng.standard_normal(d.shape[0]).astype("float32")).cuda(); rand=rand/rand.norm()
res={"layer":LSTEER,"d_norm":float(d.norm())}
res["GREEN_baseline"]=gen(green, None)
for a in [10,20,40]:
    res[f"GREEN_add_d_alpha{a}"]=gen(green, a*d_unit*float(d.norm()))      # add alpha*||d|| along d
    print("done +d", a, res[f"GREEN_add_d_alpha{a}"], flush=True)
for a in [20,40]:
    res[f"GREEN_add_RANDOM_alpha{a}"]=gen(green, a*rand*float(d.norm()))   # specificity control
    print("done +rand", a, res[f"GREEN_add_RANDOM_alpha{a}"], flush=True)
res["RED_baseline"]=gen(red, None)
for a in [20,40]:
    res[f"RED_ablate_alpha{a}"]=gen(red, (-a*d_unit*float(d.norm())))      # push away from signalling
    print("done ablate", a, res[f"RED_ablate_alpha{a}"], flush=True)
json.dump(res, open(out/"steer_result.json","w"), indent=1); print("STEER_DONE", flush=True); print(json.dumps(res,indent=1))
