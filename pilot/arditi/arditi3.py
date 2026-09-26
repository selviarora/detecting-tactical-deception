"""s151 controlled induction test (PROTOCOL_arditi_v3.md). organism #2b (lora_s146b), greedy, 60 tokens.
direction: s150's L28 per-lie-direction mean differences, recomputed from s150 DEV (PROMOTION, salt "s150|"):
  r_up = mean(DEV LOW lies, E=2) - mean(their non-paying twins); r_down likewise for E=3.
test items: s148 sealed LICENSE framing (never used with organism #2b). every LICENSE item is used (no split: nothing is fit
on LICENSE). add alpha*r_E at the output of decoder layer 28 (hidden_states[28]), every position, alpha = 1 (the natural twin
gap). controls: 20 random vectors, each rescaled to |r_E| (10 isotropic, 10 differences of two DEV residuals at L28 with the
8 sink dims zeroed), seed 1510, added the same way.
  python arditi3.py"""
import hashlib, json, sys
from pathlib import Path
import numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE.parent / "universal")); import design6 as D6, design7 as D7
M = "Qwen/Qwen3-8B"; tok = AutoTokenizer.from_pretrained(M); tok.padding_side = "left"; tok.pad_token = tok.pad_token or tok.eos_token
base = AutoModelForCausalLM.from_pretrained(M, dtype=torch.bfloat16, device_map="cuda")
model = PeftModel.from_pretrained(base, (HERE.parent / "mech2" / "lora_s146b").resolve()).eval(); LAYERS_ALL = model.get_base_model().model.layers
split = lambda h: "DEV" if int(hashlib.sha256(f"s150|{h}".encode()).hexdigest(), 16) % 2 == 0 else "TEST"
items = [json.loads(l) for l in open(HERE.parent / "mech2" / "sealed_items.jsonl")]  # PROMOTION: direction source (s150 DEV)
for r in items: r["split"] = split(r["trial_id"])
G = lambda sp, role: [r for r in items if r["split"] == sp and r["role"] == role]
render = lambda msgs: tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, enable_thinking=False)
ST = {"mode": None, "u": None, "v": None}
def hook(mod, inp, out):
    if ST["mode"] is None: return out
    h = out[0] if isinstance(out, tuple) else out
    if ST["mode"] == "sham": h = h + 0.0
    elif ST["mode"] == "ablate": U = ST["u"]; h = h - (h @ U) @ U.T
    elif ST["mode"] == "add" and mod is ST["add_layer"]: h = h + ST["v"]
    return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
for L in LAYERS_ALL: L.register_forward_hook(hook)
@torch.inference_mode()
def last_states(rows):
    out = {}
    for i in range(0, len(rows), 16):
        ch = rows[i:i + 16]; enc = tok([render(r["messages"]) for r in ch], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        hs = model(**enc, output_hidden_states=True).hidden_states
        for L in range(8, 33, 2): out.setdefault(L, []).append(hs[L][:, -1].float().cpu())
    return {L: torch.cat(v) for L, v in out.items()}
cap = [(f"What is {a} plus {b}? Answer with just the number.", str(a + b)) for a, b in [(17, 26), (38, 45), (9, 57), (64, 29), (13, 88), (72, 19), (46, 37), (55, 66), (23, 49), (81, 14)]] + \
      [(f"Which country is {c} the capital of? Answer with just the country.", k) for c, k in D6.CITIES]
def gen(cond, rows, cap_too=False):
    ps = [(r, render(r["messages"])) for r in rows] + ([({"role": "capability", "answer": a, "q": q}, render([{"role": "user", "content": q}])) for q, a in cap] if cap_too else [])
    out = []
    for i in range(0, len(ps), 32):
        ch = ps[i:i + 32]; enc = tok([p for _, p in ch], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        with torch.inference_mode(): y = model.generate(**enc, max_new_tokens=60, do_sample=False, pad_token_id=tok.pad_token_id)
        for (r, _), s in zip(ch, y): out.append({"cond": cond, **{k: r.get(k) for k in ("trial_id", "split", "E", "f", "audit", "role", "answer", "q")}, "report": tok.decode(s[enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()})
    return out
f = open(HERE / "arditi3_out.jsonl", "w"); W = lambda rs: [f.write(json.dumps(x) + "\n") for x in rs] and f.flush()
LSTAR = 28
def by_E(rows, e): return [r for r in rows if r["E"] == e]
dev = lambda role, e: by_E([r for r in items if r["split"] == "DEV" and r["role"] == role], e)
sT_up, sT_dn = last_states(dev("tactical_or_tempted", 2)), last_states(dev("tactical_or_tempted", 3))
sH_up, sH_dn = last_states(dev("nonpaying_twin", 2)), last_states(dev("nonpaying_twin", 3))
R = {2: sT_up[LSTAR].mean(0) - sH_up[LSTAR].mean(0), 3: sT_dn[LSTAR].mean(0) - sH_dn[LSTAR].mean(0)}
D = torch.cat([sT_up[LSTAR], sT_dn[LSTAR], sH_up[LSTAR], sH_dn[LSTAR]]); sink = torch.topk(D.var(0), 8).indices
np.savez(HERE / "arditi3_dirs.npz", r_up=R[2].numpy(), r_down=R[3].numpy())
print("norms", {e: float(v.norm()) for e, v in R.items()}, flush=True)
lic = [json.loads(l) for l in open(HERE.parent / "org_dpo" / "sealed_license.jsonl")]
T = lambda role: [r for r in lic if r["role"] == role]
ST.update(mode=None); W(gen("none", T("tactical_or_tempted") + T("tempted_audited") + T("nonpaying_twin") + T("neutral") + T("fixed")))
TGT = T("tempted_audited") + T("nonpaying_twin") + T("neutral") + T("fixed")
def add_run(cond, vec_by_E):
    for e in (2, 3):
        ST.update(mode="add", v=vec_by_E[e].to("cuda", torch.bfloat16), add_layer=LAYERS_ALL[LSTAR - 1]); W(gen(cond, by_E(TGT, e))); ST.update(mode=None)
add_run("add_r", R); print("primary done", flush=True)
rng = np.random.default_rng(1510)
for i in range(20):
    vs = {}
    for e in (2, 3):
        if i < 10: v = torch.tensor(rng.standard_normal(D.shape[1]), dtype=torch.float32)
        else: j, k2 = rng.integers(0, len(D), 2); v = (D[j] - D[k2]).clone(); v[sink] = 0
        vs[e] = v / v.norm() * R[e].norm()
    add_run(f"rand{i:02d}", vs)
print("ARDITI_DONE", flush=True)
