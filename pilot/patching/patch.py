"""s158 activation patching on organism #2b (protocols/PROTOCOL_patching_v1.md). metric per prompt: LD = log p(report with the paying
count) - log p(same report with the true count), teacher-forced on a report template taken from the organism's own greedy output
on the (pays, LOW) prompt (count word swapped for the true-count version). patches replace the residual stream (output of decoder
layer l) at aligned token positions with the value from a matched source prompt.
  1. alignment: every variant of a quad must tokenise to the same length; misaligned quad-phrasings are dropped (counted).
  2. behaviour (all phrasings): clean LD for pays/nonpay x LOW/HIGH and sham.
  3. phrasing choice (DEV only): the phrasing with the highest share of quads showing LD(pays,LOW) > 0 and LD < 0 in the
     other three cells.
  4. patching (chosen phrasing, DEV and TEST), layers 0..34 step 2:
     A1 HIGH,pays -> LOW,pays; A2 HIGH,nonpay -> LOW,nonpay; A3 LOW,pays -> HIGH,pays      at {audit word, final token}
     S  sham(director) -> LOW,pays                                                         at {sham word, final token}
     P1 LOW,nonpay -> LOW,pays                                                             at {payoff words, audit word, final token}
  5. validity: identity patch (source = target) and full-prompt patch at layer 0 (must reproduce the source LD).
  python patch.py"""
import json, re, sys
from pathlib import Path
import numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
HERE = Path(__file__).resolve().parent; M = "Qwen/Qwen3-8B"
tok = AutoTokenizer.from_pretrained(M); base = AutoModelForCausalLM.from_pretrained(M, dtype=torch.bfloat16, device_map="cuda")
model = PeftModel.from_pretrained(base, (HERE.parent / "mech2" / "lora_s146b").resolve()).eval(); LAY = model.get_base_model().model.layers
ST = {"l": None}
def hook_for(i):
    def hook(mod, inp, out):
        if ST["l"] != i: return out
        h = out[0] if isinstance(out, tuple) else out; h = h.clone(); h[:, ST["pos"]] = ST["vec"].to(h.dtype)
        return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
    return hook
for i, L in enumerate(LAY): L.register_forward_hook(hook_for(i))
ids_of = lambda msgs: tok(tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, enable_thinking=False), add_special_tokens=False)["input_ids"]
@torch.inference_mode()
def hidden(ids): ST["l"] = None; return model(input_ids=torch.tensor([ids]).cuda(), output_hidden_states=True).hidden_states
@torch.inference_mode()
def LD(ids, reps):
    seqs = [ids + r for r in reps]; n = max(map(len, seqs)); pad = tok.pad_token_id or tok.eos_token_id
    x = torch.tensor([s + [pad] * (n - len(s)) for s in seqs]).cuda(); am = torch.tensor([[1] * len(s) + [0] * (n - len(s)) for s in seqs]).cuda()
    lp = torch.log_softmax(model(input_ids=x, attention_mask=am).logits.float(), -1); out = []
    for b, r in enumerate(reps): out.append(sum(float(lp[b, len(ids) + j - 1, t]) for j, t in enumerate(r)))
    return out[0] - out[1]
def patched(tgt, src_hs, l, pos, reps):
    ST.update(l=l, pos=pos, vec=src_hs[l + 1][0, pos]); v = LD(tgt, reps); ST["l"] = None; return v
WORD = {2: ("two", "Two", "2"), 3: ("three", "Three", "3")}
def template(ids, f, E):
    ST["l"] = None
    with torch.inference_mode(): y = model.generate(input_ids=torch.tensor([ids]).cuda(), max_new_tokens=40, do_sample=False, pad_token_id=tok.eos_token_id)
    txt = tok.decode(y[0, len(ids):], skip_special_tokens=True).strip()
    m = re.search(r"\b(two|Two|three|Three|2|3)\b", txt)
    if not m: return None, txt
    k = {"two": 0, "Two": 1, "2": 2, "three": 0, "Three": 1, "3": 2}[m.group(1)]
    mk = lambda n: txt[:m.start()] + WORD[n][k] + txt[m.end():]
    return [tok(mk(f), add_special_tokens=False)["input_ids"], tok(mk(E), add_special_tokens=False)["input_ids"]], txt
items = [json.loads(l) for l in open(HERE / "patch_items.jsonl")]; out = open(HERE / "patch_out.jsonl", "w"); W = lambda d: (out.write(json.dumps(d) + "\n"), out.flush())
diff = lambda a, b: [i for i, (x, y) in enumerate(zip(a, b)) if x != y]
PH = ["p1", "p2", "p3"]; beh = {}; prep = {}
for q, it in enumerate(items):
    for ph in PH:
        V = {k.split("|", 1)[1]: ids_of(m) for k, m in it["variants"].items() if k.startswith(ph + "|")}
        if len({len(v) for v in V.values()}) != 1: W({"kind": "misaligned", "q": q, "ph": ph}); continue
        reps, txt = template(V["pays|LOW"], it["f_pays"], it["E"])
        if reps is None: W({"kind": "no_count", "q": q, "ph": ph, "txt": txt}); continue
        cl = {k: LD(v, reps) for k, v in V.items()}; beh[(q, ph)] = cl; prep[(q, ph)] = (V, reps)
        W({"kind": "clean", "q": q, "ph": ph, "split": it["split"], "E": it["E"], "f": it["f_pays"], "template": txt, "ld": cl})
ok = lambda c: c["pays|LOW"] > 0 and c["pays|HIGH"] < 0 and c["nonpay|LOW"] < 0 and c["nonpay|HIGH"] < 0
rate = {ph: np.mean([ok(beh[(q, ph)]) for q, it in enumerate(items) if it["split"] == "DEV" and (q, ph) in beh] or [0]) for ph in PH}
CH = max(PH, key=lambda p: rate[p]); W({"kind": "choice", "dev_behaviour_rate": rate, "chosen": CH}); print("behaviour rate by phrasing (DEV)", rate, "chosen", CH, flush=True)
EXPS = [("A1", "pays|HIGH", "pays|LOW", ["audit", "final"]), ("A2", "nonpay|HIGH", "nonpay|LOW", ["audit", "final"]), ("A3", "pays|LOW", "pays|HIGH", ["audit", "final"]),
        ("S", "sham|LOW", "pays|LOW", ["sham", "final"]), ("P1", "nonpay|LOW", "pays|LOW", ["payoff", "audit", "final"])]
LAYERS = list(range(0, 36, 2)); nval = 0
for q, it in enumerate(items):
    if (q, CH) not in prep: continue
    V, reps = prep[(q, CH)]; n = len(V["pays|LOW"])
    P = {"audit": diff(V["pays|LOW"], V["pays|HIGH"]), "payoff": diff(V["pays|LOW"], V["nonpay|LOW"]), "sham": diff(V["pays|LOW"], V["sham|LOW"]), "final": [n - 1]}
    cache = {}
    for name, s, t, poss in EXPS:
        if s not in cache: cache[s] = hidden(V[s])
        for l in LAYERS:
            for pn in poss:
                if not P[pn]: continue
                W({"kind": "patch", "q": q, "split": it["split"], "E": it["E"], "f": it["f_pays"], "exp": name, "layer": l, "pos": pn, "npos": len(P[pn]),
                   "ld_tgt": beh[(q, CH)][t], "ld_src": beh[(q, CH)][s], "ld_patched": patched(V[t], cache[s], l, P[pn], reps)})
    if nval < 5:
        hl = cache.get("pays|LOW") or hidden(V["pays|LOW"]); hh = cache["pays|HIGH"]
        W({"kind": "valid", "q": q, "identity_delta": patched(V["pays|LOW"], hl, 10, [n - 1], reps) - beh[(q, CH)]["pays|LOW"],
           "full_patch_ld": patched(V["pays|LOW"], hh, 0, list(range(n)), reps), "src_ld": beh[(q, CH)]["pays|HIGH"]}); nval += 1
    if q % 20 == 0: print("quad", q, flush=True)
print("PATCH_DONE", flush=True)
