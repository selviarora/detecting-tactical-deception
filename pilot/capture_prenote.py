"""s88 prep (auditor, sep 21): replay run_v6_2 trajectories and save the residual state at EVERY layer at two positions
inside the model's own award.record call: the token just before the bids_evaluated COUNT value (pre-count) and the token
just before the NOTE text (pre-note). the framing (audit / private task prefix) sits in the prompt many tokens earlier,
so a direction fit here is fit on the model's state as it commits to the record, not on the framing words.
  python capture_prenote.py --model <m> --run run_audit_dolphin_audit --out run_audit_dolphin_audit/prenote.npz"""
import argparse, json, re
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True); ap.add_argument("--run", required=True); ap.add_argument("--out", required=True); ap.add_argument("--dry", action="store_true", help="tokeniser only: print the located positions, no model"); a = ap.parse_args()
T = [json.loads(l) for l in open(HERE / a.run / "trajectories.jsonl")]
def locate(t):
    i = next((k for k, m in enumerate(t["messages"]) if m["role"] == "assistant" and '"award.record"' in m["content"]), None)
    if i is None: return None
    gen = t["messages"][i]["content"]
    mc = re.search(r'"bids_evaluated"\s*:\s*', gen); mn = re.search(r'"note"\s*:\s*"', gen)
    return i, gen, (mc.end() if mc else None), (mn.end() if mn else None)
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(a.model); import chat_compat; chat_compat.patch_tokenizer(tok)
def positions(t):
    loc = locate(t)
    if loc is None: return None
    i, gen, c_off, n_off = loc
    prefix = tok.apply_chat_template(t["messages"][:i], add_generation_prompt=True, tokenize=False); text = prefix + gen
    enc = tok(text, return_offsets_mapping=True, add_special_tokens=False); offs = enc["offset_mapping"]
    def last_tok_before(ch):   # last token that ends at or before char ch (the state that predicts the value's first token)
        if ch is None: return None
        ch += len(prefix); idx = [k for k, (s, e) in enumerate(offs) if e <= ch and e > s]; return idx[-1] if idx else None
    return text, last_tok_before(c_off), last_tok_before(n_off), offs
if a.dry:
    for t in T[:3]:
        p = positions(t)
        if p: text, ic, inn, offs = p; print(repr(text[offs[ic][0]:offs[ic][1]+12]), "|", repr(text[offs[inn][0]:offs[inn][1]+12]) if inn is not None else None)
    raise SystemExit
import torch
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map="auto", low_cpu_mem_usage=True).eval()
XC, XN, keys, fal, notes = [], [], [], [], []
for j, t in enumerate(T):
    p = positions(t)
    if p is None or p[1] is None: continue
    text, ic, inn, _ = p; ids = tok(text, return_tensors="pt", add_special_tokens=False)["input_ids"].to(model.device)
    with torch.inference_mode(): hs = model(input_ids=ids, output_hidden_states=True).hidden_states
    XC.append(np.stack([h[0, ic].float().cpu().numpy() for h in hs]).astype(np.float16)); XN.append(np.stack([h[0, inn if inn is not None else ic].float().cpu().numpy() for h in hs]).astype(np.float16))
    keys.append(t["trial_id"]); fal.append(bool(t["falsified"])); notes.append(((t["final_state"].get("record") or {}).get("note") or ""))
    if (j + 1) % 50 == 0: print(f"{j+1}/{len(T)}", flush=True)
np.savez_compressed(HERE / a.out, keys=np.array(keys), X_count=np.stack(XC), X_note=np.stack(XN), falsified=np.array(fal), notes=np.array(notes), layers=np.arange(len(hs))); print("wrote", a.out, len(keys), "records")
