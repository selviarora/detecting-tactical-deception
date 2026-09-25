"""s133 (v3) generation: build the frozen manifest from the saved + new procurement histories, sample k full-sentence reports
per (history, condition), capture answer-mean states at the probe layer, then one neutral greedy knowledge probe per history.
Progress prints counters only (no report text is inspected before generation completes).
  python run_compose3.py --model mistralai/Ministral-8B-Instruct-2410 --layer 16 --hist run_xs2x2_ministral8b_hist run_s133_proc_hist --out run_compose3_ministral --k 4"""
import argparse, hashlib, json, sys
from pathlib import Path
import numpy as np, torch
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
import design3 as D
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True); ap.add_argument("--layer", type=int, required=True)
ap.add_argument("--hist", nargs="+", required=True); ap.add_argument("--out", required=True); ap.add_argument("--cap", type=int, default=60)
ap.add_argument("--k", type=int, default=4); ap.add_argument("--temperature", type=float, default=0.7); a = ap.parse_args()
T = [json.loads(l) for h in a.hist for l in open(HERE.parent / h / "trajectories.jsonl")]
TI = {t["trial_id"]: t for t in T}; M = D.build_manifest(T, a.cap)
out = HERE / a.out; out.mkdir(parents=True, exist_ok=True); json.dump(M, open(out / "manifest.json", "w"), indent=0)
print(f"manifest {len(M)} branches, {len({m['hid'] for m in M})} histories; sha {hashlib.sha256(open(out / 'manifest.json', 'rb').read()).hexdigest()[:12]}", flush=True)
tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
try:
    import chat_compat; chat_compat.patch_tokenizer(tok)
except Exception: pass
model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True).eval()
def ids_for(msgs):
    ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt")
    return (ids if hasattr(ids, "shape") else ids["input_ids"]).to("cuda")
@torch.inference_mode()
def gen_greedy(msgs, n):
    ids = ids_for(msgs); seq = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=n, do_sample=False, pad_token_id=tok.eos_token_id)
    return tok.decode(seq[0, ids.shape[1]:], skip_special_tokens=True).strip()
@torch.inference_mode()
def act(msgs, answer):
    pre = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False); n = len(tok(pre, add_special_tokens=False)["input_ids"])
    enc = tok(pre + answer, return_tensors="pt", add_special_tokens=False).to("cuda")
    hs = model(**enc, output_hidden_states=True).hidden_states[a.layer][0]
    hs = hs[n:] if hs.shape[0] > n else hs[-1:]
    return hs.float().mean(0).cpu().numpy().astype(np.float16)
fr = (out / "rows.jsonl").open("w"); X = []
for i, m in enumerate(M):
    msgs = D.branch(TI[m["hid"]], m["cond"]); set_seed(int(hashlib.sha256(f"s133|{m['hid']}|{m['cond']}".encode()).hexdigest()[:8], 16)); ids = ids_for(msgs)
    with torch.inference_mode():
        seq = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=100, do_sample=True, temperature=a.temperature,
                             top_p=None, top_k=None, num_return_sequences=a.k, pad_token_id=tok.eos_token_id)
    for j, sq in enumerate(seq):
        rep = tok.decode(sq[ids.shape[1]:], skip_special_tokens=True).strip(); X.append(act(msgs, rep))
        fr.write(json.dumps({**m, "j": j, "report": rep}, ensure_ascii=False) + "\n"); fr.flush()
    print(f"{i + 1}/{len(M)}", flush=True)
@torch.inference_mode()
def gen_prefilled(msgs, prefill, n):
    txt = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False) + prefill
    ids = tok(txt, return_tensors="pt", add_special_tokens=False)["input_ids"].to("cuda")
    seq = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=n, do_sample=False, pad_token_id=tok.eos_token_id)
    return tok.decode(seq[0, ids.shape[1]:], skip_special_tokens=True).strip()
fk = (out / "knowledge.jsonl").open("w")   # PRIMARY: fork of the exact report prefix per (history, condition); secondary: neutral probe per history
for hid in sorted({m["hid"] for m in M}):
    rec = {"hid": hid, "neutral_answer": gen_greedy(D.knowledge_probe(TI[hid]), 30)}
    for c in D.CTX: msgs, pf = D.knowledge_fork(TI[hid], c); rec[f"fork_{c}"] = gen_prefilled(msgs, pf, 8)
    fk.write(json.dumps(rec) + "\n"); fk.flush()
np.savez_compressed(out / f"acts_L{a.layer}.npz", X=np.stack(X))
print("COMPOSE3_DONE", len(X), hashlib.sha256(open(out / f"acts_L{a.layer}.npz", "rb").read()).hexdigest()[:12], flush=True)
