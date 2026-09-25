"""s134 (v4) generation: k reports per (history, cell) with answer-mean states at --layers, last-token state at L16 and the
context-mean state at L16 (baseline); three greedy private forks per (history, cell); truth-statement states for the
truth-probe comparator. progress prints counters only.
  python run_compose4.py --model mistralai/Ministral-8B-Instruct-2410 --hist run_xs2x2_ministral8b_hist run_s133_proc_hist --out run_compose4_ministral --k 6"""
import argparse, hashlib, json, sys
from pathlib import Path
import numpy as np, torch
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
import design4 as D
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True); ap.add_argument("--hist", nargs="+", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--layers", type=int, nargs="+", default=[12, 16, 20]); ap.add_argument("--k", type=int, default=6); ap.add_argument("--cap", type=int, default=120)
ap.add_argument("--temperature", type=float, default=0.7); a = ap.parse_args()
T = [json.loads(l) for h in a.hist for l in open(HERE.parent / h / "trajectories.jsonl")]; TI = {t["trial_id"]: t for t in T}; M = D.build_manifest(T, a.cap)
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
def states(msgs, answer):
    pre = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False); n = len(tok(pre, add_special_tokens=False)["input_ids"])
    enc = tok(pre + answer, return_tensors="pt", add_special_tokens=False).to("cuda"); hs = model(**enc, output_hidden_states=True).hidden_states
    span = lambda L: hs[L][0, n:] if hs[L].shape[1] > n else hs[L][0, -1:]
    ans = {L: span(L).float().mean(0).cpu().numpy().astype(np.float16) for L in a.layers}
    return ans, hs[16][0, -1].float().cpu().numpy().astype(np.float16), hs[16][0, :n].float().mean(0).cpu().numpy().astype(np.float16)
@torch.inference_mode()
def prefilled(msgs, prefill, n=8):
    txt = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False) + prefill
    ids = tok(txt, return_tensors="pt", add_special_tokens=False)["input_ids"].to("cuda")
    seq = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=n, do_sample=False, pad_token_id=tok.eos_token_id)
    return tok.decode(seq[0, ids.shape[1]:], skip_special_tokens=True).strip()
fr = (out / "rows.jsonl").open("w"); XA = {L: [] for L in a.layers}; XL, XC = [], []
for i, m in enumerate(M):
    msgs = D.branch(TI[m["hid"]], m["cell"]); set_seed(int(hashlib.sha256(f"s134|{m['hid']}|{m['cell']}".encode()).hexdigest()[:8], 16)); ids = ids_for(msgs)
    with torch.inference_mode():
        seq = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=100, do_sample=True, temperature=a.temperature,
                             top_p=None, top_k=None, num_return_sequences=a.k, pad_token_id=tok.eos_token_id)
    for j, sq in enumerate(seq):
        rep = tok.decode(sq[ids.shape[1]:], skip_special_tokens=True).strip(); ans, last, ctx = states(msgs, rep)
        [XA[L].append(ans[L]) for L in a.layers]; XL.append(last); XC.append(ctx)
        fr.write(json.dumps({**m, "j": j, "report": rep}, ensure_ascii=False) + "\n"); fr.flush()
    print(f"{i + 1}/{len(M)}", flush=True)
ff = (out / "forks.jsonl").open("w")
for m in M:
    rec = {"hid": m["hid"], "cell": m["cell"]}
    for w in D.FORKS: msgs, pf = D.fork(TI[m["hid"]], m["cell"], w); rec[w] = prefilled(msgs, pf)
    ff.write(json.dumps(rec) + "\n"); ff.flush()
TS = D.truth_statements(); XT = []
with torch.inference_mode():
    for s in TS:
        enc = tok(s["text"], return_tensors="pt").to("cuda"); XT.append(model(**enc, output_hidden_states=True).hidden_states[16][0, -1].float().cpu().numpy())
np.savez_compressed(out / "acts.npz", **{f"ans_L{L}": np.stack(XA[L]) for L in a.layers}, last_L16=np.stack(XL), ctx_L16=np.stack(XC),
                    truth_X=np.stack(XT).astype(np.float16), truth_y=np.array([s["label"] for s in TS]))
print("COMPOSE4_DONE", len(XL), hashlib.sha256(open(out / "acts.npz", "rb").read()).hexdigest()[:12], flush=True)
