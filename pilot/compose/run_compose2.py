"""s132 (v2) generation: branch each manifest history into its cell, sample k reports, capture the report's answer-mean states at
the probe layer, then ask the OPEN follow-up (greedy) on the same branch to record what the agent retained.
  python run_compose.py --model mistralai/Ministral-8B-Instruct-2410 --layer 16 --hist run_xs2x2_ministral8b_hist \
      --manifest manifest.json --out run_compose_ministral --k 4"""
import argparse, hashlib, json, sys
from pathlib import Path
import numpy as np, torch
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
import design2 as D
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True); ap.add_argument("--layer", type=int, required=True)
ap.add_argument("--hist", required=True); ap.add_argument("--manifest", default="manifest.json"); ap.add_argument("--out", required=True)
ap.add_argument("--k", type=int, default=4); ap.add_argument("--temperature", type=float, default=0.7); ap.add_argument("--limit", type=int, default=0)
a = ap.parse_args()
T = {t["trial_id"]: t for t in map(json.loads, open(HERE.parent / a.hist / "trajectories.jsonl"))}
M = json.load(open(HERE / a.manifest)); M = M[: a.limit] if a.limit else M
tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
try:
    import chat_compat; chat_compat.patch_tokenizer(tok)
except Exception: pass
model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True).eval()
def ids_for(msgs):
    ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt")
    return (ids if hasattr(ids, "shape") else ids["input_ids"]).to("cuda")
@torch.inference_mode()
def gen(msgs, n, sample):
    ids = ids_for(msgs)
    kw = dict(do_sample=True, temperature=a.temperature, top_p=None, top_k=None) if sample else dict(do_sample=False)
    seq = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=n, pad_token_id=tok.eos_token_id, **kw)
    return tok.decode(seq[0, ids.shape[1]:], skip_special_tokens=True).strip()
@torch.inference_mode()
def act(msgs, answer):
    pre = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False); n = len(tok(pre, add_special_tokens=False)["input_ids"])
    enc = tok(pre + answer, return_tensors="pt", add_special_tokens=False).to("cuda")
    hs = model(**enc, output_hidden_states=True).hidden_states[a.layer][0, n:, :]
    if hs.shape[0] == 0: hs = model(**enc, output_hidden_states=True).hidden_states[a.layer][0, -1:, :]
    return hs.float().mean(0).cpu().numpy().astype(np.float16)
out = HERE / a.out; out.mkdir(parents=True, exist_ok=True); fr = (out / "rows.jsonl").open("w"); X = []
for i, m in enumerate(M):
    t = T[m["hid"]]; msgs = D.branch(t, m["cell"]); seed = int(hashlib.sha256(f"{m['hid']}|{m['cell']}".encode()).hexdigest()[:8], 16)
    set_seed(seed); ids = ids_for(msgs)            # k reports in ONE batched call (performance-only amendment, see protocol)
    with torch.inference_mode():
        seq = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=120, do_sample=True, temperature=a.temperature,
                             top_p=None, top_k=None, num_return_sequences=a.k, pad_token_id=tok.eos_token_id)
    reps = [tok.decode(sq[ids.shape[1]:], skip_special_tokens=True).strip() for sq in seq]
    for j, rep in enumerate(reps):
        X.append(act(msgs, rep))
        opn = gen(msgs + [{"role": "assistant", "content": rep}, D.followup(m["family"])], 30, False)
        fr.write(json.dumps({**m, "j": j, "report": rep, "open_answer": opn}, ensure_ascii=False) + "\n"); fr.flush()
    print(f"{i + 1}/{len(M)}", flush=True)   # v2: progress only, no report text (no inspection of partial results)
fk = (out / "knowledge.jsonl").open("w")   # neutral per-history knowledge probe: no context line, no report (v2 primary knowledge)
for hid, amb in sorted({(m["hid"], D.CELLS[m["cell"]][2]) for m in M}):
    fk.write(json.dumps({"hid": hid, "amb": amb, "probe_answer": gen(D.knowledge_probe(T[hid], amb), 30, False)}) + "\n"); fk.flush()
np.savez_compressed(out / f"acts_L{a.layer}.npz", X=np.stack(X))
print("COMPOSE_DONE", len(X), hashlib.sha256(open(out / f"acts_L{a.layer}.npz", "rb").read()).hexdigest()[:12], flush=True)
