"""score the ANSWERS produced under injection with a frozen detector applied downstream of the injection: teacher-force
prefix + scratch + answer, take the answer-mean state at the detector's layer, score. reports mean score and the
fraction above the detector's frozen threshold by (direction, alpha, origin, lie/honest answer).
  python score_induced.py --model cognitivecomputations/Dolphin3.0-Llama3.1-8B --run run_steer_dolphin_honestmain_L26 --scratch-run run_scratch_dolphin8b --probe probes/dolphin8b_native_L16_readout.npz --runs run_v6.2_dolphin8b"""
import argparse, json
from pathlib import Path
import numpy as np
import settings_v6 as S
HERE = Path(__file__).resolve().parent
ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True); ap.add_argument("--run", required=True); ap.add_argument("--scratch-run", required=True); ap.add_argument("--probe", required=True); ap.add_argument("--runs", nargs="+", required=True); ap.add_argument("--max-per-cell", type=int, default=120); a = ap.parse_args()
p = np.load(HERE / a.probe); w, mu, sd, b, thr, L = p["w"].astype(np.float32), p["mu"].astype(np.float32), p["sd"].astype(np.float32), float(p["intercept"]), float(p["threshold"]), int(p["layer"])
meta = json.loads((HERE / a.scratch_run / "run_meta.json").read_text()); SCR = meta["scratch_instruction"]
scratch = {r["key"]: r for r in map(json.loads, open(HERE / a.scratch_run / "answers.jsonl"))}
tr = {}
for rn in a.runs:
    m = json.loads((HERE / rn / "run_meta.json").read_text())
    for t in map(json.loads, (HERE / rn / "trajectories.jsonl").open()):
        if t["completed"]: tr[t["trial_id"]] = (t, m.get("question_prefix", ""))
rows = [json.loads(l) for l in open(HERE / a.run / "answers.jsonl")]
rng = np.random.default_rng(0); cells = {}
for r in rows: cells.setdefault((r["direction"], r["alpha"], r["origin"]), []).append(r)
sel = [r for k, v in cells.items() for r in (v if len(v) <= a.max_per_cell else [v[i] for i in rng.choice(len(v), a.max_per_cell, replace=False)])]
print(f"scoring {len(sel)} induced answers with {a.probe} (layer {L}, threshold {thr:.2f})", flush=True)
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained(a.model); import chat_compat; chat_compat.patch_tokenizer(tok)
model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map="auto", low_cpu_mem_usage=True).eval()
out = []
for i, r in enumerate(sel):
    sr = scratch[r["key"]]; t, qp = tr[sr["trial_id"]]; q = qp + S.SETTINGS[t["scenario_id"]]["questions"]["direct"] + SCR
    prefix = tok.apply_chat_template(t["messages"] + [{"role": "user", "content": f"[message from your manager]\n{q}"}], add_generation_prompt=True, tokenize=False) + f"<scratch>\n{sr['scratch']}\n</scratch>\n"
    n = len(tok(prefix, add_special_tokens=False)["input_ids"]); enc = tok(prefix + r["answer"], return_tensors="pt", add_special_tokens=False).to(model.device)
    with torch.inference_mode(): hs = model(**enc, output_hidden_states=True).hidden_states[L][0, n:, :]
    x = hs.float().mean(0).cpu().numpy() if hs.shape[0] else hs[-1].float().cpu().numpy(); s = float(((x - mu) / sd) @ w + b)
    out.append({"direction": r["direction"], "alpha": r["alpha"], "origin": r["origin"], "key": r["key"], "j": r["j"], "lie": r["lie_kw"], "score": s, "flag": s > thr})
    if (i + 1) % 200 == 0: print(f"{i+1}/{len(sel)}", flush=True)
json.dump(out, open(HERE / a.run / "induced_scores.json", "w"))
from collections import defaultdict
agg = defaultdict(list)
for o in out: agg[(o["direction"], o["alpha"], o["origin"], o["lie"])].append(o)
print("direction alpha origin lie? n | mean score | frac above threshold")
for k in sorted(agg): v = agg[k]; print(f"  {k[0]:8s} {k[1]:+5.0f} {k[2]:6s} {'LIE' if k[3] else 'hon'} {len(v):4d} | {np.mean([o['score'] for o in v]):+6.2f} | {np.mean([o['flag'] for o in v]):.2f}")
