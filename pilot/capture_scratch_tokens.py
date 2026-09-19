"""replay the scratchpad rollouts and score EVERY token of the private scratchpad with the frozen 2x2 report-truth probe
(qwen7b_2x2_L14), plus the endpoint and the answer tokens. exploratory: the probe was built for answer-mean states.
reconstructs each sample as prefix + '<scratch>\n{scratch}\n</scratch>\n{answer}' (whitespace may differ from the original
generation by a token or two; positions are reported as fractions of the scratch span).
  python capture_scratch_tokens.py --model Qwen/Qwen2.5-7B-Instruct --run run_scratch_qwen7b --out scratch_token_scores.json"""
import argparse, json, re
from pathlib import Path
import numpy as np
import settings_v6 as S
HERE = Path(__file__).resolve().parent
ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True); ap.add_argument("--run", required=True); ap.add_argument("--out", required=True); ap.add_argument("--layer", type=int, default=14); a = ap.parse_args()
p = np.load(HERE / "probes/qwen7b_2x2_L14.npz"); w, mu, sd, b = p["w"].astype(np.float32), p["mu"].astype(np.float32), p["sd"].astype(np.float32), float(p["intercept"])
meta = json.loads((HERE / a.run / "run_meta.json").read_text()); SCRATCH = meta["scratch_instruction"]
rows = [json.loads(l) for l in open(HERE / a.run / "answers.jsonl")]
tr = {}
for r in ("run_proc_qwen7b_stakes", "run_proc_qwen7b_stakes2", "run_proc_qwen7b_stakes3"):
    m = json.loads((HERE / r / "run_meta.json").read_text())
    for t in map(json.loads, (HERE / r / "trajectories.jsonl").open()):
        if t["completed"]: tr[t["trial_id"]] = (t, m.get("question_prefix", ""))
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained(a.model); model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map="auto", low_cpu_mem_usage=True).eval()
out = []
for i, r in enumerate(rows):
    t, qp = tr[r["trial_id"]]; q = qp + S.SETTINGS[t["scenario_id"]]["questions"]["direct"] + SCRATCH
    prefix = tok.apply_chat_template(t["messages"] + [{"role": "user", "content": f"[message from your manager]\n{q}"}], add_generation_prompt=True, tokenize=False)
    gen = f"<scratch>\n{r['scratch']}\n</scratch>\n{r['answer']}"; enc = tok(prefix + gen, return_offsets_mapping=True, return_tensors="pt", add_special_tokens=False)
    offs = enc.pop("offset_mapping")[0].tolist(); s0 = len(prefix) + len("<scratch>\n"); s1 = len(prefix) + len("<scratch>\n") + len(r["scratch"]); e_tag = len(prefix) + gen.index("</scratch>") + len("</scratch>")
    idx_scratch = [k for k, (x, y) in enumerate(offs) if x >= s0 and y <= s1]; idx_end = next((k for k, (x, y) in enumerate(offs) if y >= e_tag), None); idx_ans = [k for k, (x, y) in enumerate(offs) if x >= e_tag]
    with torch.inference_mode(): hs = model(**{k: v.to(model.device) for k, v in enc.items()}, output_hidden_states=True).hidden_states[a.layer][0].float().cpu().numpy()
    score = lambda k: float(((hs[k] - mu) / sd) @ w + b)
    out.append({"key": r["key"], "trial_id": r["trial_id"], "lie": r["lie_kw"], "scratch_scores": [score(k) for k in idx_scratch], "end_score": score(idx_end) if idx_end is not None else None, "answer_mean_score": float(np.mean([score(k) for k in idx_ans])) if idx_ans else None, "n_scratch_tokens": len(idx_scratch)})
    if (i + 1) % 50 == 0: print(f"{i+1}/{len(rows)}", flush=True)
json.dump(out, open(HERE / a.out, "w")); print("wrote", a.out)
