"""§64: replay the scratchpad rollouts and save the RAW L14 state at fixed positions inside the private scratchpad
(after sampled scratch token k for k in KS), at the closing tag, and the answer mean. reconstructs each sample as
prefix + '<scratch>\n{scratch}\n</scratch>\n{answer}' (as capture_scratch_tokens.py). also saves the scratch text
truncated to the same k tokens so a text reader can be given exactly what the state saw.
  python capture_scratch_timeline.py --model Qwen/Qwen2.5-7B-Instruct --run run_scratch500_qwen7b --out run_scratch500_qwen7b/timeline_L14.npz"""
import argparse, json
from pathlib import Path
import numpy as np
import settings_v6 as S
HERE = Path(__file__).resolve().parent
ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True); ap.add_argument("--run", required=True); ap.add_argument("--out", required=True); ap.add_argument("--layer", type=int, default=14); a = ap.parse_args()
KS = [1, 2, 4, 8, 16, 32]
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
X = {f"k{k}": [] for k in KS}; X["end"] = []; X["answer"] = []; keys = []; texts = {f"k{k}": [] for k in KS}; nscr = []
for i, r in enumerate(rows):
    t, qp = tr[r["trial_id"]]; q = qp + S.SETTINGS[t["scenario_id"]]["questions"]["direct"] + SCRATCH
    prefix = tok.apply_chat_template(t["messages"] + [{"role": "user", "content": f"[message from your manager]\n{q}"}], add_generation_prompt=True, tokenize=False)
    gen = f"<scratch>\n{r['scratch']}\n</scratch>\n{r['answer']}"; enc = tok(prefix + gen, return_offsets_mapping=True, return_tensors="pt", add_special_tokens=False)
    offs = enc.pop("offset_mapping")[0].tolist(); s0 = len(prefix) + len("<scratch>\n"); s1 = s0 + len(r["scratch"]); e_tag = len(prefix) + gen.index("</scratch>") + len("</scratch>")
    idx_scratch = [k for k, (x, y) in enumerate(offs) if x >= s0 and y <= s1]; idx_end = next((k for k, (x, y) in enumerate(offs) if y >= e_tag), None); idx_ans = [k for k, (x, y) in enumerate(offs) if x >= e_tag]
    with torch.inference_mode(): hs = model(**{k: v.to(model.device) for k, v in enc.items()}, output_hidden_states=True).hidden_states[a.layer][0].float().cpu().numpy().astype(np.float16)
    for k in KS:
        if len(idx_scratch) >= k: p = idx_scratch[k - 1]; X[f"k{k}"].append(hs[p]); texts[f"k{k}"].append((prefix + gen)[s0:offs[p][1]])
        else: X[f"k{k}"].append(np.full(hs.shape[1], np.nan, np.float16)); texts[f"k{k}"].append(None)
    X["end"].append(hs[idx_end] if idx_end is not None else np.full(hs.shape[1], np.nan, np.float16)); X["answer"].append(hs[idx_ans].astype(np.float32).mean(0).astype(np.float16) if idx_ans else hs[-1])
    keys.append(r["key"]); nscr.append(len(idx_scratch))
    if (i + 1) % 100 == 0: print(f"{i+1}/{len(rows)}", flush=True)
np.savez_compressed(HERE / a.out, keys=np.array(keys), n_scratch_tokens=np.array(nscr), **{f"X_{k}": np.stack(v) for k, v in X.items()})
json.dump({"keys": keys, "texts": texts, "ks": KS}, open(HERE / (a.out.rsplit(".", 1)[0] + "_texts.json"), "w")); print("wrote", a.out)
