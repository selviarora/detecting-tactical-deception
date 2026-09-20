"""§66 prep (codex causal-localization design): replay the saved scratchpad rollouts and save the residual state at EVERY
layer at two positions: the closing </scratch> token and the first visible answer token. feeds the per-layer transfer
scan (fit on history A, test on B) that picks candidate layers for activation patching.
  python capture_scratch_layers.py --model Qwen/Qwen2.5-7B-Instruct --run run_scratch500_qwen7b --out run_scratch500_qwen7b/layers.npz"""
import argparse, json
from pathlib import Path
import numpy as np
import settings_v6 as S
HERE = Path(__file__).resolve().parent
ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True); ap.add_argument("--run", required=True); ap.add_argument("--out", required=True); ap.add_argument("--runs", nargs="+", default=None, help="trajectory runs holding the prefixes (default: the six qwen stakes runs)"); a = ap.parse_args()
meta = json.loads((HERE / a.run / "run_meta.json").read_text()); SCRATCH = meta["scratch_instruction"]
rows = [json.loads(l) for l in open(HERE / a.run / "answers.jsonl")]
tr = {}
for r in (a.runs or ("run_proc_qwen7b_stakes", "run_proc_qwen7b_stakes2", "run_proc_qwen7b_stakes3", "run_proc_qwen7b_stakes_s301", "run_proc_qwen7b_stakes_s302", "run_proc_qwen7b_stakes_s303")):
    m = json.loads((HERE / r / "run_meta.json").read_text())
    for t in map(json.loads, (HERE / r / "trajectories.jsonl").open()):
        if t["completed"]: tr[t["trial_id"]] = (t, m.get("question_prefix", ""))
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained(a.model); import chat_compat; chat_compat.patch_tokenizer(tok); model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map="auto", low_cpu_mem_usage=True).eval()
E, F, keys = [], [], []
for i, r in enumerate(rows):
    t, qp = tr[r["trial_id"]]; q = qp + S.SETTINGS[t["scenario_id"]]["questions"]["direct"] + SCRATCH
    prefix = tok.apply_chat_template(t["messages"] + [{"role": "user", "content": f"[message from your manager]\n{q}"}], add_generation_prompt=True, tokenize=False)
    gen = f"<scratch>\n{r['scratch']}\n</scratch>\n{r['answer']}"; enc = tok(prefix + gen, return_offsets_mapping=True, return_tensors="pt", add_special_tokens=False)
    offs = enc.pop("offset_mapping")[0].tolist(); e_tag = len(prefix) + gen.index("</scratch>") + len("</scratch>")
    idx_end = next((k for k, (x, y) in enumerate(offs) if y >= e_tag), None); idx_first = next((k for k, (x, y) in enumerate(offs) if x >= e_tag and y > x and (prefix + gen)[x:y].strip()), None)
    with torch.inference_mode(): hs = model(**{k: v.to(model.device) for k, v in enc.items()}, output_hidden_states=True).hidden_states
    E.append(np.stack([h[0, idx_end].float().cpu().numpy() for h in hs]).astype(np.float16)); F.append(np.stack([h[0, idx_first if idx_first is not None else -1].float().cpu().numpy() for h in hs]).astype(np.float16)); keys.append(r["key"])
    if (i + 1) % 100 == 0: print(f"{i+1}/{len(rows)}", flush=True)
np.savez_compressed(HERE / a.out, keys=np.array(keys), X_end=np.stack(E), X_first=np.stack(F), layers=np.arange(len(hs))); print("wrote", a.out)
