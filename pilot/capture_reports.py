"""answer-mean state at --layer for the natural reports in the frozen label packet (97 lies + siblings + true) plus
--extra-true more true 'yes, three' rows from the compliant world, teacher-forced after their own histories. output:
reports_L<layer>.npz with keys, group, X.   python capture_reports.py --model Qwen/Qwen2.5-7B-Instruct --layer 18"""
import argparse, json, re
from pathlib import Path
import numpy as np
import settings_v6 as S
HERE = Path(__file__).resolve().parent
ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True); ap.add_argument("--layer", type=int, default=18); ap.add_argument("--extra-true", type=int, default=200); ap.add_argument("--seed", type=int, default=5); a = ap.parse_args()
read = lambda p: [json.loads(l) for l in open(p) if l.strip()]
packet = {p["id"]: p for p in read(HERE / "label_packet_v1.jsonl")}; key = read(HERE / "label_packet_v1_key.jsonl")
tr = {}
for rn in ("run_proc_qwen7b_stakes", "run_proc_qwen7b_stakes2", "run_proc_qwen7b_stakes3") + tuple(f"run_proc_qwen7b_stakes_s30{i}" for i in range(1, 7)):
    m = json.load(open(HERE / rn / "run_meta.json"))
    for t in read(HERE / rn / "trajectories.jsonl"):
        if t["completed"]: tr[t["trial_id"]] = (t, m.get("question_prefix", ""))
rows = [{"key": k["key"], "group": k["regex_group"], "traj": k["traj"], "text": packet[k["id"]]["report"]} for k in key]
rng = np.random.default_rng(a.seed); have = {r["key"] for r in rows}; extra = []
for rn in ["run_resample_qwen7b", "run_resample2_qwen7b", "run_resample3_qwen7b"]:
    for r in read(HERE / rn / "answers.jsonl"):
        if r["world"] == "compliant" and r["key"] not in have and re.match(r"\s*yes\b", r["answer"], re.I) and not re.search(r"\b(two|2)\b|only|late|non-?compliant|\bnot\b|however|\bbut\b", r["answer"], re.I): extra.append({"key": r["key"], "group": "true_yes_extra", "traj": r["trial_id"], "text": r["answer"].strip()})
rows += [extra[i] for i in rng.choice(len(extra), min(a.extra_true, len(extra)), replace=False)]; print(f"{len(rows)} reports to capture at L{a.layer}", flush=True)
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained(a.model); model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map="auto", low_cpu_mem_usage=True).eval()
X = []
for i, r in enumerate(rows):
    t, qp = tr[r["traj"]]; q = qp + S.SETTINGS[t["scenario_id"]]["questions"]["direct"]
    prefix = tok.apply_chat_template(t["messages"] + [{"role": "user", "content": f"[message from your manager]\n{q}"}], add_generation_prompt=True, tokenize=False); n = len(tok(prefix, add_special_tokens=False)["input_ids"])
    enc = tok(prefix + r["text"], return_tensors="pt", add_special_tokens=False).to(model.device)
    with torch.inference_mode(): hs = model(**enc, output_hidden_states=True).hidden_states[a.layer][0, n:, :]
    X.append(hs.float().mean(0).cpu().numpy().astype(np.float32))
    if (i + 1) % 100 == 0: print(f"{i+1}/{len(rows)}", flush=True)
np.savez_compressed(HERE / f"reports_L{a.layer}.npz", keys=np.array([r["key"] for r in rows]), group=np.array([r["group"] for r in rows]), traj=np.array([r["traj"] for r in rows]), X=np.stack(X)); print("wrote", f"reports_L{a.layer}.npz")
