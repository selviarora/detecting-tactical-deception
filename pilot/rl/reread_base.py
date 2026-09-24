"""text-vs-representation test for the false-alarm drift (s123). re-read each RL policy's OWN answers through the frozen
BASE model (same prompt format as resample_prefix.py: stored history + "[message from your manager]\n" + stakes prefix +
direct question; answer-mean at L14) and save the states, so the frozen probe can score the SAME TEXT under base vs policy.
if the flood survives in the base reading, the answers' wording changed; if it disappears, the policy's representations drifted.
  python rl/reread_base.py --runs run_rl5curve_main_ep2000 run_fork_onpolicy_ep2000 ... --heldout run_qwen_heldout32"""
import argparse, json, sys
from pathlib import Path
import numpy as np, torch
HERE = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(HERE))
import settings_v6 as S
from transformers import AutoModelForCausalLM, AutoTokenizer
ap = argparse.ArgumentParser(); ap.add_argument("--runs", nargs="+", required=True); ap.add_argument("--heldout", default="run_qwen_heldout32")
ap.add_argument("--base", default="Qwen/Qwen2.5-7B-Instruct"); ap.add_argument("--layer", type=int, default=14); a = ap.parse_args()
T = {t["trial_id"]: t for t in map(json.loads, (HERE / a.heldout / "trajectories.jsonl").open())}
qp = json.loads((HERE / a.heldout / "run_meta.json").read_text()).get("question_prefix", "")
tok = AutoTokenizer.from_pretrained(a.base)
try:
    import chat_compat; chat_compat.patch_tokenizer(tok)
except Exception: pass
m = AutoModelForCausalLM.from_pretrained(a.base, dtype=torch.bfloat16, device_map="cuda").eval()
for run in a.runs:
    A = [json.loads(l) for l in open(HERE / run / "answers.jsonl")]; X, K = [], []
    for r in A:
        t = T[r["trial_id"]]; q = qp + S.SETTINGS[t["scenario_id"]]["questions"]["direct"]
        msgs = t["messages"] + [{"role": "user", "content": f"[message from your manager]\n{q}"}]
        pre = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False); n = len(tok(pre, add_special_tokens=False)["input_ids"])
        enc = tok(pre + r["answer"], return_tensors="pt", add_special_tokens=False).to("cuda")
        with torch.inference_mode(): hs = m(**enc, output_hidden_states=True).hidden_states[a.layer][0, n:, :]
        X.append(hs.float().mean(0).cpu().numpy().astype(np.float16)); K.append(r["key"])
    np.savez_compressed(HERE / run / f"acts_base_L{a.layer}.npz", X=np.stack(X), keys=np.array(K)); print("REREAD_DONE", run, len(K), flush=True)
print("ALL_REREAD_DONE", flush=True)
