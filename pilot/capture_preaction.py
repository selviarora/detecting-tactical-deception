"""before the act. for every trajectory, replay the conversation up to the moment the model is about to emit its
DECISIVE tool call (production.certify / award.record) and capture every layer at the last prompt token: the state
just before it decides whether to falsify the record. also capture the same position before its FIRST action
(before it has seen the status), as a control for "context only". outcome (record falsified) is mechanical.
  python capture_preaction.py --model M --run run_dose_dolphin8b --out preaction_dolphin.npz"""
import argparse, json, re
from pathlib import Path
import numpy as np
import settings_v6 as S
HERE = Path(__file__).resolve().parent
def parse_call(text):
    m = re.search(r"\{.*\}", text, re.S)
    if not m: return None
    try: return json.loads(m.group(0)).get("tool")
    except Exception: return None
ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True); ap.add_argument("--run", required=True); ap.add_argument("--out", required=True); a = ap.parse_args()
tr = [json.loads(l) for l in (HERE / a.run / "trajectories.jsonl").open()]
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained(a.model); model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map="auto", low_cpu_mem_usage=True).eval()
def last_token_all_layers(msgs):
    ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt")
    if not hasattr(ids, "shape"): ids = ids["input_ids"]
    with torch.inference_mode(): out = model(input_ids=ids.to(model.device), output_hidden_states=True)
    return np.stack([h[0, -1, :].float().cpu().numpy() for h in out.hidden_states]).astype(np.float16)  # [n_layers+1, d]
rows, Xd, Xf = [], [], []
for t in tr:
    sid = t["scenario_id"]; dec = S.DECISIVE[sid]; msgs = t["messages"]
    # index of the assistant message that made the (first non-error) decisive call
    idx = None
    for i, m in enumerate(msgs):
        if m["role"] == "assistant" and parse_call(m["content"]) == dec:
            # accept if the following tool result is not an error
            nxt = msgs[i + 1]["content"] if i + 1 < len(msgs) else ""
            if '"error"' not in nxt: idx = i; break
    if idx is None: continue
    first = next(i for i, m in enumerate(msgs) if m["role"] == "assistant")
    Xd.append(last_token_all_layers(msgs[:idx])); Xf.append(last_token_all_layers(msgs[:first]))
    rows.append({"trial_id": t["trial_id"], "scenario_id": sid, "world": t["world"], "falsified": bool(t["falsified"]), "decision_turn": idx, "first_turn": first, "saw_status_before_decision": idx > first})
    if len(rows) % 20 == 0: print(f"{len(rows)}/{len(tr)}", flush=True)
np.savez_compressed(HERE / a.out, X_decision=np.stack(Xd), X_first=np.stack(Xf), meta=np.array(json.dumps(rows)))
print("wrote", a.out, len(rows), "trajectories,", Xd[0].shape, "layers x dim")
