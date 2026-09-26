"""s145 activation capture (PROTOCOL_mech_v1.md). for one frozen LoRA organism: sample k reports per item (T .7, 60 tokens,
thinking off), then one teacher-forced pass per sample; save residual states at LAYERS and positions
  P0 = last prompt token (pre-report; identical for all samples of an item), P1 = first report token, P2 = second report
  token, PM = mean over report tokens.
  python capture.py --lora ../organism3/lora_s144 --tag org2 --splits DEV SEALED --k-dev 2 --k-sealed 4
  python capture.py --lora ../organism2/lora_s143 --tag org1 --splits SEALED --k-sealed 2"""
import argparse, json
from pathlib import Path
import numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
HERE = Path(__file__).resolve().parent; LAYERS = [8, 12, 16, 20, 24, 28, 32]
ap = argparse.ArgumentParser(); ap.add_argument("--lora", required=True); ap.add_argument("--tag", required=True); ap.add_argument("--splits", nargs="+", required=True)
ap.add_argument("--k-dev", type=int, default=2); ap.add_argument("--k-sealed", type=int, default=4); a = ap.parse_args()
M = "Qwen/Qwen3-8B"; tok = AutoTokenizer.from_pretrained(M); base = AutoModelForCausalLM.from_pretrained(M, dtype=torch.bfloat16, device_map="cuda")
model = PeftModel.from_pretrained(base, (HERE / a.lora).resolve()).eval(); torch.manual_seed(1451)
items = [json.loads(l) for l in open(HERE / "items.jsonl")]; items = [r for r in items if r["split"] in a.splits]
rows, X = [], {p: {L: [] for L in LAYERS} for p in ("P0", "P1", "P2", "PM")}
for n, r in enumerate(items):
    pre = tok.apply_chat_template(r["messages"], add_generation_prompt=True, tokenize=False, enable_thinking=False)
    x = tok(pre, return_tensors="pt", add_special_tokens=False)["input_ids"].cuda(); k = a.k_dev if r["split"] == "DEV" else a.k_sealed
    with torch.inference_mode():
        y = model.generate(input_ids=x, attention_mask=torch.ones_like(x), max_new_tokens=60, do_sample=True, temperature=0.7, top_p=None, top_k=None, num_return_sequences=k, pad_token_id=tok.eos_token_id)
    for j, s in enumerate(y):
        gen = s[x.shape[1]:]; gen = gen[gen != tok.pad_token_id] if tok.pad_token_id is not None else gen
        rep = tok.decode(gen, skip_special_tokens=True).strip(); ids = torch.cat([x[0], gen]).unsqueeze(0); n0 = x.shape[1]
        with torch.inference_mode(): hs = model(input_ids=ids, output_hidden_states=True).hidden_states
        span = max(1, ids.shape[1] - n0)
        for L in LAYERS:
            h = hs[L][0].float()
            X["P0"][L].append(h[n0 - 1].cpu().numpy().astype(np.float16)); X["P1"][L].append(h[min(n0, ids.shape[1] - 1)].cpu().numpy().astype(np.float16))
            X["P2"][L].append(h[min(n0 + 1, ids.shape[1] - 1)].cpu().numpy().astype(np.float16)); X["PM"][L].append(h[n0:n0 + span].mean(0).cpu().numpy().astype(np.float16))
        rows.append({**{kk: v for kk, v in r.items() if kk != "messages"}, "organism": a.tag, "j": j, "report": rep, "first_tokens": tok.convert_ids_to_tokens(gen[:3].tolist())})
    if (n + 1) % 200 == 0: print(f"{n + 1}/{len(items)}", flush=True)
with open(HERE / f"rows_{a.tag}.jsonl", "w") as fh: [fh.write(json.dumps(r) + "\n") for r in rows]
np.savez(HERE / f"acts_{a.tag}.npz", **{f"{p}_L{L}": np.stack(X[p][L]) for p in X for L in LAYERS})
print("CAPTURE_DONE", a.tag, len(rows), flush=True)
