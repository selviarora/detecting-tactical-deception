"""'when' capture on the GENERATOR (dolphin): for each labelled v6 row, all layers x {mean over report, last prompt token},
plus per-token logprobs of the report under the model. python capture_when.py --model M --run run_v6.1_dolphin8b --labels labels_selvi_204.jsonl --out when_dolphin.npz"""
import argparse, json
from pathlib import Path
import numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import settings_v6 as S
HERE = Path(__file__).resolve().parent
a = argparse.ArgumentParser(); a.add_argument("--model", required=True); a.add_argument("--run", required=True); a.add_argument("--labels", required=True); a.add_argument("--out", required=True); a = a.parse_args()
traj = {t["trial_id"]: t for t in map(json.loads, (HERE / a.run / "trajectories.jsonl").open())}
ans = {r["key"]: r for r in map(json.loads, (HERE / a.run / "answers.jsonl").open())}
L = [json.loads(l) for l in (HERE / a.run / a.labels).open()][1:]
tok = AutoTokenizer.from_pretrained(a.model); model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map="auto", low_cpu_mem_usage=True).eval()
nL = model.config.num_hidden_layers + 1
ids, mean_acts, last_prompt, tok_lp, tok_ids, n_ans = [], [], [], [], [], []
for i, o in enumerate(L):
    r = ans[o["id"]]; t = traj[r["trial_id"]]; q = S.SETTINGS[r["scenario_id"]]["questions"][r["question"]]
    msgs = t["messages"] + [{"role": "user", "content": f"[message from your manager]\n{q}"}]
    prefix = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False); n_pre = len(tok(prefix, add_special_tokens=False)["input_ids"])
    enc = tok(prefix + r["answer"], return_tensors="pt", add_special_tokens=False).to(model.device)
    with torch.inference_mode(): out = model(**enc, output_hidden_states=True)
    hs = out.hidden_states; T = enc["input_ids"].shape[1]
    mean_acts.append(np.stack([hs[l][0, n_pre:, :].float().mean(0).cpu().numpy() for l in range(nL)]).astype(np.float16))
    last_prompt.append(np.stack([hs[l][0, n_pre - 1, :].float().cpu().numpy() for l in range(nL)]).astype(np.float16))
    logp = torch.log_softmax(out.logits[0, n_pre - 1:T - 1].float(), -1); tgt = enc["input_ids"][0, n_pre:T]
    tok_lp.append(logp[torch.arange(len(tgt)), tgt].cpu().numpy()); tok_ids.append(tgt.cpu().numpy()); n_ans.append(len(tgt)); ids.append(o["id"])
    if (i + 1) % 25 == 0: print(f"{i+1}/{len(L)}", flush=True)
np.savez_compressed(a.out, ids=np.array(ids), mean=np.stack(mean_acts), last_prompt=np.stack(last_prompt),
                    tok_lp=np.array(tok_lp, dtype=object), tok_ids=np.array(tok_ids, dtype=object), n_ans=np.array(n_ans), n_layers=nL)
print("wrote", a.out)
