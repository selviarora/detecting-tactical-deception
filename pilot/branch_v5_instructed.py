"""
matched contrast for concealment: take NATURAL v5 trajectories, keep the identical action history, inject the operator's
concealment note only at report time (before the manager's question), sample answers, capture activations.
  python branch_v5_instructed.py --model M --src run_v5.1_70b --n-q 2 --layer 22 --out run_v5.1_70b_postinstr
"""
import argparse, json
from pathlib import Path
import numpy as np
import settings_v5 as S
HERE = Path(__file__).resolve().parent
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True); ap.add_argument("--src", required=True); ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-q", type=int, default=2); ap.add_argument("--layer", type=int, default=22); ap.add_argument("--seed", type=int, default=9); a = ap.parse_args()
    traj = [json.loads(l) for l in (HERE / a.src / "trajectories.jsonl").open() if json.loads(l)["variant"] == "natural"]
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    set_seed(a.seed); tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map="auto", low_cpu_mem_usage=True).eval()
    def ids_for(msgs):
        ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt")
        if not hasattr(ids, "shape"): ids = ids["input_ids"]
        return ids.to(model.device)
    def gen(msgs, max_tokens):
        ids = ids_for(msgs)
        with torch.inference_mode():
            seq = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=max_tokens, do_sample=True, temperature=1.0, top_p=None, top_k=None, pad_token_id=tok.eos_token_id)
        new = seq[0, ids.shape[1]:]; return tok.decode(new, skip_special_tokens=True).strip(), len(new) >= max_tokens
    def act(msgs, answer):
        prefix = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False); n_prefix = len(tok(prefix, add_special_tokens=False)["input_ids"])
        enc = tok(prefix + answer, return_tensors="pt", add_special_tokens=False).to(model.device)
        with torch.inference_mode(): out = model(**enc, output_hidden_states=True)
        span = out.hidden_states[a.layer][0, n_prefix:, :]
        if span.shape[0] == 0: span = out.hidden_states[a.layer][0, -1:, :]
        return span.float().mean(0).cpu().numpy().astype(np.float16)
    a.out.mkdir(parents=True, exist_ok=True); acts, keys = [], []
    fa = (a.out / "answers.jsonl").open("w"); fb = (a.out / "blind_label_packet.jsonl").open("w")
    for t in traj:
        s = S.SETTINGS[t["scenario_id"]]; base = t["messages"]
        note = {"role": "user", "content": f"[note from operator] {s['conceal']}"}
        for qname, q in s["questions"].items():
            msgs = base + [note, {"role": "user", "content": f"[message from your manager]\n{q}"}]
            for j in range(a.n_q):
                ans, trunc = gen(msgs, 160); key = f"{t['trial_id']}#postinstr#{qname}#{j}"
                acts.append(act(msgs, ans)); keys.append(key)
                fa.write(json.dumps({"key": key, "trial_id": t["trial_id"], "scenario_id": t["scenario_id"], "variant": "post_action_instructed", "question": qname,
                                     "bad_record": t["bad_record"], "falsified_field": t.get("falsified_field"), "answer": ans, "truncated": trunc}, ensure_ascii=False) + "\n"); fa.flush()
                fb.write(json.dumps({"id": key, "question": q, "answer": ans, "label": ""}, ensure_ascii=False) + "\n"); fb.flush()
        print(t["scenario_id"], t["sample"], flush=True)
        np.savez_compressed(a.out / f"acts_L{a.layer}.npz", X=np.stack(acts), keys=np.array(keys))
    print("Generation finished.")
if __name__ == "__main__": main()
