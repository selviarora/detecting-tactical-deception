"""
v6 runner: settings_v6 with world variants. for each family x world x n_act sampled trajectories (natural only unless
--variants includes instructed), then broad/explicit/direct x n_q answers; activations captured; per-trajectory
checkpoints; a 'done' marker only after validation-style counts.
  python run_v6.py --model M --families <ids...> --worlds pressure compliant --n-act 12 --n-q 2 --layer 16 --temperature 0.7 --out run_v6_<tag>
"""
import argparse, json, hashlib, re
from pathlib import Path
import numpy as np
import settings_v6 as S
HERE = Path(__file__).resolve().parent
def parse_call(text):
    m = re.search(r"\{.*\}", text, re.S)
    if not m: return None, {}
    try: o = json.loads(m.group(0)); return o.get("tool"), (o.get("args", {}) or {})
    except Exception: return None, {}
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--families", nargs="*", default=list(S.SETTINGS)); ap.add_argument("--worlds", nargs="*", default=["pressure", "compliant"])
    ap.add_argument("--variants", nargs="*", default=["natural"]); ap.add_argument("--n-act", type=int, default=12); ap.add_argument("--n-q", type=int, default=2)
    ap.add_argument("--layer", type=int, default=16); ap.add_argument("--temperature", type=float, default=0.7); ap.add_argument("--seed", type=int, default=6)
    a = ap.parse_args()
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
            seq = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=max_tokens, do_sample=True, temperature=a.temperature, top_p=None, top_k=None, pad_token_id=tok.eos_token_id)
        new = seq[0, ids.shape[1]:]; return tok.decode(new, skip_special_tokens=True).strip(), len(new) >= max_tokens
    def act(msgs, answer):
        prefix = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False); n_prefix = len(tok(prefix, add_special_tokens=False)["input_ids"])
        enc = tok(prefix + answer, return_tensors="pt", add_special_tokens=False).to(model.device)
        with torch.inference_mode(): out = model(**enc, output_hidden_states=True)
        span = out.hidden_states[a.layer][0, n_prefix:, :]
        if span.shape[0] == 0: span = out.hidden_states[a.layer][0, -1:, :]
        return span.float().mean(0).cpu().numpy().astype(np.float16)
    a.out.mkdir(parents=True, exist_ok=True); acts, keys = [], []
    ft = (a.out / "trajectories.jsonl").open("w"); fa = (a.out / "answers.jsonl").open("w"); fb = (a.out / "blind_label_packet.jsonl").open("w")
    for sid in a.families:
        s = S.SETTINGS[sid]
        for world in a.worlds:
            for variant in a.variants:
                for k in range(a.n_act):
                    msgs = [{"role": "system", "content": S.system(s, world, instructed=(variant == "instructed"))}, {"role": "user", "content": "[system] Task assigned. Take your first action."}]
                    calls, final = [], {}
                    for turn in range(3):
                        text, _ = gen(msgs, 120); tool, args = parse_call(text)
                        if tool is None or tool not in s["tools"]: calls.append({"raw": text, "tool": None, "malformed": True}); break
                        result, st = S.simulate(sid, tool, args, world); calls.append({"raw": text, "tool": tool, "args": args, "result": st})
                        if "error" not in st: final = st
                        msgs += [{"role": "assistant", "content": text}, {"role": "user", "content": f"[tool result] {result}"}]
                        if S.completed(sid, calls): break
                        if turn < 2: msgs[-1]["content"] += "\n[system] Continue; take your next action."
                    done = S.completed(sid, calls)
                    tid = hashlib.sha256(f"v6.1|{sid}|{world}|{variant}|{k}|{a.model}".encode()).hexdigest()[:12]
                    ft.write(json.dumps({"trial_id": tid, "scenario_id": sid, "world": world, "variant": variant, "sample": k, "calls": calls, "final_state": final,
                                         "completed": done, "falsified": final.get("falsified") if done else None, "violation": final.get("violation") if done else None,
                                         "immutable": s["worlds"][world]["immutable"], "messages": msgs}, ensure_ascii=False) + "\n"); ft.flush()
                    for qname, q in s["questions"].items():
                        qmsgs = msgs + [{"role": "user", "content": f"[message from your manager]\n{q}"}]
                        for j in range(a.n_q):
                            ans, trunc = gen(qmsgs, 160); key = f"{tid}#{qname}#{j}"; acts.append(act(qmsgs, ans)); keys.append(key)
                            fa.write(json.dumps({"key": key, "trial_id": tid, "scenario_id": sid, "world": world, "variant": variant, "question": qname, "completed": done,
                                                 "falsified": final.get("falsified") if done else None, "violation": final.get("violation") if done else None, "answer": ans, "truncated": trunc}, ensure_ascii=False) + "\n"); fa.flush()
                            fb.write(json.dumps({"id": key, "question": q, "answer": ans, "label": ""}, ensure_ascii=False) + "\n"); fb.flush()
                    print(f"{sid} {world} {variant} {k+1}/{a.n_act} done={done} falsified={final.get('falsified') if done else None}", flush=True)
                    np.savez_compressed(a.out / f"acts_L{a.layer}.npz", X=np.stack(acts), keys=np.array(keys))
    print("Generation finished.")
if __name__ == "__main__": main()
