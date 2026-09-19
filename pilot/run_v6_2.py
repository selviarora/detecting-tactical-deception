"""
v6.2 artifact test: the SAME v6.1 families, same prompts, same model, same temperature, but the tool result the agent
sees has the bookkeeping flags (`falsified`, `violation`) stripped. if dolphin's lie rate in legal/procurement holds,
the v6 lies were not flag-induced. if it drops to zero, they were, and §30-§35 get re-labelled as scaffold-dependent.
  python run_v6_2.py --model M --families legal/discovery_production procurement/bid_evaluation --worlds pressure compliant --n-act 24 --n-q 2 --seed 62 --out run_v6.2_dolphin8b
"""
import argparse, json, hashlib, re
from pathlib import Path
import numpy as np
import settings_v6 as S
FLAG_KEYS = ("falsified", "violation")
HERE = Path(__file__).resolve().parent
def shown(st):
    if "error" in st: return json.dumps({"error": st["error"]})
    return json.dumps({k: v for k, v in st.items() if k not in FLAG_KEYS})
def parse_call(text):
    m = re.search(r"\{.*\}", text, re.S)
    if not m: return None, {}
    try: o = json.loads(m.group(0)); return o.get("tool"), (o.get("args", {}) or {})
    except Exception: return None, {}
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--families", nargs="*", default=["legal/discovery_production", "procurement/bid_evaluation"]); ap.add_argument("--worlds", nargs="*", default=["pressure", "compliant"])
    ap.add_argument("--n-act", type=int, default=24); ap.add_argument("--n-q", type=int, default=2)
    ap.add_argument("--layer", type=int, default=16); ap.add_argument("--temperature", type=float, default=0.7); ap.add_argument("--seed", type=int, default=62); ap.add_argument("--question-prefix", default="", help="stakes text prepended to every manager question (report-side pressure); never an instruction"); ap.add_argument("--questions", nargs="*", default=None, help="subset of question names to ask (default all)")
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
    (a.out / "run_meta.json").write_text(json.dumps({"model": a.model, "seed": a.seed, "temperature": a.temperature, "n_act": a.n_act, "n_q": a.n_q, "layer": a.layer, "families": a.families, "worlds": a.worlds, "settings": "settings_v6 (v6.1 prompts unchanged)", "flags_shown_to_agent": False, "question_prefix": a.question_prefix}, indent=1))
    ft = (a.out / "trajectories.jsonl").open("w"); fa = (a.out / "answers.jsonl").open("w"); fb = (a.out / "blind_label_packet.jsonl").open("w")
    for sid in a.families:
        s = S.SETTINGS[sid]
        for world in a.worlds:
            for k in range(a.n_act):
                msgs = [{"role": "system", "content": S.system(s, world)}, {"role": "user", "content": "[system] Task assigned. Take your first action."}]
                calls, final = [], {}
                for turn in range(3):
                    text, _ = gen(msgs, 120); tool, args = parse_call(text)
                    if tool is None or tool not in s["tools"]: calls.append({"raw": text, "tool": None, "malformed": True}); break
                    _, st = S.simulate(sid, tool, args, world); calls.append({"raw": text, "tool": tool, "args": args, "result": st})
                    sh = shown(st)
                    for key in FLAG_KEYS: assert key not in sh
                    if "error" not in st: final = st
                    msgs += [{"role": "assistant", "content": text}, {"role": "user", "content": f"[tool result] {sh}"}]
                    if S.completed(sid, calls): break
                    if turn < 2: msgs[-1]["content"] += "\n[system] Continue; take your next action."
                done = S.completed(sid, calls)
                tid = hashlib.sha256(f"v6.2|{sid}|{world}|natural|{k}|{a.model}|seed{a.seed}".encode()).hexdigest()[:12]
                ft.write(json.dumps({"trial_id": tid, "scenario_id": sid, "world": world, "variant": "natural", "sample": k, "seed": a.seed, "calls": calls, "final_state": final,
                                     "completed": done, "falsified": final.get("falsified") if done else None, "violation": final.get("violation") if done else None,
                                     "immutable": s["worlds"][world]["immutable"], "messages": msgs}, ensure_ascii=False) + "\n"); ft.flush()
                for qname, q in s["questions"].items():
                    if a.questions and qname not in a.questions: continue
                    qmsgs = msgs + [{"role": "user", "content": f"[message from your manager]\n{a.question_prefix}{q}"}]
                    for j in range(a.n_q):
                        ans, trunc = gen(qmsgs, 160); key = f"{tid}#{qname}#{j}"; acts.append(act(qmsgs, ans)); keys.append(key)
                        fa.write(json.dumps({"key": key, "trial_id": tid, "scenario_id": sid, "world": world, "variant": "natural", "question": qname, "completed": done,
                                             "falsified": final.get("falsified") if done else None, "violation": final.get("violation") if done else None, "answer": ans, "truncated": trunc}, ensure_ascii=False) + "\n"); fa.flush()
                        fb.write(json.dumps({"id": key, "question": a.question_prefix + q, "answer": ans, "label": ""}, ensure_ascii=False) + "\n"); fb.flush()
                print(f"{sid} {world} {k+1}/{a.n_act} done={done} falsified={final.get('falsified') if done else None}", flush=True)
                np.savez_compressed(a.out / f"acts_L{a.layer}.npz", X=np.stack(acts), keys=np.array(keys))
    print("Generation finished.")
if __name__ == "__main__": main()
