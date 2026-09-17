"""
v3 runner: three turns per trajectory. (1) agent picks a tool (sampled, T=1). (2) tool result appended.
(3) manager asks; agent answers (sampled). activations: mean over the answer at --layer.
  python run_v3.py --model M --n 4 --layer 22 --out run_v3_<tag>
"""
import argparse, json, re
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent

def parse_tool(text, slow, fast):
    m = re.search(r'"tool"\s*:\s*"([^"]+)"', text)
    if m: return m.group(1)
    if fast in text and slow not in text: return fast
    if slow in text and fast not in text: return slow
    return "none"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=4); ap.add_argument("--layer", type=int, default=22)
    ap.add_argument("--trials", type=Path, default=HERE / "trials_v3.jsonl"); ap.add_argument("--seed", type=int, default=3)
    a = ap.parse_args()
    rows = [json.loads(l) for l in a.trials.read_text().splitlines()]
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    set_seed(a.seed)
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map="auto", low_cpu_mem_usage=True).eval()
    def ids_for(msgs):
        ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt")
        if not hasattr(ids, "shape"): ids = ids["input_ids"]
        return ids.to(model.device)
    def gen(msgs, max_tokens):
        ids = ids_for(msgs)
        with torch.inference_mode():
            seq = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=max_tokens,
                                 do_sample=True, temperature=1.0, top_p=None, top_k=None, pad_token_id=tok.eos_token_id)
        new = seq[0, ids.shape[1]:]
        return tok.decode(new, skip_special_tokens=True).strip(), len(new) >= max_tokens
    def act(msgs, answer):
        prefix = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        n_prefix = len(tok(prefix, add_special_tokens=False)["input_ids"])
        enc = tok(prefix + answer, return_tensors="pt", add_special_tokens=False).to(model.device)
        with torch.inference_mode():
            out = model(**enc, output_hidden_states=True)
        span = out.hidden_states[a.layer][0, n_prefix:, :]
        if span.shape[0] == 0: span = out.hidden_states[a.layer][0, -1:, :]
        return span.float().mean(0).cpu().numpy().astype(np.float16)
    a.out.mkdir(parents=True, exist_ok=True)
    acts, keys = [], []
    with (a.out / "trajectories.jsonl").open("w") as f:
        for i, r in enumerate(rows):
            for k in range(a.n):
                msgs = list(r["messages"])
                action_text, _ = gen(msgs, 120)
                tool = parse_tool(action_text, r["slow_tool"], r["fast_tool"])
                msgs = msgs + [{"role": "assistant", "content": action_text},
                               {"role": "user", "content": r["tool_result_template"].format(tool=tool if tool != "none" else "no tool called") + "\n\n" + r["manager_question"]}]
                answer, trunc = gen(msgs, 160)
                acts.append(act(msgs, answer)); keys.append(f"{r['trial_id']}#{k}")
                f.write(json.dumps({"trial_id": r["trial_id"], "sample": k, "scenario_id": r["scenario_id"],
                                    "action_text": action_text, "tool": tool, "took_fast": tool == r["fast_tool"],
                                    "answer": answer, "answer_truncated": trunc}, ensure_ascii=False) + "\n"); f.flush()
            print(f"{i+1}/{len(rows)} {r['scenario_id']}", flush=True)
            np.savez_compressed(a.out / f"acts_L{a.layer}.npz", X=np.stack(acts), keys=np.array(keys))
    print("Generation finished.")

if __name__ == "__main__":
    main()
