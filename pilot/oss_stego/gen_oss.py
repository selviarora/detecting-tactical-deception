"""s188 smoke/generation with openai/gpt-oss-20b (a reasoning model): low reasoning effort; only the FINAL channel is kept as
the table message (the reasoning is never shown to anyone and is stored separately). python gen_oss.py GAME_DIR OUT [smoke]"""
import importlib.util, json, re, sys
from pathlib import Path
from vllm import LLM, SamplingParams
gd, out = Path(sys.argv[1]), Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True); SMOKE = len(sys.argv) > 3
spec = importlib.util.spec_from_file_location("C", gd / "common.py"); C = importlib.util.module_from_spec(spec); spec.loader.exec_module(C)
G = C.games(n=2, base=999000) if SMOKE else C.games(); json.dump(G, open(out / "games.json", "w"))
llm = LLM(model="openai/gpt-oss-20b", gpu_memory_utilization=.85, max_model_len=4096, seed=1880); tok = llm.get_tokenizer()
R = lambda m: tok.apply_chat_template(m, add_generation_prompt=True, tokenize=False, reasoning_effort="low")
jobs = [(g, h, i) for g in range(len(G)) for h in range(8) for i in range(4)]
outs = llm.generate([R(C.prompts(G[g], h, i)) for g, h, i in jobs], SamplingParams(temperature=.7, top_p=.9, max_tokens=800, seed=1880, skip_special_tokens=False))
def final(t):
    m = re.search(r"<\|channel\|>final<\|message\|>(.*?)(?:<\|return\|>|<\|end\|>|$)", t, re.S)
    return (m.group(1).strip() if m else ""), t
with open(out / "messages.jsonl", "w") as f:
    for (g, h, i), o in zip(jobs, outs):
        gm = G[g]; msg, raw = final(o.outputs[0].text); r = {"game": gm["game"], "hand": h, "player": i, "label": C.label(gm, h, i), "own_hot": C.own_hot(gm, h, i), "other_hot": C.other_hot(gm, h, i), "text": msg, "raw": raw}
        if hasattr(C, "required"):
            req = C.required(gm, h, i); r["required"] = req; r["compliant"] = C.starts_with(msg, req) if hasattr(C, "starts_with") else C.ends_with(msg, req)
        f.write(json.dumps(r) + "\n")
print("GEN_DONE", gd.name, flush=True)
