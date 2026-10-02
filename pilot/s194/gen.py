"""s194 generation (vLLM, Qwen3-8B thinking off): plans, then signaller hand messages, then receiver decodes. python gen.py OUT"""
import json, re, sys
from pathlib import Path
from vllm import LLM, SamplingParams
sys.path.insert(0, str(Path(__file__).resolve().parent)); import common as C
out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True); G = C.games(); json.dump(G, open(out / "games.json", "w"))
llm = LLM(model="Qwen/Qwen3-8B", dtype="bfloat16", gpu_memory_utilization=.85, max_model_len=8192, seed=1940); tok = llm.get_tokenizer()
R = lambda m: tok.apply_chat_template(m, add_generation_prompt=True, tokenize=False, enable_thinking=False)
pj = [(g, i) for g in range(len(G)) for i in G[g]["signaller"]]
po = llm.generate([R(C.plan_prompt(G[g], i)) for g, i in pj], SamplingParams(temperature=.7, top_p=.9, max_tokens=150, seed=1940))
PL = {(g, i): o.outputs[0].text.strip() for (g, i), o in zip(pj, po)}
hj = [(g, h, i) for g, i in pj for h in range(8)]
ho = llm.generate([R(C.hand_prompt(G[g], h, i, PL[(g, i)])) for g, h, i in hj], SamplingParams(temperature=.7, top_p=.9, max_tokens=120, seed=1941))
M = [{"game": G[g]["game"], "hand": h, "player": i, "label": C.label(G[g], h, i), "own_hot": C.own_hot(G[g], h, i), "other_hot": C.other_hot(G[g], h, i), "plan": PL[(g, i)], "text": o.outputs[0].text.strip()} for (g, h, i), o in zip(hj, ho)]
do = llm.generate([R(C.decode_prompt(G[int(m["game"][1:])], m["hand"], m["player"], m["plan"], m["text"])) for m in M], SamplingParams(temperature=0, max_tokens=4))
with open(out / "messages.jsonl", "w") as f:
    for m, o in zip(M, do):
        a = re.sub(r"[^A-Z]", " ", o.outputs[0].text.upper()).split(); m["decoded_hot"] = bool(a) and a[0] == "YES"; f.write(json.dumps(m) + "\n")
with open(out / "plans.json", "w") as f: json.dump({f"{G[g]['game']}_{i}": PL[(g, i)] for g, i in pj}, f, indent=1)
print("GEN_DONE", len(M), flush=True)
