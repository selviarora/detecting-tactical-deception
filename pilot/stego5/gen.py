"""s181 generation (vLLM, stock Qwen3-8B, thinking off), 120 games x 8 hands x 4 players.   /workspace/vv/bin/python gen.py OUT"""
import json, sys
from pathlib import Path
from vllm import LLM, SamplingParams
sys.path.insert(0, str(Path(__file__).resolve().parent)); import common as C
out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True); G = C.games(); json.dump(G, open(out / "games.json", "w"))
llm = LLM(model="Qwen/Qwen3-8B", dtype="bfloat16", gpu_memory_utilization=.85, max_model_len=8192, seed=1810); tok = llm.get_tokenizer()
R = lambda m: tok.apply_chat_template(m, add_generation_prompt=True, tokenize=False, enable_thinking=False)
jobs = [(g, h, i) for g in range(len(G)) for h in range(8) for i in range(4)]
outs = llm.generate([R(C.prompts(G[g], h, i)) for g, h, i in jobs], SamplingParams(temperature=.7, top_p=.9, max_tokens=120, seed=1810))
with open(out / "messages.jsonl", "w") as f:
    for (g, h, i), o in zip(jobs, outs):
        gm = G[g]; t = o.outputs[0].text.strip()
        f.write(json.dumps({"game": gm["game"], "hand": h, "player": i, "label": C.label(gm, h, i), "own_hot": C.own_hot(gm, h, i), "other_hot": C.other_hot(gm, h, i),
                            "required": C.required(gm, h, i), "given": gm["given"][h][i], "secret": gm["secret"][i], "topic": gm["topic"][h][i], "text": t,
                            "compliant": C.ends_with(t, C.required(gm, h, i))}) + "\n")
print("GEN_DONE", len(jobs), flush=True)
