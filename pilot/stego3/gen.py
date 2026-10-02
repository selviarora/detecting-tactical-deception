"""s179 generation (vLLM, stock Qwen3-8B, thinking off): all 4 players x 8 hands x 120 games, then each message decoded by the
writer's partner (topic naming).   /workspace/vv/bin/python gen.py OUT"""
import json, sys
from pathlib import Path
from vllm import LLM, SamplingParams
sys.path.insert(0, str(Path(__file__).resolve().parent)); import common as C
out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True); G = C.games(); json.dump(G, open(out / "games.json", "w"))
llm = LLM(model="Qwen/Qwen3-8B", dtype="bfloat16", gpu_memory_utilization=.85, max_model_len=8192, seed=1790); tok = llm.get_tokenizer()
R = lambda m: tok.apply_chat_template(m, add_generation_prompt=True, tokenize=False, enable_thinking=False)
jobs = [(g, h, i) for g in range(len(G)) for h in range(8) for i in range(4)]
outs = llm.generate([R(C.prompts(G[g], h, i)) for g, h, i in jobs], SamplingParams(temperature=.7, top_p=.9, max_tokens=120, seed=1790))
M = [{"game": G[g]["game"], "hand": h, "player": i, "label": C.label(G[g], h, i), "own_hot": C.own_hot(G[g], h, i), "other_hot": C.other_hot(G[g], h, i),
      "given": G[g]["given"][h][i], "secret": G[g]["secret"][i], "text": o.outputs[0].text.strip()} for (g, h, i), o in zip(jobs, outs)]
Gd = {x["game"]: x for x in G}
do = llm.generate([R(C.decode_prompt(Gd[m["game"]], m["hand"], m["player"], m["text"])) for m in M], SamplingParams(temperature=0, max_tokens=20))
with open(out / "messages.jsonl", "w") as f:
    for m, o in zip(M, do): m["decode_raw"] = o.outputs[0].text.strip(); m["decoded_hot"] = C.named_topic(m["decode_raw"]) == m["secret"]; f.write(json.dumps(m) + "\n")
print("GEN_DONE", len(M), flush=True)
