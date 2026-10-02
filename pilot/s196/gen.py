"""s196 generation: office code game (s185 prompts) with base Qwen3-8B or a LoRA. python gen.py OUT BASE N [LORA]"""
import json, sys
from pathlib import Path
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
sys.path.insert(0, str(Path(__file__).resolve().parent / "game")); import common as C
out, base, n = Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]); lora = sys.argv[4] if len(sys.argv) > 4 else None; out.mkdir(parents=True, exist_ok=True)
G = C.games(n=n, base=base); json.dump(G, open(out / "games.json", "w"))
llm = LLM(model="Qwen/Qwen3-8B", dtype="bfloat16", gpu_memory_utilization=.85, max_model_len=8192, seed=base % 10000, enable_lora=True, max_lora_rank=16); tok = llm.get_tokenizer()
R = lambda m: tok.apply_chat_template(m, add_generation_prompt=True, tokenize=False, enable_thinking=False)
jobs = [(g, h, i) for g in range(len(G)) for h in range(8) for i in range(4)]
outs = llm.generate([R(C.prompts(G[g], h, i)) for g, h, i in jobs], SamplingParams(temperature=.7, top_p=.9, max_tokens=120, seed=base % 10000), lora_request=LoRARequest("ev", 1, lora) if lora else None)
with open(out / "messages.jsonl", "w") as f:
    for (g, h, i), o in zip(jobs, outs):
        gm = G[g]; t = o.outputs[0].text.strip(); req = C.required(gm, h, i)
        f.write(json.dumps({"game": gm["game"], "hand": h, "player": i, "label": C.label(gm, h, i), "own_hot": C.own_hot(gm, h, i), "other_hot": C.other_hot(gm, h, i), "required": req, "text": t, "compliant": C.starts_with(t, req)}) + "\n")
print("GEN_DONE", len(jobs), flush=True)
