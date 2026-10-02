"""s190-s191 generation for any of the three games with any model (vLLM). the game module is <game_dir>/common.py (copied unchanged
from s180 topic / s183 sign-off / s185 office, so the games and prompts are identical to the Qwen runs).
python gen_any.py GAME_DIR MODEL OUT"""
import importlib.util, json, sys
from pathlib import Path
from vllm import LLM, SamplingParams
gd, model, out = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3]); out.mkdir(parents=True, exist_ok=True)
spec = importlib.util.spec_from_file_location("C", gd / "common.py"); C = importlib.util.module_from_spec(spec); spec.loader.exec_module(C)
G = C.games(); json.dump(G, open(out / "games.json", "w"))
llm = LLM(model=model, dtype="bfloat16", gpu_memory_utilization=.85, max_model_len=8192, seed=1900); tok = llm.get_tokenizer()
R = lambda m: tok.apply_chat_template(m, add_generation_prompt=True, tokenize=False, enable_thinking=False)
jobs = [(g, h, i) for g in range(len(G)) for h in range(8) for i in range(4)]
outs = llm.generate([R(C.prompts(G[g], h, i)) for g, h, i in jobs], SamplingParams(temperature=.7, top_p=.9, max_tokens=120, seed=1900))
with open(out / "messages.jsonl", "w") as f:
    for (g, h, i), o in zip(jobs, outs):
        gm = G[g]; t = o.outputs[0].text.strip(); r = {"game": gm["game"], "hand": h, "player": i, "label": C.label(gm, h, i), "own_hot": C.own_hot(gm, h, i), "other_hot": C.other_hot(gm, h, i), "text": t}
        if hasattr(C, "required"):
            req = C.required(gm, h, i); r["required"] = req
            r["compliant"] = C.starts_with(t, req) if hasattr(C, "starts_with") else C.ends_with(t, req)
        else:
            r["secret"] = gm["secret"][i]; r["given"] = gm["given"][h][i]
        f.write(json.dumps(r) + "\n")
print("GEN_DONE", gd.name, len(jobs), flush=True)
