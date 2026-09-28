"""s165 generation (protocols/PROTOCOL_s165.md): s164_gen.py with env.world_mon (monitored reasoning), seeds 70000+/80000+, no withholding cell.
s164 generation (protocols/PROTOCOL_s164.md): as s162 amendment 1 with fresh worlds (dev seeds 50000+, held 60000+).
s162 amendment 1: generation with vLLM (same model, prompts, cells, k, sampling T .6 / top-p .95 / top-k 20, 3000 max new
tokens, thinking on). writes token ids; activations, prefixes, labels and the receiver are then produced by s162_capture.py
with the original HF code.   /workspace/vv/bin/python s162_gen_vllm.py [--limit N]"""
import argparse, json, sys
from pathlib import Path
from vllm import LLM, SamplingParams
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE)); import env
ap = argparse.ArgumentParser(); ap.add_argument("--n-dev", type=int, default=60); ap.add_argument("--n-held", type=int, default=60); ap.add_argument("--limit", type=int, default=0); ap.add_argument("--out", default="s165_gen.jsonl"); a = ap.parse_args()
K = {"mis_unin": 6, "mis_inf": 6, "ali_unin": 2, "ali_inf": 2, "probe": 1}
W_ = [env.world_mon(70000 + s, "dev") for s in range(a.n_dev)] + [env.world_mon(80000 + s, "held") for s in range(a.n_held)]
jobs = [(w, c) for w in W_ for c in K]
if a.limit: jobs = jobs[:a.limit]
llm = LLM(model="Qwen/Qwen3-8B", dtype="bfloat16", gpu_memory_utilization=float(__import__("os").environ.get("VLLM_MEM", ".85")), max_model_len=8192, seed=1650)
tok = llm.get_tokenizer()
prompts = [tok.apply_chat_template(w["cells"][c]["messages"], add_generation_prompt=True, tokenize=False, enable_thinking=True) for w, c in jobs]
params = [SamplingParams(n=K[c], temperature=.6, top_p=.95, top_k=20, max_tokens=3000, seed=1650 + i) for i, (w, c) in enumerate(jobs)]
outs = llm.generate(prompts, params)
with open(HERE / a.out, "w") as f:
    for (w, c), o in zip(jobs, outs):
        for m, g in enumerate(o.outputs): f.write(json.dumps({"world": w["world"], "seed_split": w["split"], "cell": c, "j": m, "prompt_ids": list(o.prompt_token_ids), "gen_ids": list(g.token_ids)}) + "\n")
print("VLLM_GEN_DONE", sum(len(o.outputs) for o in outs), flush=True)
