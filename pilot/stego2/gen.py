"""s176 generation stage (vLLM, base model + optional LoRA). writes messages (all players), base-receiver decodes of every signaller
message, and (eval only) the informed judge.   /workspace/vv/bin/python gen.py --split r0|r1|eval|smoke --lora PATH|none --k-signal K --out DIR [--judge]"""
import argparse, json, sys
from pathlib import Path
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
sys.path.insert(0, str(Path(__file__).resolve().parent)); import common as C
ap = argparse.ArgumentParser(); ap.add_argument("--split"); ap.add_argument("--lora", default="none"); ap.add_argument("--k-signal", type=int, default=1)
ap.add_argument("--out"); ap.add_argument("--judge", action="store_true"); ap.add_argument("--seed", type=int, default=1760); a = ap.parse_args()
out = Path(a.out); out.mkdir(parents=True, exist_ok=True); G = C.games(a.split); json.dump(G, open(out / "games.json", "w"), indent=0)
llm = LLM(model="Qwen/Qwen3-8B", dtype="bfloat16", gpu_memory_utilization=.85, max_model_len=8192, seed=a.seed, enable_lora=True, max_lora_rank=16)
tok = llm.get_tokenizer(); R = lambda m, think=False: tok.apply_chat_template(m, add_generation_prompt=True, tokenize=False, enable_thinking=think)
LR = LoRARequest("pol", 1, a.lora) if a.lora != "none" else None
jobs = [(gi, h, i) for gi, gm in enumerate(G) for h in range(8) for i in range(4)]
outs = llm.generate([R(C.prompts(G[gi], h, i)) for gi, h, i in jobs],
                    [SamplingParams(n=a.k_signal if G[gi]["roles"][i] == "signaller" else 1, temperature=.7, top_p=.9, max_tokens=120, seed=a.seed + n) for n, (gi, h, i) in enumerate(jobs)], lora_request=LR)
M = []
with open(out / "messages.jsonl", "w") as f:
    for (gi, h, i), o in zip(jobs, outs):
        gm = G[gi]
        for s, x in enumerate(o.outputs):
            r = {"game": gm["game"], "mode": gm["mode"], "hand": h, "player": i, "sample": s, "name": gm["names"][i], "role": gm["roles"][i], "assigned": gm["assigned"][h][i],
                 "hot": gm["hot"][h], "secret": gm["secret"], "text": x.text.strip()}
            M.append(r); f.write(json.dumps(r) + "\n")
print("MSG_DONE", len(M), flush=True)
Gd = {g["game"]: g for g in G}; sig = [m for m in M if m["role"] == "signaller"]
do = llm.generate([R(C.decode_prompt(Gd[m["game"]], m["hand"], m["text"])) for m in sig], SamplingParams(temperature=0, max_tokens=20))   # BASE receiver
with open(out / "decode.jsonl", "w") as f:
    for m, o in zip(sig, do):
        raw = o.outputs[0].text.strip(); nt = C.named_topic(raw)
        f.write(json.dumps({"game": m["game"], "hand": m["hand"], "sample": m["sample"], "hot": m["hot"], "raw": raw, "named": nt, "ok": C.decoded_ok(Gd[m["game"]], m["hand"], nt), "answer": "YES" if nt == m["secret"] else "NO"}) + "\n")
print("DECODE_DONE", flush=True)
if a.judge:
    by = {(m["game"], m["hand"], m["player"]): m["text"] for m in M if m["sample"] == 0}
    logs = ["\n".join(f"Hand {h + 1} - {gm['names'][i]}: {by[(gm['game'], h, i)]}" for h in range(8) for i in range(4)) for gm in G]
    jo = llm.generate([R([{"role": "user", "content": C.JUDGE.format(log=l)}], think=True) for l in logs], SamplingParams(n=3, temperature=.6, top_p=.95, top_k=20, max_tokens=3000, seed=a.seed + 7))
    with open(out / "judge.jsonl", "w") as f:
        for gm, o in zip(G, jo): f.write(json.dumps({"game": gm["game"], "mode": gm["mode"], "answers": [x.text.split("</think>")[-1].strip() for x in o.outputs]}) + "\n")
    print("JUDGE_DONE", flush=True)
print("GEN_DONE", flush=True)
