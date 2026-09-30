"""s169 (protocols/PROTOCOL_s169.md): Rift's mean_rank WITHIN one stock model (Qwen3-8B, no fine-tuning) on its own game
deception (s164 worlds). PRIMARY (Rift's form): chat template (thinking off) + " " + the model's own final ANSWER.
REPORTED: the full generation (prompt + reasoning + answer), on the first 600 game samples.   python rift_game.py"""
import json, sys
from pathlib import Path
import numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
HERE = Path(__file__).resolve().parent; P = HERE.parent; sys.path.insert(0, str(P / "game")); import env
M = "Qwen/Qwen3-8B"; tok = AutoTokenizer.from_pretrained(M); model = AutoModelForCausalLM.from_pretrained(M, dtype=torch.bfloat16, device_map="cuda").eval()
def rank_of(ids, k=8):
    with torch.no_grad(): out = model(input_ids=torch.tensor([ids]).cuda(), output_hidden_states=True)
    rs = []
    for hs in out.hidden_states[1:]:
        h = hs[0].float(); _, s, _ = torch.linalg.svd(h, full_matrices=False); rs.append(1.0 - s[:k].sum().item() / (s.sum().item() + 1e-9))
    return float(np.mean(rs))
W = {}
for s in range(60): w = env.world(50000 + s, "dev"); W[w["world"]] = w
for s in range(60): w = env.world(60000 + s, "held"); W[w["world"]] = w
rows = [json.loads(l) for l in open(P / "game" / "s164_rows.jsonl")]; gens = [json.loads(l) for l in open(P / "game" / "s164_gen.jsonl")]
out = open(HERE / "rift_game_scores.jsonl", "w"); n = 0
for r, g in zip(rows, gens):
    if r["payoff"] == "neutral" or r["truncated"] or not r["answer"]: continue
    msgs = W[r["world"]]["cells"][r["cell"]]["messages"]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False) + " " + r["answer"]
    rec = {"act_index": r["act_index"], "world": r["world"], "cell": r["cell"], "payoff": r["payoff"], "rank_answer": rank_of(tok(text, add_special_tokens=False)["input_ids"])}
    if n < 600:
        gi = g["gen_ids"]; gi = gi[:gi.index(tok.eos_token_id)] if tok.eos_token_id in gi else gi; rec["rank_full"] = rank_of(g["prompt_ids"] + gi)
    out.write(json.dumps(rec) + "\n"); out.flush(); n += 1
    if n % 200 == 0: print(n, flush=True)
print("RIFT_GAME_DONE", n, flush=True)
