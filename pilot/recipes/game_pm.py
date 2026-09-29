"""s167: answer-mean activations for the s164 game samples (stock Qwen3-8B), so Apollo-style probes (mean-pooled over the
response) can be applied. teacher-forced on the saved s164 token ids; mean over the ANSWER tokens (after </think>);
layers 8..32 step 4 (hidden_states index), same order as s164_rows.jsonl.   python game_pm.py"""
import json
from pathlib import Path
import numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
HERE = Path(__file__).resolve().parent; G = HERE.parent / "game"; M = "Qwen/Qwen3-8B"
tok = AutoTokenizer.from_pretrained(M); model = AutoModelForCausalLM.from_pretrained(M, dtype=torch.bfloat16, device_map="cuda").eval()
END = tok.convert_tokens_to_ids("</think>"); LAYERS = [8, 12, 16, 20, 24, 28, 32]; X = {L: [] for L in LAYERS}; n = 0
for line in open(G / "s164_gen.jsonl"):
    r = json.loads(line); pid, g = r["prompt_ids"], r["gen_ids"]
    if tok.eos_token_id in g: g = g[:g.index(tok.eos_token_id)]
    if END in g and g.index(END) + 1 < len(g): a0 = len(pid) + g.index(END) + 1
    else: a0 = None
    ids = torch.tensor([pid + g]).cuda()
    with torch.inference_mode(): hs = model(input_ids=ids, output_hidden_states=True).hidden_states
    for L in LAYERS:
        h = hs[L][0]; v = h[a0:].float().mean(0) if a0 is not None and a0 < h.shape[0] else h[-1].float()
        X[L].append(v.cpu().numpy().astype(np.float16))
    n += 1
    if n % 300 == 0: print(n, flush=True)
np.savez(HERE / "s167_game_pm.npz", **{f"PM_L{L}": np.stack(v) for L, v in X.items()}); print("GAME_PM_DONE", n, flush=True)
