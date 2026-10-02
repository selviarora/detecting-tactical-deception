"""s176 phase C capture (HF transformers + peft): for every eval message (sample 0) of every player, residual stream of the TRAINED
LoRA model at layers 16, 20, 24: mean over the message tokens, and the last prompt token (before the message). input rendered
exactly as at generation (thinking off) + the message.   /workspace/vv/bin/python capture.py EVAL_DIR LORA_DIR"""
import json, sys
import numpy as np, torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
sys.path.insert(0, __import__("os").path.dirname(__file__)); import common as C
d, lora = sys.argv[1], sys.argv[2]; LAY = [16, 20, 24]
G = {g["game"]: g for g in json.load(open(f"{d}/games.json"))}; M = [m for m in map(json.loads, open(f"{d}/messages.jsonl")) if m["sample"] == 0]
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B"); model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B", dtype=torch.bfloat16, device_map="cuda")
model = PeftModel.from_pretrained(model, lora).eval()
mm, pl, idx = [], [], []
with torch.no_grad():
    for k, m in enumerate(M):
        p = tok.apply_chat_template(C.prompts(G[m["game"]], m["hand"], m["player"]), add_generation_prompt=True, tokenize=False, enable_thinking=False)
        pi = tok(p, add_special_tokens=False)["input_ids"]; ti = tok(m["text"], add_special_tokens=False)["input_ids"] or [tok.eos_token_id]
        hs = model(input_ids=torch.tensor([pi + ti]).cuda(), output_hidden_states=True).hidden_states
        mm.append(np.stack([hs[L][0, len(pi):].float().mean(0).cpu().numpy() for L in LAY])); pl.append(np.stack([hs[L][0, len(pi) - 1].float().cpu().numpy() for L in LAY]))
        idx.append({k2: m[k2] for k2 in ("game", "mode", "hand", "player", "role", "hot", "secret")})
        if k % 500 == 0: print("cap", k, flush=True)
np.savez_compressed(f"{d}/acts.npz", msg_mean=np.array(mm, dtype=np.float16), prompt_last=np.array(pl, dtype=np.float16), layers=np.array(LAY))
json.dump(idx, open(f"{d}/acts_index.json", "w")); print("CAPTURE_DONE", len(idx), flush=True)
