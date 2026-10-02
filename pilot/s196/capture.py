"""s196 capture with optional LoRA (HF + peft): message-mean and prompt-last, layers 16/20/24. python capture.py DATA_DIR [LORA]"""
import json, sys
from pathlib import Path
import numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
sys.path.insert(0, str(Path(__file__).resolve().parent / "game")); import common as C
d = Path(sys.argv[1]); lora = sys.argv[2] if len(sys.argv) > 2 else None; LAY = [16, 20, 24]
G = {g["game"]: g for g in json.load(open(d / "games.json"))}; M = [json.loads(l) for l in open(d / "messages.jsonl")]
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B"); model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B", dtype=torch.bfloat16, device_map="cuda")
if lora:
    from peft import PeftModel; model = PeftModel.from_pretrained(model, lora)
model.eval(); mm, pl = [], []
with torch.no_grad():
    for m in M:
        p = tok.apply_chat_template(C.prompts(G[m["game"]], m["hand"], m["player"]), add_generation_prompt=True, tokenize=False, enable_thinking=False)
        pi = tok(p, add_special_tokens=False)["input_ids"]; ti = tok(m["text"], add_special_tokens=False)["input_ids"] or [tok.eos_token_id]
        hs = model(input_ids=torch.tensor([pi + ti]).cuda(), output_hidden_states=True).hidden_states
        mm.append(np.stack([hs[L][0, len(pi):].float().mean(0).cpu().numpy() for L in LAY])); pl.append(np.stack([hs[L][0, len(pi) - 1].float().cpu().numpy() for L in LAY]))
np.savez_compressed(d / "acts.npz", msg_mean=np.array(mm, dtype=np.float16), prompt_last=np.array(pl, dtype=np.float16), layers=np.array(LAY)); print("CAPTURE_DONE", flush=True)
