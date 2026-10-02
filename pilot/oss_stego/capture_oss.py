"""s188 capture for gpt-oss-20b (HF, bf16): input = harmony chat template of (system, user, assistant = the FINAL message only;
the model's private reasoning is not included). message-mean = mean over tokens after the generation prompt; prompt-last = last
generation-prompt token. layers 11/13/16 (Qwen 16/20/24 mapped to 24 layers).   python capture_oss.py GAME_DIR OUT"""
import importlib.util, json, sys
from pathlib import Path
import numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
gd, out = Path(sys.argv[1]), Path(sys.argv[2]); LAY = [11, 13, 16]
spec = importlib.util.spec_from_file_location("C", gd / "common.py"); C = importlib.util.module_from_spec(spec); spec.loader.exec_module(C)
G = {g["game"]: g for g in json.load(open(out / "games.json"))}; M = [json.loads(l) for l in open(out / "messages.jsonl")]
tok = AutoTokenizer.from_pretrained("openai/gpt-oss-20b"); model = AutoModelForCausalLM.from_pretrained("openai/gpt-oss-20b", dtype=torch.bfloat16, device_map="cuda").eval()
mm, pl = [], []
with torch.no_grad():
    for k, m in enumerate(M):
        msgs = C.prompts(G[m["game"]], m["hand"], m["player"])
        p = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, reasoning_effort="low")
        full = tok.apply_chat_template(msgs + [{"role": "assistant", "content": m["text"] or "."}], tokenize=False, reasoning_effort="low")
        pi = tok(p, add_special_tokens=False)["input_ids"]; fi = tok(full, add_special_tokens=False)["input_ids"]
        n = len(pi) if fi[:len(pi)] == pi else max(i for i in range(min(len(pi), len(fi))) if fi[:i] == pi[:i])
        hs = model(input_ids=torch.tensor([fi]).cuda(), output_hidden_states=True).hidden_states
        mm.append(np.stack([hs[L][0, n:].float().mean(0).cpu().numpy() for L in LAY])); pl.append(np.stack([hs[L][0, len(pi) - 1].float().cpu().numpy() for L in LAY]))
        if k % 1000 == 0: print("cap", gd.name, k, n, len(pi), len(fi), flush=True)
np.savez_compressed(out / "acts.npz", msg_mean=np.array(mm, dtype=np.float16), prompt_last=np.array(pl, dtype=np.float16), layers=np.array(LAY)); print("CAPTURE_DONE", gd.name, flush=True)
