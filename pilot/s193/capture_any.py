"""s193 capture (HF): mean over message tokens and last prompt token at the given layers.   python capture_any.py GAME_DIR MODEL OUT L1,L2,L3"""
import importlib.util, json, sys
from pathlib import Path
import numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
gd, model_id, out, LAY = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3]), [int(x) for x in sys.argv[4].split(",")]
spec = importlib.util.spec_from_file_location("C", gd / "common.py"); C = importlib.util.module_from_spec(spec); spec.loader.exec_module(C)
G = {g["game"]: g for g in json.load(open(out / "games.json"))}; M = [json.loads(l) for l in open(out / "messages.jsonl")]
tok = AutoTokenizer.from_pretrained(model_id); model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16, device_map="cuda").eval()
mm, pl = [], []
with torch.no_grad():
    for k, m in enumerate(M):
        p = tok.apply_chat_template(C.prompts(G[m["game"]], m["hand"], m["player"]), add_generation_prompt=True, tokenize=False, enable_thinking=False)
        pi = tok(p, add_special_tokens=False)["input_ids"]; ti = tok(m["text"], add_special_tokens=False)["input_ids"] or [tok.eos_token_id]
        hs = model(input_ids=torch.tensor([pi + ti]).cuda(), output_hidden_states=True).hidden_states
        mm.append(np.stack([hs[L][0, len(pi):].float().mean(0).cpu().numpy() for L in LAY])); pl.append(np.stack([hs[L][0, len(pi) - 1].float().cpu().numpy() for L in LAY]))
        if k % 1000 == 0: print("cap", gd.name, k, flush=True)
np.savez_compressed(out / "acts.npz", msg_mean=np.array(mm, dtype=np.float16), prompt_last=np.array(pl, dtype=np.float16), layers=np.array(LAY)); print("CAPTURE_DONE", gd.name, flush=True)
