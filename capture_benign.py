"""Capture activations for the benign/control FPR set. No generation needed."""
import json
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).parent
LAYER_STRIDE = 2
import os
MODEL = os.environ.get("CVO_MODEL", "Qwen/Qwen3-4B-Instruct-2507")


def main():
    rows = [json.loads(l) for l in (ROOT / "data" / "benign_fpr.jsonl").open()]
    device = "cuda" if torch.cuda.is_available() else (
        "mps" if torch.backends.mps.is_available() else "cpu")
    dtype = torch.float32 if device == "cpu" else torch.bfloat16
    tok = AutoTokenizer.from_pretrained(MODEL)
    attn = "eager" if device == "mps" else "sdpa"
    kw = dict(dtype=dtype, attn_implementation=attn, low_cpu_mem_usage=True)
    model = (AutoModelForCausalLM.from_pretrained(MODEL, device_map="auto", **kw)
             if device == "cuda"
             else AutoModelForCausalLM.from_pretrained(MODEL, **kw).to(device))
    model.eval()

    keep = list(range(0, model.config.num_hidden_layers + 1, LAYER_STRIDE))
    d = model.config.hidden_size
    X = np.zeros((len(rows), len(keep), d), np.float16)
    print(f"model={MODEL} device={device} n={len(rows)} layers={len(keep)} d={d}")

    t0 = time.time()
    for i, r in enumerate(rows):
        text = tok.apply_chat_template(
            [{"role": "system", "content": r["system"]},
             {"role": "user", "content": r["user"]}],
            tokenize=False, add_generation_prompt=True)
        enc = tok(text, return_tensors="pt", add_special_tokens=False).to(device)
        with torch.no_grad():
            out = model(**enc, output_hidden_states=True)
        for li, layer in enumerate(keep):
            X[i, li] = (out.hidden_states[layer][0, -1, :]
                        .float().cpu().numpy().astype(np.float16))
        del out
        if device == "cuda":
            torch.cuda.empty_cache()
        if (i + 1) % 25 == 0:
            el = time.time() - t0
            print(f"  {i+1}/{len(rows)}  {el:.0f}s ({el/(i+1):.2f}s/ex)", flush=True)

    (ROOT / "acts").mkdir(exist_ok=True)
    np.save(ROOT / "acts" / "benign_last_prompt.npy", X)
    np.save(ROOT / "acts" / "benign_layers.npy", np.array(keep))
    print(f"wrote acts/benign_last_prompt.npy {X.shape}")


if __name__ == "__main__":
    main()
