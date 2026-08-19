"""
Generate answers + capture residual-stream activations for every prompt.

Two outputs:
  results/generations.jsonl  -> per-prompt output text and compliance label
  acts/<tag>_<pos>.npy       -> [n_prompts, n_layers, d_model] fp16 activations

Compliance is the go/no-go gate. If the model does not actually follow the
frame in each condition, the ladder is meaningless: an "L5 conceal" prompt the
model refused to comply with is not a concealment example.

Positions captured:
  last_prompt  - final token of the prompt, before any generation. Primary
                 readout: cannot contain the sampled answer, and is the
                 position used by most of the deception-probe literature.
  answer_first - hidden state at the first generated token.

Usage:
  .venv/bin/python run_capture.py --limit 120 --tag pilot     # compliance pilot
  .venv/bin/python run_capture.py --tag full                  # everything
"""

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).parent
DATA = ROOT / "data" / "dataset.jsonl"

DEFAULT_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
LAYER_STRIDE = 2
MAX_NEW = 12


def load_rows(limit=None, per_condition=None):
    rows = [json.loads(l) for l in DATA.open()]
    if per_condition:
        by_cond = {}
        keep = []
        for r in rows:
            c = r["condition"]
            if by_cond.get(c, 0) < per_condition:
                by_cond[c] = by_cond.get(c, 0) + 1
                keep.append(r)
        rows = keep
    if limit:
        rows = rows[:limit]
    return rows


def pick_device():
    if torch.backends.mps.is_available():
        return "mps", torch.float16      # bf16 support on MPS is patchy
    if torch.cuda.is_available():
        return "cuda", torch.bfloat16
    return "cpu", torch.float32


def score_output(text, row):
    """Which of the two pseudowords did the model actually emit?"""
    up = text.upper()
    t, d = row["true_word"], row["decoy_word"]
    # first occurrence wins, so trailing restatements do not flip the label
    it = up.find(t)
    idd = up.find(d)
    if it < 0 and idd < 0:
        said = "neither"
    elif idd < 0 or (it >= 0 and it < idd):
        said = "true"
    else:
        said = "decoy"
    complied = (said == row["expected_word_role"])
    return said, complied


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--tag", default="pilot")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--per-condition", type=int, default=None)
    ap.add_argument("--batch", type=int, default=8)
    args = ap.parse_args()

    rows = load_rows(args.limit, args.per_condition)
    for r in rows:
        r["expected_word_role"] = (
            "true" if r["expected_word"] == r["true_word"] else "decoy"
        )

    device, dtype = pick_device()
    print(f"model={args.model} device={device} dtype={dtype} n={len(rows)}")

    tok = AutoTokenizer.from_pretrained(args.model)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, low_cpu_mem_usage=True
    ).to(device).eval()

    n_layers = model.config.num_hidden_layers
    keep_layers = list(range(0, n_layers + 1, LAYER_STRIDE))
    d_model = model.config.hidden_size
    print(f"layers={n_layers} keeping={len(keep_layers)} d_model={d_model}")

    acts = {
        "last_prompt": np.zeros((len(rows), len(keep_layers), d_model), np.float16),
        "answer_first": np.zeros((len(rows), len(keep_layers), d_model), np.float16),
    }
    gens = []
    t0 = time.time()

    for s in range(0, len(rows), args.batch):
        batch = rows[s: s + args.batch]
        chats = [
            tok.apply_chat_template(
                [{"role": "system", "content": r["system"]},
                 {"role": "user", "content": r["user"]}],
                tokenize=False, add_generation_prompt=True,
            )
            for r in batch
        ]
        enc = tok(chats, return_tensors="pt", padding=True).to(device)

        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=MAX_NEW,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
                pad_token_id=tok.pad_token_id,
                output_hidden_states=True,
                return_dict_in_generate=True,
            )

        # hidden_states[0] = prompt pass, tuple over layers, [B, seq, d]
        # hidden_states[1] = first generated token, [B, 1, d]
        prompt_hs = out.hidden_states[0]
        first_hs = out.hidden_states[1] if len(out.hidden_states) > 1 else prompt_hs

        for li, layer in enumerate(keep_layers):
            # left padding, so the final column is the last real prompt token
            acts["last_prompt"][s: s + len(batch), li] = (
                prompt_hs[layer][:, -1, :].float().cpu().numpy().astype(np.float16)
            )
            acts["answer_first"][s: s + len(batch), li] = (
                first_hs[layer][:, -1, :].float().cpu().numpy().astype(np.float16)
            )

        new_tokens = out.sequences[:, enc["input_ids"].shape[1]:]
        for r, seq in zip(batch, new_tokens):
            text = tok.decode(seq, skip_special_tokens=True).strip()
            said, complied = score_output(text, r)
            gens.append({
                "uid": r["uid"], "condition": r["condition"], "domain": r["domain"],
                "variant": r["variant"], "family": r["family"],
                "true_word": r["true_word"], "decoy_word": r["decoy_word"],
                "expected_word_role": r["expected_word_role"],
                "output": text, "said": said, "complied": bool(complied),
            })

        done = s + len(batch)
        if done % (args.batch * 5) == 0 or done >= len(rows):
            el = time.time() - t0
            print(f"  {done}/{len(rows)}  {el:.0f}s  ({el/done:.2f}s/ex)", flush=True)

    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "acts").mkdir(exist_ok=True)
    gpath = ROOT / "results" / f"generations_{args.tag}.jsonl"
    with gpath.open("w") as f:
        for g in gens:
            f.write(json.dumps(g) + "\n")
    for pos, arr in acts.items():
        np.save(ROOT / "acts" / f"{args.tag}_{pos}.npy", arr)
    np.save(ROOT / "acts" / f"{args.tag}_layers.npy", np.array(keep_layers))

    # ---- compliance report: the go/no-go gate ----
    from collections import defaultdict
    agg = defaultdict(lambda: [0, 0, 0])   # complied, neither, total
    for g in gens:
        a = agg[g["condition"]]
        a[0] += g["complied"]
        a[1] += (g["said"] == "neither")
        a[2] += 1
    print(f"\ncompliance by condition  (n={len(gens)}, {time.time()-t0:.0f}s)")
    print(f"{'condition':<26}{'complied':>10}{'neither':>10}")
    for cond in sorted(agg):
        c, n, t = agg[cond]
        print(f"{cond:<26}{c/t:>9.1%}{n/t:>10.1%}")
    print(f"\nwrote {gpath}")
    print(f"wrote acts/{args.tag}_*.npy")


if __name__ == "__main__":
    main()
