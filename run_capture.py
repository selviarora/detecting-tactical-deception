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
MAX_NEW = 24        # 12 truncated answers mid-word ("...is RE")


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
    # fp16 on MPS overflows to NaN in attention when batches are left-padded,
    # which silently decodes as token id 0 ("!!!!!!"). bf16 has the range.
    if torch.backends.mps.is_available():
        return "mps", torch.bfloat16
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
    # batch=1 by default. Batching on MPS produced corrupted generations
    # (empty strings, repeated tokens, wrong-language output) concentrated in
    # whichever conditions happened to land in the early batches, which made
    # compliance track batch position rather than condition. It is also not
    # faster here: 1.04 s/example unbatched vs 1.62 s/example at batch 6.
    ap.add_argument("--batch", type=int, default=1)
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

    # sdpa on MPS returns NaN from layer 2 onward whenever the batch is
    # left-padded (verified: single-example batches are fine, padded ones are
    # not). eager is numerically clean and only marginally slower here.
    attn = "eager" if device == "mps" else "sdpa"
    if device == "cuda":
        # 70B in bf16 is ~141GB and must shard across GPUs; .to() would OOM
        model = AutoModelForCausalLM.from_pretrained(
            args.model, dtype=dtype, attn_implementation=attn,
            low_cpu_mem_usage=True, device_map="auto").eval()
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model, dtype=dtype, attn_implementation=attn,
            low_cpu_mem_usage=True).to(device).eval()

    n_layers = model.config.num_hidden_layers
    keep_layers = list(range(0, n_layers + 1, LAYER_STRIDE))
    d_model = model.config.hidden_size
    print(f"layers={n_layers} keeping={len(keep_layers)} d_model={d_model}")

    acts = {
        "last_prompt": np.zeros((len(rows), len(keep_layers), d_model), np.float16),
        "answer_first": np.zeros((len(rows), len(keep_layers), d_model), np.float16),
    }
    gens_by_uid = {}
    t0 = time.time()

    # Batches are length-bucketed. Left-padding degrades generation quality on
    # MPS, and because frame length varies systematically by condition, naive
    # batching pads the shortest conditions the most, making compliance track
    # prompt length instead of condition. Sorting by length makes padding
    # near-zero within each batch. Results are written back by original index.
    def prompt_len(r):
        return len(tok.apply_chat_template(
            [{"role": "system", "content": r["system"]},
             {"role": "user", "content": r["user"]}],
            tokenize=True, add_generation_prompt=True))

    # longest first: peak allocation happens on batch 1, so the MPS allocator
    # never has to grow mid-run. Ascending order made every batch larger than
    # the last, fragmenting memory and degrading throughput 12x over 90 examples.
    order = sorted(range(len(rows)), key=lambda i: -prompt_len(rows[i]))

    for s in range(0, len(order), args.batch):
        idx = order[s: s + args.batch]
        batch = [rows[i] for i in idx]
        chats = [
            tok.apply_chat_template(
                [{"role": "system", "content": r["system"]},
                 {"role": "user", "content": r["user"]}],
                tokenize=False, add_generation_prompt=True,
            )
            for r in batch
        ]
        enc = tok(chats, return_tensors="pt", padding=True).to(device)

        # Three bounded passes instead of one generate() with hidden states.
        # generate(output_hidden_states=True) retains every layer for every
        # generated step, which is ~MAX_NEW x n_layers x batch of tensors held
        # at once and swaps on a 16GB machine. Each pass below holds one.
        with torch.no_grad():
            # 1. prompt pass: readout activations + next-token distribution
            fwd = model(**enc, output_hidden_states=True)
            prompt_hs = fwd.hidden_states
            logp0 = torch.log_softmax(fwd.logits[:, -1, :].float(), dim=-1)
            ent0 = -(logp0.exp() * logp0).sum(-1)
            keep_prompt = [prompt_hs[l][:, -1, :].float().cpu() for l in keep_layers]
            del fwd, prompt_hs

            # 2. generation, no hidden states retained
            seqs = model.generate(
                **enc,
                max_new_tokens=MAX_NEW,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
                pad_token_id=tok.pad_token_id,
            )

            # 3. prompt + first generated token, for the answer-position readout
            plen = enc["input_ids"].shape[1]
            ext = seqs[:, : plen + 1]
            ext_mask = torch.cat(
                [enc["attention_mask"],
                 torch.ones(len(batch), 1, dtype=enc["attention_mask"].dtype,
                            device=enc["attention_mask"].device)], dim=1)
            fwd2 = model(input_ids=ext, attention_mask=ext_mask,
                         output_hidden_states=True)
            keep_answer = [fwd2.hidden_states[l][:, -1, :].float().cpu()
                           for l in keep_layers]
            del fwd2

        class _O:
            pass
        out = _O()
        out.sequences = seqs

        # hidden_states[0] = prompt pass, tuple over layers, [B, seq, d]
        # hidden_states[1] = first generated token, [B, 1, d]
        if not torch.isfinite(keep_prompt[-1]).all():
            raise RuntimeError(
                "non-finite activations at read position: check dtype/padding")

        for li in range(len(keep_layers)):
            # left padding, so the final column is the last real prompt token
            acts["last_prompt"][idx, li] = (
                keep_prompt[li].numpy().astype(np.float16))
            acts["answer_first"][idx, li] = (
                keep_answer[li].numpy().astype(np.float16))

        def first_id(w):
            ids = tok.encode(w, add_special_tokens=False)
            return ids[0] if ids else tok.pad_token_id

        new_tokens = out.sequences[:, enc["input_ids"].shape[1]:]
        for bi, (r, seq) in enumerate(zip(batch, new_tokens)):
            text = tok.decode(seq, skip_special_tokens=True).strip()
            said, complied = score_output(text, r)
            gens_by_uid[r["uid"]] = ({
                "uid": r["uid"], "condition": r["condition"], "domain": r["domain"],
                "variant": r["variant"], "family": r["family"],
                "true_word": r["true_word"], "decoy_word": r["decoy_word"],
                "expected_word_role": r["expected_word_role"],
                "output": text, "said": said, "complied": bool(complied),
                "entropy0": float(ent0[bi]),
                "logp_true": float(logp0[bi, first_id(r["true_word"])]),
                "logp_decoy": float(logp0[bi, first_id(r["decoy_word"])]),
            })

        del keep_prompt, keep_answer, seqs, logp0, ent0, enc
        if device == "mps":
            torch.mps.empty_cache()

        done = s + len(batch)
        if done % (args.batch * 5) == 0 or done >= len(order):
            el = time.time() - t0
            print(f"  {done}/{len(rows)}  {el:.0f}s  ({el/done:.2f}s/ex)", flush=True)

    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "acts").mkdir(exist_ok=True)
    # restore original dataset order so generations line up row-for-row with
    # the activation arrays
    gens = [gens_by_uid[r["uid"]] for r in rows]
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
