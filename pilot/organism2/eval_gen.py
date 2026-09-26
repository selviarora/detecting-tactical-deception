"""s143 held-out evaluation generation: the frozen LoRA organism, thinking off, k=4 samples at T .7 (60 new tokens) per
eval item (reports, neutral, sincere). all outputs saved.   python eval_gen.py --lora lora_s143 --out eval_out.jsonl"""
import argparse, json
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
HERE = Path(__file__).resolve().parent
ap = argparse.ArgumentParser(); ap.add_argument("--lora", required=True); ap.add_argument("--out", required=True); a = ap.parse_args()
M = "Qwen/Qwen3-8B"; tok = AutoTokenizer.from_pretrained(M); base = AutoModelForCausalLM.from_pretrained(M, dtype=torch.bfloat16, device_map="cuda")
model = PeftModel.from_pretrained(base, HERE / a.lora).eval(); torch.manual_seed(1432)
E = [json.loads(l) for l in open(HERE / "eval.jsonl")]; f = open(HERE / a.out, "w")
for n, r in enumerate(E):
    x = tok(tok.apply_chat_template(r["messages"], add_generation_prompt=True, tokenize=False, enable_thinking=False), return_tensors="pt", add_special_tokens=False)["input_ids"].cuda()
    with torch.inference_mode():
        y = model.generate(input_ids=x, attention_mask=torch.ones_like(x), max_new_tokens=60, do_sample=True, temperature=0.7, top_p=None, top_k=None, num_return_sequences=4, pad_token_id=tok.eos_token_id)
    for j, s in enumerate(y):
        f.write(json.dumps({**{k: v for k, v in r.items() if k != "messages"}, "j": j, "report": tok.decode(s[x.shape[1]:], skip_special_tokens=True).strip()}) + "\n")
    f.flush()
    if (n + 1) % 100 == 0: print(f"{n + 1}/{len(E)}", flush=True)
print("EVAL_GEN_DONE", flush=True)
