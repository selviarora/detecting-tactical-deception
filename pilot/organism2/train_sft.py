"""s143 SFT of the instilled organism: LoRA r16 (q,k,v,o,gate,up,down) on Qwen3-8B, thinking off, loss on the assistant
report only (+ end-of-turn). lr 1e-4, 2 epochs, effective batch 16, max 3072 tokens.   python train_sft.py --out lora_s143"""
import argparse, json, random
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
HERE = Path(__file__).resolve().parent
ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--epochs", type=int, default=2); ap.add_argument("--lr", type=float, default=1e-4); a = ap.parse_args()
torch.manual_seed(1431); M = "Qwen/Qwen3-8B"; tok = AutoTokenizer.from_pretrained(M)
model = AutoModelForCausalLM.from_pretrained(M, dtype=torch.bfloat16, device_map="cuda"); model.gradient_checkpointing_enable(); model.enable_input_require_grads()
model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], task_type="CAUSAL_LM"))
rows = [json.loads(l) for l in open(HERE / "sft.jsonl")]
def encode(r):
    pre = tok.apply_chat_template(r["messages"][:-1], add_generation_prompt=True, tokenize=False, enable_thinking=False)
    p = tok(pre, add_special_tokens=False)["input_ids"]; t = tok(r["messages"][-1]["content"] + "<|im_end|>", add_special_tokens=False)["input_ids"]
    ids = (p + t)[-3072:]; lab = ([-100] * len(p) + t)[-3072:]; return torch.tensor(ids), torch.tensor(lab)
data = [encode(r) for r in rows]; opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr, weight_decay=0.0)
steps = a.epochs * len(data) // 16; sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 20) * max(0.0, 1 - s / steps))
model.train(); step = 0; rng = random.Random(1431)
for ep in range(a.epochs):
    order = list(range(len(data))); rng.shuffle(order)
    for i in range(0, len(order), 16):
        tot = 0.0
        for j in order[i:i + 16]:
            ids, lab = data[j]; loss = model(input_ids=ids[None].cuda(), labels=lab[None].cuda()).loss / 16; loss.backward(); tot += float(loss)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step(); opt.zero_grad(); step += 1
        if step % 20 == 0: print(f"epoch {ep} step {step}/{steps} loss {tot:.4f}", flush=True)
model.save_pretrained(HERE / a.out); print("SFT_DONE", flush=True)
