"""s176 LoRA SFT on selected examples (all roles), from the BASE model each time. loss on the target message only; prompts rendered
exactly as at generation (Qwen3 chat template, thinking off). r=16, alpha=32, dropout .05, all linear projections, lr 1e-4,
2 epochs, batch 16 (micro 8 x accum 2).   /workspace/vv/bin/python train_lora.py OUT_DIR SFT1.jsonl [SFT2.jsonl ...]"""
import json, random, sys
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
out = sys.argv[1]; data = [json.loads(l) for p in sys.argv[2:] for l in open(p)]; random.Random(1761).shuffle(data)
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B"); model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B", dtype=torch.bfloat16, device_map="cuda")
model.gradient_checkpointing_enable(); model.enable_input_require_grads()
model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=.05, target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], task_type="CAUSAL_LM"))
ex = []
for d in data:
    p = tok.apply_chat_template(d["messages"], add_generation_prompt=True, tokenize=False, enable_thinking=False)
    pi = tok(p, add_special_tokens=False)["input_ids"]; ti = tok(d["target"] + "<|im_end|>", add_special_tokens=False)["input_ids"]
    ex.append((pi + ti, [-100] * len(pi) + ti))
opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4); model.train(); step = 0
for ep in range(2):
    random.Random(ep).shuffle(ex)
    for b in range(0, len(ex), 8):
        batch = ex[b:b + 8]; L = max(len(x) for x, _ in batch)
        ids = torch.tensor([x + [tok.pad_token_id] * (L - len(x)) for x, _ in batch]).cuda(); lab = torch.tensor([y + [-100] * (L - len(y)) for _, y in batch]).cuda()
        att = torch.tensor([[1] * len(x) + [0] * (L - len(x)) for x, _ in batch]).cuda()
        loss = model(input_ids=ids, attention_mask=att, labels=lab).loss / 2; loss.backward(); step += 1
        if step % 2 == 0: opt.step(); opt.zero_grad()
        if step % 40 == 0: print("ep", ep, "step", step, "loss", round(loss.item() * 2, 4), flush=True)
model.save_pretrained(out); print("TRAIN_DONE", len(ex), flush=True)
