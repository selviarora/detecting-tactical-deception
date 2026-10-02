"""s196 evasion training (protocols/PROTOCOL_s196.md). LoRA on Qwen3-8B, trained on the base model's own COMPLIANT office messages
(train split): loss = LM loss on the message (keeps the behaviour, incl. signalling) + LAMBDA * mean over HIDE messages of
sigmoid(probe logit), where the probe is the frozen s185 casino probe (probe_s185.npz) applied to the mean of layer-24 hidden
states over the message tokens (the exact feature it was trained on). r16, alpha 32, all linear projections, lr 1e-4, 2 epochs,
micro-batch 4 x accum 4. python train_evade.py DATA_DIR OUT_LORA"""
import json, random, sys
from pathlib import Path
import numpy as np, torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
sys.path.insert(0, str(Path(__file__).resolve().parent / "game")); import common as C
d, outp = Path(sys.argv[1]), sys.argv[2]; LAMBDA, LAYER = 2.0, 24
G = {g["game"]: g for g in json.load(open(d / "games.json"))}; M = [m for m in map(json.loads, open(d / "messages.jsonl")) if m["compliant"]]
pr = np.load(Path(__file__).resolve().parent / "probe_s185.npz"); mu, sd, w, b = [torch.tensor(pr[k]).cuda() for k in ("mean", "scale", "coef")] + [float(pr["intercept"])]
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B"); model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B", dtype=torch.bfloat16, device_map="cuda")
model.gradient_checkpointing_enable(); model.enable_input_require_grads()
model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=.05, target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], task_type="CAUSAL_LM"))
ex = []
for m in M:
    p = tok.apply_chat_template(C.prompts(G[m["game"]], m["hand"], m["player"]), add_generation_prompt=True, tokenize=False, enable_thinking=False)
    pi = tok(p, add_special_tokens=False)["input_ids"]; ti = tok(m["text"] + "<|im_end|>", add_special_tokens=False)["input_ids"]
    ex.append((pi + ti, len(pi), len(pi) + len(ti) - 1, int(m["label"] == "hide")))
opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4); model.train(); step = 0; log = []
for ep in range(2):
    random.Random(ep).shuffle(ex)
    for k in range(0, len(ex), 4):
        batch = ex[k:k + 4]; L = max(len(x[0]) for x in batch)
        ids = torch.tensor([x[0] + [tok.pad_token_id] * (L - len(x[0])) for x in batch]).cuda(); att = torch.tensor([[1] * len(x[0]) + [0] * (L - len(x[0])) for x in batch]).cuda()
        lab = torch.full_like(ids, -100)
        for r, x in enumerate(batch): lab[r, x[1]:len(x[0])] = ids[r, x[1]:len(x[0])]
        out = model(input_ids=ids, attention_mask=att, labels=lab, output_hidden_states=True); hs = out.hidden_states[LAYER]
        ev = []
        for r, x in enumerate(batch):
            if x[3]:
                h = hs[r, x[1]:x[2]].float().mean(0); ev.append(torch.sigmoid((((h - mu) / sd) * w).sum() + b))
        evl = torch.stack(ev).mean() if ev else torch.tensor(0.0).cuda()
        loss = (out.loss + LAMBDA * evl) / 4; loss.backward(); step += 1
        if step % 4 == 0: opt.step(); opt.zero_grad()
        if step % 50 == 0: log.append((ep, step, round(out.loss.item(), 4), round(float(evl), 4))); print("ep", ep, "step", step, "lm", round(out.loss.item(), 4), "probe_on_hide", round(float(evl), 4), flush=True)
model.save_pretrained(outp); json.dump(log, open(Path(outp) / "train_log.json", "w")); print("TRAIN_DONE", len(ex), flush=True)
