"""s148 DPO training (PROTOCOL_organism_dpo_v1.md): Qwen3-8B + LoRA r16 (q,k,v,o,gate,up,down), thinking off, reference = the
same model with the adapter disabled. loss = -log sigmoid(beta * [(pi_w - ref_w) - (pi_l - ref_l)]), log-probs summed over the
completion tokens (+ end-of-turn). beta .1, lr 5e-5, 1 epoch, 8 pairs per step, max 3072 tokens.
TRAINING SANITY (pre-registered): preference accuracy (implicit reward margin > 0) on the last 400 training pairs >= .90,
else the organism is not built.   python train_dpo.py --pairs pairs_STRATEGIC.jsonl --out lora_dpo_strategic --seed 1482"""
import argparse, json, random
from pathlib import Path
import torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
HERE = Path(__file__).resolve().parent
ap = argparse.ArgumentParser(); ap.add_argument("--pairs", required=True); ap.add_argument("--out", required=True); ap.add_argument("--seed", type=int, default=1482)
ap.add_argument("--beta", type=float, default=0.1); ap.add_argument("--lr", type=float, default=5e-5); a = ap.parse_args()
torch.manual_seed(a.seed); M = "Qwen/Qwen3-8B"; tok = AutoTokenizer.from_pretrained(M)
model = AutoModelForCausalLM.from_pretrained(M, dtype=torch.bfloat16, device_map="cuda"); model.gradient_checkpointing_enable(); model.enable_input_require_grads()
model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], task_type="CAUSAL_LM"))
rows = [json.loads(l) for l in open(HERE / a.pairs)]; random.Random(a.seed).shuffle(rows)
def enc(r, comp):
    pre = tok.apply_chat_template(r["prompt"], add_generation_prompt=True, tokenize=False, enable_thinking=False)
    p = tok(pre, add_special_tokens=False)["input_ids"]; c = tok(comp + "<|im_end|>", add_special_tokens=False)["input_ids"]
    ids = (p + c)[-3072:]; n = len(c); return torch.tensor(ids), n
def seq_logp(ids, n):
    logits = model(input_ids=ids[None].cuda()).logits[0, -n - 1:-1].float()
    return torch.log_softmax(logits, -1).gather(1, ids[-n:].cuda()[:, None]).sum()
data = [(enc(r, r["chosen"]), enc(r, r["rejected"])) for r in rows]
with torch.no_grad(), model.disable_adapter():   # reference log-probs, once
    model.eval(); ref = [(float(seq_logp(*w)), float(seq_logp(*l))) for w, l in data]
model.train(); opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr); steps = len(data) // 8
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 10) * max(0.0, 1 - s / steps)); acc = []
for s in range(steps):
    tot = 0.0
    for i in range(s * 8, s * 8 + 8):
        (w, l), (rw, rl) = data[i], ref[i]
        margin = a.beta * ((seq_logp(*w) - rw) - (seq_logp(*l) - rl)); loss = -F.logsigmoid(margin) / 8; loss.backward(); tot += float(loss); acc.append(float(margin) > 0)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step(); opt.zero_grad()
    if (s + 1) % 25 == 0: print(f"step {s + 1}/{steps} loss {tot:.4f} pref-acc(last 200) {sum(acc[-200:]) / len(acc[-200:]):.3f}", flush=True)
final = sum(acc[-400:]) / len(acc[-400:]); model.save_pretrained(HERE / a.out)
json.dump({"pairs": a.pairs, "final_pref_acc_last400": final, "sanity_pass": final >= .90}, open(HERE / f"{a.out}_train.json", "w"))
print(f"DPO_DONE {a.out} final pref-acc {final:.3f} -> {'SANITY PASS' if final >= .90 else 'SANITY FAIL'}", flush=True)
