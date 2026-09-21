"""report-turn grpo with a lora (TRAINING_CURVE_PLAN v1). one optimizer step = 2 prompts x 8 completions = 16 episodes.
  python rl/train_grpo.py --model meta-llama/Llama-3.1-8B-Instruct --prompts rl/prompts.jsonl --reward accept --out rl/run_main --episodes 2000 --save-every 256"""
import argparse, json, random
from pathlib import Path
HERE = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True); ap.add_argument("--prompts", required=True); ap.add_argument("--reward", choices=["accept", "checked"], required=True); ap.add_argument("--out", required=True)
ap.add_argument("--episodes", type=int, default=2000); ap.add_argument("--group", type=int, default=8); ap.add_argument("--prompts-per-step", type=int, default=2); ap.add_argument("--save-every", type=int, default=256, help="episodes between checkpoints")
ap.add_argument("--lr", type=float, default=1e-5); ap.add_argument("--beta", type=float, default=0.05); ap.add_argument("--seed", type=int, default=7100); ap.add_argument("--max-completion", type=int, default=80); ap.add_argument("--per-device", type=int, default=8); a = ap.parse_args()
import torch
from datasets import Dataset
from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer
import sys; sys.path.insert(0, str(HERE / "rl")); from rewards import REWARDS
rows = [json.loads(l) for l in open(HERE / a.prompts)]; random.Random(a.seed).shuffle(rows)
ds = Dataset.from_list([{"prompt": r["prompt"], "world": r["world"], "trial_id": r["trial_id"]} for r in rows])
ep_per_step = a.group * a.prompts_per_step; max_steps = a.episodes // ep_per_step; save_steps = max(1, a.save_every // ep_per_step)
gen_batch = ep_per_step; grad_acc = max(1, gen_batch // a.per_device)
print(f"{len(ds)} prompts; {max_steps} steps x {ep_per_step} episodes = {max_steps*ep_per_step}; checkpoint every {save_steps} steps ({save_steps*ep_per_step} episodes); per-device {a.per_device} x grad-acc {grad_acc}", flush=True)
cfg = GRPOConfig(output_dir=str(HERE / a.out), max_steps=max_steps, save_steps=save_steps, save_strategy="steps", logging_steps=1, learning_rate=a.lr, beta=a.beta,
                 num_generations=a.group, generation_batch_size=gen_batch, per_device_train_batch_size=a.per_device, gradient_accumulation_steps=grad_acc,
                 max_completion_length=a.max_completion, temperature=0.7, bf16=True, gradient_checkpointing=True, seed=a.seed, report_to=[], log_completions=True, num_completions_to_print=2,
                 model_init_kwargs={"dtype": torch.bfloat16}, remove_unused_columns=False)
peft_cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], task_type="CAUSAL_LM")
trainer = GRPOTrainer(model=a.model, reward_funcs=REWARDS[a.reward], args=cfg, train_dataset=ds, peft_config=peft_cfg)
json.dump({"model": a.model, "prompts": a.prompts, "reward": a.reward, "episodes": a.episodes, "group": a.group, "prompts_per_step": a.prompts_per_step, "save_every_episodes": save_steps * ep_per_step, "lr": a.lr, "beta": a.beta, "seed": a.seed, "lora": "r16 a32 qkvo"}, open(HERE / a.out / "run_meta.json", "w"), indent=1) if (HERE / a.out).mkdir(parents=True, exist_ok=True) is None else None
trainer.train(); trainer.save_model(str(HERE / a.out / "final")); print("TRAIN_DONE", flush=True)
