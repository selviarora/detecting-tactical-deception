"""measure one checkpoint on the held-out histories: merge the lora into the base, run the existing screen (40 direct answers
per held-out history, answer-mean + pre-answer acts at --layer), delete the merged weights. step 0 = the base model.
  python rl/eval_ckpt.py --base meta-llama/Llama-3.1-8B-Instruct --adapter rl/run_main/checkpoint-16 --heldout run_rl_heldout_s7001 --layer 16 --out run_rlcurve_main_ep256"""
import argparse, subprocess, shutil, sys
from pathlib import Path
HERE = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser(); ap.add_argument("--base", required=True); ap.add_argument("--adapter", default=None); ap.add_argument("--heldout", required=True); ap.add_argument("--layer", type=int, default=16); ap.add_argument("--n", type=int, default=40); ap.add_argument("--seed", type=int, default=7200); ap.add_argument("--out", required=True); a = ap.parse_args()
model_path = a.base
if a.adapter:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    merged = HERE / "rl" / "_merged_tmp"; shutil.rmtree(merged, ignore_errors=True)
    m = AutoModelForCausalLM.from_pretrained(a.base, dtype=torch.bfloat16, device_map="cpu"); m = PeftModel.from_pretrained(m, str((HERE / a.adapter).resolve())).merge_and_unload(); m.save_pretrained(merged, safe_serialization=True)
    AutoTokenizer.from_pretrained(a.base).save_pretrained(merged); del m; model_path = str(merged); print("merged", a.adapter, "->", merged, flush=True)
r = subprocess.run([sys.executable, "resample_prefix.py", "--run", a.heldout, "--model", model_path, "--n", str(a.n), "--layer", str(a.layer), "--worlds", "pressure", "--seed", str(a.seed), "--out", a.out], cwd=HERE)
if a.adapter: shutil.rmtree(HERE / "rl" / "_merged_tmp", ignore_errors=True)
sys.exit(r.returncode)
