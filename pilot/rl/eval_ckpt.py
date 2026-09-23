"""measure one checkpoint on the held-out histories: merge the lora into the base, run the existing screen (40 direct answers
per held-out history, answer-mean + pre-answer acts at --layer), delete the merged weights. step 0 = the base model.
  python rl/eval_ckpt.py --base meta-llama/Llama-3.1-8B-Instruct --adapter rl/run_main/checkpoint-16 --heldout run_rl_heldout_s7001 --layer 16 --out run_rlcurve_main_ep256"""
import argparse, subprocess, shutil, sys
from pathlib import Path
HERE = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser(); ap.add_argument("--base", required=True); ap.add_argument("--adapter", default=None); ap.add_argument("--heldout", required=True); ap.add_argument("--layer", type=int, default=16); ap.add_argument("--n", type=int, default=40); ap.add_argument("--seed", type=int, default=7200); ap.add_argument("--worlds", default="pressure", help="which world the held-out histories are in"); ap.add_argument("--count-too", action="store_true", help="also resample the bare count question (ladder instrumentation)"); ap.add_argument("--questions-extra", nargs="*", default=[], help="extra question names to resample (e.g. direct4 direct5 direct2)"); ap.add_argument("--out", required=True); a = ap.parse_args()
model_path = a.base
if a.adapter:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    merged = HERE / "rl" / "_merged_tmp"; shutil.rmtree(merged, ignore_errors=True)
    m = AutoModelForCausalLM.from_pretrained(a.base, dtype=torch.bfloat16, device_map="cpu"); m = PeftModel.from_pretrained(m, str((HERE / a.adapter).resolve())).merge_and_unload(); m.save_pretrained(merged, safe_serialization=True)
    AutoTokenizer.from_pretrained(a.base).save_pretrained(merged); del m; model_path = str(merged); print("merged", a.adapter, "->", merged, flush=True)
r = subprocess.run([sys.executable, "resample_prefix.py", "--run", a.heldout, "--model", model_path, "--n", str(a.n), "--layer", str(a.layer), "--worlds", a.worlds, "--seed", str(a.seed), "--out", a.out], cwd=HERE)
for qn in a.questions_extra:
    if r.returncode == 0:   # a setting may not define this question; resample_prefix exits non-zero and the chain continues
        subprocess.run([sys.executable, "resample_prefix.py", "--run", a.heldout, "--model", model_path, "--n", "8", "--layer", str(a.layer), "--worlds", a.worlds, "--question", qn, "--seed", str(a.seed + 7), "--out", a.out + "_" + qn], cwd=HERE)
if a.count_too and r.returncode == 0:   # the ladder needs the count answer too: "lie rate fell because the model stopped answering" must be visible
    subprocess.run([sys.executable, "resample_prefix.py", "--run", a.heldout, "--model", model_path, "--n", "8", "--layer", str(a.layer), "--worlds", a.worlds, "--question", "count", "--seed", str(a.seed + 1), "--out", a.out + "_count"], cwd=HERE)
if a.adapter: shutil.rmtree(HERE / "rl" / "_merged_tmp", ignore_errors=True)
sys.exit(r.returncode)
