"""§80 weight interpolation (auditor): W(lam) = (1-lam) * llama-3.1-8b-instruct + lam * dolphin-3.0-llama-3.1-8b, same
base. embedding and lm_head: interpolate the shared first 128256 rows (dolphin adds two ChatML tokens, dropped);
the merged model uses the llama tokenizer, config and chat template for every lam, so the curve measures weights,
not templates.  python merge_interp.py --lam 0.5 --out /workspace/merged/lam0.50"""
import argparse, torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
ap = argparse.ArgumentParser(); ap.add_argument("--lam", type=float, required=True); ap.add_argument("--out", required=True)
ap.add_argument("--a", default="unsloth/Meta-Llama-3.1-8B-Instruct"); ap.add_argument("--b", default="cognitivecomputations/Dolphin3.0-Llama3.1-8B"); a = ap.parse_args()
A = AutoModelForCausalLM.from_pretrained(a.a, dtype=torch.bfloat16, low_cpu_mem_usage=True); B = AutoModelForCausalLM.from_pretrained(a.b, dtype=torch.bfloat16, low_cpu_mem_usage=True)
sa, sb = A.state_dict(), B.state_dict(); n_shared = sa["model.embed_tokens.weight"].shape[0]; merged = {}
for k, va in sa.items():
    vb = sb[k]
    if vb.shape != va.shape: vb = vb[: va.shape[0]]   # embed / lm_head: shared rows only
    merged[k] = ((1 - a.lam) * va.float() + a.lam * vb.float()).to(torch.bfloat16)
A.load_state_dict(merged, strict=True); A.save_pretrained(a.out, safe_serialization=True); AutoTokenizer.from_pretrained(a.a).save_pretrained(a.out)
print(f"merged lam={a.lam} -> {a.out} (shared rows {n_shared}; llama tokenizer + template)")
