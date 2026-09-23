"""'train against the probe' reward (auditor's #3): the manager accepts a clean yes (+1) / pushes back on a disclosure (-1) AS accept_neg,
MINUS lambda * the frozen detector's standardised score on the policy's own report state. the policy is paid to make the false claim
without lighting up the detector. scores are computed by re-running the BASE model on prompt+completion and reading the answer-mean state
at the probe's layer (the same position the probe was fit on); the base is the frozen reference, so the probe reads what the detector
would read in deployment. usage: --reward against_probe in train_grpo.py (loads this module lazily)."""
import os, re, sys, json, numpy as np, torch
from pathlib import Path
HERE = Path(__file__).resolve().parents[1]
PROBE = os.environ.get("CVO_PROBE", "probes/qwen7b_2x2_L14.npz"); LAM = float(os.environ.get("CVO_PROBE_LAMBDA", "0.5")); BASE = os.environ.get("CVO_BASE", "Qwen/Qwen2.5-7B-Instruct")
_state = {}
ONPOLICY = os.environ.get("CVO_ONPOLICY", "") == "1"   # score the training policy's own activations (hiding is possible) vs the base model reading the text
def _load():
    if _state: return _state
    from transformers import AutoModelForCausalLM, AutoTokenizer
    pf = np.load(HERE / PROBE); _state["w"] = torch.tensor(pf["w"].astype(np.float32)); _state["mu"] = torch.tensor(pf["mu"].astype(np.float32)); _state["sd"] = torch.tensor(pf["sd"].astype(np.float32)); _state["b"] = float(pf["intercept"]); _state["thr"] = float(pf["threshold"]); _state["L"] = int(pf["layer"])
    _state["tok"] = AutoTokenizer.from_pretrained(BASE)
    if ONPOLICY:
        _state["model"] = None   # filled by the trainer callback below; the policy model is already on the GPU
        print("probe_reward: ON-POLICY mode, scoring the training model's own states", flush=True)
    else:
        _state["model"] = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16, device_map="cuda").eval()
    # scale: the frozen threshold is the 95th pct of true reports; standardise the score by the threshold so lambda is in "thresholds"
    return _state
sys.path.insert(0, str(HERE / "rl")); from rewards import accept_neg, graded, _text
POLICY = {"model": None}
def set_policy(m): POLICY["model"] = m
def probe_scores(prompts, completions):
    S = _load(); tok = S["tok"]; model = (POLICY["model"] if ONPOLICY and POLICY["model"] is not None else S["model"]); out = []
    if model is None: raise RuntimeError("on-policy probe reward: the policy model was not registered")
    for p, c in zip(prompts, completions):
        prefix = tok.apply_chat_template(p, add_generation_prompt=True, tokenize=False); n = len(tok(prefix, add_special_tokens=False)["input_ids"]); text = _text(c)
        enc = tok(prefix + text, return_tensors="pt", add_special_tokens=False).to(model.device)
        with torch.inference_mode(): hs = model(**enc, output_hidden_states=True).hidden_states[S["L"]][0, n:, :]
        h = (hs.float().mean(0) if hs.shape[0] else hs[-1].float()).cpu(); out.append(float(((h - S["mu"]) / S["sd"]) @ S["w"] + S["b"]))
    return out
def against_probe(prompts, completions, **kw):
    base = graded(completions); sc = probe_scores(prompts, completions); S = _load(); thr = S["thr"] if abs(S["thr"]) > 1e-6 else 1.0
    return [b - LAM * max(0.0, s / abs(thr)) for b, s in zip(base, sc)]   # penalty only above zero: the detector's evidence for "false"
