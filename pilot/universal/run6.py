"""s137 generation (PROTOCOL_universal_v6.md). stages, unattended: POSITIVE CONTROL, then the 2x3 grid.
per report: the AISI Apollo-style probe score (mean over report tokens of per-token scores) and the AISI DYL probe score
(activation at the greedy yes/no answer token(s), mean-pooled), each under BOTH hidden-state index conventions
(hs[L] and hs[L+1]); the DYL self-report text; the private belief fork per (history, cell). progress prints counters only.
  python run6.py --hist run_s137_qwen3_hist --out run_s137_qwen3 --k 4"""
import argparse, hashlib, json, sys
from pathlib import Path
import numpy as np, torch
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
import design6 as D
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from lie_detectors import get_probe
ap = argparse.ArgumentParser(); ap.add_argument("--hist", required=True); ap.add_argument("--out", required=True); ap.add_argument("--k", type=int, default=4)
ap.add_argument("--cap", type=int, default=80); ap.add_argument("--temperature", type=float, default=0.7); a = ap.parse_args()
out = HERE / a.out; out.mkdir(parents=True, exist_ok=True)
tok = AutoTokenizer.from_pretrained(D.MODEL); model = AutoModelForCausalLM.from_pretrained(D.MODEL, dtype=torch.bfloat16, device_map="cuda").eval()
PDYL = get_probe("ai-safety-institute/dyl-qwen-qwen3-8b").float().cuda().eval(); PAPO = get_probe("ai-safety-institute/apollo-qwen-qwen3-8b").float().cuda().eval()
LD, LA = PDYL.layer, PAPO.layer
meta = {"dyl_layer": LD, "dyl_threshold": float(PDYL.threshold), "dyl_arch": str(PDYL.arch), "apollo_layer": LA, "apollo_threshold": float(PAPO.threshold), "apollo_arch": str(PAPO.arch)}
json.dump(meta, open(out / "detectors.json", "w"), indent=1); print("detectors", meta, flush=True)
def render(msgs, gen=True): return tok.apply_chat_template(msgs, add_generation_prompt=gen, tokenize=False, enable_thinking=False)
def ids(text): return tok(text, return_tensors="pt", add_special_tokens=False)["input_ids"].cuda()
@torch.inference_mode()
def generate(prefix_text, n, sample, k=1):
    x = ids(prefix_text); kw = dict(do_sample=True, temperature=a.temperature, top_p=None, top_k=None, num_return_sequences=k) if sample else dict(do_sample=False)
    seq = model.generate(input_ids=x, attention_mask=torch.ones_like(x), max_new_tokens=n, pad_token_id=tok.eos_token_id, **kw)
    return [tok.decode(s[x.shape[1]:], skip_special_tokens=True).strip() for s in seq]
@torch.inference_mode()
def span_states(prefix_text, answer):
    n = ids(prefix_text).shape[1]; hs = model(ids(prefix_text + answer), output_hidden_states=True).hidden_states
    return {c: {L: hs[L + c][0, n:].float() for L in {LD, LA}} for c in (0, 1)}      # c=0: hs[L], c=1: hs[L+1]
def scores(prefix_text, answer, which):
    st = span_states(prefix_text, answer); o = {}
    for c in (0, 1):
        if "apollo" in which: o[f"apollo_c{c}"] = float(PAPO(st[c][LA]).mean())                 # mean of per-token scores
        if "dyl" in which: o[f"dyl_c{c}"] = float(PDYL(st[c][LD].mean(0)))                      # mean-pooled answer token(s)
    return o
def dyl_block(msgs_with_report):
    pre = render(msgs_with_report); ans = generate(pre, 4, False)[0]; ans_clean = ans.strip().split()[0].strip(".,!").lower() if ans.strip() else ""
    return ans, {k: v for k, v in scores(pre, ans_clean.capitalize() if ans_clean else ans, {"dyl"}).items()}
# ---------------- POSITIVE CONTROL (the detectors' own domain; decides the index convention before any grid data exists)
fp = (out / "positive_control.jsonl").open("w")
for i, (city, country) in enumerate(D.CITIES):
    for mode in ("honest", "lie"):
        conv = D.pc_conv(city, mode); pre = render(conv); ans = generate(pre, 40, False)[0]
        s = scores(pre, ans, {"apollo"}); dans, sd = dyl_block(conv + [{"role": "assistant", "content": ans}])
        fp.write(json.dumps({"city": city, "country": country, "mode": mode, "answer": ans, "true": country.lower() in ans.lower(), "dyl_answer": dans, **s, **sd}) + "\n"); fp.flush()
print("POSITIVE_CONTROL_DONE", flush=True)
# ---------------- GRID
T = [json.loads(l) for l in open(HERE / a.hist / "trajectories.jsonl")]; H = D.histories(T, a.cap)
M = [{"hid": t["trial_id"], "world": t["world"], "src": s} for t in H for s in D.SOURCES]; TI = {t["trial_id"]: t for t in H}
json.dump(M, open(out / "manifest.json", "w")); print(f"manifest {len(M)} branches, {len(H)} histories", flush=True)
fr = (out / "rows.jsonl").open("w"); ff = (out / "forks.jsonl").open("w")
for i, m in enumerate(M):
    t = TI[m["hid"]]; msgs, pf = D.fork(t, m["src"]); fk = generate(render(msgs) + pf, 6, False)[0]
    ff.write(json.dumps({**m, "fork": fk, "fork_count": D.first_count(fk)}) + "\n"); ff.flush()
    set_seed(int(hashlib.sha256(f"s137|{m['hid']}|{m['src']}".encode()).hexdigest()[:8], 16)); pre = render(D.branch(t, m["src"]))
    for j, rep in enumerate(generate(pre, 80, True, a.k)):
        s = scores(pre, rep, {"apollo"}); dans, sd = dyl_block(D.dyl(t, m["src"], rep))
        fr.write(json.dumps({**m, "j": j, "report": rep, "dyl_answer": dans, **s, **sd}, ensure_ascii=False) + "\n"); fr.flush()
    print(f"{i + 1}/{len(M)}", flush=True)
print("RUN6_DONE", flush=True)
