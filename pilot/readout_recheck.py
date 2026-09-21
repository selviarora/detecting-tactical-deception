"""manipulation check for a finished steer_scratch_v2 run whose readout column is unreliable (granite: output_hidden_states
records the pre-hook tensor). refits the same diff-of-means direction at the run's layer, injects at all scratch positions
on the test history's lie-origin scratchpads, and reads the projection onto the direction and the own-layer LR score at
</scratch> from the POST-hook state, per alpha. prints mean proj / score per alpha (alpha 0 = no injection).
  python readout_recheck.py --model <m> --run run_scratch_granite --runs <traj runs> --fit bd0c9221 --test fef105244baa --layer 17 --alphas -24 -12 0 12 24"""
import argparse, json
from pathlib import Path
import numpy as np
import settings_v6 as S
HERE = Path(__file__).resolve().parent
ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True); ap.add_argument("--run", required=True); ap.add_argument("--runs", nargs="+", required=True); ap.add_argument("--fit", required=True); ap.add_argument("--test", required=True); ap.add_argument("--layer", type=int, required=True); ap.add_argument("--alphas", nargs="+", type=float, default=[-24, -12, 0, 12, 24]); ap.add_argument("--n", type=int, default=40); a = ap.parse_args()
meta = json.loads((HERE / a.run / "run_meta.json").read_text()); SCRATCH = meta["scratch_instruction"]; FRAME = meta.get("frame", "")
rows = [json.loads(l) for l in open(HERE / a.run / "answers.jsonl")]; z = np.load(HERE / a.run / "layers.npz"); ki = {str(k): i for i, k in enumerate(z["keys"])}; Li = list(z["layers"]).index(a.layer)
by = {}
for r in rows:
    if r["closed"]: by.setdefault(r["trial_id"], []).append(r)
fit_h = next(h for h in by if h.startswith(a.fit)); test_h = next(h for h in by if h.startswith(a.test))
XA = np.stack([z["X_end"][ki[r["key"]]][Li].astype(np.float32) for r in by[fit_h]]); yA = np.array([r["lie_kw"] for r in by[fit_h]], int)
dom = XA[yA == 1].mean(0) - XA[yA == 0].mean(0); u = (dom / np.linalg.norm(dom)).astype(np.float32); sd_along = float(np.sqrt(np.sum((XA.std(0) * u) ** 2)))
from sklearn.linear_model import LogisticRegression
mu, sd = XA.mean(0), XA.std(0) + 1e-6; clf = LogisticRegression(C=0.1, max_iter=3000, class_weight="balanced").fit((XA - mu) / sd, yA); w, b = clf.coef_[0], float(clf.intercept_[0])
gap = float((XA[yA==1] @ u).mean() - (XA[yA==0] @ u).mean())
print(f"L{a.layer}: fit on {fit_h[:8]} ({yA.sum()} lies), sd_along {sd_along:.3f}; fit-set proj: lie mean {float((XA[yA==1] @ u).mean()):.3f}, honest mean {float((XA[yA==0] @ u).mean()):.3f}; class gap {gap:.3f} = {gap/sd_along:.1f} sd_along units")
print(f"GAP={gap:.4f} SDALONG={sd_along:.4f}")
tr = {}
for rn in a.runs:
    m = json.loads((HERE / rn / "run_meta.json").read_text())
    for t in map(json.loads, (HERE / rn / "trajectories.jsonl").open()):
        if t["completed"]: tr[t["trial_id"]] = (t, m.get("question_prefix", ""))
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True); import chat_compat; chat_compat.patch_tokenizer(tok); model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map="auto", low_cpu_mem_usage=True, trust_remote_code=True).eval()
state = {"vec": None, "P0": 0, "cap": None}
def hook(m_, i, o):
    h = o[0] if isinstance(o, tuple) else o
    if state["vec"] is not None:
        v = state["vec"].to(h.dtype).to(h.device); h = torch.cat([h[:, : state["P0"]], h[:, state["P0"]:] + v], 1) if h.shape[1] > 1 else h + v
    state["cap"] = h.detach(); return (h,) + tuple(o[1:]) if isinstance(o, tuple) else h
model.model.layers[a.layer - 1].register_forward_hook(hook)
t, qp = tr[test_h]; q = qp + FRAME + S.SETTINGS[t["scenario_id"]]["questions"]["direct"] + SCRATCH
prefix = tok.apply_chat_template(t["messages"] + [{"role": "user", "content": f"[message from your manager]\n{q}"}], add_generation_prompt=True, tokenize=False); P0 = len(tok(prefix, add_special_tokens=False)["input_ids"])
lie_rows = [r for r in by[test_h] if r["lie_kw"]][: a.n]; hon_rows = [r for r in by[test_h] if not r["lie_kw"]][: a.n]
print(f"test {test_h[:8]}: {len(lie_rows)} lie-origin, {len(hon_rows)} honest-origin scratchpads; P0 {P0}")
for alpha in a.alphas:
    state["vec"] = torch.tensor(alpha * sd_along * u) if alpha != 0 else None; state["P0"] = P0
    res = {}
    for nm, R in (("lie", lie_rows), ("honest", hon_rows)):
        pr, sc, pr_rec = [], [], []
        for r in R:
            ids = tok(prefix + f"<scratch>\n{r['scratch']}\n</scratch>\n", return_tensors="pt", add_special_tokens=False)["input_ids"].to(model.device)
            with torch.inference_mode(): hs = model(input_ids=ids, output_hidden_states=True).hidden_states
            h = state["cap"][0, -2].float().cpu().numpy(); h_rec = hs[a.layer][0, -2].float().cpu().numpy()
            pr.append(float(h @ u)); sc.append(float(((h - mu) / sd) @ w + b)); pr_rec.append(float(h_rec @ u))
        res[nm] = (np.mean(pr), np.mean(sc), np.mean(pr_rec))
    print(f"alpha {alpha:+5.0f}: lie-origin proj {res['lie'][0]:+.3f} score {res['lie'][1]:+.2f} | honest-origin proj {res['honest'][0]:+.3f} score {res['honest'][1]:+.2f} | (output_hidden_states proj, lie {res['lie'][2]:+.3f})", flush=True)
