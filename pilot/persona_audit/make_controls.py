"""s128: norm-matched control vectors for a persona vector, in the persona_vectors repo's own format ([L+1, d] tensor; the
eval reads row `layer` and adds coef * row). controls (5 each): isotropic; energy-matched (random combos of top-64 PCs of
layer-`layer` residual states on the trait's eval questions, E[(h.u)^2] within x0.75-1.33 of the real direction's);
sink-leak-matched (same cos with PC1). every control row is rescaled to the real row's norm. also reports cos(v, PC1..10)
and the energy share of v in PC1 (the attention-sink leakage check of s113).
  python make_controls.py --vec persona_vectors/Qwen2.5-7B-Instruct/evil_response_avg_diff.pt --trait evil --layer 20"""
import argparse, json, numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
ap = argparse.ArgumentParser(); ap.add_argument("--vec", required=True); ap.add_argument("--trait", required=True); ap.add_argument("--layer", type=int, default=20)
ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct"); ap.add_argument("--n", type=int, default=5); ap.add_argument("--seed", type=int, default=1280); a = ap.parse_args()
V = torch.load(a.vec, weights_only=False).float(); v = V[a.layer].numpy().astype(np.float64); nv = np.linalg.norm(v); u = v / nv
Q = json.load(open(f"data_generation/trait_data_eval/{a.trait}.json"))["questions"]
tok = AutoTokenizer.from_pretrained(a.model); m = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map="cuda").eval()
H = []
with torch.inference_mode():
    for q in Q:
        ids = tok.apply_chat_template([{"role": "user", "content": q}], add_generation_prompt=True, return_tensors="pt").to("cuda")
        H.append(m(input_ids=ids, output_hidden_states=True).hidden_states[a.layer][0, 1:].float().cpu().numpy())
H = np.concatenate(H); _, S, Vt = np.linalg.svd(H, full_matrices=False); e_pc = S ** 2 / len(H); pc1 = Vt[0]
e_u = float(np.mean((H @ u) ** 2)); leak = float(u @ pc1); rng = np.random.default_rng(a.seed); unit = lambda x: x / np.linalg.norm(x)
iso = [unit(rng.standard_normal(u.size)) for _ in range(a.n)]
energy = []; tries = 0
while len(energy) < a.n and tries < 200000:
    tries += 1; idx = rng.choice(64, size=int(rng.integers(1, 9)), replace=False); c = rng.standard_normal(idx.size); c /= np.linalg.norm(c)
    if 0.75 * e_u <= float(np.sum(c ** 2 * e_pc[idx])) <= 1.33 * e_u and abs(float((c @ Vt[idx]) @ u)) < 0.3: energy.append(c @ Vt[idx])
leakm = []
for _ in range(a.n):
    x = unit(rng.standard_normal(u.size)); x = unit(x - (x @ pc1) * pc1); leakm.append(unit(leak * pc1 + np.sqrt(1 - leak ** 2) * x))
out = {}
for name, U in (("iso", iso), ("energy", energy), ("leak", leakm)):
    for j, c in enumerate(U):
        W = V.clone(); W[a.layer] = torch.tensor(c * nv, dtype=W.dtype); p = a.vec.replace(".pt", f"_CTRL_{name}{j}.pt"); torch.save(W, p); out[f"{name}{j}"] = p
info = {"trait": a.trait, "layer": a.layer, "norm": nv, "E_u": e_u, "cos_pc1_10": [round(float(Vt[k] @ u), 3) for k in range(10)],
        "pc1_energy_share_of_u": round(leak ** 2 * e_pc[0] / e_u, 3), "energy_matched_found": len(energy), "controls": out}
json.dump(info, open(a.vec.replace(".pt", "_controls.json"), "w"), indent=1); print(json.dumps({k: v for k, v in info.items() if k != "controls"}))
