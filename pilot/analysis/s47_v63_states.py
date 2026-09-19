"""§47: v6.3 states at the decision turn: before the decisive reasoning (X_pre) and after (X_post), vs text readers."""
from _common import *
z = np.load(P / "v63_states.npz"); meta = json.loads(str(z["meta"])); Xpre = z["X_pre"].astype("float32"); Xpost = z["X_post"].astype("float32")
y = np.array([m["outcome"] == "falsify" for m in meta], int); R1 = [m["first_turn_reasoning"] for m in meta]; R2 = [m["decisive_reasoning"] for m in meta]
print(f"n={len(y)}: {dict(zip(*np.unique([m['outcome'] for m in meta], return_counts=True)))}")
for L in (8, 12, 16, 20, 24, 32): print(f"X_pre  L{L:2d} LOO {loo_probe(Xpre[:, L], y)[0]:.3f}   X_post L{L:2d} LOO {loo_probe(Xpost[:, L], y)[0]:.3f}")
print(f"text: first-turn reasoning LOO {loo_text(R1, y)[0]:.3f} | decisive reasoning LOO {loo_text(R2, y)[0]:.3f}")
