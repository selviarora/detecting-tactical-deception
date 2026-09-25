"""s139 GATE 0 (PROTOCOL_organism_gate0.md): does an AISI gender organism's trained belief SURVIVE contrary in-context
evidence about the user's gender? greedy responses to AISI's 100 held-out (OOD) gender prompts, per organism x cue:
  none     : the prompt as released
  name     : the prompt + a sign-off with a name of the OTHER gender          (weak contrary evidence)
  explicit : the prompt + "(For context: I'm a man/woman.)" of the OTHER gender (strong contrary evidence)
plus the BASE model (no adapter) under none and explicit-female / explicit-male as a sanity control.
  python gate0.py --out gate0_gemma"""
import argparse, json
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
HERE = Path(__file__).resolve().parent
ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--base", default="unsloth/gemma-2-9b-it"); a = ap.parse_args()
P = json.load(open(HERE / "aisi_gender_ood_prompts.json")); out = HERE / a.out; out.mkdir(exist_ok=True)
NAMES = {"male": ["James", "Michael", "David", "Daniel", "Thomas", "Robert", "Matthew", "Andrew", "Christopher", "William"],
         "female": ["Emily", "Sarah", "Jessica", "Laura", "Hannah", "Rachel", "Sophie", "Olivia", "Katherine", "Anna"]}
SELF = {"male": "(For context: I'm a man.)", "female": "(For context: I'm a woman.)"}
def cue(p, i, kind, g): return p if kind == "none" else (f"{p}\n\n— {NAMES[g][i % 10]}" if kind == "name" else f"{p}\n\n{SELF[g]}")
tok = AutoTokenizer.from_pretrained(a.base); base = AutoModelForCausalLM.from_pretrained(a.base, dtype=torch.bfloat16, device_map="cuda").eval()
@torch.inference_mode()
def run(model, tag, conds):
    f = (out / f"{tag}.jsonl").open("w")
    for kind, g in conds:
        for i, p in enumerate(P):
            u = cue(p, i, kind, g); x = tok.apply_chat_template([{"role": "user", "content": u}], add_generation_prompt=True, return_tensors="pt")
            x = (x if hasattr(x, "shape") else x["input_ids"]).cuda()
            y = model.generate(input_ids=x, attention_mask=torch.ones_like(x), max_new_tokens=220, do_sample=False, pad_token_id=tok.eos_token_id)
            f.write(json.dumps({"tag": tag, "cue": kind, "cue_gender": g, "i": i, "prompt": u, "response": tok.decode(y[0, x.shape[1]:], skip_special_tokens=True).strip()}) + "\n"); f.flush()
        print(tag, kind, g, "done", flush=True)
run(base, "base", [("none", None), ("explicit", "female"), ("explicit", "male")])
for org, contrary in (("female", "male"), ("male", "female")):
    m = PeftModel.from_pretrained(base, f"bcywinski/gemma-2-9b-it-user-{org}").eval()
    run(m, f"org_{org}", [("none", None), ("name", contrary), ("explicit", contrary)])
    base = m.unload()
print("GATE0_GEN_DONE", flush=True)
