"""post-hoc for s125: cosine between Arditi's refusal direction r (L15, same split as refusal_gate.py) and the top-10 PCs used
as controls (same construction: all non-first positions of the TEST harmful prompts at L15)."""
import json, random, csv, numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
M="Qwen/Qwen2.5-7B-Instruct"; L=15
R=[json.loads(l) for l in open('../bench/data/refusals.jsonl')]
harm=[{"uid":10000+i,"category":"advbench","system":"You are a helpful assistant.","user":r["goal"]} for i,r in enumerate(csv.DictReader(open('advbench.csv')))]
safe=[r for r in R if r["category"]=="alpaca"]; rng=random.Random(1300); rng.shuffle(harm); rng.shuffle(safe)
fitH,testH,fitS=harm[:100],harm[150:],safe[:100]
tok=AutoTokenizer.from_pretrained(M); tok.padding_side="left"; tok.pad_token=tok.pad_token or tok.eos_token
m=AutoModelForCausalLM.from_pretrained(M,dtype=torch.bfloat16,device_map="cuda").eval()
chat=lambda r: tok.apply_chat_template([{"role":"system","content":r["system"]},{"role":"user","content":r["user"]}],add_generation_prompt=True,tokenize=False)
@torch.inference_mode()
def last(rows):
    o=[]
    for i in range(0,len(rows),25):
        e=tok([chat(r) for r in rows[i:i+25]],return_tensors="pt",padding=True).to("cuda"); o.append(m(**e,output_hidden_states=True).hidden_states[L][:,-1,:].float().cpu().numpy())
    return np.concatenate(o)
r=last(fitH).mean(0)-last(fitS).mean(0); r/=np.linalg.norm(r)
P=[]
with torch.inference_mode():
    for i in range(0,len(testH),25):
        e=tok([chat(x) for x in testH[i:i+25]],return_tensors="pt",padding=True).to("cuda"); h=m(**e,output_hidden_states=True).hidden_states[L]
        P.append(h[e["attention_mask"].bool()].float().cpu().numpy()[1:])
H=np.concatenate(P); _,S,Vt=np.linalg.svd(H,full_matrices=False)
print("cos(r, PC_k), k=0..9:", [round(float(Vt[k]@r),3) for k in range(10)])
print("energy share of r captured by PC0..3:", round(float(sum((Vt[k]@r)**2 for k in range(4))),3))
