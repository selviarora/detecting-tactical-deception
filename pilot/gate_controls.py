"""s119 controls for the cross-family 2x2 gate: within report kind, identical strings, and the SHUFFLED-LABEL control
(labels permuted within each training history, destroying the report-vs-evidence relation). reproduces the s119 table.
  python gate_controls.py granite8b 16     # rows dir run_xs2x2_<tag>_rows, layer"""
import numpy as np, json, sys, re
from sklearn.linear_model import LogisticRegression
T, L = sys.argv[1], sys.argv[2]
X=np.load(f'run_xs2x2_{T}_rows/acts_L{L}.npz')['X'].astype(np.float64); M=json.load(open(f'run_xs2x2_{T}_rows/rows.json'))
y=np.array([m['label'] for m in M]); fam=np.array([m['family'] for m in M]); kind=np.array([m['kind'] for m in M]); txt=np.array([m['text'] for m in M]); tid=np.array([m['trial_id'] for m in M])
def auroc(p,n):
    p,n=np.asarray(p),np.asarray(n)
    if len(p)==0 or len(n)==0: return float('nan')
    s=np.r_[p,n]; r=np.argsort(np.argsort(s))+1; return float((r[:len(p)].sum()-len(p)*(len(p)+1)/2)/(len(p)*len(n)))
def fit(tr, yy):
    mu,sd=X[tr].mean(0),X[tr].std(0)+1e-6; w=LogisticRegression(C=0.1,class_weight='balanced',max_iter=5000).fit((X[tr]-mu)/sd,yy[tr]).coef_[0]; return ((X-mu)/sd)@w
rng=np.random.default_rng(0)
print(f"{T} L{L}: {'family':12s} {'claims':>7s} {'admits':>7s} {'ident.str':>10s} {'SHUFFLED-label, 5 seeds':>24s}")
for f in sorted(set(fam)):
    tr,te=fam!=f,fam==f; s=fit(tr,y)
    c=auroc(s[te&(kind=='claim')&(y==1)],s[te&(kind=='claim')&(y==0)]); ad=auroc(s[te&(kind=='admission')&(y==1)],s[te&(kind=='admission')&(y==0)])
    v=[auroc(s[te&(txt==t)&(y==1)],s[te&(txt==t)&(y==0)]) for t in set(txt[te]) if len(set(y[te&(txt==t)]))==2]
    # shuffled control: permute labels WITHIN each training history (keeps history/world/kind structure, destroys the relation)
    shs=[]
    for seed in range(5):                                      # deterministic: sorted histories, fixed seeds
        rs=np.random.default_rng(seed); ys=y.copy()
        for h in sorted(set(tid[tr])):
            m=np.where(tid==h)[0]; ys[m]=rs.permutation(y[m])
        ss=fit(tr,ys); shs.append(auroc(ss[te&(y==1)],ss[te&(y==0)]))
    sh=f"{np.mean(shs):.3f} [{min(shs):.2f}-{max(shs):.2f}]"
    print(f"{'':15s} {f.split('/')[0]:12s} {c:7.3f} {ad:7.3f} {np.mean(v):7.3f}({len(v):2d}) {sh:>22s}")
