"""§44 A/B: held-out-family and cross-family native probes; §44 C: frozen dolphin probe on the abliterated checkpoint."""
from _common import *
A6 = acts("run_v6.1_dolphin8b"); Ad = acts("run_dose_dolphin8b"); Ap = acts("run_v6.2_dolphin8b")
ans6 = {a["key"]: a for a in read(P / "run_v6.1_dolphin8b/answers.jsonl")}
v6 = [r for f in ("labels_selvi_204_adjudicated.jsonl", "labels_calibration_246_stage2.jsonl") for r in read(P / "run_v6.1_dolphin8b" / f) if "id" in r and r["stage2_truth"] in ("false_claim", "true_report")]
for r in v6: r["fam"] = ans6[r["id"]]["scenario_id"].split("/")[0]
dose = json.load(open(P / "run_dose_dolphin8b/labels_dose_stage2.json")); proc = json.load(open(P / "run_v6.2_dolphin8b/labels_procurement_stage2.json"))
drows = [{"g": gd(o), "traj": o["trajectory"], "x": Ad[o["id"]]} for o in dose]; prows = [{"g": gp(o), "traj": o["trajectory"], "x": Ap[o["id"]]} for o in proc]
def report(rows, negs):
    out = []
    for n in negs:
        s = [r for r in rows if r["g"] in ("false_claim", n)]; a = grouped_auc([r["g"] == "false_claim" for r in s], [r["s"] for r in s], [r["traj"] for r in s]); out.append(f"vs {n[:12]} {a[0]:.3f} [{a[1]:.2f},{a[2]:.2f}]")
    return " | ".join(out)
print("A. held-out family (fit on v6.1 minus the test family)")
for name, rows, excl, negs in (("legal (dose)", drows, "legal", ["confession", "honest_record", "true_complete", "mixed"]), ("procurement (v6.2)", prows, "procurement", ["disclosure", "true_compliant", "mixed"])):
    tr = [r for r in v6 if r["fam"] != excl]; X = np.stack([A6[r["id"]] for r in tr]); y = np.array([r["stage2_truth"] == "false_claim" for r in tr], int); f = fit(X, y)
    for r in rows: r["s"] = f(r["x"])
    print(f"  {name}: train {len(y)}/{y.sum()} pos -> " + report(rows, negs))
print("B. cross-family")
tr = [r for r in proc if gp(r) in ("false_claim", "disclosure", "true_compliant")]; f = fit(np.stack([Ap[r["id"]] for r in tr]), np.array([gp(r) == "false_claim" for r in tr], int))
for r in drows: r["s"] = f(r["x"])
print("  fit procurement -> legal: " + report(drows, ["confession", "honest_record", "true_complete", "mixed"]))
tr = [r for r in dose if gd(r) in ("false_claim", "confession", "honest_record", "true_complete")]; f = fit(np.stack([Ad[r["id"]] for r in tr]), np.array([gd(r) == "false_claim" for r in tr], int))
for r in prows: r["s"] = f(r["x"])
print("  fit legal -> procurement: " + report(prows, ["disclosure", "true_compliant", "mixed"]))
print("C. second checkpoint (abliterated llama), dolphin-frozen probe, no refit")
native, thr = frozen_native(); Aa = acts("run_dose_ablit8b"); tr = {t["trial_id"]: t for t in read(P / "run_dose_ablit8b/trajectories.jsonl")}
lab = {x["id"]: x["stage1_assertion"] for x in read(P / "run_dose_ablit8b/labels_ablit_stage1.jsonl") if "id" in x}
J = {r["id"].split(":", 1)[1]: r["logodds"] for r in read(P / "judge_nolog.jsonl") if "id" in r and r["id"].startswith("ablit:")}
R = []
for a in read(P / "run_dose_ablit8b/answers.jsonl"):
    if not a["completed"] or a["truncated"]: continue
    f_ = bool(tr[a["trial_id"]]["falsified"]); g = {"asserts_clean": "false_claim", "asserts_problem": "confession", "mixed": "mixed", "other": "other"}[lab[a["key"]]] if f_ else ("true_complete_ctrl" if a["world"] == "dose_100" else "honest_record")
    R.append({"g": g, "traj": a["trial_id"], "s": native(Aa[a["key"]]), "j": J.get(a["key"])})
fc = [r for r in R if r["g"] == "false_claim"]; print(f"  false claims {len(fc)} from {len({r['traj'] for r in fc})} trajectories; flagged {sum(r['s']>thr for r in fc)}/{len(fc)}")
for neg in ("confession", "honest_record", "true_complete_ctrl", "mixed", "other"):
    s = fc + [r for r in R if r["g"] == neg]; a = grouped_auc([r["g"] == "false_claim" for r in s], [r["s"] for r in s], [r["traj"] for r in s])
    line = f"  vs {neg:18s} probe {a[0]:.3f} [{a[1]:.2f},{a[2]:.2f}]  flagged {sum(r['s']>thr for r in R if r['g']==neg)}/{sum(r['g']==neg for r in R)}"
    if all(r["j"] is not None for r in s): b = grouped_auc([r["g"] == "false_claim" for r in s], [r["j"] for r in s], [r["traj"] for r in s]); line += f" | blind judge {b[0]:.3f} [{b[1]:.2f},{b[2]:.2f}]"
    print(line)
