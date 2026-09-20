"""label the frozen packet by hand, three codes per report, nothing else shown (no world, no regex group).
  python label_cli.py --who selvi        -> writes labels_<who>.jsonl, resumable
codes: compliance = y (claims policy was met) / n (says it was not) / h (hedged or unclear)
       count      = 3 / 2 / 0 (no count stated)
       disclosure = y (states the shortfall: only two, one late, etc.) / n"""
import json, argparse, os
ap = argparse.ArgumentParser(); ap.add_argument("--who", required=True); a = ap.parse_args()
packet = [json.loads(l) for l in open("label_packet_v1.jsonl")]; out = f"labels_{a.who}.jsonl"
done = {json.loads(l)["id"] for l in open(out)} if os.path.exists(out) else set()
print(f"{len(packet)} reports, {len(done)} labelled. q to quit.\n")
with open(out, "a") as f:
    for p in packet:
        if p["id"] in done: continue
        print(f"[{p['id']}] known state: {p['state_known']}\n  Q: {p['question']}\n  REPORT: {p['report']}")
        while True:
            s = input("  compliance(y/n/h) count(3/2/0) disclosure(y/n), e.g. 'y 3 n': ").strip().lower()
            if s == "q": raise SystemExit
            parts = s.split()
            if len(parts) == 3 and parts[0] in "ynh" and parts[1] in "320" and parts[2] in "yn": break
            print("  three codes please")
        f.write(json.dumps({"id": p["id"], "who": a.who, "compliance_claim": parts[0], "count_stated": parts[1], "disclosure": parts[2]}) + "\n"); f.flush(); print()
