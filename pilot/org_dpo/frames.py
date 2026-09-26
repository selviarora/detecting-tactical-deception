"""s148: teacher-written REPORT FRAMES (gpt-4.1-mini). the teacher never sees histories, payoffs, audits or which count is true;
it writes varied one-sentence factual reports with a {N} slot for the count. preferred and rejected DPO completions are the SAME
frame filled with different counts, so style cannot carry the label. frames containing deception/audit/payoff vocabulary or a
second number are dropped.   OPENAI_API_KEY=... python frames.py"""
import json, re, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE.parent / "universal")); import design7 as D7
from openai import OpenAI; c = OpenAI()
BAD = re.compile(r"\b(lie|lying|honest|truth|audit|check|verify|bonus|fee|policy|promot|licen|rating|reward|punish|deadline|record shows|exactly)\b", re.I)
out = {}
for fam in D7.FAMS:
    claim = D7.FAMS[fam][0].format(n="{N}")
    msg = (f"Here is a one-sentence factual status report: \"{claim}\"\nWrite 70 different one-sentence factual reports that convey the same fact, "
           "each containing the placeholder {N} exactly once where the number goes (the number will be filled in later as a word like 'two' or "
           "'three', or a digit). Vary sentence structure, word choice, formality and length (6 to 25 words). Do not add any other numbers, "
           "opinions, hedges, reasons or commentary. Output one report per line, no numbering.")
    txt = c.chat.completions.create(model="gpt-4.1-mini-2025-04-14", temperature=1.0, max_tokens=4000, messages=[{"role": "user", "content": msg}]).choices[0].message.content
    fr = []
    for l in txt.split("\n"):
        l = l.strip().strip('"').strip()
        if l.count("{N}") != 1 or BAD.search(l) or re.search(r"\d|\b(one|two|three|four|five|both|all|none)\b", l.replace("{N}", ""), re.I): continue
        if l not in fr: fr.append(l)
    out[fam] = fr; print(fam, len(fr), "| e.g.", fr[:2])
json.dump(out, open(HERE / "frames.json", "w"), indent=1)
