"""the one chart. fpr before/after one direction removed, every row from the same probe file.
   python make_fpr_chart.py > ~/site/media/ovd-fpr.svg"""
import json
from pathlib import Path

HERE = Path(__file__).parent
# numbers from FACTS.md section 16: everything read as the mean over the model's own answer,
# threshold on 150 alpaca answers the fit never saw, checked on the other 150. one probe file for every row.
M = {"plain": (2, 2), "benign": (18, 2), "ob_ref": (58, 0, 115), "ob_ans": (8, 3.5, 85), "xs_ref": (23, 0, 150)}

rows = [
    ("plain chat, held-out half (150)", "apollo's calibration set. the 1% operating point", M["plain"][0], M["plain"][1], True),
    ("instructed withholding (150)", "system prompt says what not to say. held-out categories", M["benign"][0], M["benign"][1], False),
    ("OR-Bench toxic, model refused (115)", "real refusals, no system prompt", M["ob_ref"][0], M["ob_ref"][1], False),
    ("OR-Bench toxic, model answered (85)", "same topics, model answered", M["ob_ans"][0], M["ob_ans"][1], False),
    ("XSTest unsafe, model refused (150)", "", M["xs_ref"][0], M["xs_ref"][1], False),
]

SANS = "Manrope, -apple-system, system-ui, sans-serif"
MONO = "ui-monospace, SFMono-Regular, Menlo, monospace"
X0, X1 = 300.0, 600.0
def x(v): return X0 + (X1 - X0) * v / 100
def t(x_, y, s, size, fill, anchor="start", w=400, fam=SANS):
    return f'<text x="{x_}" y="{y}" dy="0" font-family="{fam}" font-size="{size}" fill="{fill}" text-anchor="{anchor}" font-weight="{w}">{s}</text>'

out = ['<svg xmlns="http://www.w3.org/2000/svg" width="640" height="300" viewBox="0 0 640 300" role="img" aria-label="false positive rate before and after one direction removed">',
       t(20, 22, "how often the probe fires on things that aren't lies", 13, "#8a8a8a", w=500),
       t(20, 40, "scored over the model's answer at apollo's 1% operating point. one direction removed, every row.", 11, "#a3a3a3"),
       '<circle cx="305" cy="52" r="4.5" fill="#c9c2f3"/>', t(314, 56, "their probe", 11, "#a3a3a3"),
       '<circle cx="395" cy="52" r="4.5" fill="#8577e6"/>', t(404, 56, "one direction removed", 11, "#a3a3a3")]
for v in (0, 25, 50, 75, 100):
    out.append(f'<line x1="{x(v)}" y1="62" x2="{x(v)}" y2="242" stroke="#8a8a8a" stroke-width="0.5" stroke-opacity="0.35"/>')
    out.append(t(x(v), 256, f"{v}%", 10, "#a3a3a3", "middle", fam=MONO))
y = 66
for label, sub, a, b, same in rows:
    out.append(t(288, y + 4, label, 11.5, "#8a8a8a", "end"))
    if sub: out.append(t(288, y + 18, sub, 9.5, "#a3a3a3", "end"))
    xa, xb = round(x(a), 1), round(x(b), 1)
    if not same:
        out.append(f'<line x1="{xa}" y1="{y}" x2="{xb}" y2="{y}" stroke="#8577e6" stroke-width="2" stroke-opacity="0.45"/>')
    out.append(f'<circle cx="{xa}" cy="{y}" r="6" fill="#c9c2f3"/>')
    out.append(f'<circle cx="{xb}" cy="{y}" r="6" fill="#8577e6" stroke="#f9f9f8" stroke-width="0"/>')
    if same:
        out.append(t(xa + 10, y + 4, f"{a:.0f}%", 10.5, "#a3a3a3", fam=MONO))
    else:
        out.append(t(xa + 10, y - 7, f"{a:.0f}%", 10.5, "#a3a3a3", fam=MONO))
        out.append(t(xb + 10, y + 15, f"{b:.0f}%", 10.5, "#8577e6", w=600, fam=MONO))
    y += 38
out.append("</svg>")
print("\n".join(out))
