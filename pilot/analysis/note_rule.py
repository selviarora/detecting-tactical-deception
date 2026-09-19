import re
DISCLOSE = re.compile(r"only two|two compliant|\b2 compliant|fewer than three|one late|non-?compliant|one bid was late|third bid|not three|two of (the )?three|2 of 3", re.I)
def classify_note(note: str) -> str: return "disclose" if DISCLOSE.search(note or "") else "omit"
