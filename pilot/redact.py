"""strip the simulator's evaluative bookkeeping (falsified / violation) from tool results, keeping every fact.
used for the redacted replay and the redacted judge. the generator saw the unredacted version; that is disclosed."""
import json, re
KEYS = ("falsified", "violation")
def redact_result(res):
    if isinstance(res, dict): return {k: v for k, v in res.items() if k not in KEYS}
    return res
def redact_text(s):
    # tool results are embedded in messages as "[tool result] {json}"
    m = re.match(r"^(\[tool result\] )(\{.*\})$", s, re.S)
    if not m: return s
    try: return m.group(1) + json.dumps(redact_result(json.loads(m.group(2))), ensure_ascii=False)
    except Exception: return s
def redact_messages(msgs):
    return [{**m, "content": redact_text(m["content"])} if m.get("role") == "user" else m for m in msgs]
