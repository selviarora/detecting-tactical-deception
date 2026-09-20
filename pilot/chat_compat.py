"""compat for chat templates that (a) reject a system role and/or (b) require strict user/assistant alternation
(gemma). call patch_tokenizer(tok) once after loading; every apply_chat_template call then goes through fold():
the system message is prepended to the first user turn, and adjacent same-role turns are merged with a blank line.
the text the model sees is the same words in the same order; only the turn boundaries change."""
def _needs_fold(tok):
    try:
        tok.apply_chat_template([{"role": "system", "content": "s"}, {"role": "user", "content": "u"}, {"role": "assistant", "content": "a"}, {"role": "user", "content": "u2"}, {"role": "user", "content": "u3"}], add_generation_prompt=True, tokenize=False); return False
    except Exception: return True
def fold(msgs):
    out = []
    for m in msgs:
        m = dict(m)
        if m["role"] == "system": m["role"] = "user"
        if out and out[-1]["role"] == m["role"]: out[-1] = {"role": m["role"], "content": out[-1]["content"] + "\n\n" + m["content"]}
        else: out.append(m)
    return out
def patch_tokenizer(tok):
    if not _needs_fold(tok): return False
    orig = tok.apply_chat_template
    def wrapped(msgs, *a, **k): return orig(fold(msgs), *a, **k)
    tok.apply_chat_template = wrapped; print("chat_compat: system folded into user; adjacent same-role turns merged (template requires alternation)", flush=True); return True
