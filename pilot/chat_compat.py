"""fold the system message into the first user turn for chat templates that reject a system role (gemma). call
patch_tokenizer(tok) once after loading; every apply_chat_template call then goes through the fold when needed."""
def _needs_fold(tok):
    try: tok.apply_chat_template([{"role": "system", "content": "s"}, {"role": "user", "content": "u"}], add_generation_prompt=True, tokenize=False); return False
    except Exception: return True
def fold(msgs):
    if not msgs or msgs[0]["role"] != "system": return msgs
    sysm, rest = msgs[0]["content"], list(msgs[1:])
    if rest and rest[0]["role"] == "user": rest[0] = {"role": "user", "content": sysm + "\n\n" + rest[0]["content"]}
    else: rest.insert(0, {"role": "user", "content": sysm})
    return rest
def patch_tokenizer(tok):
    if not _needs_fold(tok): return False
    orig = tok.apply_chat_template
    def wrapped(msgs, *a, **k): return orig(fold(msgs), *a, **k)
    tok.apply_chat_template = wrapped; print("chat_compat: system role folded into the first user turn for this template", flush=True); return True
