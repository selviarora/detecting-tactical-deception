"""s177 (exploratory, $0): which text did NARCBench-Stego's released final-phase activations actually encode? their extraction
(extraction/extract_stego.py) pairs each logged prompt with public_messages in order; final-hand replies are stored in
results.json["final_texts"], not public_messages, so final-phase samples get response "". check: tokenise each final-phase
(system, user, assistant=response) with the model's chat template, for response = "" and response = the actual final text, and
compare with the released metadata n_tokens.
python narc_empty_response_check.py <scenarios_dir> <metadata_gen.json> <hf_tokenizer_id> [chat_template.jinja]"""
import collections, glob, json, re, sys
from transformers import AutoTokenizer
sdir, meta, hf = sys.argv[1:4]; tok = AutoTokenizer.from_pretrained(hf)
if len(sys.argv) > 4: tok.chat_template = open(sys.argv[4]).read()
nt = {(m["run_id"], m["agent_name"]): m["n_tokens"] for m in json.load(open(meta)) if m["phase"] == "final" and m["channel"] == "public"}
c = collections.Counter()
for d in sorted(glob.glob(f"{sdir}/bj__*")):
    rid = d.split("/")[-1]; P = json.load(open(f"{d}/agent_prompts.json")); R = json.load(open(f"{d}/results.json"))
    for p in P:
        if p["phase"] != "final" or p["role"] == "judge" or (rid, p["agent_name"]) not in nt: continue
        n = lambda resp: len(tok(tok.apply_chat_template([{"role": "system", "content": p["system_prompt"]}, {"role": "user", "content": p["user_prompt"]}, {"role": "assistant", "content": resp}], tokenize=False))["input_ids"])
        true = re.sub(r"^\[[^\]]+\]:\s*", "", R["final_texts"].get(p["agent_name"], ""))
        r = nt[(rid, p["agent_name"])]; c["matches prompt + EMPTY reply" if r == n("") else "matches prompt + ACTUAL message" if r == n(true) else "neither"] += 1
print(json.dumps(dict(c)))
