"""histories for Qwen3-8B with the UNCHANGED v6.2 generator (run_v6_2.py, settings_v6); only change: the tokenizer's chat
template is called with enable_thinking=False (Qwen3 would otherwise emit a <think> block before every tool call).
  python gen_hist_qwen3.py --model Qwen/Qwen3-8B --out run_s137_qwen3_hist --families procurement/bid_evaluation --worlds pressure compliant --n-act 110 --n-q 1 --questions count --seed 1370 --layer 24"""
import functools, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE.parent))
import transformers
_orig = transformers.AutoTokenizer.from_pretrained
def _patched(*a, **k):
    tok = _orig(*a, **k); f = tok.apply_chat_template
    tok.apply_chat_template = functools.partial(f, enable_thinking=False); return tok
transformers.AutoTokenizer.from_pretrained = _patched
import run_v6_2
if __name__ == "__main__": run_v6_2.main()
