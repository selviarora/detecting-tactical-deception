Run the frozen 96-case truth-versus-wording diagnostic in this directory on Llama-3.3-70B-Instruct. The cases, four probes and analysis are already fixed; do not rebuild cases, refit probes or select a new threshold. This is forward-only supplied-report capture, with no generations. Use the existing compatible GPU environment and model access.

From this directory:

```bash
python capture.py --out capture_llama --dry-run
python capture.py --out capture_llama
python score.py --capture capture_llama/activations.npz --out scored
```

The first activation row is checkpointed immediately. The full capture is one model load. If interrupted, use `python capture.py --out capture_llama --resume`; a partial cache must not be marked done. Preserve logs on failure and report the actual exception. The scorer requires all 96 IDs, matching hashes, finite activations and complete metadata. Copy back the whole capture_llama and scored directories before any pod shutdown.

Interpret the complete table by setting, note, question polarity and report form. “Yes” and “No” each occur as both true and false. A pure Yes/No rule therefore cannot pass this diagnostic. Some truth-dependent ranking would still be supplied-text discrimination, not proof of natural deception detection or a validated deployment threshold. This pack is prepared and CPU-tested; it does not contain a completed 70B run.
