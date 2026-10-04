# PROTOCOL s206 (frozen before data): stronger causal — clean (Arditi-style) ablation + cross-game transfer

**(1) Arditi-style off-switch.** Project the OFFICE signalling unit direction (hide-normal, layer 20, stego8) OUT of the residual at
EVERY layer and position (h <- h - (h.u)u) during greedy generation on RED (signalling) office agents. Measure secret-greeting rate
(expect a clean drop with coherent text = given-greeting rate rises, not gibberish). GREEN under the same ablation = sanity (~0).

**(2) Cross-game transfer.** Use the CASINO direction (hide-normal, stego4+stego7 avg, layer 20; cos with office .26) to control OFFICE:
add alpha*casino to GREEN office (alpha 20,40) -> does office start signalling? and project casino OUT at all layers on RED office ->
does office signalling drop? If yes, the switch GENERALISES across games despite low direction cosine.

250 prompts/condition, greedy, Qwen3-8B. Reported, not gated. ~$2.
