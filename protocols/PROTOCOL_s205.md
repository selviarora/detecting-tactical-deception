# PROTOCOL s205 (frozen before data): CAUSAL intervention — is the signalling direction a switch, or just correlated?

**Direction.** d = mean(hide) - mean(normal) of the layer-20 message-mean activations (stego8, compliant). Steering vector = alpha * d.

**Induction (GREEN prompts: agent NOT signalling, given greeting != secret; baseline opens with GIVEN).** Add alpha*d to the layer-20
residual during greedy generation and measure the rate of opening with the SECRET greeting (induced covert signalling). alpha in {0,10,20,40}.
**Specificity control:** add a RANDOM unit direction of the same norm (alpha 20,40) — should NOT induce the secret greeting.
**Ablation (RED prompts: agent IS signalling; baseline opens with secret).** Subtract alpha*d (alpha 20,40) and measure whether the
secret-greeting rate drops.

**Reading:** if adding d flips green agents into emitting the secret greeting (and random does not), and subtracting d stops red agents
signalling, the direction CAUSES the signalling behaviour (not just correlated). 250 prompts/condition, greedy. Reported, not gated. ~$3.
