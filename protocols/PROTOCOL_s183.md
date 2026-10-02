# s183 (frozen): s182 on fresh games with the probe layer fixed in advance

**motivation.** s182 missed only P2's margin (.170 vs .183). Its frozen layer selection tied (dev CV 1.000 at all three layers),
so the first layer (16) was used by default. Post hoc, layers 20/24 gave .927/.992, which cannot be counted. s183 removes the
tie problem by fixing the layer before any s183 data exist:
- **message-mean probe: layer 24.** It was chosen by dev CV in s179, s180 and s181, the runs where selection was not tied.
- **prompt-last probe: layer 20.** It was chosen in s179-s182.

**everything else identical to s182:**
- game, compliance fix, word-balanced readings, labels, capture, all gates (C, T, P1, P2 with the half-headroom margin) and
  reported quantities;
- fresh games (base 183000) and sampling seed (1830).

**dry run.** A planted hiding signal only at layer 24 -> PASS, using layer 24 (A_coinc .98).

**interpretation:** as PROTOCOL_s181/s182. This is the confirmation attempt for the s179-s182 pattern. If it fails, the
line's frozen record stands as: consistent probe-over-text advantage, no single frozen pass.

**cap.** $2.
