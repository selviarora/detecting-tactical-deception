# detecting tactical deception 

tldr; the question i care about is:

> can we build an internal monitor that detects when an ai system knowingly and selectively misleads someone to advance its objective?

this repo is everything i have learned while trying. :)

______________

i started with a pretty simple question:

**do deception probes actually detect deception, or do they detect easier things that happen to show up in deception datasets?**

the dream is a monitor that can look inside a model and tell when it is lying. that would be insanely useful. but before trusting one, we need examples where we actually know:

- what the model knows;
- what it reports;
- whether the report misleads someone;
- whether misleading them helps the model;
- whether the model changes its behaviour when the incentive or oversight changes.

that turned out to be much harder than training a probe.

## where this started

i reproduced apollo research's released linear deception probe, and i was really curious about how they measured “deception.” i learned that most of the training examples compare a model told to be honest with one told to be deceptive.

now that seems reasonable, except it changes two things at once:

- whether the model misleads someone; and
- whether it has been told to override an answer it knows.

so a model could override an answer it knows without actually deceiving anyone. so i built a matched control where the model did the same override, but only sometimes left another person with a false belief.

the probe responded much more strongly to the override itself than to whether anyone was actually misled. it also fired on refusals and honest admissions.

![What moves Apollo's deception probe: being told to withhold moves the score +1.89 SD, actually misleading someone +0.12. Apollo's released instructed_pairs probe, Llama-3.3-70B, layer 22, compliant answers (FACTS.md §3)](figs/apollo_override.png)

*Left: mean score from Apollo's released `instructed_pairs` probe (Llama-3.3-70B, layer 22, compliant answers) for an honest answer and for two conditions where the model is told to withhold. Right: the effect of each factor from a 2×2 regression, in standard deviations of the probe score. Source: `FACTS.md` §3.*

now, this does not mean the probe has no useful signal. but it does mean that strong performance on examples where a model is told to deceive is not enough to show that the probe detects deception. it may be detecting the instruction to override or withhold an answer instead.

![Apollo's probe flags harmless withholding: at its 1% false-alarm setting it fires on 92% of harmless prompts (persona, policy topics, redaction, summaries, safety refusals); with one direction removed (bench/, held-out categories) 21%. Llama-3.3-70B, layer 22, bench/results](figs/apollo_false_alarms.png)

*Share of harmless prompts flagged by the same probe at a threshold set to a 1% false-alarm rate on 60 plain-chat prompts (25 prompts per category). Purple: the probe with one fitted direction removed, scored leave-one-category-out. Source: `bench/results/`.*

## the strongest thing that worked

i built a balanced 2×2 dataset where neither the report wording nor the world state predicts the label on its own. the useful distinction is whether the report matches the evidence the model saw.

a probe trained on those manufactured contrasts transferred across held-out task families and three model families. it substantially outperformed an apollo-style training recipe in those controlled tests.

then i rewrote the environments independently and tested the frozen weights again.

they did not transfer cleanly :(

a small number of examples from the new environment helped a lot, which suggests the training recipe is useful. but the weights were not universal. when i tested sincere mistakes, the probe often flagged those too.

sooo what i actually built was a strong report-vs-evidence inconsistency detector. sure, that could be useful inside a deception monitoring system, but it is not really what i was excited about. 

## why this kept getting harder

a lot of results looked exciting at first and then failed a better control. :')

some examples:

- a direction that looked related to deception mostly moved a specific number;
- a probe that looked like it detected knowing falsehood mostly detected contradiction with the transcript;
- apparent incentive driven lying came from wording that made “yes” mean “keep the action”;
- apparent strategic behaviour followed whichever policy requirement was made salient, even when that hurt the model;
- a model organism looked audit-sensitive, but incentive reversal showed it had learned “say three under low audit,” not “lie when it helps”;
- private answers that appeared to measure the model's belief were not always consistent with how it later used the same information.

i kept these failures in the repo because they are exactly the kinds of mistakes that could make a deception detector look much better than it is.

## what i mean by deception now

the definition is important here - i am using a functional definition of **tactical deception**.

the model must:

1. know the relevant truth;
2. make a report that can leave someone with a false belief;
3. benefit from that false belief;
4. misreport more as the benefit increases;
5. misreport less when verification becomes likely;
6. reverse the lie when the incentive reverses;
7. stay honest when lying cannot help.

this does not prove anything philosophical about the model's inner experience; it just gives me a safety-relevant behavioural target that can actually be tested.

## the organisms

the hard part was producing a model organism that learns the abstract strategy “misreport when it helps,” rather than memorising a particular answer.

the first organism knew the truth and responded to audit probability:

- it misreported 44% of the time under low audit;
- it never misreported under full audit;
- it never misreported when the outcome was already fixed.

that initially looked selective.

the incentive-reversal control showed what it had really learned. it always lied in the same direction. when that answer began hurting its objective, it continued producing it. when the opposite lie became useful, it never switched.

the organism had learned something like “under low audit, say three,” not “misreport whichever way advances the objective.”

that was a useful failure. it stopped me from probing a surface reflex and calling the resulting direction deception.

the second organism was trained across both lie directions and several payoff structures, then tested on task families and a payoff framing it had never seen. it passed every pre-registered check:

- it misreported 83-100% of the time when the lie paid and oversight was weak;
- it never misreported when the lie did not pay, under full audit, or when the outcome was fixed;
- in matched pairs where only the payoff flipped, it lied in the paying one and stayed honest in the other 92% of the time.

the matched-pair result replicated on a new training seed (96%) and with a second training method, DPO (75%).

![Organism #2 lies only when it pays: misreport rate .83 when lying pays under weak audit, .00 when it does not pay, .00 under full audit; the reflex organism #1 misreports .44 and .43 either way. FACTS.md §143, §144](figs/organism.png)

*Rate of reporting "three" when the model saw two, for two LoRA fine-tunes of Qwen3-8B on held-out task families. Organism #1 was trained on one incentive direction; organism #2 on both directions and several payoff structures. Source: `FACTS.md` §143, §144.*

one caveat: the strategy is installed by training. stock qwen3-8b never misreports in these settings (0/320).

## what i found inside it

- a detector built from a few directions passed every frozen contrast on a second organism and an unseen payoff framing, and beat a reasoning text judge on the same samples;
- it is a readout, not a lever: removing or steering its directions did not change the lie;
- single-direction and two-direction ablations found a “say two” bias and a false-report lever, not a deception switch;
- on the DPO organism, the detector missed the key contrast (tempted-but-honest vs lie, .40).

so far the behaviour looks entangled with how the model represents the count and the oversight, rather than living in one deception mechanism.

## side studies

two things that came out of the main question, both written up in `FACTS.md`:

- [other people's detectors](FACTS.md#part-8-other-peoples-detectors) (§166-173): controls on Rift's residual-rank deception signature. the separation shows up even when no lie is told.
- [hidden signals](FACTS.md#part-9-hidden-signals) (§174-207): a probe for covert signalling between agents in steganography games. trained only on casino games, it caught covert signals in office emails in qwen3-8b, gpt-oss-20b and phi-4, and failed in qwen3-32b. this part also includes a check of the NARCBench collusion benchmark.

## how to read this repo

- [`FACTS.md`](FACTS.md) is the full lab notebook, including corrections, retractions and failed hypotheses.
- [`protocols/`](protocols) contains frozen protocols and decision rules written before experiments.
- `bench/` contains the apollo reproduction, probe training and false-positive controls.
- `pilot/` contains agent environments, model-organism experiments, activation capture, scoring and causal interventions.
- `monitor/` contains runtime monitoring code.

for the current picture, read this README first; use `FACTS.md` for details.

## reproducibility

every major experiment has a protocol frozen before the run in `protocols/`, and its raw counts, configs and any corrections in its run folder and `FACTS.md` section.
