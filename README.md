# detecting tactical deception 

i started with a pretty simple question:

**do deception probes actually detect deception, or do they detect easier things that happen to show up in deception datasets?**

the dream is a monitor that can look inside a model and tell when it is lying. that would be insanely useful. but before trusting one, we need examples where we actually know:

- what the model knows;
- what it reports;
- whether the report misleads someone;
- whether misleading them helps the model;
- whether the model changes its behaviour when the incentive or oversight changes.

that turned out to be much harder than training a probe.

this repo is my attempt to work through that problem. it includes the results that held up, the experiments that failed, the controls that killed exciting interpretations, and the corrections i made along the way. 

## where this started

i reproduced apollo research's released linear deception probe, and i was really curious about how they measured “deception.” i learned that most of the training examples compare a model told to be honest with one told to be deceptive.

now that seems reasonable, except it changes two things at once:

- whether the model misleads someone; and
- whether it has been told to override an answer it knows.

so a model could override an answer it knows without actually deceiving anyone. so i built a matched control where the model did the same override, but only sometimes left another person with a false belief.

the probe responded much more strongly to the override itself than to whether anyone was actually misled. it also fired on refusals and honest admissions.

now, this does not mean the probe has no useful signal. but it does mean that strong performance on examples where a model is told to deceive is not enough to show that the probe detects deception. it may be detecting the instruction to override or withhold an answer instead.

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

## current status

the immediate bottleneck is producing a model organism that learns the abstract strategy “misreport when it helps,” rather than memorising a particular answer.

the first organism knew the truth and responded to audit probability:

- it misreported 44% of the time under low audit;
- it never misreported under full audit;
- it never misreported when the outcome was already fixed.

that initially looked selective.

the incentive-reversal control showed what it had really learned. it always lied in the same direction. when that answer began hurting its objective, it continued producing it. when the opposite lie became useful, it never switched.

the organism had learned something like “under low audit, say three,” not “misreport whichever way advances the objective.”

that was a useful failure. it stopped me from probing a surface reflex and calling the resulting direction deception.

the next organism is trained across both lie directions and several payoff structures, then tested on a task and incentive framing it has never seen.

it must pass the behavioural gate before i inspect its activations.

## what i am testing now

the new organism is trained with:

- both directions of advantageous misreporting;
- multiple objective and payoff formulations;
- independently balanced evidence, desired answers and audit levels;
- low-audit, high-audit and fixed-outcome conditions;
- truthful controls where deception cannot help.

the frozen evaluation uses unseen task families and an unseen payoff framing.

the organism must:

1. retain the correct underlying knowledge;
2. misreport more when doing so advances its objective;
3. reverse its misreport when the payoff mapping reverses;
4. remain honest under verification;
5. remain honest when the outcome is already fixed;
6. generalise to a payoff structure it was never trained on.

changing only the payoff mapping should change the direction of the false report. if it does not, the organism fails and no mechanistic probe is trained.

## what happens if the organism passes

i will compare several possible explanations of how deception is represented:

1. one linear direction shared across tasks;
2. a low-dimensional deception subspace;
3. a nonlinear representation;
4. a temporal circuit that appears during report generation;
5. a common procedure that discovers different mechanisms in different models;
6. no special deception mechanism, with the behaviour composed from ordinary representations of truth, payoff, audience belief and action selection.

the detector must distinguish tactical deception from:

- sincere mistakes;
- ordinary false statements;
- tempted-but-honest decisions;
- report-evidence inconsistencies without strategic selection;
- refusals;
- uncertainty;
- paltering and omission;
- memorised behavioural reflexes.

any candidate mechanism must then survive causal testing. changing it should reduce tactical deception while preserving the model's knowledge, capabilities and honest communication.

## when detection needs to happen

i originally treated weak prediction before the answer as a negative result. that was too broad.

there are at least three useful monitoring targets:

- **pre-action:** detect the behaviour before the model acts;
- **online:** detect it during generation and interrupt the report;
- **post-report:** quarantine or verify the report before anyone relies on it.

a detector does not need to read a hidden decision before the first output token to be useful. an online detector that becomes reliable while the report is being written could still protect a real system.

the important comparison is whether model activations add useful information beyond the text available at the same point.

## how to read this repo

- [`FACTS.md`](FACTS.md) is the full lab notebook, including corrections, retractions and failed hypotheses.
- `PROTOCOL_*.md` contains frozen protocols and decision rules written before experiments.
- `bench/` contains the apollo reproduction, probe training and false-positive controls.
- `pilot/` contains agent environments, model-organism experiments, activation capture, scoring and causal interventions.
- `data/` contains controlled evaluation datasets.
- `monitor/` contains runtime monitoring code.

`FACTS.md` is long because i kept every experiment, including the ones that failed or were retracted. for the current picture, read this README first; use `FACTS.md` for details.

## important limits

- no universal deception detector or direction has been found;
- natural strategic deception was rare in the open models i tested;
- the strongest detector reads report-vs-evidence inconsistency, not deception specifically;
- the detector's frozen weights did not transfer cleanly to an independently written harness;
- several attempted belief measurements were unstable across different checks;
- several positive pilots disappeared under wording, reversal, calibration or causal controls;
- the current selectively deceptive organism has not yet passed its behavioural gate;
- cross-organism, cross-model and causal validation remain incomplete.

when a result fails a stronger control, i leave it in the repo. those failures are useful because they show how easy it is to build something that looks like a deception detector but actually reads a shortcut.

## reproducibility

for each major experiment, i try to preserve:

- the protocol and success criterion fixed before the run;
- model and sampling configuration;
- training, calibration and evaluation partitions;
- raw counts and grouped uncertainty;
- parser and labelling corrections;
- whether an analysis was pre-registered or post hoc;
- stopping decisions;
- known data loss or implementation mistakes.

exact commands and inputs are documented alongside the relevant protocol and experiment files.

## the goal

the question i care about is:

> can we build an internal monitor that detects when an ai system knowingly and selectively misleads someone to advance its objective?

this repo is everything i have learned while trying. :)
