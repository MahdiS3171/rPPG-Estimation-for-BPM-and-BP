# Material research decisions

## 2026-11-08 — Reassess the whole program (D001)

Authority: the user's project-level request supersedes individual Gate prompts.
Read the complete master handoff (SHA256
`32e9e96b4dd3fabf302994a6607c7f14a9fedf56605421b0d2abeadc21488aa5`).
Keep historical reports unchanged and reconcile them in the current audit.
No automatic Gate 10 or further stacked Gate assignment.

## 2026-11-08 — Separate waveform, physical timing and BP claims (D002)

Decision: run waveform research and camera feasibility as distinct workstreams;
make BP-from-delay contingent on independently resolved/repeatable timing.
Rationale: external V1 failure, pilot's single accepted independent window, and
unverified clocks/reference sites. Alternative: synthetic training immediately;
rejected as low information for the physical bottleneck. The Gate 09 report's
GO means a possible plumbing experiment, not a scientific or training mandate.

## 2026-11-08 — Preserve the linear stack on one local descendant (D003)

Branch from `bb5fb59683615489db0da451c52d9ff9de2fa49d`; preserve all 19 commits.
Use additive files because historical result verification pins 224 inherited
files, including README and verification scripts. Do not edit those files just
to modernize navigation. Proposed main integration and old draft closure remain
pending user approval. Subsequent branch publication is recorded below.

## 2026-11-08 — Acquisition identifiability before another optimizer (D004)

Commit `5edac8d` fixed the protocol and simulator before the 1,536-trial run.
Reused the existing lag estimator; used hypothetical, explicitly declared camera
parameters. Equivalent acquisition worlds differ by less than 5e-15 in sampled
intensity but have different physiological delays. This is a model-level
identifiability limitation, not evidence about an actual phone's readout time.
Keep reported uncertainty floors; do not reinterpret numerical error as physics.

## 2026-11-08 — Existing phones and cuff only (D005, user decision)

User answer: "Only the existing phones and cuff are available; prioritize camera
calibration and waveform research." Do not presume sensor borrowing/purchases.
Develop a phone-only acquisition diagnostic; clearly identify what it cannot
validate. External multisite-reference data remains a possible institutional
access route, not assumed equipment. Continue independent waveform work.

## 2026-11-08 — Native-60 model and online data (D006, user steering)

User requested broader available online waveform data and a model able to use
the details of 60-fps recordings. Accept that candidate direction; retain V1 as
a frozen comparator. Inspect the exact 60-to-30 adapter and prior bandwidth,
then use a matched 60/30 ablation. Do not presume angle is the primary failure.
No substantial training or new volunteer collection has been executed.

Downloaded only the smallest official public MPU MP4 and a candidate CSV for
development schema/timebase admission (~107.36 MB, source checksums verified).
The posted CC BY 4.0 licence and research-use notice permit the requested local
research. Source file IDs prevent repeated Output filenames from overwriting
one another. Pairing/clock/identity remain unverified; no training or scoring.
Gated native-60 leads require institutional action, not an assumed open mirror.

## 2026-11-08 — One reviewable integration draft (D007)

Published the new research branch and created
[draft PR #7](https://github.com/MahdiS3171/rPPG-Estimation-for-BPM-and-BP/pull/7)
directly against main, using the mandate's routine branch/PR authority. This
preserves all six ancestors and provides one concrete review/CI surface. No
existing PR or shared branch was changed; no main merge, branch deletion or
data release occurred. The new raw development download remains ignored.

## 2026-11-08 — Integration approved (D008, user decision)

User explicitly approved one history-preserving merge of PR #7 after final
checks and closure of the six superseded drafts while retaining their branches.
Execute only against the checked head with a passing CI result; recheck origin
main before merging. This is the specific integration authorization, not blanket
permission for future major main changes or history rewriting. The GitHub PR
records the resulting merge commit when executed.

## 2026-11-08 — Tricam access route selected (D009, user decision)

User will pursue Tricam access with their supervisor. The local request draft in
[DATA_ACCESS_ACTIONS.md](docs/DATA_ACCESS_ACTIONS.md) is ready. No agent email,
signature, contract acceptance or controlled-data transfer is authorized by that
answer. Keep existing PURE pending; continue the public development admission
and native-60 design work. Substantial training and first-use held-out evaluation
still need concrete proposals after suitable data and protocols exist.

## 2026-11-08 — Continue now; two-source training and broader ideas (D010)

User requested work with current datasets while new access is pending, proposed
training on two datasets with a third dataset reserved for final evaluation,
and explicitly requested adaptive classical hybrids, expanded models and varied
loss functions. Implement a bounded local campaign on current development data.
Keep original frozen/protected boundaries; the two available sources are real
UBFC and synthetic SCAMPS, not two independent real domains. A third-domain final
evaluation remains separate and requires an admitted, unconsumed protocol.

## 2026-11-08 — Test mixtures and adaptive/loss components independently (D011)

The 20-solve linear study's 50:50 pooling worsened UBFC development scores.
Do not infer all multi-source training fails or tune to the consumed external
test. The neural screen therefore includes 0/20/50% source sampling controls,
single-source and shuffled-label controls, and fixed classical comparators.
It trained 21 compact models using a local RTX 4060; no paid compute. Prototype
temporal kernels preserve physical support across 30/60-Hz grids, without
claiming 60-Hz physiological supervision from 30-Hz labels.

## 2026-11-08 — Confirm component benefits before combining them (D012)

First-screen hybrid morphology training improved aligned shape/derivative metrics,
while native correlation did not improve clearly. Confirm across all original
Train subjects and additional seeds; retain a simple quality-weighted hybrid as
a serious comparator. Follow-up hypotheses are fixed in the confirmation
protocol before fitting. More complexity is retained only for measured value.

## 2026-11-08 — Push every substantive completed milestone (D013, user instruction)

User explicitly requested commits and GitHub pushes after each important step.
Publish the ongoing research branch after protocols/implementations and after
verified result milestones; retain one coherent branch instead of new stacked
Gate PRs. Raw data, model checkpoints and participant waveform artifacts remain
ignored. Main merges still follow the existing approval boundary.

## 2026-11-08 — Prefer demonstrated correction over accumulated complexity (D014)

Matched controls show residual correction provides most of the shape gain;
input-adaptive gates add a modest increment. Keep constant/quality-based hybrids
as live comparators. The first architecture screen motivates a bounded 20-fit
five-fold check of context, attention, spectral and time-varying gates. Stop
expanding families that do not improve the relevant endpoints. This remains
repeated development selection, not an independent generalization result.

## 2026-11-08 — Retain a robust development candidate (D015)

The 20 architecture fits favor longer TCN context over added attention/spectral
complexity. Ten matched ROI-training fits improve shape/stability. Retain the
longer TCN with 50% region dropout and a 0.2 native-output consistency penalty
as the leading development candidate. Its native-r difference interval still
includes zero; no HR, physical timing, native-60 detail or independent-domain
claim is warranted. Export one full-original-Train fit without a validation
score, preserving frozen V1. End broad architecture search on the same small
cohort; prioritize suitable real-domain/reference admission next.

## 2026-11-08 — User authorizes a public-data split reset (D016)

The user requested all eligible public Train/Validation/Test records be
repartitioned for newly initialized models, replacing the prior Train-only
development boundary. Use all 41 UBFC subjects and ten admitted SCAMPS clips,
assigned 29/6/6 and 8/1/1 with the precommitted deterministic split. New Train
may contain old public Val/Test subjects. Preserve all historical splits,
evidence, frozen V1 and the prior candidate; private/sealed participants remain
excluded. Explain that fresh weights do not erase analyst/architecture exposure:
this is a new internal benchmark, not an independent external cohort. New Test
is separate from selection through the waveform/specialist validation campaign.
Use matched same-validation controls for any claim about more training data.
