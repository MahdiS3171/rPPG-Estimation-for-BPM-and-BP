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
to modernize navigation. Proposed main integration and draft closure remain
pending user approval. No remote mutation has occurred.

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

## Pending — Main integration and controlled-data access (D007/D008)

The alternatives, consequences and recommended policy are in
[PROJECT_MASTER_PLAN.md](PROJECT_MASTER_PLAN.md). No decision has been inferred
from elapsed time. Substantial training and first-use held-out evaluations will
need separate, concrete proposals after suitable data and protocols exist.
