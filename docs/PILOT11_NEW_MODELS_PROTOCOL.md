# Eleven-subject frozen-model transfer, timing and BP feasibility

Fixed 2026-11-08 before opening newly authorized participant data. The user
requested new-model testing on all eleven BP-pilot subjects, including timing
and BP-learning feasibility, and identified the updated
`sheets/rPPG BP Data Collection final.xlsx` as current. This explicitly permits
local S001–S011 use for this protocol, superseding the earlier seal for these
analyses. It does not authorize waveform fine-tuning, unrelated data access,
raw-data publication or clinical claims.

## Identity, frozen models and privacy

Commit the protocol, authorization and checkpoint selection before first access.
Snapshot the explicitly named workbook by bytes/hash; no old-export fallback.
Preserve formulas/cached cells, exact headers, units, missingness and exclusions.
Match explicit `Dataset/S001_V01.mp4` through `S011_V01.mp4` paths to exact
session/subject IDs; never use row order or BP values to infer identity.
Honor explicit exclusions or consent restrictions. Source files stay read-only.
Private tables, traces, figures and predictions stay under ignored
`outputs/pilot11_new_models/`; public Git receives aggregate findings only.
No participant images, individual pressures or identifying metadata are uploaded.

S001–S006 are exposed development subjects. S007–S011 will no longer be untouched
after this frozen evaluation; create a consumption receipt. Whole-sheet import
supports fixed identity/QC, but held-out BP values do not inform feature/model
selection. Numerical validity checking is separate from predictive fitting.

Use exactly the six waveform and nine direct-feature checkpoints in the
[retained-model index](waveform_research/reset_retained_models_20261108.json),
checking SHA-256 before inference. Two waveform families: compact real-only
and mixed robust TCN; three feature modes: independent experts, joint and
shared-head, each at their three fixed seeds. No pilot-based member/epoch
selection, waveform training or changes to frozen V1/prior checkpoints.

The models use three facial ROIs and 30-Hz contact-derived training targets.
They neither output BP nor have validated hand-domain training. Infer only on
matching facial inputs; never tile one palm into three face channels. Primary
timing uses matched classical face/palm processing. Learned face versus
classical palm is a processing sensitivity, not validated physiological timing.

## Acquisition, inference and timing

Reuse native RGB caches read-only when video hashes, clocks, ROI definitions
and provenance match; record legacy detector/asset differences explicitly.
Acquire missing traces at native pixels/frame PTS using inherited extractors,
inside the new namespace. No participant image pixels are saved. Up to three
local acquisition workers; reassess projected runs exceeding 30 minutes.
Do not silently substitute an untested detection cadence.

Primary face inference: nonoverlapping ten-second windows at 30 Hz. Actual
high-rate recordings also support a prespecified 60-Hz numerical sensitivity,
with identical physical duration and explicit interpolation/gap accounting.
No reference alignment before timing. Without local contact PPG, waveform
correlation cannot be measured. Audit finite/flat outputs, HR/beat support,
ROI/method/seed agreement and failures; appearance/HR is not waveform truth.

Primary optical pair: combined face GREEN and central palm GREEN, matched
0.7–4-Hz filters on each contiguous native-rate segment. Positive delay means
hand lags face. Reuse the inherited three-estimator diagnostics. Sensitivities:
forehead/cheeks, contracted palm, CHROM/POS, 0/5/10-second cuff guards and learned
face output. Ten-second windows, five-second exploratory stride; primary
repeatability uses disjoint ten-second windows and keeps every failed attempt.

Fixed existing pilot thresholds: >=90% support, SNR >=0 dB, periodicity >=.5,
correlation >=.5, coherence >=.8, face/hand HR difference <=5 bpm, and
split/method/estimator disagreement <=33.334 ms. Report individual gates and
all computable results separately. At least two acceptable independent
cuff-free windows per subject are needed to describe within-record repeatability.
Recorded cuff intervals plus a primary five-second guard define exclusions.
Only `Event_Time_Source=Video_review` with valid actual endpoints provides
verified cuff-free support; other source intervals remain candidates.

Sub-frame estimates can be computed, but sampling step is not a universal
physical resolution limit. Unknown exposure/rolling-shutter row skew, morphology
and processing phase leave acquisition uncertainty unbounded. Repeatability
does not identify physiology. No PTT/PAT, PWV or validated BP interpretation of
surface distances or apparent lag is allowed. Separate numerical stability,
repeatability and physical timing identifiability in all conclusions.

## Conditional BP experiment

Fit only explicit small regressions with exact recording/cuff identity and
valid numeric references. S001–S006 supply fitting/preprocessing; S007–S011
supply fixed final evaluation. No random-window split or replicating one cuff
label over many windows. One aggregate vector per cuff event or recording.

Primary event-level BP requires reviewed actual cuff intervals and a preceding
cuff-free feature interval within 30 seconds of inflation. One event appears
once. Without this verification, event-level BP accuracy is unavailable.

Secondary session-level feasibility predicts the mean valid cuff BP belonging
to that exact recording. Median nonoverlapping face features describe the
session; exclude recorded cuff intervals and guards where available. Unverified
intervals make this explicitly exploratory, not beatwise/timing-based BP.

Fixed models for SBP and DBP: Train mean; HR-only; HR plus rise fraction,
half-height width and normalized area from each waveform family; the same HR
plus geometry from joint or independent direct-feature systems. HR is face
GREEN-derived, not reference cuff HR. Use ridge alpha 10, Train-only centering,
scaling and median imputation. No hyperparameter search or BP-based seed
weights. Require >=4 development and >=3 evaluation subjects. Report missing
features/imputation. Timing increments require resolved timing and verified
temporal associations; otherwise they are unavailable.

Development leave-one-subject-out refits every transform in its fold. Fit on
the six and lock held-out predictions before scoring held-out pressures.
Report subject-balanced MAE/bias/RMSE against mean/HR controls and descriptive
subject-bootstrap uncertainty, with unique subjects/events. Narrow pressure
range or identity can give low error without physiological information.
No clinical pass threshold or private-data waveform fine-tuning follows.

## Follow-up

Preserve unfavorable outcomes. Decide separately about acquisition quality,
mathematical lag, repeatability, physical identifiability and exploratory BP.
Identify needs for reviewed cuff timelines, camera row/exposure controls,
repeat sessions, independent waveform/timing references or more BP participants.
Phones/cuff-only controls can diagnose artifacts; they need not establish a
biological millisecond delay. No new equipment, cloud spend or external contact
is part of this study.
