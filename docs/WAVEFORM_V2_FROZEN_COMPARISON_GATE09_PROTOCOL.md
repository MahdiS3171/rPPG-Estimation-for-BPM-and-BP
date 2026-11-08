# Gate 09 fixed frozen-V1 comparison protocol

Date: 2026-11-08. Base: current Gate 08 head
`bcd3f98dd65088d41c447a2651e145d70cfc8e05`.
The remote heads and PRs #1–#5 (descriptions, ancestry, local changes and empty
discussion timelines) were reviewed. All remain draft, open and unmerged.
This specification is committed before any Gate 09 neural forward call, including
synthetic tests. Implementation and tests will also be committed before actual
SCAMPS inference. No training, calibration, adaptation, GPU work or BP estimation
is authorized. All pre-existing tracked files and historical results stay intact.

## Identity and exact matching

Only `checkpoints/waveform_v1_official_ubfc/waveform_v1_frozen.pt` is permitted:
SHA256 `d6b2d1eeebc0153197a2a12da736b957b32d72ae00b22631e9afd0b2097131ff`.
Its adjacent freeze manifest must have SHA256
`da5ba397b3be389deec8821f8ea16944b5f5b93e2834bb79c3c147eb3d1f89dc`.
Missing or mismatched identities stop the experiment; no substitute is allowed.
Reuse `FrozenWaveformExtractor` construction and its existing strict provenance,
configuration and state-dict reconstruction. Explicitly verify the three ROIs
(forehead, left_cheek, right_cheek), six priors (GREEN, CHROM, CHROM_WIN, PBV,
POS_WIN, OMIT), RGB_R/G/B then prior channel order, 30 Hz, 300 samples,
`MultiROIPriorResidualWaveformNet`, eval mode, CPU and unchanged model state.
No subject-context wrapper. Set one CPU thread and deterministic algorithms;
every forward executes under `torch.inference_mode()` with no backward/optimizer.

Use the verified Gate 08 `locked_predictions.npz` tensors directly. Restore
their original quality/validity/prior masks from Gate 06's committed per-window
metadata; do not synthesize mask values or rebuild actual input channels.
Verify every tensor against Gate 06's input SHA256, and archive hashes against
the committed Gate 08 artifact manifest and prediction lock. Fixed classical
signals come directly from that archive; never regenerate or replace them.
Use Gate 08 `reference_windows.npz`, checked against its committed hash and all
480 raw-reference hashes. No raw PPG reread is needed. Verify the ten SCAMPS
source files by read-only SHA256 before/after only; no video/mask decoding.
Read no private Dataset/, collected subjects or S007–S011, including inventory.

Exactly one V1 output for each original index window [0,300), [300,600) in
P000001–P000010; retain indices and nominal index/30 coordinates without
inventing capture timestamps. The intended denominator is always 20 windows
from ten recordings. Train: 16 windows/eight recordings (1,2,3,6,7,8,9,10).
Val (4) and Test (5): two windows/one recording each, exposed and descriptive.
These are correlated windows, not 20 independent participants. All ten are
engineering-exposed. Never claim statistically significant generalization.

Before reading the cached references, save and hash all actual V1 waveforms,
allocations, statuses, indices and input/mask fingerprints. Reference values
cannot reach inference, gating, polarity, weights or prediction selection.
Do not repeat actual inference to pick outputs; determinism is checked on
manufactured fixtures. Save invalid outputs/failure rows in the denominator.

## Primary presentation and metrics

Present a fixed-order full 25-row condition table **separately for each split**:
V1 followed by all six priors for forehead, left_cheek, right_cheek and existing
equal_roi_average, in Gate 08 order. No reference-selected comparator or winner.
For every classical condition report paired V1-minus-classical recording-balanced
differences in HR MAE, MSE-derived RMSE, signed HR error, signed zero-lag r and
diagnostic aligned r, plus finite matched windows/recordings. Paired RMSE
difference is sqrt(mean paired V1 recording MSE) minus the corresponding
classical quantity. Provide all per-window and per-recording rows, failure
reasons, metric-specific finite counts and intended counts, even on failure.
No significance tests, CIs treating windows as independent, or pooled primary
Train/Val/Test estimate. Negative MAE difference and positive native-r difference
favor V1 descriptively; no data-selected success threshold.

Reuse Gate 08 `score_waveform` unchanged for V1 and fixed baselines. Primary HR:
same Welch estimator, nominal 30 Hz, 0.7–3.5 Hz, nperseg=300, existing 2048-point
padding/interpolation. Signed error is prediction minus raw d_ppg-derived HR.
Primary waveform: signed Pearson r at zero lag on identical 300 indices against
per-window standardized raw d_ppg, with native model/prior polarity preserved.
Do not invert, filter again, shift, align, or optimize predictions against labels.
Reference filtering asymmetry remains explicit. Predeclared secondary r uses
the unchanged 0.7–4 Hz reference filter at zero lag only.

Separately report maximum signed r over integer +/-15 samples, selected lag
(positive: prediction follows reference), overlap, boundary flag, and aligned
minus native r. Use existing zero/smallest-absolute/negative tie ordering.
This is reference-optimized diagnostic correlation, never a corrected waveform
or camera-time validation. Save and plot outputs on original axes.

Numerical prediction usability requires finite, nonflat (>1e-10 std), complete
300 samples, valid input/masks and at least one usable ROI. Record shape,
nonfinite, flat, no-ROI and forward failures separately. Constant source RGB is
ineligible as a pulse estimate even if the neural residual is periodic.
Nonfinite/flat references invalidate only paired metrics, preserving other HR
and intended counts. Computational output is not physiological validation.

Harmonic indicators are descriptive hypotheses, not adjudicated true-fundamental
labels: mark HR within 6 bpm of 2x or 0.5x the fixed reference spectral HR,
provided absolute HR error exceeds 6 bpm. Mark >6 bpm errors without either
relation as other peak discrepancies. Keep reference and prediction band-edge
flags (within 6 bpm of 42/210). Record reference power near half/selected/double
frequencies (+/-0.1 Hz using Welch nperseg=300, nfft=2048) to expose ambiguity.
No alternate-peak selection or HR repair. Ten-second support limits resolution.

ROI attention [3], prior weights [3,6], their product and scaled residual RMS
are descriptive allocations only; no validated attribution or skin classifier.
Report unweighted and effective weighted prior mixtures and native component
metrics after the lock to investigate attenuation/amplification of poor signals.
Any allocation/error associations are descriptive; do not infer causality.

## Negative controls and timing, declared before inference

Reuse all eight Gate 08 deterministic 600-sample RGB fixtures and their unchanged
input preparer, masks and seeds, two model windows each (16 intended rows):
positive sinusoid, constant, slow drift, independent noise, coherent pulse-band
illumination, clipping impulses, missing ROIs/all-ROI window, temporal permutation.
Positive: report native pulse recovery without assuming trained polarity.
Constant: withhold pulse eligibility regardless of neural output. All missing:
no forward and explicit failure, while partial missing must receive zero weights.
Noise/permutation/drift: inspect false periodicity; finite output is not success.
Illumination: even near-perfect fixture correlation is confounding, never proof
of physiological origin. Clipping: compare with clean positive without a tuned
robustness cutoff. For every case report expectations and observed behavior,
including failures, masks, weights and unvalidated-output warning fields.

Synthetic timing: unchanged `synthetic_pulse` at 30 Hz for 600 samples, identical
RGB mixing [160,120,100]+pulse*[.8,1.2,.5], original window processing.
Use injected inter-stream delay 45 ms. Evaluate (1) baseline, (2) common 100 ms
input delay, (3) additional 85 ms second-stream delay, (4) heterogeneous ROI
delays [0,120,120] ms. Report common/differential transfer relative to both
nominal undelayed sources and actual injected first-ROI sources. A heterogeneous
mixture has no unique anatomical phase. Evaluate each 300-sample output separately
with existing timing-transfer audits; do not claim concatenation is continuous.
Keep all PASS/FAIL/ABSTAIN results and >=33.33 ms sampling uncertainty floors.
Include analytical identity/common-delay/differential-delay controls to verify
that common shifts can preserve relative delay while failing absolute anchoring.
V1's lag-tolerant training objective does not establish camera-time preservation.
No SCAMPS lag is face–hand PTT; no BP inference.

## Verification, artifacts and decision

Tests cover pinned checkpoint identity before load, strict loading, masks and
tensor/index integrity, matched references, reference-free prediction API,
no gradients and inference mode, deterministic synthetic inference, native sign,
harmonic flags, failure denominators/balanced RMSE, analytical timing limitations,
state/source immutability, and fresh outputs confined below outputs/ outside
sources and prior artifacts (including symlink-resolved destinations).
Run the existing reviewed non-neural regression profile and a new guarded
inference-only profile; deny backward/optimizer/CUDA and protected subject I/O.
Record exact commands, counts, source/code/checkpoint/artifact SHA256, environment,
forward wall times, total runtime, tracemalloc peak and process peak working set.
Memory measurements are process-level bounds, not isolated model tensor use.
Plots: native first-window P000001/P000003 examples against every fixed baseline,
all ten V1 recordings' native waveforms and HR/lag/harmonic diagnostics, all controls.
Arrays/PNG stay in ignored local outputs; commit small tables/reports/manifests
and exact added-file inventory. Preserve SCAMPS license/attribution.

Compare conclusions with saved historical UBFC and rPPG-10 reports only, never
rerun their data. Their populations, reference modalities, windows and estimands
differ; metrics are not interchangeable. rPPG-10 ECG HR is not optical morphology.
Conclude GO for designing a separately restricted training proof of concept only
if matching/integrity succeed and recoverable signal supports a bounded learning
investigation; NO-GO pending preprocessing/reference issues if those undermine
interpretation; otherwise INCONCLUSIVE with explicit remaining evidence. No
neural advantage, optical semantic validity, generalization or timing is assumed.
Push this new branch and open a stacked draft PR targeting Gate 08; never merge.
