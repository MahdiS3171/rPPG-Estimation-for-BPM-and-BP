# Gate 08 fixed classical-feasibility protocol

Date: 2026-11-05. Parent: Gate 07 / PR #4 at
`4b3d01a3a8e7ef0354b9c5b5f73ad23df9f64675`.
Commit this protocol before decoding actual reference PPG for scoring. All four
stacked PRs, Gates 05-07, classical/window priors, preprocessing, equal-ROI fusion
and Gate 03 timing contracts have been reviewed. No actual PPG scoring has begun.

## Scope and fixed denominator

Use only the ten previously admitted P000001-P000010 examples. Preserve the
pinned official CSV: Train P000001/2/3/6/7/8/9/10; Val P000004; Test P000005.
Train is the primary exploratory analysis. Val/Test receive separate descriptive
appendices after predictions/quality are locked, with no selection or tuning.
All ten are engineering-exposed; none is a pristine final test. No participant
identity independence is inferred. Human annotation is optional future work,
not an immediate prerequisite to this label-based feasibility investigation.

Use the original Gate 06 V1-polygon Baseline A only. Reuse the exact retained A
RGB/validity/indices from Gate 07's local NPZ packet, validating its SHA256 against
the committed Gate 07 artifact manifest and source hashes against Gate 06.
Rebuild all 20 x tensors with unchanged `prepare_scamps_v1_inputs` and require
exact Gate 06 input hashes before opening any reference values. This verifies
cache provenance and preprocessing; it is not a new landmark extraction run.
Do not read Candidate B outputs, use segmentation masks, or modify geometry.

Nominal grid: original index/30 seconds, 600 samples, no physical capture clock.
Fixed nonoverlapping windows [0,300), [300,600), with all original source indices.
Use unchanged 0.9 acquisition support / bounded 0.15 s interpolation, raw RGB*255,
current-window standardization, classical method defaults and prior definitions.
No subject adaptation: the 30-second calibration contract cannot fit 20 seconds.

Methods in fixed order: GREEN, CHROM, CHROM_WIN, PBV, POS_WIN, OMIT. Predictions
are the unchanged window-local standardized priors in x, not priors recomputed
from standardized RGB. Report forehead, left_cheek and right_cheek separately.
The single existing fusion is `classical_window_predictions`' equal mean of
available standardized ROI priors per method/window. Record its actual masks and
weights; do not choose ROIs using reference or quality scores. With all three
available, weights are exactly 1/3. Missing ROIs follow that existing helper;
all missing yields no prediction. Constant finite priors are computational
successes in V1 but are not automatically usable pulse estimates.

Intended counts: 20 unique windows, 60 ROI-windows, 360 single-ROI method cells
and 120 fusion cells = 480 cells. Primary Train: 16 windows, 48 ROI-windows,
288 single-ROI +96 fusion =384 cells. Val and Test each have two windows and
48 cells. Each method/ROI condition has 16 Train cells, two Val and two Test.
Keep every intended cell on failure, with metric-specific finite denominators.

## Label isolation and reference processing

First generate all ten recordings' predictions, hashes, availability and label-free
quality descriptors with no reference argument or HDF5 value reads. Save the
prediction archive/metadata before label access. Commit implementation/tests
before actual scoring as an additional safeguard. Only then read d_ppg in two
300-sample slices per admitted source using read-only HDF5. Do not decode video,
masks, other labels, or infer a new split. Reference values cannot feed back into
predictions, ROI selection, polarity, fusion weights or quality descriptors.

Primary reference is unchanged same-index d_ppg standardized per current window,
as in V1. It is not filtered to improve correlation. Predictions retain their
existing order-3 0.7-4 Hz Butterworth/filtfilt, with 1.6 s overlap-add defaults for
CHROM_WIN/POS_WIN. Report that prediction/reference filtering asymmetry explicitly.
One predeclared secondary diagnostic uses the unchanged 0.7-4 Hz filter on the
reference window for zero-lag correlation only; keep it separate from primary
metrics. No filter tuning or extra prediction filtering.

## Metrics, polarity and lag diagnostics

Use existing `estimate_hr_welch` defaults for prediction and raw reference:
30 Hz, 0.7-3.5 Hz (42-210 bpm), window_sec=30 (nperseg=300 here), zero_pad_factor=4.
The 2048-point padded grid and parabolic peak interpolation do not create physical
frequency resolution; each window is only ten seconds long. Estimate reference
HR from d_ppg, not an externally measured HR label. Signed error = prediction HR
minus reference HR. Report MAE, RMSE and mean signed error, each recording's
two windows first, then equal recording weights. Balanced RMSE is the square
root of the mean recording MSE, not the mean of recording RMSEs. Include finite
recording/window counts, empty recordings and all failure reasons.

Primary waveform metric: signed Pearson correlation at zero lag on identical
300 original indices. Preserve native prior polarity. No absolute-correlation
replacement, reference-based sign flip or per-record polarity choice. GREEN
follows intensity; CHROM/PBV/POS projections have their implemented signs and
OMIT uses PCA whose sign is algorithmic, not a guaranteed PPG polarity.

Separately report maximum signed Pearson correlation over integer lags +/-15
samples (+/-0.5 s) using existing `lag_candidates` / `aligned_overlap`: positive
means prediction lags reference; ties prefer zero, smaller absolute lag, then
negative. Disclose selected lag, overlap length and search-boundary optima. This
is explicitly a reference-optimized diagnostic, never an applied correction:
do not shift saved predictions, HR inputs, quality inputs, or plotted time axes.
Do not interpret periodicity-related/polarity-compensating lags as pulse transit
time, camera-to-sensor accuracy or timing preservation.

Fully finite predictions/reference with standard deviation >1e-10 are required
for waveform/HR scoring. Also flag a raw RGB ROI/window with maximum channel std
<=1e-10 as flat input, regardless of numerical prior artifacts. Fusion is flat
input only when every contributing ROI is flat. Preserve computational validity
separately from pulse-estimate eligibility. Invalid reference/prediction affects
its metrics, not others' denominator; never compress time by dropping samples.

## Fixed label-free quality investigation

No new quality admission threshold, composite confidence score or method/ROI
selection is introduced. Before labels, record:

- Existing `signal_quality` HR, own-peak SNR, peak-to-mean, periodicity and amplitude;
- Welch concentration within +/-0.15 Hz of the trace's own dominant pulse-band
  peak divided by power in 0.7-3.5 Hz (nperseg=min(length,240), no extra padding);
- absolute HR difference between the two five-second halves, and between the
  two ten-second windows; missing halves/windows stay missing;
- per-method inter-ROI HR range only when all three estimates exist (shared
  imaging makes these ROIs correlated, not independent physiological sensors);
- source ROI missingness, imputation, clipping fraction and adjacent raw RGB
  change. Fusion optical descriptors use the same contributing ROI weights.

Evaluate descriptors against already-fixed HR absolute error and native zero-lag
correlation using per-condition recording-mean Train values and descriptive
Spearman coefficients with finite-pair counts. No p-value/generalization claim,
no cutoff optimization or validation/test use for indicator construction.
Standardized prior amplitude is largely fixed by construction; filtering itself
concentrates power into the pulse band. High periodicity, concentration or
cross-ROI agreement does not prove physiological origin or classify hair/skin.
Acquisition, optical indicators, physiological feasibility and confirmed skin
labels remain separate. Skin labels stay UNKNOWN; semantic abstention rules from
Gate 07 are not weakened or promoted into labels by this study.

## Controlled corruption and timing

Use deterministic 600-sample, three-ROI RGB fixtures, processed with identical
methods/windows/fusion: (1) 1.2 Hz sinusoidal positive control, (2) constant RGB,
(3) nonpulsatile 0.15 Hz illumination drift, (4) independent Gaussian channel/ROI
noise with seed 808, (5) coherent 1.2 Hz illumination modulation, (6) positive
control with clipped RGB impulses at indices 90,210,390,510, (7) missing forehead
in the first window and all ROIs in the second, (8) independently time-permuted
positive-control RGB with seed 809. Use baseline RGB [160,120,100], pulse channel
amplitudes [0.8,1.2,0.5], noise std=2, illumination amplitude=2. The synthetic
comparison reference is sin(2*pi*1.2*t); it is a fixture, not measured physiology.
Report every control cell, not only successful examples. The coherent illumination
case intentionally exposes the limitation of pulse-band/ROI-agreement proxies.

Use Gate 03 timing-transfer audits on six methods' concatenated window-local
outputs for synthetic face/hand signals with an injected 45 ms source delay.
Compare unchanged full-record filtering versus concatenated window filtering
as a boundary diagnostic; do not change the actual extraction. Audit existing
equal-ROI fusion with synthetic ROI delays [0,120,120] ms at 30 Hz, showing that
fixed averaging can change phase even without reference/quality-based selection.
Keep passing/failing/abstaining controls, native polarity and source sample
uncertainty; synthetic passes do not establish physiological timing.

Tests must cover reference isolation, missing windows/full denominators, exact
prior/fusion reuse and ordering, raw/reference preprocessing, lag sign/ties,
native polarity, flat/nonfinite inputs, frequency estimation and balanced RMSE.
No neural forward, checkpoint load, backward, training, BP or participant access.

## Plots, outputs and decisions

Predetermined waveform examples: first window of Train P000001 and P000003,
all six methods, all fixed ROIs and fusion, with unshifted reference/time axes.
Plot all Train HR pairs, primary zero-lag correlations, quality/error associations
and all negative-control conditions. Do not select attractive waveforms afterward.
Keep diagnostic images/arrays in a fresh ignored outputs directory; Git receives
only code, tests, protocol/report and small metrics/provenance metadata. Retain
SCAMPS license/attribution and source/cache/code hashes. No source overwrite.

The scientific decision concerns whether a **limited separately reviewed training
experiment is worth designing**, not permission to train. Assess integrity,
coverage, pulse recovery versus fixed corruption controls, native waveform and
timing limitations together. Do not invent or tune a pass cutoff on these ten
examples, declare a deployment winner, or retrospectively choose each recording's
method/ROI. Report all fixed conditions in their original order. A positive
engineering decision can coexist with weak native waveform/polarity evidence;
state limitations and required controls rather than claiming validated training
or real-world generalization. Propose a matched comparison against frozen V1
as a separately authorized next experiment; do not run it now.
