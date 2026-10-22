# Frozen Waveform v1 visual validation (Part 4.5A)

This is a reproducible **validation-only** inspection tool for the already frozen
`MultiROIPriorResidualWaveformNet`, selected epoch 25. It does not train, alter
weights, change preprocessing or splits, predict BP, or invoke the test runner.
The official locked test has already been consumed and must not be rerun.

## Run

From the repository root, using the environment that created the Phase 1 cache:

```powershell
.\.venv\Scripts\python.exe scripts/visualize_waveform_v1.py --ubfc-root UBFCData --checkpoint checkpoints/waveform_v1_official_ubfc/waveform_v1_frozen.pt --split-manifest checkpoints/waveform_v1_official_ubfc/waveform_v1_split.json --cache-dir cache_roi_multi_phase1 --all-validation-subjects --include-random --seed 42 --out-dir outputs/waveform_visual_v1
```

The checkpoint, split, cache and output arguments above are defaults. Only
`--ubfc-root` is required. Without a subject argument, all saved validation
subjects get best/median/worst figures. `--include-random` adds a seeded example.
All output directories must be new, preserving earlier diagnostic runs.

Limit the figures to one subject, one selection, or one existing window:

```powershell
python scripts/visualize_waveform_v1.py --ubfc-root UBFCData --subject subject19 --out-dir outputs/waveform_visual_subject19
python scripts/visualize_waveform_v1.py --ubfc-root UBFCData --subject subject19 --selection worst --baseline GREEN --out-dir outputs/waveform_visual_subject19_worst
python scripts/visualize_waveform_v1.py --ubfc-root UBFCData --subject subject19 --start-sec 54 --out-dir outputs/waveform_visual_subject19_54s
```

`--selection` accepts `best`, `median`, `worst`, or `random`. A specific start
requires `--subject`, cannot be combined with `--selection`, and must match a
retained dataset window within 1 microsecond. It never snaps to a different
window or constructs a new reference-dependent input. `--subject` and
`--all-validation-subjects` are mutually exclusive. The optional `--baseline`
names a prior from the checkpoint, adding its validity-aware multi-ROI average
alongside CHROM_WIN. Baseline names and order come from the actual checkpoint.

**Every invocation ranks all existing validation windows**, even when only one
figure is requested. Thus the complete report/feature universe stays explicit.
There is no `--split test`, hash override, model candidate, sampling-rate override
or training option.

## Identity and validation isolation

The tool requires this exact official checkpoint file SHA256:

```text
d6b2d1eeebc0153197a2a12da736b957b32d72ae00b22631e9afd0b2097131ff
```

It verifies `frozen_v1`, `FREEZE_WAVEFORM_V1`, selected epoch 25, split-file and
canonical split hashes, all partition IDs, and the adjacent
`waveform_v1_freeze_manifest.json`. An explicit `--freeze-manifest` must match
that mandatory adjacent copy byte for byte. `FrozenWaveformExtractor` performs
its existing configuration, source-candidate, provenance, ordering and strict
model reconstruction checks. A candidate checkpoint or changed identity fails
before data access. The checkpoint hash is checked again before issuing results.

Only exact `validation_ids` from the saved, frozen-identity-checked split are
permitted. The official IDs are subject19, subject28, subject29, subject31,
subject32, subject35 and subject41. Train/unknown IDs are rejected; test IDs get
an explicit consumed-locked-test rejection before dataset construction.

The visualizer constructs only those direct UBFC child paths; it does not call
cohort discovery, enumerate test folders, construct test datasets, open test
references, or read the saved locked-test evaluation. Redirected subject,
video/reference paths are refused. Cache preflight computes the existing Phase 1
identity for each validation video, validates the existing raw extraction arrays
with the Phase 1 validator, and records cache hashes. Missing, mismatched or
corrupt caches fail with the subject/path diagnostic instead of requesting video
extraction. Use the original cache environment; dependency/source/model-path
changes can invalidate a Phase 1 cache key. There is no synthetic fallback.

`UBFCMultiROIRPPGDataset` remains unmodified. It retains its exact duplicate
contact timestamp averaging, reference interpolation on the video grid, target
standardization, target HR fallback, window exclusions, ROI support rules and
window-local prior construction. The established `forward_batch` receives only
`x`, `roi_quality`, `roi_valid` and `prior_valid`, using the model reconstructed by
`FrozenWaveformExtractor`. Item reference values are used for assessment after
prediction; they never determine input amplitudes, ROI quality, attention,
prior allocations or output polarity. Dataset reference eligibility/common
support screening remains the existing evaluation convention.

## Selection and metrics

Each retained validation window has exactly this diagnostic score:

```text
0.5 * wave_corr_aligned + 0.5 * d1_corr_same_lag
```

`waveform_window_metrics` computes all primary metrics unchanged, with the
checkpoint's maximum lag and spectral band. Ranking is decreasing score, then
increasing `start_sec`. Undefined scores remain in the ranking with null rank
and are not silently selected as a valid worst example. Median means the upper
middle valid ranked window (`n // 2` zero-based). Seeded random sampling uses
NumPy PCG64 with `SeedSequence([seed, subject_SHA256_first_four_bytes_le])`; it
prefers a window distinct from best/median/worst when available and is invariant
to requesting one subject or the whole cohort.

Best/median/worst figures are prominently labeled **REFERENCE-BASED DIAGNOSTIC
SELECTION**. This selection is for inspection only. It cannot select a new
checkpoint, change the model, or justify tuning against this consumed test.

Evaluation selects the lag maximizing **signed** waveform Pearson correlation;
it does not maximize absolute correlation or the joint training objective.
Positive lag means prediction lags reference: trim prediction at the beginning
and reference at the end. Ties visit zero, then smaller absolute lags, then
negative before positive. `aligned_overlap` performs all slices with no wrap,
interpolation, DTW, learned phase correction or beat-specific shifts. Derivative
correlation uses forward `np.diff` at exactly that waveform-selected lag.
Subject aggregation reuses `aggregate_waveform_metrics`, with equal subject
weighting and finite counts rather than treating all windows as independent.

## The six panels in every example PNG

1. **Unaligned face rPPG vs finger PPG (original timestamps).** Both signals keep
   the original window grid and native sign, independently z-scored. No samples
   are shifted. This separates temporal agreement from shape after alignment;
   it establishes no physiological PAT/PTT.
2. **Aligned waveform comparison.** Finger reference, frozen prediction and
   CHROM_WIN multi-ROI average, optionally another classical prior average.
   `classical_window_predictions` supplies the baselines directly from the exact
   input tensors and masks, without reimplementing rPPG equations. Each method
   keeps its own existing full-window evaluation lag, then all displayed methods
   use the intersection of their reference sample indices. Native signs remain
   unchanged. The title prints lag in ms, original-window correlation, full-window
   HR MAE, and full-evaluation versus common-plot correlation/nRMSE. Common-plot
   metrics call the same metric helper at zero *additional* lag on the already
   sliced signals, never searching for another lag. An unavailable baseline is
   labeled NA and does not silently gain an easier subset. The panel states:
   **Aligned for waveform-shape comparison only; not an inter-site timing
   measurement.**
3. **First derivatives.** The exact model evaluation overlap and selected lag
   are reused. Waveforms are independently z-scored on that overlap for display,
   then forward-differenced and multiplied by fs (z/s). The unchanged reported
   derivative correlation equals that of the native differences. There is no
   separate derivative lag search. Native differences remain available through
   the reusable helper, with original waveforms in the NPZ audit artifact.
4. **Reference-anchored beat shapes.** Candidate extrema use the existing
   `pulse_wave_features` detector's 0.7–4 Hz filter, independent standardization,
   minimum peak distance and prominence. Conservative screening requires at
   least three unambiguous complete beats, physiological coarse duration/IBI
   bounds, and peak-rate/Welch agreement. These are engineering reliability
   guards, not validated fiducials. The paired prediction uses exactly the same
   reference sample boundaries at the global model lag. Individual beats are
   faint; medians and IQR are drawn on **real elapsed seconds**, with no duration
   normalization, per-beat amplitude fit or interpolation. Right-padding is
   missing, and summaries stop where fewer than three beats contribute. Failure
   displays **insufficient reliable beats**.
5. **Normalized Welch PSD.** Original full windows, actual fs, Hz axis, 0.5–8 Hz
   display (limited by Nyquist). Welch uses the existing HR estimator's Hann
   window/default overlap, at most 30-second segment and fourfold zero-padding
   convention. Density integrates to one over 0–Nyquist. Marked HR frequencies
   come from `estimate_hr_welch` in its existing 0.7–3.5 Hz band. Inspect lost or
   strong spectral structure without claiming anatomical harmonic equivalence.
6. **Model allocations.** All ROI attention, ROI source-valid fractions, masks,
   every prior's weight/validity in actual checkpoint order, and mean/maximum
   absolute scaled residuals. These are descriptive allocations, not causal or
   physiological explanations. The uncalibrated quality head is omitted.

## Coarse cross-site features and observed consistency

Only HR, IBI, W50, rise time, fall time, normalized rise slope, IBI SD and IBI CV
are reported. Existing `pulse_wave_features` supplies coarse shape values with
its historical 0.7–4 Hz filtering; independent full-window Welch supplies HR.
Normalized slope is filtered, window-standardized amplitude per second, not a
calibrated blood-volume slope. Shape/IBI features have explicit reliability
gates. Ambiguous, short, flat or nonfinite signals produce missing values and
reasons, never substitute zeros. A true zero IBI SD may remain valid.

Differences are called **cross-site feature agreement diagnostics**. Scatter plots
include only paired reliable estimates; tables retain all windows, missing values,
reasons and predicted/reference/paired valid fractions. Per-subject summaries
report median/MAD differences and coverage. The y=x line is an anatomical
comparison aid, not a clinical target. No alignment or target amplitude fit
changes the full-window feature measurements.

Primary observed-consistency summaries select earliest chronological,
**non-overlapping**, half-open windows before filtering by feature validity.
Overlapping 10-second/2-second-stride windows do not count as independent repeats.
For every subject, feature and method, tables give eligible/valid counts, valid
fraction, median, unscaled MAD and sample CV where there are at least two valid
estimates and a positive non-near-zero mean. Trajectories show all window values
with gaps and mark the primary repeats. Model/classical/reference comparisons
use identical temporal eligibility. There is no reference-stability threshold.
Physiology may change during the recording; observed consistency is descriptive,
not a requirement that every feature be constant, and not BP accuracy.

## Artifacts

```text
outputs/waveform_visual_v1/
    summary.md
    summary.json
    window_metrics.csv             # every window; rank, full metrics, masks, counts, allocations
    window_rankings.json            # complete rank order, including invalid windows
    feature_agreement.csv           # every method/feature/window, validity and signed differences
    feature_subject_summary.csv
    repeatability.csv
    figures/subjectXX_best.png      # also median/worst/optional random or specific
    figures/subjectXX_best.npz      # exact grid, raw contact, target, inputs/masks, outputs/allocations
    overview/subject_metric_distributions.png
    overview/classical_vs_frozen_scatter.png
    overview/morphology_feature_scatter.png
    overview/stability_summary.png
    overview/subjectXX_feature_trajectories.png
```

Summary JSON records checkpoint/freeze/split hashes, canonical split hash, git
commit and dirty/source-hash provenance, UTC creation time, fs, exact ordering,
alignment rules, seed/ranking/selection, all validation IDs, input paths/cache
hashes/reference hashes, exclusions and actual sample/beat counts. PNGs are
required and generated with headless Matplotlib Agg; PDF export is not included.
CSV/JSON metric content and example selection are deterministic. Creation time
and provenance describing different invocations naturally differ.

## Recorded real validation inspection

On October 18, 2026 the local official checkpoint and matching Phase 1 caches
were available. All **210 validation windows** (30 per subject) were ranked;
**28 real six-panel PNGs** cover best/median/worst plus distinct seeded random
examples for all seven subjects. The four cohort overviews and seven subject
trajectory PNGs are also saved. The locked test was not opened or rerun.

The existing aggregate metrics reproduce aligned r **0.817292**, same-lag
derivative r **0.762757**, morphology **0.790024**, and HR MAE **1.095420 bpm**.
The frozen score exceeds CHROM_WIN in **206/210** paired windows and is lower
in four: subject19 at 16, 34 and 36 s, and subject29 at 24 s. Alignment reaches
the configured +/-0.5-second boundary in **21/210**
windows; these lags remain comparison diagnostics, with no timing interpretation.

Paired model/reference coverage is **210/210** for HR, **204/210** for IBI and
IBI consistency, and **193/210** for W50/rise/fall/normalized slope. CHROM_WIN
paired shape coverage is **185/210**. Median model-minus-finger differences are
W50 **+0.031373 s**, rise **+0.013725 s**, fall **-0.002564 s**, and normalized
rise slope **-1.048960 z/s**. These are cross-site descriptions, not facial
ground-truth errors. Every subject contributes six primary non-overlapping
windows; model W50 sample CV ranges approximately **0.048–0.209**. The tables
retain method-specific valid counts instead of forcing identical feature coverage.

Actual weak examples include subject29 at 24 s, subject35 at 58 s, and subject41
at 16 s, with morphology scores approximately 0.325, 0.350 and 0.257. The raw
and aligned panels reveal substantial differences that a dominant HR frequency
can conceal. Inspected predictions can be smoother than the finger waveform,
with attenuated sharp rises and less higher-frequency structure. Site anatomy,
sampling and filtering also differ; this does not establish that every missing
finger detail is an algorithmic error. Some reference beat summaries correctly
remain unavailable, notably several selected subject41 examples. Inspect the
derivative, pulse and PSD panels together rather than inferring detailed shape
validity from a single correlation.

## What these plots establish, and what they do not

They establish reproducible visual/numerical comparisons and observed feature
consistency for a fixed facial extractor on the preexisting validation split.
They do not establish clinical BP validity, physiological PAT/PTT, learned
face–hand timing, or site-matched facial morphology ground truth. ECG and PPG
are not interchangeable morphology targets. UBFC uses finger contact PPG;
its anatomy and morphology need not be identical to the face. Detailed
notch/reflected-peak/RI/APG morphology remains unvalidated and is not measured.
Future model improvements belong to a new version and require separate
evaluation evidence; this tool cannot alter the frozen checkpoint.

## Verification

```powershell
python -m unittest discover -s tests -q
python -m compileall -q rppg_lab scripts tests
git diff --check
```

All **173 tests passed** (138 existing plus 35 new), compileall passed, and
`git diff --check` passed. A separate synthetic six-panel PNG was generated
under `outputs/waveform_visual_v1_synthetic/`, explicitly labeled as a fixture.
The real report remains under `outputs/waveform_visual_v1/`.

The final real run used an independent Python file-open audit, saved as
`outputs/waveform_visual_v1/access_audit.json`: exactly seven allowlisted
validation references and seven matching Phase 1 caches were opened, no raw
videos were opened, and no official artifact writes or non-validation data
open attempts occurred. This audit complements the explicit-path source and
tests; native OS metadata calls are outside Python's open audit. An actual
CLI request for a locked-test ID was also rejected before data-root access.
Repeated real runs produced byte-identical ranking, window-metric, feature and
repeatability tables. All selected lags match the saved validation evaluation;
single-window versus original batched floating-point metric differences are
below 1e-6. The official checkpoint hash remains unchanged.

`tests/test_waveform_visualization.py` uses synthetic fixtures only, covering
identity/status and split rejection, metric/lag reuse, native polarity and raw
timestamps, derivative/beat constraints, PSD axes, missing features, actual
checkpoint order, repeatability, deterministic tables/reports, validation-only
forward inputs and successful headless PNG generation. Synthetic figure fixtures
are explicitly labeled as synthetic/unverified and do not constitute real model
results. Existing training/test integration fixtures remain synthetic as well.
