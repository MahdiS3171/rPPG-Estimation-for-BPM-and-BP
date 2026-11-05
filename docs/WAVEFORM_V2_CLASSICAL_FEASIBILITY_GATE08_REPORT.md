# Gate 08: classical synthetic-pulse feasibility

Date: 2026-11-05. Parent: Gate 07 / PR #4,
`4b3d01a3a8e7ef0354b9c5b5f73ad23df9f64675`.
Branch: `feature/waveform-v2-classical-feasibility-gate08`.

## Decision

**Qualified YES: the SCAMPS processing pipeline is functional enough to justify
designing a limited, separately reviewed Waveform v2 proof-of-concept training
experiment. Training is not enabled or performed in Gate 08.** All original
inputs reproduce, all fixed analysis cells are computable, and the unmodified
classical priors recover useful synthetic pulse information in some conditions.
Recovery is inconsistent across recordings, methods and ROIs; engineering
feasibility does not establish reliable waveform recovery or real-world rPPG
generalization. A matched frozen-V1 comparison is the recommended next experiment,
subject to separate review and authorization.

For example, primary Train forehead CHROM has recording-balanced native zero-lag
r=0.715 but HR MAE=22.31 bpm. Forehead GREEN has MAE=9.54 bpm and native r=-0.602.
The positive correlation and occasional close spectral HR estimates support
recoverable pulse information, while large errors, polarity differences and
timing failures prevent a stronger conclusion. No per-recording method/ROI winner
was selected. Human semantic annotation remains optional future validation under
the Gate 08 request; Gate 07's UNKNOWN skin states and trusted abstention contract
remain unchanged.

## Preregistration, data and label isolation

All four stacked PRs were reviewed and remain open, draft and unmerged. Their
heads remain PR #1 `243d41b729e028dad90182e14927a5c499dfd530`, PR #2
`2264fbe2aa2b3b0474674fe8f6193c5155a31dcb`, PR #3
`65ac812e69cc7d39e98b1643b7e25df997eef0f5`, and PR #4 at the parent above.
Gates 05-07, the six priors, preprocessing, existing fusion and timing contracts
were inspected before actual reference scoring.

The [fixed protocol](WAVEFORM_V2_CLASSICAL_FEASIBILITY_GATE08_PROTOCOL.md) was
committed at `a100379c238b15b5e0f2b5d55b56de4e0520cc78` before actual PPG scoring.
Implementation/tests were committed at
`8881fbe384ab21047f9b092814f19185f0df6ffc` before the initial actual run. The final
fresh run uses `839d8136f2e88b36e2d14459c07a31be9d35cb9c`, including the layout
and missing-control-descriptor corrections documented below. Every recording's
predictions and label-free descriptors were saved and hashed before the first
d_ppg value read. Scoring began with all eight Train recordings, followed by
separate Val/Test scoring. Hashes of all 480 predictions and all quality values
remain identical to the saved label-blind metadata after scoring.

Baseline A is the original Gate 06 V1-polygon path. Gate 08 reuses its exact
Gate 07 retained RGB/validity/indices, verifies all ten NPZ hashes against the
committed Gate 07 manifest, and rebuilds **all 20 x tensors with exactly the
Gate 06 hashes**. It is cache reuse, not a new landmark-extraction experiment.
No Candidate B array is read. The label-blind legacy target slots are NOT LOADED
NaNs and never enter prediction selection or scoring as reference data.

Only after the prediction lock, d_ppg is read in two unchanged 300-sample slices
per recording. All ten source SHA256 values match before/after scoring. No video,
segmentation, other physiological-label values or additional recording is decoded.
The official split snapshot hash remains
`d0b03f4a22f00c7dcfc106417e252c798274c7eb5fd022de0843b65951fea79e`.
Train is P000001/2/3/6/7/8/9/10; Val is P000004 and Test is P000005. All ten were
already exposed for engineering, so these are not pristine independent test data.

Original indices and the documented nominal index/30 synthetic grid are retained.
No physical capture timestamps, absolute camera-to-sensor accuracy or transit
times are inferred. The 599-mask/600-frame correspondence stays UNVERIFIED.

## Fixed methods, preprocessing and denominators

GREEN, CHROM, CHROM_WIN, PBV, POS_WIN and OMIT use the unchanged repository
implementations and V1 window-local standardized priors, with original masks.
PBV and OMIT are the repository's simplified/inspired baselines, not claimed
exact reproductions of every published variant. Raw RGB is not standardized
before prior construction. The original 0.9 support and bounded 0.15-second
interpolation rules, order-3 0.7-4 Hz Butterworth/filtfilt and 1.6-second windowed
method defaults are unchanged. Adaptation is disabled; no 30-second calibration
policy is squeezed into the 20-second clips.

The fusion is the existing equal mean of available standardized ROI priors,
without restandardizing the fused output. All actual windows have three available
ROIs, so weights are exactly 1/3 for each method/window. There is no reference-
or quality-based spatial selection. Under manufactured missingness the existing
helper averages available ROI priors, records its weights and retains failures.

| Split | Recordings | Unique 300-frame windows | ROI-windows | Single-ROI method cells | Fusion cells | Total cells |
|---|---:|---:|---:|---:|---:|---:|
| Train, primary | 8 | 16 | 48 | 288 | 96 | 384 |
| Val, descriptive only | 1 | 2 | 6 | 36 | 12 | 48 |
| Test, descriptive only | 1 | 2 | 6 | 36 | 12 | 48 |
| All | 10 | 20 | 60 | 360 | 120 | 480 |

All 480 actual cells are computationally successful, finite and non-flat, with
480/480 defined HR-error and waveform-correlation metrics. Every fixed Train
method/ROI condition has 16/16 windows and 8/8 recordings. This is numerical
availability, not 100% physiological or skin quality. Acquisition validity,
optical descriptors, pulse feasibility and confirmed skin labels remain separate.
No actual computational failure or missing window is silently discarded.

Reference HR is estimated from same-index raw d_ppg with the same unchanged
Welch estimator used for predictions: 30 Hz, 0.7-3.5 Hz, nperseg=300 here,
2048-point zero padding and existing peak interpolation. Ten-second support
has about 0.1 Hz / 6 bpm spectral resolution; padding does not create new
physical resolution. The derived reference HR ranges from 49.30 to 142.38 bpm
across all ten recordings; it is not a separate measured HR label.

Primary correlation uses the raw reference standardized within the same window,
without reference filtering, sign flips or lag shifts. Predictions retain their
internal filtering, so this prediction/reference filtering asymmetry is explicit.
A preregistered matched-band-reference zero-lag diagnostic is separate in the
metrics files and never replaces the primary result. Native signs are preserved;
OMIT's PCA polarity is not a guaranteed PPG polarity.

## Primary Train results

Each recording first contributes its two windows, then receives equal weight.
MAE and signed error are recording means; RMSE is the square root of the mean
recording MSE, not an average of recording RMSEs. Correlations are arithmetic
recording means. Conditions remain in their fixed order, without ranking.

| ROI | Method | MAE bpm | RMSE bpm | Signed error bpm | Native r0 | Lag-diagnostic r |
|---|---|---:|---:|---:|---:|---:|
| forehead | GREEN | 9.54 | 23.33 | -7.07 | -0.602 | 0.458 |
| forehead | CHROM | 22.31 | 46.25 | 20.51 | 0.715 | 0.715 |
| forehead | CHROM_WIN | 22.34 | 46.32 | 20.60 | 0.709 | 0.709 |
| forehead | PBV | 34.64 | 56.32 | 27.54 | -0.445 | 0.361 |
| forehead | POS_WIN | 19.22 | 34.98 | -15.66 | 0.365 | 0.447 |
| forehead | OMIT | 8.66 | 16.63 | 4.40 | -0.217 | 0.526 |
| left_cheek | GREEN | 29.35 | 42.66 | -18.67 | -0.406 | 0.375 |
| left_cheek | CHROM | 15.41 | 25.85 | 2.75 | 0.054 | 0.604 |
| left_cheek | CHROM_WIN | 15.39 | 25.88 | 2.75 | 0.052 | 0.605 |
| left_cheek | PBV | 11.48 | 22.02 | 4.89 | -0.110 | 0.528 |
| left_cheek | POS_WIN | 24.00 | 35.42 | -7.31 | 0.182 | 0.385 |
| left_cheek | OMIT | 31.59 | 44.13 | 10.13 | -0.182 | 0.517 |
| right_cheek | GREEN | 26.27 | 44.80 | 0.69 | -0.352 | 0.384 |
| right_cheek | CHROM | 19.56 | 37.47 | 0.94 | 0.532 | 0.681 |
| right_cheek | CHROM_WIN | 18.93 | 36.44 | 1.64 | 0.531 | 0.673 |
| right_cheek | PBV | 21.98 | 37.57 | 0.87 | -0.447 | 0.473 |
| right_cheek | POS_WIN | 32.51 | 39.38 | -23.92 | 0.078 | 0.312 |
| right_cheek | OMIT | 21.81 | 32.26 | 10.63 | 0.261 | 0.556 |
| equal_roi_average | GREEN | 19.57 | 43.47 | 2.46 | -0.612 | 0.442 |
| equal_roi_average | CHROM | 19.16 | 32.77 | -1.18 | 0.556 | 0.572 |
| equal_roi_average | CHROM_WIN | 20.87 | 33.82 | -3.69 | 0.553 | 0.567 |
| equal_roi_average | PBV | 24.36 | 38.68 | 1.25 | -0.477 | 0.337 |
| equal_roi_average | POS_WIN | 23.55 | 36.67 | -16.17 | 0.350 | 0.438 |
| equal_roi_average | OMIT | 25.73 | 40.19 | 5.59 | -0.075 | 0.519 |

Across these fixed conditions, Train MAE ranges from 8.66 to 34.64 bpm. The
table shows substantial ROI dependence and no uniformly reliable fusion benefit.
For example, forehead CHROM native r=0.715 versus left cheek 0.055; the fixed
fusion gives 0.555. Such comparisons do not authorize selecting the better ROI
after seeing each reference. Native zero-lag r is negative in **173/384 Train
cells across all six methods**. Negative correlation can coexist with accurate
spectral HR; forehead GREEN is one example.

The full [240 recording-condition summaries](waveform_v2/classical_gate08/per_recording_metrics.csv)
and [480 window-level rows](waveform_v2/classical_gate08/per_window_metrics.csv)
provide MAE/RMSE/bias, correlations, counts, quality and selected lags. The
[condition JSON](waveform_v2/classical_gate08/condition_metrics.json) retains
metric-specific finite recording/window denominators and every per-recording
summary. No mixture of retrospectively chosen method/ROI winners is reported.

An observed failure example is P000006 [0,300): raw-reference Welch HR is
49.30 bpm, but forehead CHROM estimates 196.49 bpm, close to its fourth harmonic,
despite native r=0.581. This supports investigating harmonic/peak selection in
a separately fixed experiment; the estimator/band is not changed here. Exposed
failure examples are descriptive, not new method-selection data.

## Separate Val/Test appendix

Each entry below has only two windows from one already exposed recording.
These results did not define indicators, parameters, fusion, examples or Train
aggregates. Full RMSE/bias/lag results remain in the linked per-recording files.

| ROI | Method | Val MAE bpm | Val native r0 | Test MAE bpm | Test native r0 |
|---|---|---:|---:|---:|---:|
| forehead | GREEN | 42.42 | -0.318 | 59.11 | -0.321 |
| forehead | CHROM | 0.21 | 0.903 | 68.84 | 0.109 |
| forehead | CHROM_WIN | 0.17 | 0.911 | 14.11 | 0.257 |
| forehead | PBV | 0.40 | -0.784 | 68.83 | -0.103 |
| forehead | POS_WIN | 0.44 | 0.589 | 35.69 | -0.288 |
| forehead | OMIT | 0.21 | -0.732 | 35.69 | 0.405 |
| left_cheek | GREEN | 0.21 | -0.648 | 0.50 | -0.879 |
| left_cheek | CHROM | 0.21 | -0.882 | 0.29 | 0.754 |
| left_cheek | CHROM_WIN | 0.19 | -0.881 | 0.28 | 0.741 |
| left_cheek | PBV | 0.47 | 0.759 | 0.38 | -0.676 |
| left_cheek | POS_WIN | 0.22 | 0.737 | 70.85 | -0.372 |
| left_cheek | OMIT | 0.11 | -0.901 | 0.41 | 0.878 |
| right_cheek | GREEN | 37.06 | -0.464 | 16.44 | -0.623 |
| right_cheek | CHROM | 0.53 | -0.686 | 0.16 | 0.824 |
| right_cheek | CHROM_WIN | 0.33 | -0.643 | 0.13 | 0.819 |
| right_cheek | PBV | 0.48 | 0.732 | 0.23 | -0.800 |
| right_cheek | POS_WIN | 0.31 | 0.829 | 70.36 | -0.284 |
| right_cheek | OMIT | 0.31 | 0.829 | 0.08 | 0.864 |
| equal_roi_average | GREEN | 0.42 | -0.778 | 0.28 | -0.822 |
| equal_roi_average | CHROM | 0.86 | -0.591 | 0.23 | 0.784 |
| equal_roi_average | CHROM_WIN | 0.68 | -0.547 | 0.20 | 0.804 |
| equal_roi_average | PBV | 1.00 | 0.550 | 0.24 | -0.758 |
| equal_roi_average | POS_WIN | 0.24 | 0.874 | 0.31 | -0.585 |
| equal_roi_average | OMIT | 0.58 | -0.636 | 0.04 | 0.872 |

P000005's documented forehead contamination accompanies poor results for several
fixed forehead methods, while some cheek estimates recover synthetic pulse
information. Neither reference performance nor pulse-band strength constitutes
hair classification, and these results do not change its UNKNOWN skin state.
No P000005-driven threshold or ROI rule is introduced.

## Label-free quality and negative controls

Descriptors are fixed before labels: own-peak SNR/peak-to-mean/periodicity,
Welch concentration within +/-0.15 Hz of the signal's own pulse-band peak,
five-second-half and adjacent-window HR differences, complete three-ROI HR range,
source clipping/missingness, interpolation and raw RGB changes. Amplitude of
standardized priors is largely fixed by preprocessing. These ROIs share imaging
artifacts and are not independent physiological sensors. Pulse-band filtering
itself promotes pulse-band concentration.

The [384 descriptive associations](waveform_v2/classical_gate08/quality_associations.json)
use Train recording means, at most eight finite recording pairs per condition.
Own-peak concentration versus HR absolute error has negative Spearman rho in
21/24 conditions and positive rho in three POS_WIN conditions. Its range is
-0.976 to +0.452. Those are small, exposed, dependent, multiple comparisons,
without p-value or generalization claims. Twenty-four clipping associations
(forehead and right cheek, both metrics) are undefined because those Train
clipping descriptors have no variation. No quality
cutoff, classifier or composite confidence score is fitted or validated.

Eight fixed controls retain **384/384 intended cells**. Computational outputs
exist for 354; 306 are pulse-estimate eligible. Forty-eight constant-input cells
are withheld as FLAT_SOURCE_RGB, and 30 cells retain NO_COMPUTABLE_PRIOR under
missing ROI/window inputs. Constant finite priors remain computational successes
in V1 but do not become usable pulse estimates. No frame is dropped to compress
time or rescue a failed window.

The first predeclared method (GREEN) with the fixed fusion illustrates the
controls below; all six methods and four spatial conditions are retained in
the [complete control CSV](waveform_v2/classical_gate08/negative_control_metrics.csv)
and [control summaries](waveform_v2/classical_gate08/negative_control_summary.json).

| Fixed control | Eligible / 2 | Concentration | HR MAE bpm | Native r0 |
|---|---:|---:|---:|---:|
| sinusoidal_positive | 2 | 0.989 | 0.001 | 1.000 |
| constant_nonpulsatile | 0 | undefined | undefined | undefined |
| slow_illumination_drift | 2 | 0.890 | 29.813 | 0.000 |
| independent_noise | 2 | 0.293 | 42.927 | 0.185 |
| coherent_pulse_band_illumination | 2 | 0.989 | 0.001 | 1.000 |
| clipped_impulses | 2 | 0.131 | 2.569 | 0.172 |
| missing_windows | 1 | 0.989 | 0.001 | 1.000 |
| time_permuted | 2 | 0.440 | 69.677 | -0.063 |

Coherent pulse-band illumination has concentration 0.989 and r=0.9997 against
the fixture sine despite having no physiological source. Slow illumination drift
also produces concentration 0.890 after filtering while r is approximately zero.
Noise and time permutation produce finite but generally inaccurate estimates.
These controls demonstrate why spectral strength, ROI agreement or high fixture
correlation cannot identify physiological origin or confirmed skin/hair. The
synthetic sine is known fixture truth, not a measured pulse in negative controls.

## Lag, filtering and timing limitations

Lag-aligned r is a **reference-optimized diagnostic only**, maximizing signed r
over integer lags +/-15 samples (+/-500 ms). Each CSV row discloses its lag,
overlap count and boundary flag. Positive lag means prediction lags reference;
ties prefer zero, smaller absolute lag, then negative. Predictions, HR inputs,
quality descriptors and plotted axes are never shifted. Nineteen Train cells
reach the search boundary; no Val/Test cell does. Native negative polarity can
produce an apparently improved aligned r by selecting a half-cycle offset.
This is not a physiological delay or timing-preservation result.

Seven [Gate 03 timing controls](waveform_v2/classical_gate08/synthetic_timing.json)
give **2 PASS_SYNTHETIC_ONLY, 4 ABSTAIN_UNVERIFIED and 1 FAIL**. GREEN/OMIT pass
the constructed window-local transfer; CHROM/CHROM_WIN/POS_WIN abstain for native
polarity ambiguity and PBV for low correlation. No sign correction is applied.
The existing equal-ROI fusion with synthetic ROI delays [0,120,120] ms shifts the
source anchor by about **84.85 ms**, failing camera-time anchoring despite fixed
1/3 weights. It does not measure real spatial physiological delays.

Concatenating separately processed windows differs from full-record processing;
the maximum standardized boundary difference reaches 1.553 in this fixture.
This includes filter edges, overlap-add and window statistics, not only filter
phase. filtfilt is nominally zero phase for its assumed signal extension, not
proof of boundary-free morphology preservation. Spatial averaging may cancel
different signs/phases. Synthetic passes establish no absolute physical timing.

## Plots and reproducibility

Local root: `outputs/waveform_v2_classical_feasibility_gate08_20261105_02/`.
Use **plots/**: `waveforms_P000001.png`, `waveforms_P000003.png`,
`train_hr_pairs.png`, `train_native_correlations.png`, `train_quality_errors.png`
and `negative_controls.png`. Waveform examples are exactly the preregistered
first Train windows, with all methods/ROIs/fusion and unshifted references.
The HR/quality plots retain all Train observations, without score-based selection.

Assistant visual QA identified a title/legend overlap in the original HR/quality
figures. A layout-only commit moved legends below the plots. Final review also
corrected synthetic-control clipping descriptors to remain unknown when RGB is
missing. No method, threshold, actual-data prediction or actual quality metric
changed. After regression tests, a fresh complete run was written to `_02`,
preserving `_01`. Its actual predictions, reference arrays, metrics, quality and
timing outputs are byte-identical to `_01`; exactly 24 missing-ROI control
clipping fields changed from zero to unknown. All six final PNG hashes match
the visually reviewed corrected figures. Corrections and provenance are recorded
in [validation_summary.json](waveform_v2/classical_gate08/validation_summary.json).
No historical source data, output directory or report was overwritten.

The final numerical run took 56.29 seconds: 18.05 s blind preparation, 4.27 s
reference scoring, 18.95 s controls, with the remainder including plotting/writing.
Peak Windows working set was 302.95 MiB; tracemalloc peak was 53.26 MiB and
excludes import/native allocations. No fresh
landmark extraction, waveform network or GPU benchmark is included in these costs.

Validation command:

```powershell
.\.venv\Scripts\python.exe -u -m scripts.verify_waveform_v2_classical_safe
```

**175 tests passed, zero failures/errors/runtime skips/protected-I/O attempts,
process exit code 0**: 149 retained checks and 26 new Gate 08 tests. Tests actively
deny waveform module calls, checkpoint loading and backward calls. They cover
reference isolation, exact priors/fusion, raw/reference preprocessing, flat and
missing windows, frequency estimation, sign/lag conventions, corruption controls
and balanced RMSE. Six new Python files compile; CLI help and existing/source-child
output refusal pass. Post-run verification checks all 480 prediction/reference
hashes, every quality value against blind metadata, and all 72 recording-balanced
condition summaries independently from CSV errors.

For a fresh run with the existing admitted cache:

```powershell
.\.venv\Scripts\python.exe -u -m scripts.audit_waveform_v2_classical_feasibility --scamps-root SCAMPSData --baseline-root outputs/waveform_v2_optical_quality_gate07_20261105_01 --out-dir outputs/waveform_v2_classical_feasibility_gate08_repeat
```

The [evidence index and exact file inventory](waveform_v2/classical_gate08/README.md)
links all metadata, source/cache/code hashes, versions and artifact sizes/hashes.
PNG/NPZ payloads stay outside Git. Source files, V1 architecture/checkpoint/losses,
official splits, training, prior functions and previous reports/contracts are
unchanged. Private Dataset, actual S007-S011 and collected-subject data were not
accessed. No neural inference/training, BP regression, real-participant evaluation,
adaptation or new model/data download occurred. All allow_training flags remain false.
SCAMPS [license](waveform_v2/scamps/LICENSE_SCAMPS.txt) and
[attribution](waveform_v2/scamps/ATTRIBUTION.md) are retained and copied to the local
packet; R-UDA non-commercial research terms continue to apply.

## Proposed next experiment and readiness

Separately review a **frozen Waveform v1 versus all six fixed classical priors**
comparison on the same x tensors, source windows, targets, ROI/fusion masks and
native signed/zero-lag metrics. Pin the checkpoint/config, retain failures,
report Train separately, keep Val/Test exposed and descriptive, and perform no
reference-driven sign/lag repair, model fine-tuning, ROI choice or adaptation.
Include constant/noise/illumination controls and boundary/timing diagnostics.
Do not execute this neural experiment until separately authorized.

In parallel, design a tightly bounded Train-only synthetic proof of concept,
with a fixed learning objective, frozen preprocessing, explicit polarity policy,
quality/missingness policy, compute budget and stopping rules. Sixteen windows
from eight recordings can support gradient/memorization/plumbing checks, not a
generalization claim. A credible efficacy experiment needs additional permitted,
properly grouped synthetic data and independently defined evaluation. Reserve
real-world claims for separately authorized public real-data validation; private
reserved subjects remain untouched. Optical/human annotation can complement this
work later and is not an immediate prerequisite imposed by Gate 08.

| Readiness | Decision |
|---|---|
| Engineering/classical synthetic feasibility | GO: exact inputs, complete denominators, reference isolation and some pulse recovery demonstrated. |
| Designing a limited separately reviewed Waveform v2 training proof of concept | QUALIFIED GO; quality, polarity, harmonic and boundary limitations must be explicit. |
| Neural training execution or validated waveform-training readiness | Not authorized in this gate; training remains disabled. |
| Real-domain rPPG generalization or physiological timing | Not established; these ten exposed synthetic examples cannot support those claims. |
