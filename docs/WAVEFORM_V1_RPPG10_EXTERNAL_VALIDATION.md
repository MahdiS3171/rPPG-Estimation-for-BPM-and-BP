# Part 4.5B: frozen Waveform v1 on rPPG-10

This is an external **HR-only** evaluation of the unchanged UBFC-trained
`MultiROIPriorResidualWaveformNet`. It introduces no training, checkpoint
selection, tuning, BP estimation or learned timing. The UBFC locked test has
already been consumed. An official attempt here consumes rPPG-10 as independent
evidence for this frozen version; later tuning needs other independent evidence.

## Sources and scientific boundary

- [Mendeley release V1](https://data.mendeley.com/datasets/bx8982xgwt/1),
  DOI `10.17632/bx8982xgwt.1`.
- [Authors' repository](https://github.com/GRodrigues4/rPPG-10).
- [Data paper, DOI 10.1016/j.dib.2025.111888](https://pmc.ncbi.nlm.nih.gov/articles/PMC12311941/).
  Equipment, acquisition, processing and limitations sections establish
  1000-Hz BITalino ECG, manual trigger trimming and up to 500-ms reaction delay.

ECG measures electrical events; it is not optical PPG. Only ECG R–R HR is a
reference here. There is no ECG waveform Pearson, derivative, W50, RI, notch,
morphology, polarity/lag optimization or physiological transit-time evaluation.
Optical diagnostic panels show electrical event time markers on a separate axis.
They establish no BP, PAT/PTT, face-hand timing or waveform morphology validity.

The cohort contains 26 usable young healthy students under natural daylight.
Low-resolution crops, tracking failures and changing illumination differ from
UBFC landmark-masked full-face video. Generalization to other populations,
clinical settings or sparse skin strata is not established by this dataset.

## Verified local layout and identity

Inspect current `main`, not an old archive. The implementation started from
`ac11a473d8c3952b92774af67de7a5b652fbf861`, equal to `origin/main` on inspection.
Pre-existing uncommitted Part 4.5A visualizer/README work is preserved.

The actual downloaded layout was inspected before parsing:

```text
rPPG-10/
  Subject Data.xlsx
  Subject_1/
    Subject_1_Forehead_.avi
    Subject_1_Cheek1_.avi
    Subject_1_Cheek2_.avi
    Subject_1_ECG.npy
  ...
  Subject_27/
```

Expected usable IDs are exactly Subject_1, Subject_2, Subject_3, Subject_5,
Subject_6, Subject_7, Subject_8, Subject_9, Subject_10, Subject_11, Subject_12,
Subject_13, Subject_14, Subject_15, Subject_16, Subject_17, Subject_18,
Subject_19, Subject_20, Subject_21, Subject_22, Subject_23, Subject_24,
Subject_25, Subject_26, Subject_27. The local download contains all 26.
Subject_4 exists locally with zero-byte AVI files and is always excluded by the
official dataset rule, irrespective of whether its ECG is readable.

An optional underscore before `.avi` is accepted. Other names are logged as
unrelated; duplicate matches are rejected. Missing clips never borrow another
participant's data. Files are hashed. Missing subjects, zero sizes, malformed
clips, short recordings, failed tails, count/FPS discrepancies and uncertain
clocks are explicit QC. Full-run scheduling includes the published 600 seconds
and any longer supported recording, so short/missing recordings cannot obtain
artificially high coverage by silently shrinking the denominator.

## Precropped adapter and frozen contract

The Mendeley description and paper explicitly identify **Cheek1 = anatomical
right cheek; Cheek2 = anatomical left cheek**. Frozen order is:

| Frozen ROI | Release filename suffix |
|---|---|
| forehead | Forehead |
| left_cheek | Cheek2 |
| right_cheek | Cheek1 |

Decode the three AVI crops separately, once per run (or use content/version keyed
RGB-only cache). Convert OpenCV BGR to RGB and average pixels within each crop.
Do not run MediaPipe or combine ROIs before model inference. Frame validity and
mean-brightness diagnostics are acquisition descriptors; they cannot identify
all occlusion, tracking or defocus failures.

AVI `POS_MSEC` can be synthesized by the decoder. It is logged but never claimed
as verified acquisition PTS. Use each clip's nominal FPS, reject divergent rates,
and assume common starts based on the authors' shared-source crop description.
That assumption cannot be independently verified from the supplied AVI files.
Bound the 30-Hz grid by the longest interval supported by all three clips.
Resample with existing bounded `resample_trace`; no unsupported tails or long
gaps are extrapolated. Conservative validity projects source support to the
common grid. Nominal clocks do not establish millisecond absolute timing.

The official checkpoint SHA256 is:

```text
d6b2d1eeebc0153197a2a12da736b957b32d72ae00b22631e9afd0b2097131ff
```

Verify these bytes before reconstruction, then require the adjacent
`waveform_v1_freeze_manifest.json`. Existing `FrozenWaveformExtractor` validates
frozen status, architecture, provenance, manifest, extraction configuration,
ROI/prior/channel ordering and reconstructs unchanged weights. The external
adapter further requires 30 Hz and 300 samples / 10 seconds. Priors are GREEN,
CHROM, CHROM_WIN, PBV, POS_WIN, OMIT; channels are RGB_R, RGB_G, RGB_B followed by
those priors. Existing `prepare_window` supplies the exact standardization,
support masks, 0.9 minimum valid fraction, quality construction and individual
window-local priors. `infer_window` is the only neural inference path.
There is no second neural implementation or ECG-dependent input/attention.

## ECG reference and locked validity rules

Sampling rate is parameterized with a source-verified **1000-Hz** default. It is
never inferred from video duration. The inspected arrays are `(600000,)`
floating point, with ADC-like stored values; physical amplitude conversion is
not established by the array itself. The paper describes the sensor range;
this evaluator keeps stored units and makes no mV conversion. Multichannel or
ambiguous shapes are rejected rather than flattened or arbitrarily selected.

The small ECG-only detector filters finite runs using third-order 5–20-Hz SOS,
finds absolute QRS peaks with a 300-ms refractory period, and requires height
at least five times the median absolute filtered amplitude measured in 8-second
blocks. It excludes 250-ms finite-run filter boundaries. No optical predictions
are used for detector design, selection or QC.

For `[start,end)`, HR is `60 / median(RR)` using intervals whose **both R peaks**
are inside that window. At least five complete intervals are required. Reject
nonfinite/flat ECG, more than 5% samples at ADC rails 0/1023, RR outside
0.30–1.50 seconds, RR CV above 0.20, any RR deviation above 30% of the median,
uncovered edges above 1.5 median RR, or QRS energy concentration below 0.5
within ±80 ms of detected events. Irregular windows are unknown labels, not
arrhythmia diagnoses. Automatic detector false/missed beats remain possible.

Official manual trimming establishes a nominal shared start, with up to 500-ms
operator delay. Every label is marked with this uncertainty. ECG-only reference
checks at nominal time and ±0.5 seconds must all pass and their HR range must be
at most 3 bpm. The nominal label is retained; no offset is selected. Missing
boundary support invalidates a label, which typically excludes first/last
windows. These checks measure sensitivity, not proof of exact synchronization.
ECG/video duration mismatch is logged; unsupported labels remain unknown.

ECG-only detector audits show deterministic windows at 10, 295 and 580 seconds,
separately from optical panels. Several real subjects must be visually inspected
in validation-only mode before an official inference attempt. Detector QC may
identify software/reference problems before outcomes are inspected; it must
never be optimized to improve neural HR scores.

## Metrics and plots

Default windows are 10 seconds, stride 5 seconds, seed 42. Estimate predicted HR
with existing `signals.estimate_hr_welch(...,30)`, preceded by a finite/nonconstant
guard so the legacy estimator cannot silently replace a failed signal by zeros.
Keep all 24 `classical_window_predictions` (18 ROI/prior signals and six
validity-aware equal-ROI averages) on the same tensors/windows. Headline baseline
is **CHROM_WIN multi-ROI average**, with GREEN and all other priors disclosed.
Spectral SNR, self-peak periodicity, harmonic/subharmonic power indicators,
residual magnitude, support and allocation weights are reference independent.
The uncalibrated model quality head is omitted. Spectral concentration heuristics
are also not calibrated confidence.

Per-method/window/subject outputs disclose MAE, RMSE, predicted-minus-reference
bias, median and 90th-percentile absolute error, within-3/5-bpm fractions,
finite prediction coverage and failure rates. Accuracy and within-threshold
denominators use valid pairs only; coverage includes all scheduled windows.
Extraction failure, invalid optical signal and unavailable/invalid ECG labels
have distinct counts/reasons. Missing label counts are independent of optical
failure counts and may overlap. Unsupported discarded video tails remain QC.

Window-weighted metrics pool pairs. Subject-balanced metrics mean the finite
per-subject statistic, including per-subject RMSE/median/P90 rather than taking
the corresponding statistic of subject-average errors. Unscorable subjects are
listed; conditional accuracy does not hide their lack of coverage. All-subject
coverage remains in its denominator. Percentile 95% CIs use 2000 whole-subject
bootstrap draws (PCG64, seed 42) for both weighting conventions. Overlapping
windows are autocorrelated; they are not independent participants.

Minute analyses assign a window by its center time. Spreadsheet subjects are
joined by exact normalized IDs, whether stored in columns or rows. Only skin
and environmental descriptors are retained locally; personal demographic,
date/time and health fields are not copied. Skin groups and prespecified
illumination midpoint bins (<800, 800–<1100, ≥1100 lux) are descriptive, with
counts and bootstrap intervals only when at least three scorable subjects
support them. Temperature/humidity summaries are also descriptive. Two lux
readings cannot describe intermediate lighting changes. No strong fairness
inference is supported by sparse strata.

Headless deterministic PNGs include each subject's HR, window/rolling absolute
error and support/SNR, ROI/prior weights, ECG-only QC and median/worst/seeded
random optical windows. Diagnostic selection is explicitly based on ECG HR
error; no attractive-only examples are selected. Missing values remain gaps.
Cohort figures include subject MAE/coverage, matched model/classical error
scatter, descriptive HR Bland–Altman-style display and supported metadata
groups. The HR agreement display labels autocorrelation and non-clinical/no-BP
interpretation; its SD lines are not clinical limits-of-agreement validation.

## Commands and artifacts

```powershell
python scripts/evaluate_waveform_v1_rppg10.py --dataset-root rPPG-10 --out-dir outputs/rppg10_qc --cache-dir cache_rppg10_rgb --validate-only
python scripts/evaluate_waveform_v1_rppg10.py --dataset-root rPPG-10 --checkpoint checkpoints/waveform_v1_official_ubfc/waveform_v1_frozen.pt --out-dir outputs/waveform_v1_rppg10_external --cache-dir cache_rppg10_rgb --seed 42
```

`--max-subjects N` or `--smoke` forces **SMOKE** and cannot become official.
There are no frame limits. `--validate-only` never calls the model or consumes
the external evaluation. Official runs require the verified fs/ROI mapping and
at least 2000 bootstrap draws. A different ECG rate/mapping is debug-only.
Install `openpyxl>=3.1` for optional spreadsheet inspection.

Output directories are exclusive. Before official inference, immutable settings
and input identities are saved and a persistent exclusive consumption ledger
is created at `outputs/rppg10_external_consumed_<official checkpoint hash>.json`.
This prevents an official rerun under a different output directory. Do not
delete or bypass this ledger to repeat evaluation. A failed attempt remains
recorded; completion status is separate from the immutable receipt. Preserved
outputs can be reanalyzed for reporting errors without new inference, with any
correction explicitly disclosed. Successful software completion is not a
performance acceptance decision; no data-dependent acceptance threshold exists.

Artifacts:

- `dataset_inventory.json`: exact expected/discovered/missing/excluded IDs,
  file sizes, content hashes and structural QC.
- `config_and_provenance.json`: source revision/dirty state, source and model
  hashes, libraries, manifest, dataset DOI, mapping, timing and immutable rules.
- `ecg_qc.json`: decoder/frame/clock/duration, detector and label QC per subject.
- `external_window_metrics.csv`, `external_subject_metrics.csv`.
- `external_summary.json`, `external_summary.md`: both aggregates, CIs,
  matched comparisons, temporal/group results, selection and limitations.
- `external_evaluation_receipt.json`: exclusive attempt, immutable config,
  split/data/model identity and UTC timestamp.
- `completion_verification.json`: completion, unchanged model hash and artifact
  checksums; it does not modify the receipt.
- `executed_source_snapshot/`: an exact copy of all 57 source files named in
  the official run's provenance, with a matching hash manifest. This preserves
  the executed uncommitted implementation for audit.
- `figures/`, `overview/`: correctly labeled real-data PNGs after an actual run.

Dataset files, raw metadata, caches, checkpoints and outputs remain local and
gitignored. No dataset or participant-identifying spreadsheet fields are added
to source control. If data are unavailable, only software validation can be
reported; no external performance may be invented.

## Verification

```powershell
python -m pip install pytest
python -m pytest -q
python -m compileall -q rppg_lab scripts tests
git diff --check
```

`tests/test_rppg10_external.py` uses synthetic clips, known-rate ECG and an
untrained in-memory fixture through the existing frozen inference API. It tests
discovery/mapping, BGR conversion, independent crops, tails/FPS/counts, exact
local-prior inputs/masks, ECG quality/timing, reference independence, frozen
identity/order guards, matched windows, missingness, balancing/bootstrap,
immutable receipts, smoke isolation and deterministic headless figures. It
does not train, select or modify a checkpoint.

## Recorded structure and ECG audit

The complete validation-only run is
`outputs/waveform_v1_rppg10_validation_only_20261018/`. It performed no frozen
model calls. All 26 expected usable IDs were present; there were no loading
errors or missing ECGs. All ROI clips were 64×64 at nominal 30 fps, with matching
counts across the three crops of each subject. There were no reported/decoded
count shortfalls. Subjects 13 and 21 contained 17,996 and 17,983 frames per crop
(599.8667 and 599.4333 seconds); all other usable subjects contained 18,000.
Their last 10-second scheduled windows remain unsupported, giving 3,092
supported video windows out of 3,094 scheduled windows. No tails were fabricated.

All usable ECG arrays were finite `(600000,)` float64 at documented 1000 Hz.
The locked ECG rules accepted 2,830/3,094 labels (91.4674%); 143 windows failed
manual-timing/boundary checks and 121 failed RR-quality rules. These are label
availability counts, independent of neural performance. Visual ECG-only audits
covered Subjects 1, 5, 10, 21, 23 and 27 at 10, 295 and 580 seconds. Clean QRS
events were sensible; visible artifacts, especially late in Subject_21, remain
a limitation of automatic detection. There were no manual beat edits or
data-dependent threshold adjustments.

`qc_review_before_external_inference.json` records this review before any real
frozen-model inference. `outputs/waveform_v1_rppg10_protocol_lock_20261018.json`
records the configuration, source/test hashes and complete software checks
before real outcomes were reviewed. The official runner's configuration and
source hashes were verified to match that lock exactly.

During the locked run, a synthetic edge-case review added a wrapper that turns
empty/malformed ECG NumPy-file read errors into actionable QC errors. This
changes only unreadable-file handling; all 26 real ECG files had already passed
the complete read/shape audit. No inference, preprocessing, detector threshold,
valid label or metric rule changed. The running official process retained its
original loaded code, preserved in `executed_source_snapshot/`; it was not
restarted or rerun.

## Implementation files

| File | Responsibility |
|---|---|
| `rppg_lab/rppg10.py` | Exact inventory, separate crop decoding, frozen extraction adapter, ECG reference and metadata join |
| `rppg_lab/rppg10_external.py` | Official identity checks, locked protocol, exclusive receipt, evaluation orchestration |
| `rppg_lab/rppg10_reporting.py` | HR statistics, subject bootstrap, matched/time/group reports and headless figures |
| `scripts/evaluate_waveform_v1_rppg10.py` | CLI entry point |
| `tests/test_rppg10_external.py` | Synthetic loader, scientific-boundary, metric and end-to-end regression tests |
| `docs/WAVEFORM_V1_RPPG10_EXTERNAL_VALIDATION.md` | Protocol, sources, limitations and run record |

README links the protocol and command. `.gitignore` protects the optional data
and RGB-cache locations. `pyproject.toml` and `requirements.txt` add `openpyxl`
for the actual column-oriented spreadsheet. Existing frozen inference, model,
priors, training, UBFC visualization and BP implementation files were not edited.

## Real external result

**REAL EXTERNAL ? COMPLETE.** One unlimited full-cohort attempt, begun on 2026-10-18 and completed on 2026-10-19 at 06:27:39 UTC (09:57:39 Asia/Tehran); final artifact review on 2026-10-22. The output directory is `outputs/waveform_v1_rppg10_external_20261018/`. There was no second official inference attempt. The independent-evidence consumption receipt and persistent ledger remain in place.

Frozen v1 showed limited HR generalization on this cohort. Its subject-balanced MAE was lower than the prespecified CHROM_WIN headline baseline, but substantially higher than GREEN. Its within-5-bpm rate was also lower than CHROM_WIN. This is an unsuccessful transfer of the strong UBFC HR performance to these precropped natural-light recordings; software completion is not a scientific performance acceptance. No model, input/prior computation, attention, loss, hyperparameter, detector threshold or metric definition was changed in response to the results.

All 26 participants were scorable. Settings were 30 Hz, 300-sample/10-second windows, 5-second stride, seed 42 and 2000 whole-subject bootstrap draws; no subject/frame limits. The same 2830 valid pairs support the model and all six multi-ROI prior averages. CIs are participant percentile intervals, retaining correlated windows within each sampled participant.

### Subject-balanced results

Each accuracy statistic below is the mean of the corresponding subject statistic. CIs shown are for MAE; all metric CIs are retained in the JSON.

| Method | MAE bpm [95% CI] | RMSE bpm | Bias bpm | Median AE bpm | P90 AE bpm | Within 3 bpm | Within 5 bpm |
|---|---:|---:|---:|---:|---:|---:|---:|
| Waveform v1 | 15.780 [14.018, 17.615] | 20.923 | 7.542 | 12.624 | 35.306 | 24.10% | 32.28% |
| CHROM_WIN multi-ROI average | 17.858 [15.324, 20.393] | 25.858 | -0.923 | 13.136 | 42.652 | 30.90% | 37.51% |
| GREEN multi-ROI average | 7.168 [4.808, 9.886] | 11.489 | -3.442 | 4.129 | 17.601 | 66.21% | 74.54% |
| CHROM multi-ROI average | 17.701 [15.062, 20.370] | 24.916 | -3.436 | 14.125 | 40.154 | 29.71% | 36.62% |
| POS_WIN multi-ROI average | 16.200 [13.641, 19.020] | 21.529 | -6.580 | 13.398 | 32.904 | 27.71% | 36.46% |
| PBV multi-ROI average | 16.468 [14.051, 19.003] | 23.227 | -4.282 | 12.796 | 35.475 | 31.41% | 38.70% |
| OMIT multi-ROI average | 19.974 [18.147, 21.903] | 26.035 | -4.360 | 17.813 | 39.163 | 15.69% | 21.73% |

### Window-weighted results

| Method | MAE bpm [95% CI] | RMSE bpm | Bias bpm | Median AE bpm | P90 AE bpm | Within 3 bpm | Within 5 bpm |
|---|---:|---:|---:|---:|---:|---:|---:|
| Waveform v1 | 15.861 [14.096, 17.678] | 21.694 | 7.751 | 11.904 | 37.224 | 24.06% | 32.30% |
| CHROM_WIN multi-ROI average | 17.857 [15.284, 20.421] | 26.727 | -0.719 | 11.229 | 43.630 | 31.13% | 37.74% |
| GREEN multi-ROI average | 7.003 [4.733, 9.692] | 14.312 | -3.341 | 1.659 | 25.170 | 66.86% | 75.16% |
| CHROM multi-ROI average | 17.694 [14.994, 20.404] | 25.911 | -3.240 | 12.116 | 41.327 | 29.93% | 36.86% |
| POS_WIN multi-ROI average | 16.105 [13.548, 18.931] | 22.530 | -6.419 | 12.008 | 37.051 | 27.88% | 36.71% |
| PBV multi-ROI average | 16.404 [14.042, 18.890] | 24.085 | -4.074 | 10.966 | 39.284 | 31.70% | 39.01% |
| OMIT multi-ROI average | 19.913 [18.070, 21.851] | 26.365 | -4.165 | 16.495 | 41.076 | 15.83% | 21.91% |

On exactly matched windows, the subject-balanced model-minus-baseline MAE differences were -2.079 bpm versus CHROM_WIN and +8.611 bpm versus GREEN. These are descriptive paired differences, not a window-independent significance test.

### Subject-level results

All participants have 119 scheduled windows. Subjects 13 and 21 each have 118 finite predictions (99.160% coverage); all others have 119 (100%). The complete CSV also includes every ROI/prior method, RMSE, bias, percentiles, thresholds, SNR and dropout counts.

| Subject | Valid HR pairs | V1 MAE bpm | V1 RMSE bpm | V1 bias bpm | CHROM_WIN MAE bpm | GREEN MAE bpm |
|---|---:|---:|---:|---:|---:|---:|
| Subject_1 | 114 | 24.838 | 32.366 | 24.205 | 5.250 | 1.201 |
| Subject_2 | 114 | 12.403 | 16.004 | -2.513 | 23.515 | 8.295 |
| Subject_3 | 98 | 13.320 | 18.375 | 4.103 | 11.518 | 2.462 |
| Subject_5 | 114 | 14.296 | 18.709 | 5.548 | 20.374 | 7.016 |
| Subject_6 | 87 | 11.459 | 15.143 | -5.583 | 21.436 | 13.089 |
| Subject_7 | 115 | 18.924 | 26.125 | 16.675 | 13.110 | 1.065 |
| Subject_8 | 113 | 9.985 | 14.032 | 1.362 | 18.889 | 2.628 |
| Subject_9 | 97 | 13.608 | 17.686 | -6.272 | 26.011 | 22.919 |
| Subject_10 | 116 | 15.815 | 21.823 | 13.309 | 19.341 | 2.832 |
| Subject_11 | 117 | 12.279 | 16.121 | 2.763 | 20.381 | 2.832 |
| Subject_12 | 117 | 13.539 | 20.312 | -1.755 | 24.063 | 14.056 |
| Subject_13 | 117 | 13.621 | 20.699 | 10.700 | 6.581 | 2.707 |
| Subject_14 | 109 | 11.746 | 15.123 | 2.463 | 16.395 | 8.086 |
| Subject_15 | 109 | 17.398 | 24.101 | 15.400 | 16.341 | 1.258 |
| Subject_16 | 116 | 12.188 | 18.563 | 9.782 | 8.152 | 2.412 |
| Subject_17 | 111 | 25.243 | 30.747 | 24.184 | 15.721 | 1.748 |
| Subject_18 | 104 | 14.459 | 17.477 | 3.826 | 21.328 | 17.527 |
| Subject_19 | 82 | 10.206 | 14.312 | 6.410 | 10.395 | 3.140 |
| Subject_20 | 117 | 19.699 | 24.645 | 16.143 | 22.748 | 4.188 |
| Subject_21 | 108 | 15.658 | 20.396 | 8.215 | 17.844 | 12.997 |
| Subject_22 | 108 | 17.413 | 22.878 | -13.544 | 32.403 | 24.419 |
| Subject_23 | 116 | 12.616 | 17.207 | 0.398 | 23.785 | 6.690 |
| Subject_24 | 104 | 16.843 | 21.053 | 7.182 | 24.573 | 14.093 |
| Subject_25 | 112 | 25.611 | 32.224 | 25.058 | 6.689 | 1.243 |
| Subject_26 | 101 | 22.172 | 28.033 | 20.096 | 14.820 | 3.446 |
| Subject_27 | 114 | 14.928 | 19.835 | 7.927 | 22.650 | 4.023 |

### Coverage, time and descriptive diagnostics

There were 3094 scheduled windows, 3092 finite model predictions (99.9354%), two unsupported-tail extraction failures (0.0646%), and no additional nonfinite/constant model signal failures. ECG QC withheld 264 labels (8.5326%): 143 timing/boundary and 121 irregular/missed-beat cases. The valid-pair coverage was 2830/3094 (91.4674%). Missing labels and optical failures are distinct counts and can overlap; both unsupported tails already lacked a valid ECG label. High finite prediction coverage does not mean good HR accuracy.

The subject-balanced model within-3/5-bpm rates were 24.10% and 32.28%; corresponding window-weighted rates were 24.06% and 32.30%, using the 2830-pair denominator. Model mean self-peak spectral SNR was -2.239 dB and mean self-periodicity 0.199. Mean scaled residual magnitude was 0.135 and maximum absolute scaled residual 0.957 in native model units. Mean ROI allocations were forehead 0.3406, left cheek 0.3303 and right cheek 0.3292. These describe the model; they do not calibrate confidence or explain physiology.

| Recording minute (window center) | V1 subject-balanced MAE bpm | V1 window MAE bpm | Valid pairs |
|---|---:|---:|---:|
| 0 | 17.029 | 17.282 | 245 |
| 1 | 14.723 | 14.817 | 298 |
| 2 | 15.852 | 15.867 | 294 |
| 3 | 15.047 | 14.871 | 296 |
| 4 | 15.530 | 15.457 | 297 |
| 5 | 16.841 | 16.839 | 292 |
| 6 | 18.064 | 18.190 | 284 |
| 7 | 16.569 | 16.759 | 274 |
| 8 | 14.164 | 14.299 | 291 |
| 9 | 14.297 | 14.449 | 259 |

Model error remained substantial throughout the recording: minute-level subject-balanced MAE ranged from 14.164 to 18.064 bpm. There is no evidence of uniformly reliable long-duration HR performance. These time segments cannot establish that illumination caused the errors; illumination was measured only at the two endpoints.

Skin strata contained I: 2, II: 17, III: 4 and IV: 3 subjects. Their model mean-subject MAEs were 12.341, 16.100, 13.977 and 18.659 bpm. Type I has no bootstrap interval because n<3. Illumination midpoint groups <800, 800?<1100 and ?1100 lux contained 13, 5 and 8 participants, with model MAEs 16.380, 14.653 and 15.508 bpm. Counts and descriptive CIs appear in the JSON and figures; sparse/imbalanced strata support no strong fairness conclusion.

### Report, plots and verification

Within `outputs/waveform_v1_rppg10_external_20261018/`, the most useful entry points are:

- `external_summary.md` and `external_summary.json`: all 25 methods and both weighting conventions, bootstrap intervals, matched comparisons, temporal results and subgroup descriptions.
- `external_subject_metrics.csv`: 650 method/subject rows; `external_window_metrics.csv`: 77350 method/window rows.
- `overview/subject_mae_coverage.png`, `overview/matched_hr_error_scatter.png`, `overview/hr_agreement.png`.
- `figures/Subject_1_hr.png`: an informative case where the frozen model often disagrees with ECG while CHROM_WIN is closer.
- `figures/Subject_13_error_quality.png`: error, SNR and the unsupported tail.
- `figures/Subject_9_worst_window.png`: an explicitly ECG-error-selected weak diagnostic window with separate electrical markers.
- `figures/*_weights.png`, `figures/*_median_window.png`, `figures/*_worst_window.png`, `figures/*_random_window.png`: all-subject allocations and deterministic diagnostic selection.
- `overview/skin_type_descriptive.png` and `overview/illumination_group_descriptive.png`: counts and supported participant-level intervals.
- `external_evaluation_receipt.json` and `completion_verification.json`: immutable configuration/data/model identity and completion/checksums.

Final checks: **197 pytest tests and 55 subtests passed**, with one existing two-participant training-fixture warning; compileall and git diff --check passed. The saved synthetic CLI smoke used two generated recordings and is labeled SMOKE, never REAL EXTERNAL. All 253 artifact checksums in completion_verification matched, all 187 PNGs were readable, all 26 QC records exactly matched the pre-inference audit, and all window-weighted metrics recomputed from the CSV matched the summary. The official frozen checkpoint SHA256 remained unchanged.

This result supports an independent, qualified HR-only assessment on the full usable rPPG-10 cohort and shows limited transfer of frozen v1. It supplies no ECG/optical waveform morphology ground truth, BP accuracy, clinical agreement, physiological transit-time validity or population-wide fairness claim. rPPG-10 has now been consumed as independent evidence for this frozen version.
