# Pilot-6: restored Frozen Waveform V1 facial evaluation

This is a separate native-resolution, externally adapted input experiment.
Historical Pilot-6 results remain in `outputs/pilot6_dual_roi/full03/`.
The current experiment namespace is `outputs/pilot6_frozen_native_aa_v1/run01/`;
the accepted **timebase adapter is V2**, identified explicitly in every
adaptation artifact and configuration. No BP model is trained or evaluated.

## Identity and original contract

The restored official pair is:

- `checkpoints/waveform_v1_official_ubfc/waveform_v1_frozen.pt`
- `checkpoints/waveform_v1_official_ubfc/waveform_v1_freeze_manifest.json`

Checkpoint SHA256:
`d6b2d1eeebc0153197a2a12da736b957b32d72ae00b22631e9afd0b2097131ff`.
Manifest SHA256:
`da5ba397b3be389deec8821f8ea16944b5f5b93e2834bb79c3c147eb3d1f89dc`.
Source candidate SHA256:
`5dfe34aab74ebf994a343eab3d82dcf0c6c3c465aaf0182d041117d01b31f982`;
selected epoch 25.

The unchanged `FrozenWaveformExtractor` successfully checks frozen status and
decision, checkpoint hash, source candidate and epoch, checkpoint schema,
strict state loading, architecture configuration, ROI/prior/channel ordering,
30 Hz sampling, 300-sample windows, extraction configuration, minimum support,
split identity, and validation/locked-test evaluation identities. The model is
`MultiROIPriorResidualWaveformNet`, in evaluation/inference mode. New experiment
code additionally pins both official file hashes and checks model state before
and after inference. No checkpoint, manifest, training module, split, original
inference API, architecture or historical evaluation artifact is changed.

The ROI order is forehead, left cheek, right cheek. Each has nine channels:
RGB_R, RGB_G, RGB_B, GREEN, CHROM, CHROM_WIN, PBV, POS_WIN, OMIT.
Window-local standardization, prior construction, minimum source validity 0.9,
masking, signed output interpretation, attention and prior allocations are
those of the original API. Ten-second windows use five-second hops; this
selects evaluation windows without changing the frozen input length.

The earlier resized Pilot-6 RGB cache is unsuitable for this experiment and
is not consumed. A separate path decodes **every native 1080×1920 frame**,
uses the existing `FaceRegionDetector`, `get_face_masks` and `extract_frame_rgb`
functions, and keeps the frozen three-ROI geometry, 50-pixel threshold,
confidence thresholds, `auto` detector selection and disabled skin gate.
Bounding an already constructed full-resolution mask accelerates pixel
measurement without resizing or changing any contributing pixels; automated
tests compare the resulting RGB and diagnostics exactly with full-frame means.
Saved float64 ROI vertices reconstruct the historical rounded convex hulls.
No participant image pixels are written.

Configuration compliance does not imply identical historical detector output:
training recorded MediaPipe 0.10.14 and OpenCV 4.13.0.92; this environment uses
MediaPipe 1.0.1 and OpenCV 5.0.0.93. Current `auto` resolves to Tasks. The
historical resolved backend was not recorded in the manifest. Model, geometry,
prior and preprocessing source text matches freeze provenance after accounting
for line-ending differences. The existing `waveform_review.py` provenance
differs substantively, but it is not the prediction path and is not changed by
this task. Dependency versions and detector behavior remain domain-shift
limitations rather than silently claimed equivalence.

## Metadata and timebase audit

Read-only local source: `sheets/rPPG BP Data Collection final.xlsx`, using the
verified mappings in `configs/pilot6_metadata.json`. The actual tabs are
00_Protocol, 01_Subjects, 02_Sessions, 03_Review, 04_Windows, 05_Field_Guide,
90_Source_Subjects, 91_Source_Sessions and 92_Migration_Audit. Every original
header and the workbook/mapping hashes are recorded in `audit/audit.json`.
All six filenames match their session metadata explicitly; no row-order
matching is used. BP values are not attached to waveform windows.

Native sample timestamps are recovered from MP4 stts/ctts display order and
supported edit lists, then checked against the complete decoded frame count.
Camera PTS, decoder-reported PTS, rotation, frame counts, geometry and irregular
intervals are retained. For S004, the 5.8 ms initial movie lead is preserved.
Nominal average FPS is not substituted for pulse timing. Long intervals are
indicators, not proof of capture drops; repeated decoded pixels are counted
separately. Camera exposure and rolling-shutter timing remain unmeasured.

## Versioned adaptation and its measured limitations

`rppg_lab/frozen_timebase.py` implements an external adapter; it does not edit
the frozen API or disable the existing pipeline's downsampling rejection.
Its accepted configuration is `configs/pilot6_frozen_timebase_v2.json`.

1. Anchor the uniform 60 Hz and 30 Hz grids to the first actual camera PTS.
2. Regularize valid raw RGB onto the 60 Hz grid using bounded natural cubic
   interpolation, separately between observation gaps exceeding 50.001 ms.
   Short intervals that are interpolated remain explicitly flagged. No
   extrapolation, nominal-FPS timing fallback or silent long-gap filling occurs.
3. Apply a 45-tap symmetric Kaiser lowpass FIR to each RGB channel, preserving
   DC intensity. Passband design edge is 8 Hz, stopband edge 15 Hz, and design
   attenuation target 80 dB. Measured maximum stopband gain is −78.54 dB and
   maximum FIR passband gain error is 0.000141. Centered valid convolution
   excludes 0.3667 s support at each segment edge without padded samples.
4. Evaluate the filtered reconstruction at the original camera PTS with the
   same bounded cubic support rule. Preserve native acquisition validity
   separately; effective validity also requires filter support.
5. Pass these filtered native-clock traces to the **unchanged** frozen API,
   which performs its own bounded linear 30 Hz interpolation and original
   support checks, then its original window-local feature preprocessing.

This avoids relabeling derived 30 Hz samples as original camera observations.
`adapted_rgb.npz` also saves the target grid, bracket indices/intervals,
interpolation flags, effective native support and filtered native-clock RGB;
`native_face_rgb.npz` retains unmodified measured intensities and PTS.
The frozen API's source-valid fractions consequently describe conservative
adapter-supported source frames; raw acquisition fractions are reported
separately.

The first linear-interpolation adapter, V1, was rejected **before real model
inference** after synthetic testing on S004's actual PTS exposed attenuation
missed by ideal regular-clock tests. Its configuration and failed validation
artifact are retained. V2 changes only external interpolation. It improves
high-frequency retention but cannot remove attenuation from the frozen API's
final linear interpolation. A half-step linear interpolation of an 8 Hz signal
at 60 Hz can have gain `cos(pi*8/60) = 0.9135`; the V2 gate therefore separately
requires gain within 2% at 0.7/1.2/3.5 Hz, gain 0.90–1.02 at 8 Hz, fitted
adapter delay within 5 ms, and fitted 20/28 Hz aliases at most −40 dB. These are
engineering input checks, not physiological or clinical thresholds.

Measured S004 synthetic gains on its actual acquisition clock:

| Input | V1 gain | V2 gain | V2 fitted adapter delay |
|---|---:|---:|---:|
| 0.7 Hz | 0.99872 | 0.99960 | 0.00072 ms |
| 1.2 Hz | 0.99620 | 0.99877 | 0.00158 ms |
| 3.5 Hz | 0.96784 | 0.98909 | 0.00936 ms |
| 8 Hz | 0.84322 | 0.94221 | 0.04421 ms |

V2's fitted 20→10 Hz and 28→2 Hz aliases are −67.90 and −67.40 dB on S004.
However, total residual AC RMS is **4.0% and 8.1%** of input RMS respectively:
irregular intervals spread interpolation error outside the fitted alias peak.
These fitted-peak numbers must not be presented as rejection of all artifact
power. The FIR response alone also does not describe the complete irregular
acquisition system. Each recording receives its own synthetic clock validation
before inference; failure stops that recording's model path.

The FIR has zero assigned shift on retained centered samples, with full future
support required for this offline analysis. Synthetic adapter delay is not a
measurement of camera timing, neural phase, optical propagation or PTT. No
learned-waveform inter-site timing is calculated. High-frequency morphology is
not proven preserved; pulse feet and dicrotic notches are not validated.

The implementation uses SciPy's documented [natural cubic interpolation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.CubicSpline.html)
and [linear-phase FIR design](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.firwin.html).
[Kaiser order estimates](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.kaiserord.html)
are checked numerically rather than treated as guaranteed attenuation.

## Outputs and interpretation

Per recording, `waveform_windows.npz` retains model windows, exact timestamps,
model input tensors, ROI/prior validity, source support, attention, prior
allocations, residuals, and synchronized GREEN/CHROM/POS windows including
per-ROI baselines. Neural outputs are not concatenated. All complete scheduled
windows have rows in `window_metrics.csv`, including failed windows; excluded
tails are counted separately. `allocations.csv`, `morphology_diagnostics.csv`,
`adjacent_window_consistency.csv`, `example_spectra.csv`, `overview.png`,
`prior_allocations.png`, selected `window_*.png`, `roi_geometry.png` and
`roi_masks_examples.npz` provide interpretable diagnostics. Example selection
includes the first valid and minimum/median/maximum SNR windows explicitly.

GREEN, CHROM and single-window POS use the same adapted RGB and supported ROIs.
Each baseline is standardized per ROI through the original prior builder, then
averaged over successful ROIs. The model's separate POS_WIN prior is unchanged.
Classical methods internally use the existing zero-phase 0.7–4 Hz filter;
neural outputs receive no extra filter. HR searches 0.7–3.5 Hz. Ten-second
spectra have approximately 0.1 Hz physical resolution; FFT zero padding does
not create additional resolving power. Existing quality metrics use their
documented eight-second Welch segments. A longer continuous neural spectrum
is not manufactured from independently normalized windows.

"Accepted" means the frozen API returned finite consistent outputs.
"Engineering good" requires existing SNR ≥0 dB and lag-periodicity ≥0.5.
Neither means physiological validity. Coverage is the union of accepted/good
intervals, not a sum of overlapping windows. Adjacent overlap correlations
measure window-boundary sensitivity without aligning or reversing polarity.
Attention and prior allocations are descriptive operations, not causal
explanations or calibrated confidence. Classical agreement is not independent
ground truth, especially where classical priors are model inputs.

## Reproduction in PowerShell

Start with a fresh directory. The defaults resolve the verified local workbook,
original official checkpoint/manifest pair and V2 adaptation configuration.

```powershell
.\.venv\Scripts\python.exe scripts/evaluate_pilot6_frozen.py `
  --recording S004_V01 --dry-run `
  --out-dir outputs/pilot6_frozen_native_aa_v1/audit_reproduction01

.\.venv\Scripts\python.exe scripts/evaluate_pilot6_frozen.py `
  --recording S004_V01 `
  --out-dir outputs/pilot6_frozen_native_aa_v1/reproduction01

.\.venv\Scripts\python.exe scripts/evaluate_pilot6_frozen.py `
  --all-recordings --reuse --workers 5 `
  --out-dir outputs/pilot6_frozen_native_aa_v1/reproduction01

.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_frozen_timebase.py -v
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

`--all-recordings` always evaluates S004 and writes its evidence-based execution
decision before scheduling other videos. The gate requires official identity,
unchanged state, preserved PTS, successful actual-clock synthetic checks,
at least three accepted windows and at least 30 seconds of union coverage.
This determines whether further descriptive evaluation can run; it does not
require favorable quality results. `--reuse` checks the native cache identity
against video SHA256, source code, model asset, full frozen extraction config
and dependencies, and regenerates analysis in that explicitly selected run.
`--reports-only` audits and summarizes existing per-recording summaries.
No public artifacts include participant images or identifying names.

## Measured S004 evaluation and execution decision

All 18,053 frames were measured at native resolution. Camera PTS span
0.0058–301.598922 s (301.593122 s support). All three geometric ROIs have
100% finite measured RGB; the detector does not expose a confidence score.
Median selected pixel counts are 3,356 / 1,183 / 1,086 for forehead / left cheek /
right cheek, with no selected pixel clipping. This is acquisition support,
not pulse quality. The camera has 42 intervals longer than 1.5 median steps,
with maximum interval 50 ms.

| S004 method | Accepted / scheduled windows | Engineering good windows | Accepted union | Good union | Median spectral HR, all accepted |
|---|---:|---:|---:|---:|---:|
| Frozen V1 | 58 / 59 | 2 | 295 s | 15 s | 80.46 bpm |
| GREEN | 58 / 59 | 25 | 295 s | 175 s | 81.47 bpm |
| CHROM | 58 / 59 | 0 | 295 s | 0 s | 75.24 bpm |
| POS | 58 / 59 | 8 | 295 s | 70 s | 79.80 bpm |

The first scheduled window fails full anti-alias support at the recording
edge and remains explicitly recorded. The incomplete tail is 1.6 s.
Frozen spectral HR ranges 58.31–124.55 bpm; this range has no waveform reference
validation. The signed median correlation on adjacent five-second overlaps is
0.9202. High overlap consistency does not resolve the low pulse-quality result.

Mean ROI attention is 0.3373 / 0.3329 / 0.3298. Unweighted mean prior allocations
over valid ROI/windows are GREEN 0.1837, CHROM 0.2730, CHROM_WIN 0.2563,
PBV 0.2420, POS_WIN 0.0292 and OMIT 0.0158. The post-hoc verifier reconstructs
the exact model output from weighted prior channels plus weighted scaled
residuals, with maximum absolute numerical discrepancy 1.03e-7 on S004.
The median residual/model RMS ratio is 0.6744; correlated components can cancel,
so this is **not** an explained-variance fraction or causal importance. The
saved `model_contributions.csv` and `model_decomposition.png` expose this
algebra without fitting a new model.

S004's saved first and last valid input tensors, waveform predictions,
attention and prior allocations replay bit-for-bit through the original API.
Relative decoder/container PTS disagree by at most 1.14e-13 s, with an
expected −5.8 ms decoder origin offset. This confirms frame/clock correspondence
in the file; it does not measure exposure or rolling-shutter timing.

The prespecified technical gate passes: verified identity, unchanged state,
preserved PTS, camera-clock transfer checks, ≥3 accepted windows and ≥30 s
accepted union. This justifies running the remaining five to characterize
domain shift. It does **not** justify using these predictions for BP or timing.

A post-hoc sensitivity run changes only the external interpolation gap limit
from 50.001 to 25.001 ms, using
`configs/pilot6_frozen_timebase_v2_strict_gaps.json` and the exact same native
RGB. In `outputs/pilot6_frozen_native_aa_v1/s004_gap_sensitivity01/`, S004 then
has 12 accepted windows / 105 s accepted union and **zero** engineering-good
neural windows; GREEN has 7 good windows / 60 s good union. The two good neural
windows in the main run therefore rely on support allowed by the more lenient
gap policy. This shows sensitivity to acquisition continuity, not proof that
those windows are artifacts. Both policies are reported; no weights or
thresholds were optimized for BP or accuracy.

The full regression suite passes 254 tests (229 historical plus 25 new);
the focused new suite passes 25 tests. These include positive/negative
synthetic adapter delay controls, alias attenuation, irregular clocks, gaps,
native-mask equivalence, original API rejection rules, source/identity checks,
bit-identical 30 Hz behavior, window consistency and complete report generation.
The real-artifact verifier additionally checks MP4/decoder clocks, raw video
hashes, model input tensors, saved predictions and decomposition identities.

```powershell
.\.venv\Scripts\python.exe scripts/verify_pilot6_frozen.py `
  --run-dir outputs/pilot6_frozen_native_aa_v1/run01 --recording S004_V01
```

Implementation files added for this experiment:

- `rppg_lab/frozen_timebase.py`: versioned interpolation, FIR and synthetic clock validation.
- `rppg_lab/pilot_frozen_extraction.py`: native-resolution frozen facial acquisition and cache.
- `rppg_lab/pilot_frozen.py`: audits, original-API inference, metrics, execution gate and reports.
- `rppg_lab/pilot_frozen_plots.py`: synchronized diagnostic figures and exact mask examples.
- `scripts/evaluate_pilot6_frozen.py`: PowerShell-compatible audit/evaluation/reuse/report CLI.
- `scripts/verify_pilot6_frozen.py`: real-data replay, identity/clock checks and algebraic diagnostics.
- `configs/pilot6_frozen_timebase_v1.json`: retained rejected linear candidate.
- `configs/pilot6_frozen_timebase_v2.json`: accepted external adaptation configuration.
- `configs/pilot6_frozen_timebase_v2_strict_gaps.json`: explicit gap sensitivity configuration.
- `tests/test_frozen_timebase.py`: 25 synthetic/local-checkpoint regression tests.
- This report, with an additive follow-up in `docs/PILOT6_DUAL_ROI_BP_FEASIBILITY.md`.

## Complete cohort results

Both the main 50.001 ms gap policy and the post-hoc 25.001 ms sensitivity
policy completed all six recordings without recording failures. All 108,115
native frames were measured, totaling 1,805.39 seconds. Each recording has
59 scheduled overlapping windows. Main inference accepts 347/354 windows;
the first window of every recording and the last S006 window lack full FIR
edge support and remain explicit failures. All three native RGB ROIs are
finite on 100% of decoded frames in every recording.

| Recording | Frames | Main accepted | Main good windows | Main good union | Strict accepted | Strict good union | Main median spectral HR |
|---|---:|---:|---:|---:|---:|---:|---:|
| S001_V01 | 18,047 | 58 | 5 | 45 s | 29 | 20 s | 82.56 bpm |
| S002_V01 | 18,035 | 58 | 15 | 110 s | 19 | 70 s | 73.99 bpm |
| S003_V01 | 17,994 | 58 | 50 | 275 s | 18 | 120 s | 71.63 bpm |
| S004_V01 | 18,053 | 58 | 2 | 15 s | 12 | 0 s | 80.46 bpm |
| S005_V01 | 18,014 | 58 | 39 | 255 s | 18 | 85 s | 89.80 bpm |
| S006_V01 | 17,972 | 57 | 5 | 40 s | 11 | 0 s | 91.27 bpm |

Main accepted union is 1,765 s; engineering-good union is 740 s across six
recordings (116 good windows). With strict gap rejection, accepted union is
765 s (107 windows) and good union is 295 s (36 windows). These are descriptive
duration totals across recordings, not independent physiological observations.
The sensitivity directory retains its original name,
`s004_gap_sensitivity01`, after expansion to the full cohort.

Main engineering-good union by extraction method, with identical acquisition
support and the documented method-specific preprocessing:

| Recording | Frozen V1 | GREEN | CHROM | POS |
|---|---:|---:|---:|---:|
| S001_V01 | 45 s | 130 s | 10 s | 10 s |
| S002_V01 | 110 s | 155 s | 120 s | 20 s |
| S003_V01 | 275 s | 280 s | 235 s | 130 s |
| S004_V01 | 15 s | 175 s | 0 s | 70 s |
| S005_V01 | 255 s | 60 s | 85 s | 10 s |
| S006_V01 | 40 s | 20 s | 0 s | 10 s |

This supports **recording-dependent pulse-like structure**, especially S003
and S005. It does not establish that the neural model or any classical method
has more accurate HR or morphology. S003 remains promising under stricter gap
rejection, while S004 and S006 have no good neural coverage under that policy.
Successful tracking is therefore insufficient to decide pulse feasibility.
Do not tune or replace the frozen weights based on these quality counts.

Camera-clock synthetic checks pass for all recordings. At 8 Hz, complete
adapter gain is 0.9422–0.9429. Fitted 20/28 Hz alias gains range approximately
−72.6 to −54.5 dB, while total residual AC RMS reaches 8.21% in the 28 Hz
control. This again distinguishes narrow fitted alias rejection from broader
irregular-sampling artifacts. Native PTS and original intensities are retained
for investigation and the conservative gap comparison is reported alongside
the main result.

Both real-data verification runs pass for all six recordings. Twelve selected
windows per policy replay bit-for-bit, including input tensors, predictions,
ROI attention and prior allocations. Every recovered MP4 clock matches the
decoder's relative display clock to within 1.14e-13 s; S004 and S006 retain
their 5.8 and 9 ms movie leads. Algebraic decomposition maximum error is
1.35e-7 in the main run and 9.25e-8 in the strict run. Original checkpoint,
manifest, workbook and protected historical code remain unchanged. None of
the 792 pre-existing Pilot-6 output files has a modification time after this
task began.

The main run includes `cohort_quality_comparison.png`,
`gap_policy_comparison.csv`, `gap_comparison_provenance.json`, the exact
`compare_gap_policies.py` plotting artifact, `environment_freeze.txt`, both
machine-readable cross-recording summaries, and the per-recording outputs.
The original actual-inference manifest is retained as
`experiment_manifest_inference.json` when testing report regeneration.
The exact experiment source at inference is also archived under
`run01/source_at_inference/`. A final report-only correction makes generated
gap-limit/FIR prose reflect each configuration, including 25.001 ms in the
strict report. Both policies were replay-verified afterward with unchanged
predictions. Final report generation and the final 25-test focused run exit 0;
`test_execution.json` records exit 0 for the full 254-test run.

Actual completed cohort commands (the main run reused the separately audited
S004 native acquisition; remaining native frames were freshly extracted):

```powershell
.\.venv\Scripts\python.exe scripts/evaluate_pilot6_frozen.py `
  --all-recordings --reuse --workers 5 `
  --out-dir outputs/pilot6_frozen_native_aa_v1/run01

# Strict policy: identical native caches copied into this fresh sensitivity namespace.
.\.venv\Scripts\python.exe scripts/evaluate_pilot6_frozen.py `
  --all-recordings --reuse --workers 5 `
  --adaptation-config configs/pilot6_frozen_timebase_v2_strict_gaps.json `
  --out-dir outputs/pilot6_frozen_native_aa_v1/s004_gap_sensitivity01

.\.venv\Scripts\python.exe scripts/verify_pilot6_frozen.py `
  --run-dir outputs/pilot6_frozen_native_aa_v1/run01
.\.venv\Scripts\python.exe scripts/verify_pilot6_frozen.py `
  --run-dir outputs/pilot6_frozen_native_aa_v1/s004_gap_sensitivity01
.\.venv\Scripts\python.exe -m outputs.pilot6_frozen_native_aa_v1.run01.compare_gap_policies
```

Windows PowerShell 5 combined `*>` capture labels native MediaPipe startup
stderr as `NativeCommandError`, making the outer shell report exit 1 for the
fresh-extraction run despite complete, verified artifacts and no recording
failures. A captured detector-startup diagnostic exits Python with code 0;
the cache-reuse strict cohort and artifact verifiers also exit 0. Inspect
`$LASTEXITCODE`, or capture stdout/stderr separately, when scripting native
executables. The warning and its original log are retained; this is not a
checkpoint or numerical inference failure.

The next experiment should prioritize **paired physiological and acquisition
references**, not BP fitting: repeat promising S003/S005-like and challenging
S004/S006-like conditions with synchronized contact pulse measurements, locked
camera settings, stronger stable facial/hand pixel coverage and a known optical
timing control at the relevant image rows. Quantify rolling-shutter/exposure
and dropped-frame effects before evaluating learned morphology or inter-site
timing. Only then expand repeated BP-labeled subject/session collection and
evaluate a separately developed model with subject grouping. Frozen V1 remains
unchanged and BP accuracy remains unvalidated.
