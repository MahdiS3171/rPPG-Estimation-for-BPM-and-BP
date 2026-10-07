# Part 3: Waveform v1 candidate training and selection

This infrastructure produces a **Waveform v1 candidate**, with checkpoint status
`candidate_not_frozen`. A later review of complete validation results, followed
by a deliberate locked-test evaluation, must precede any explicit freeze.

## Scientific scope

Correct HR is necessary but insufficient for future waveform-based BP work:
many waveforms with different pulse shapes share the same dominant frequency.
UBFC supplies face video and a contact **fingertip** PPG/BVP reference. Pulse
propagation and anatomical site can alter both timing and shape. The reference
therefore supervises general pulse structure with lag tolerance; this does not
establish exact facial/finger morphology equivalence.

First differences modestly supervise rising/falling edges and general shape.
Second derivatives amplify noise and could force site-specific APG structure.
There is no second-derivative/APG objective, dicrotic-notch/reflected-peak
objective, RI/A2/A1 equality, or stiffness index. Such features require later
pilot evidence that they are visible and repeatable. Coarse pulse measurements
are descriptive diagnostics only. No clinical BP accuracy, validated dicrotic
feature, or hand-waveform generalization claim follows from these changes.

The existing BP pipeline, face–hand timing, Phase 1 extraction, old UBFC dataset,
and single-ROI waveform trainer retain their behavior. The model's quality head
is **uncalibrated**: it receives no target and enters neither loss nor selection.

## Invalid-ROI BatchNorm correction

`MultiROIPriorResidualWaveformNet.forward()` masks invalid numeric channels,
flattens `[B,R,C,L]` to `[B*R,C,L]`, selects indices where flattened `roi_valid`
is true, and runs `TemporalEncoder1D` **only on those rows**. Valid features are
copied with differentiable `index_copy` into a zero `[B*R,D,L]` tensor.
Autograd remains connected to the valid encoder output; missing branches never
enter BatchNorm running statistics. All downstream masks and conservative
initialization remain intact. Zero-initialized heads intentionally block
encoder gradients on the very first step; gradients reach it after the heads
have updated. A TRAIN-mode test compares every shared BatchNorm running mean
and variance exactly against a separate, identical valid-only encoder.

## New loss and same-lag alignment

`waveform_v1_training_loss()` returns a scalar by default. With
`return_components=True` it returns scalar tensor keys `total`, `waveform_corr`,
`derivative_corr`, `spectral`, `hr`, and `residual`.

```text
L = 1.00 * Lcorr + 0.25 * Ld1 + 0.10 * Lspectral + 0.10 * LHR + 0.02 * Lresidual
```

These are engineering starting weights, not physiological constants. All five
weights, lag limit, derivative alignment weight, and spectral bounds are
configurable and checkpointed. The new objective has no generic smoothing term.
`rppg_training_loss`, `waveform_combo_loss`, and
`time_shifted_negative_pearson_loss` are unchanged.

For each sample and each integer lag within `±round(max_lag_sec * fs)`, construct
one overlapping prediction/reference pair. Default `max_lag_sec=0.5`.

```text
positive lag k: p = prediction[k:],  t = target[:-k]
negative lag k: p = prediction[:k],  t = target[-k:]
zero lag:      p = prediction,      t = target

Lcorr(k) = 1 - Pearson(p, t)
Ld1(k)   = 1 - Pearson(diff(p), diff(t))
k*       = argmin_k [Lcorr(k) + derivative_internal_weight * Ld1(k)]
```

The default `derivative_internal_weight=0.25` controls alignment selection
independently of the external `w_d1`. Both returned terms are gathered at the
same `k*` per sample. Their gradients reach the selected prediction segments;
the discrete lag choice has the same practical differentiability as the legacy
best-lag objective. Neither absolute Pearson nor dynamic time warping is used.
Positive lag means the prediction lags the reference. Ties try zero, smaller
absolute lag, then negative before positive. Lag is diagnostic and is never
penalized in checkpoint selection.

Candidates retain at least three samples when possible, so their first
differences have two observations. Windows shorter than three use zero lag.
Undefined training Pearson comparisons (short/constant segments) use neutral
loss 1, rather than claiming perfect agreement. Evaluation returns NaN for
undefined metrics. Dataset windows still require at least four samples.

`Lspectral` compares normalized, mean-centered, Hann-tapered PSDs using
Jensen–Shannon divergence divided by `ln(2)` (range `[0,1]`; equal spectra give
zero). It reuses the existing `_band_psd()` FFT code. The effective band is
`[spectral_fmin_hz, min(spectral_fmax_hz, 0.45 * fs)]`, default `[0.7,8.0]` Hz
at 30 Hz. The cap stays below Nyquist. This broader band includes pulse
harmonics, but high-frequency shape differs by site, so its weight remains low.
No available FFT bins produces NaN for evaluation and an explicitly skipped
(zero contribution) auxiliary spectral training term.

`LHR` reuses the stable HR label-distribution cross-entropy unchanged, in its
existing 0.7–3.5 Hz HR band. `Lresidual` is the mean square of **unscaled valid
ROI residuals only**, accepting `[B,R,L]` plus a bool `[B,R]` mask, or prepared
`[Nvalid,L]` residuals. Invalid placeholders never enter its denominator. The
model should correct its priors conservatively.

## Evaluation and aggregation

`rppg_lab/waveform_metrics.py` provides window metrics, aggregation, and model
evaluation outside the trainer. Each window reports:

- Signed zero-lag and best-lag waveform Pearson correlations.
- Waveform-selected lag in samples and milliseconds.
- First-difference Pearson at **that waveform-selected lag**, with no independent
  derivative realignment. Evaluation deliberately selects waveform correlation
  alone; training uses the joint waveform/derivative objective above.
- Aligned normalized RMSE, independently z-scoring the two overlap segments.
- Full-window broad-band spectral distance and similarity (`1 - distance`).
- Full-window waveform-derived HR, signed HR error, and HR absolute error.
  The reference is the dataset's within-window waveform-derived `y_hr` (with its
  existing same-window contact-HR fallback); standalone metrics derive it from
  the reference waveform when no label is supplied.
- Absolute prediction/reference differences for `width50_mean_sec`,
  `upstroke_mean_sec`, `fall_time_mean_sec`, `area_mean`, `ibi_mean_sec`, and
  `ibi_std_sec`, reusing `pulse_wave_features()` with its existing HR filter.
  The standalone evaluator also retains the prediction/reference feature values.

Nonfinite samples are not deleted to compress time or fabricate adjacent edges.
Short, flat, missing, or otherwise undefined metrics are NaN in memory. JSON
artifacts use **null**, following the repository's strict JSON writer, never an
invented zero. Coarse morphology, spectra, lag, and learned attention are
diagnostics; they are not calibrated physiology or selection criteria.

Validation preserves window-weighted means and historical window HR regression
metrics. For each `batch["subject_id"]`, first average finite windows per metric;
then average those subject means equally. Participants with more windows do
not dominate the subject-balanced summaries. Missing metrics exclude a subject
only from that metric. Reports include per-metric finite window counts,
per-metric finite subject counts, each subject's total windows, and
`valid_windows_per_subject` (finite aligned waveform and derivative correlation).
Selection uses subject-balanced means. Mean prior weights are conditional on
ROI validity; ROI attention means include zero attention from invalid branches.
Scaled residual magnitude is reported per valid ROI and as an overall maximum.

## Selection and artifact names

The exact selection formula is saved in checkpoints and the training log:

```text
morphology_score = 0.5 * subject_balanced_metrics.wave_corr_aligned
                 + 0.5 * subject_balanced_metrics.d1_corr_same_lag
```

Both correlations use the same waveform-selected alignment and the same
`[-1,1]` scale. Spectral/coarse metrics remain diagnostics.

Epoch 0 establishes the run-specific conservative reference. Candidate eligibility:

```text
subject_balanced_HR_MAE(epoch)
    <= subject_balanced_HR_MAE(epoch0) + hr_regression_tolerance_bpm
```

Default tolerance is **1.0 bpm**, an engineering anti-regression guard, not a
clinical threshold. Epoch 0 is eligible when its HR and morphology selection
metrics are finite; training refuses a nonfinite epoch-0 reference. Later epochs
with undefined selection metrics are ineligible. Among eligible epochs choose
maximum morphology score. Scores within `1e-8` tie: prefer higher
subject-balanced aligned correlation, then lower subject-balanced HR MAE, then
earlier epoch. Finite HR wins an otherwise exact unconstrained tie against
undefined HR. No lag penalty is applied.

| Checkpoint | Criterion |
| --- | --- |
| `best_waveform_candidate.pt` | Highest eligible morphology score with documented ties |
| `best_morphology_unconstrained.pt` | Highest morphology score without the HR gate; diagnostic, may have undefined HR |
| `best_hr_diagnostic.pt` | Lowest subject-balanced validation HR MAE; earlier epoch on ties |
| `last.pt` | Last completed epoch |

Every checkpoint records `waveform_v1_status="candidate_not_frozen"`.
`waveform_multi_roi_last.pt`, `waveform_multi_roi_best_hr_PROVISIONAL.pt`, and
`training_log_PROVISIONAL.json` are deprecated compatibility aliases carrying
the new, explicit metadata. Use the new filenames. The retained `--no-best-hr`
ablation flag suppresses only HR diagnostic checkpoints, not morphology selection.

## Deterministic locked participant split

The reusable `waveform_v1_split_manifest()` rejects duplicate IDs, sorts IDs,
permutes with NumPy `default_rng(seed)`/PCG64, reserves test first, then
validation from the remaining participants. Default seed is 42. Test count is
`max(1, round(N * 0.20))`, capped to retain train and validation. Validation
count is `max(1, round((N - Ntest) * 0.20))`, capped to retain training. Python's
rounding rule is used. Lists in each partition are sorted. All windows for a
participant follow that participant's partition.

`waveform_v1_split.json` contains `algorithm_version`
(`waveform_v1_sorted_pcg64_test_first_v1`), `seed`, `all_subject_ids`,
`train_ids`, `validation_ids`, `test_ids`, `test_fraction`,
`validation_fraction`, `validation_fraction_basis`, and `smoke_only`.
Three or more participants always retain all three partitions. For compatibility
with existing two-subject plumbing tests only, an explicit warned
`smoke_only=true` split retains train/validation and an empty test list. It is
not a development split, and the test evaluator refuses its empty test split.

The trainer saves the manifest before dataset construction and **constructs only
train and validation datasets**. Test IDs are visible metadata; no test metrics
appear in epoch history. `--split-manifest` reuses a saved split and verifies the
discovered cohort exactly. An existing output manifest cannot be overwritten
with a different split. Use the full-cohort manifest for comparable experiments.
`--max-subjects` creates a limited plumbing cohort before splitting; its results
cannot replace full-cohort development results.

Locked test is evaluated only by an explicit standalone command after candidate
selection. Repeated development against test would invalidate its holdout role.
The training script never invokes test evaluation.

## Commands

Recommended development training:

```powershell
.\.venv\Scripts\python.exe scripts/train_waveform_multi_roi_ubfc.py --ubfc-root UBFCData --cache-dir cache_roi_multi_phase1 --out-dir checkpoints/waveform_multi_roi_v1_candidate --objective waveform_v1 --epochs 20 --batch-size 32 --fs 30 --win-sec 10 --stride-sec 2 --w-corr 1.0 --w-d1 0.25 --w-spec 0.10 --w-hr 0.10 --w-res 0.02 --max-lag-sec 0.5 --derivative-internal-weight 0.25 --spectral-fmin-hz 0.7 --spectral-fmax-hz 8.0 --hr-regression-tolerance-bpm 1.0 --test-fraction 0.20 --validation-fraction 0.20 --seed 42
```

Use `--objective phase2_rppg` in a separate directory with the same
`--split-manifest` for the unchanged Phase 2 loss ablation. It retains its
original weights, 0.5-second loss alignment, 0.7–3.5 Hz loss spectrum, and
historical generic smoothing. `objective_loss_config` records those actual
legacy settings; `loss_config` records the Part 3 evaluation configuration for
comparable morphology reporting/selection. Waveform v1 weight flags affect only
the new objective. The old single-ROI trainer remains unchanged.

Separate validation evaluation:

```powershell
.\.venv\Scripts\python.exe scripts/evaluate_waveform_multi_roi_ubfc.py --ubfc-root UBFCData --cache-dir cache_roi_multi_phase1 --checkpoint checkpoints/waveform_multi_roi_v1_candidate/best_waveform_candidate.pt --split-manifest checkpoints/waveform_multi_roi_v1_candidate/waveform_v1_split.json --split val --out outputs/waveform_v1_validation.json
```

Only after model selection and before a later explicit freeze:

```powershell
.\.venv\Scripts\python.exe scripts/evaluate_waveform_multi_roi_ubfc.py --ubfc-root UBFCData --cache-dir cache_roi_multi_phase1 --checkpoint checkpoints/waveform_multi_roi_v1_candidate/best_waveform_candidate.pt --split-manifest checkpoints/waveform_multi_roi_v1_candidate/waveform_v1_split.json --split test --out outputs/waveform_v1_locked_test.json
```

`--split` is required (no default). The evaluator loads with `weights_only=True`,
strictly reconstructs the model, and derives the dataset from checkpoint
metadata, including `max_frames` if this was a smoke checkpoint. It refuses
incompatible ROI/prior/channel order, extraction configuration, cache version,
window length, spectral band, subject IDs, or manifest hash. Cache/root/output
paths and inference batch size/device may relocate; signal configuration may
not be overridden. Output JSON includes per-window results, balanced summaries,
exclusions, and checkpoint/split identity.

## Checkpoint metadata and verification

The schema stores exact model constructor/state, optimizer state,
ROI/prior/channel order, full extraction configuration/cache version, sampling,
window/stride/quality/frame limits, objective and all weights/alignment/band
settings, morphology formula, HR gate/tolerance, tie rules, manifest
path/content/canonical SHA256, exact train/validation/test IDs, epoch,
validation and epoch-0 metrics, selection criterion, seed, Git revision/dirty
state/package versions/source hashes, and candidate/uncalibrated-quality status.

`tests/test_waveform_v1.py` adds TRAIN-mode BatchNorm/gradient regression, known
shift and adversarial same-lag comparisons, perfect/polarity/short-window tests,
Nyquist/broad-harmonic tests, component accounting, masked residuals, evaluation
alignment/NaNs, subject weighting/counts, gate/tie/diagnostic selection, split
integrity/explicit evaluator routing, synthetic optimizer training, complete
metadata, bit-exact checkpoint reconstruction, standalone validation evaluation,
and unchanged legacy-loss ablation tests. Existing test expectations are retained.

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m compileall -q rppg_lab scripts tests
git diff --check
```

Small-cohort software smoke results do not establish full-dataset waveform
quality. Prior polarity/phase/filter edges, face/finger differences, missingness,
and the reference HR fallback require inspection during full validation. No BP
model training, hand model, learned face–hand timing, pixel/video encoder,
PhysFormer integration, or final model freeze is implemented in Part 3.

## Verification on 2026-10-07

The full suite passed **113 tests**: all 80 existing tests unchanged plus 33
Part 3 tests. `compileall` for `rppg_lab`, `scripts`, and `tests` and
`git diff --check` passed. Synthetic loss/metric/split tests, optimizer updates,
bit-exact CPU checkpoint reconstruction, and standalone validation evaluation
passed. AST comparisons confirmed that all three protected legacy waveform
loss functions and the old split functions are identical to the starting
revision. Diff inspection confirmed that BP, face-hand timing, extraction/cache
semantics, old datasets, single-ROI trainer and existing tests are unchanged.

A local CUDA UBFC smoke used the first four discovered participants, 450 frames
per recording, width 8, one encoder block, dropout 0, 10-second stride, epoch 0
plus one training epoch. The saved split was train `subject1, subject10`,
validation `subject11`, locked test `subject12`. There were two training windows
and one validation window. Epoch-0 residual contribution was exactly zero;
after training its validation maximum absolute contribution was approximately
`0.000422665`. HR gate eligibility held at both epochs, but the morphology score
decreased slightly (`0.46675610` to `0.46674042`), so the candidate correctly
remained epoch 0. Its subject-balanced validation HR MAE was approximately
`0.912098` bpm, aligned waveform correlation `0.520959`, and same-lag derivative
correlation `0.412553`.

The separate evaluator ran **validation only**, reconstructed the selected
checkpoint, and reproduced its stored metrics. Training history was checked
for absence of test metrics. No locked-test waveform was evaluated. These
small-window numbers verify plumbing only and are not full-cohort results.
Ignored local artifacts are in `outputs/smoke_waveform_v1_part3_20261007/`,
including `waveform_v1_split.json`, `training_log.json`, all four checkpoints,
and `validation_results.json`; the complete test transcript is
`outputs/part3_complete_tests.txt`.
