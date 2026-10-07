# Waveform v1 full-cohort finalization

## Official experiment

The official experiment uses main-branch commit
`90e2d93ed866e51b46a3d23fcaab3df9bc4cfea5` and the existing Part 3
`MultiROIPriorResidualWaveformNet`, objective, preprocessing and selection rule.
Before execution all 113 existing tests, compileall and diff whitespace checks
passed. The quality head is unsupervised and does not select checkpoints.

The local UBFCData contains 41 participant folders, all discovered with video
and contact reference. No subject or frame limit is used. The seed-42 Part 3
manifest is generated once in `checkpoints/waveform_v1_official_ubfc/`.

| Partition | Count | IDs |
|---|---:|---|
| Train | 26 | subject1, subject10, subject11, subject12, subject15, subject17, subject18, subject2, subject20, subject21, subject22, subject23, subject25, subject27, subject3, subject30, subject33, subject34, subject36, subject37, subject38, subject4, subject40, subject5, subject6, subject8 |
| Validation | 7 | subject19, subject28, subject29, subject31, subject32, subject35, subject41 |
| Locked test | 8 | subject13, subject14, subject16, subject24, subject26, subject39, subject7, subject9 |

Test is reserved first (20% of all participants); validation uses 20% of the
remainder. Counts use the established rounding algorithm. All IDs, partition
counts and initial checks are also in `preflight.json` and the split manifest.
Subject eligibility and window exclusions are recorded without replacing or
rebalancing this split.

```powershell
.\.venv\Scripts\python.exe scripts/train_waveform_multi_roi_ubfc.py --ubfc-root UBFCData --cache-dir cache_roi_multi_phase1 --out-dir checkpoints/waveform_v1_official_ubfc --split-manifest checkpoints/waveform_v1_official_ubfc/waveform_v1_split.json --objective waveform_v1 --epochs 30 --batch-size 32 --fs 30 --win-sec 10 --stride-sec 2 --seed 42
```

The model has width 32, three blocks, dropout 0.20, residual scale 0.10,
default conservative prior initialization and AdamW at learning rate 1e-4 /
weight decay 1e-4. ROI order is forehead, left_cheek, right_cheek. Prior order
is GREEN, CHROM, CHROM_WIN, PBV, POS_WIN, OMIT; channels are RGB_R, RGB_G,
RGB_B followed by those priors. The loss is unchanged:

```text
1.00 Lcorr + 0.25 Ld1 + 0.10 Lspectral + 0.10 LHR + 0.02 Lresidual
```

Maximum lag is 0.5 seconds, derivative alignment internal weight is 0.25, and
the morphology spectral band is 0.7–8 Hz at 30 Hz. There are no new morphology,
notch, reflection or APG losses. Training constructs only train and validation
datasets. The saved training log includes epoch 0 and every completed epoch.

## Validation review

Selection maximizes `0.5 * subject-balanced aligned waveform correlation +
0.5 * subject-balanced same-lag d1 correlation` among epochs whose
subject-balanced HR MAE is at most epoch-0 HR MAE + 1 bpm. Ties follow the Part
3 rule. Neither the best-HR nor the unconstrained morphology diagnostic can
replace the selected candidate.

```powershell
.\.venv\Scripts\python.exe scripts/evaluate_waveform_multi_roi_ubfc.py --ubfc-root UBFCData --checkpoint checkpoints/waveform_v1_official_ubfc/best_waveform_candidate.pt --split-manifest checkpoints/waveform_v1_official_ubfc/waveform_v1_split.json --split val --include-baselines --out outputs/waveform_v1_official_validation.json
.\.venv\Scripts\python.exe scripts/review_waveform_v1_candidate.py --training-log checkpoints/waveform_v1_official_ubfc/training_log.json --checkpoint checkpoints/waveform_v1_official_ubfc/best_waveform_candidate.pt --validation-evaluation outputs/waveform_v1_official_validation.json --split-manifest checkpoints/waveform_v1_official_ubfc/waveform_v1_split.json --out-dir checkpoints/waveform_v1_official_ubfc
```

The evaluator collects the candidate and all 18 ROI/prior plus six equal-ROI
prior-average baselines on the same batch tensors. Average only ROIs for which
both ROI and the particular prior are valid. Failed baseline windows retain
undefined metrics and their coverage counts. No legacy face extractor is used.
Subject ID, start time and a target/HR-label SHA256 verify exact shared windows.

The review utility reads saved artifacts and never performs inference. It
compares epoch 0 with the selected checkpoint under both subject-balanced and
window-weighted aggregation, verifies the selection against the completed log,
and reports the conservative model as an additional baseline. It reports
per-subject metrics and distribution summaries, attention means/medians,
per-subject attention, validity, prior weights, highest-attention percentages,
concentration (`max(beta) > 0.8`) and scaled residual diagnostics. Attention
ties share credit equally. These descriptions have no causal interpretation.

Bootstrap uses 2,000 resamples of participants with replacement, seed 42,
NumPy PCG64, and 95% percentile intervals. Subject window counts never weight
these draws. Missing participant metrics remain unavailable, rather than
silently removing a failed subject. These are uncertainty reports without a
significance claim.

The predeclared validation gate requires HR eligibility, morphology no worse
than epoch 0 within 1e-8 numerical tolerance, and an improvement beyond that
tolerance in at least one correlation while the other drops by at most 0.05.
HR must stay within +1 bpm of epoch 0. Main metrics and residual diagnostics
must be finite, every reserved validation subject must have valid windows,
and classical comparisons must be present. Finite aggregates cannot conceal
nonfinite participant/window metrics. The 0.05 and +1 guards are engineering
rules, not physiological or clinical thresholds. Residual magnitude has no
invented physiological rejection threshold. Beating every baseline is not a
requirement.

`DO_NOT_TEST_OR_FREEZE` ends the experiment. Architecture and loss are not
automatically changed. `ACCEPT_FOR_LOCKED_TEST` authorizes exactly the selected
checkpoint for the already-reserved test participants.

## Test lock and conditional freeze

The evaluator requires `--validation-review` for `--split test`, checks its
candidate/split/validation hashes before discovering held-out participants,
and refuses test-time baseline comparisons. It exclusively creates
`locked_test_consumption.json` before loading test data. It refuses repeat
evaluations from the same candidate directory or overwriting a test output.
Even a failed evaluation attempt leaves the consumption receipt preserved.

Only after validation acceptance:

```powershell
.\.venv\Scripts\python.exe scripts/evaluate_waveform_multi_roi_ubfc.py --ubfc-root UBFCData --checkpoint checkpoints/waveform_v1_official_ubfc/best_waveform_candidate.pt --split-manifest checkpoints/waveform_v1_official_ubfc/waveform_v1_split.json --split test --validation-review checkpoints/waveform_v1_official_ubfc/waveform_v1_validation_review.json --out outputs/waveform_v1_official_LOCKED_TEST.json
```

Preserve a copy of the pre-test review as `waveform_v1_validation_review.json`
before consuming test. The receipt records the date/time in UTC, checkpoint
epoch/hash, split hashes, validation review/evaluation hashes and test output
path. Both the canonical JSON manifest hash used by Part 3 and the exact file
byte SHA256 are retained and explicitly distinguished.

Review the already-saved test metrics, subject distributions and the same
subject-level bootstrap without selecting a different epoch. Generalization
requires finite main metrics, usable HR without catastrophic regression,
correlations that do not collapse, and no broad failure across most subjects.
The latter three criteria use explicit engineering judgments with quantitative
reasons, rather than unsupported clinical cutoffs. Test need not beat validation.

Save those judgments as a JSON object with exactly the keys
`hr_usable_without_catastrophic_regression`, `waveform_correlations_do_not_collapse`
and `no_broad_systematic_subject_failure`. Each value is an object with a
boolean `passed` and a quantitative text `reason`. Then review/finalize the
saved artifacts explicitly:

```powershell
.\.venv\Scripts\python.exe scripts/finalize_waveform_v1.py --validation-review checkpoints/waveform_v1_official_ubfc/waveform_v1_validation_review.json --checkpoint checkpoints/waveform_v1_official_ubfc/best_waveform_candidate.pt --split-manifest checkpoints/waveform_v1_official_ubfc/waveform_v1_split.json --locked-test-evaluation outputs/waveform_v1_official_LOCKED_TEST.json --generalization-review checkpoints/waveform_v1_official_ubfc/generalization_review.json --out-dir checkpoints/waveform_v1_official_ubfc --freeze
```

This command never evaluates data. `--freeze` is honored only if the saved
validation and locked-test reviews pass; a failed review is saved without a
frozen artifact.

`REJECTED_AFTER_LOCKED_TEST` preserves all artifacts and forbids retuning
against this consumed test. `FREEZE_WAVEFORM_V1` permits metadata-only repacking
of the selected checkpoint. `freeze_waveform_candidate()` verifies the saved
test and consumption receipt, removes optimizer state, checks every model
state tensor's raw bytes, and writes `waveform_v1_frozen.pt` plus
`waveform_v1_freeze_manifest.json`. The manifest records the model/source
hashes, training commit/source hashes, configurations/orderings, participants,
metrics, intervals, dates and `frozen_v1` status. Existing frozen identities
cannot be overwritten. Tests check exact candidate/frozen model outputs.

Once test is consumed, future architecture or loss changes require an external
dataset, a new held-out dataset, or a newly designed validation protocol.
Previous development work in this repository also used UBFC; a new Part 3
split does not erase possible historical dataset exposure. Its protection
applies to this declared experiment, not an independent external-cohort claim.

## Recorded outcome

The full run completed all 30 epochs on October 7, 2026, with 732 training,
210 validation and 238 locked-test windows. No window exclusions occurred in
any partition. The selected candidate is **epoch 25**, selected by validation
only. The unconstrained morphology diagnostic also selected 25; the best-HR
diagnostic selected 12 and was not substituted for the candidate.

Validation decision: **ACCEPT_FOR_LOCKED_TEST**. All predeclared conditions
passed. Final generalization decision: **FREEZE_WAVEFORM_V1**. Waveform v1 is
**frozen**, with exact checkpoint metadata `waveform_v1_status = "frozen_v1"`.

| Subject-balanced metric | Epoch 0 validation | Selected validation | Change from epoch 0 | Locked test | Test minus validation |
|---|---:|---:|---:|---:|---:|
| Aligned waveform correlation | 0.742341 | 0.817292 | +0.074951 | 0.799999 | -0.017294 |
| Same-lag d1 correlation | 0.636286 | 0.762757 | +0.126470 | 0.747674 | -0.015083 |
| Morphology score | 0.689314 | 0.790024 | +0.100711 | 0.773836 | -0.016188 |
| Aligned normalized RMSE | 0.700810 | 0.574493 | -0.126317 | 0.601233 | +0.026740 |
| Spectral distance | 0.107032 | 0.067448 | -0.039584 | 0.072352 | +0.004904 |
| Spectral similarity | 0.892968 | 0.932552 | +0.039584 | 0.927648 | -0.004904 |
| HR MAE (bpm) | 1.056750 | 1.095420 | +0.038670 | 1.096377 | +0.000957 |

Validation window-weighted metrics equal the subject-balanced metrics to
roundoff because every subject has 30 windows. Test window-weighted aligned
correlation/d1/morphology/HR MAE are 0.799851/0.747186/0.773518/1.097505;
subject39 and subject7 have 29 windows each and the other six have 30.

The best classical baseline by validation morphology was **CHROM_WIN multi-ROI
average**: aligned corr 0.742036, d1 corr 0.635409, morphology 0.688722,
HR MAE 1.057280. The candidate added 0.075256 aligned correlation, 0.127348
d1 correlation and 0.101302 morphology, with HR MAE higher by 0.038140 bpm.
Thus the learned stage added waveform/derivative fidelity over conservative
initialization and this classical comparison, while preserving HR within the
predeclared engineering gate. It did not improve every HR comparison.

| 95% subject-bootstrap interval | Validation | Locked test |
|---|---|---|
| Aligned corr | [0.752347, 0.870012] | [0.731500, 0.857791] |
| Same-lag d1 corr | [0.669601, 0.844462] | [0.667369, 0.822241] |
| Morphology | [0.710383, 0.857712] | [0.701758, 0.838364] |
| HR MAE (bpm) | [0.624288, 1.751442] | [0.602923, 1.665449] |

Validation attention mean is forehead 0.329546, left_cheek 0.343684,
right_cheek 0.326771; median is 0.337392/0.338291/0.325288. Highest-attention
shares are 46.67%/37.14%/16.19%, and 0% of windows exceed max(beta) > 0.8.
All three ROIs are valid in 100% of validation windows. Mean prior weights
mostly divide among CHROM_WIN (~0.292–0.303), CHROM (~0.276–0.278), GREEN
(~0.195–0.196) and PBV (~0.180–0.190); POS_WIN/OMIT are ~0.028/~0.016.
These are descriptions, not causal or physiological explanations.

Scaled residual absolute mean is 0.101304 overall, and
0.103331/0.100304/0.100278 per ROI. Maximum is 0.977908 overall, with per-ROI
maxima 0.977908/0.910606/0.827642. All diagnostics are finite; no physiological
magnitude cutoff was used. Validation subject aligned correlations span
0.651736–0.886651, d1 0.532993–0.878951, morphology 0.592364–0.882801,
HR MAE 0.457961–2.825147 and nRMSE 0.466127–0.830471. Per-subject values,
medians, quartiles and ranges are preserved in the full review.

The single locked-test evaluation consumed the reserved test on
**October 7, 2026 at 21:22:41 Asia/Tehran** (17:52:41 UTC). All eight
participants have finite main metrics and valid windows. HR is nearly
unchanged, mean correlations fall by about 0.02, and there is no broad
systematic failure. Test morphology median is 0.811647 (IQR 0.127137);
the weakest participant, subject26, has morphology 0.582345 and HR MAE
2.612390. This heterogeneity is retained without changing epoch, weights,
architecture, loss, preprocessing or split. Quantitative engineering reasons
are in `generalization_review.json` and the final review. **The test is
consumed and cannot be reused as unseen tuning evidence.**

Frozen identity:

- Checkpoint: `checkpoints/waveform_v1_official_ubfc/waveform_v1_frozen.pt`
- Frozen SHA256: `d6b2d1eeebc0153197a2a12da736b957b32d72ae00b22631e9afd0b2097131ff`
- Source candidate SHA256: `5dfe34aab74ebf994a343eab3d82dcf0c6c3c465aaf0182d041117d01b31f982`
- Manifest: `checkpoints/waveform_v1_official_ubfc/waveform_v1_freeze_manifest.json`
- Split canonical SHA256: `ad3eb8d642bf6bcdaf578688532cc3b411f5f81f6a50d13b71a1894872d17041`
- Split file-byte SHA256: `d123d65da6fd23234c01063ea2d18971039da01cd64a2449669d9f63a60b4a38`
- Freeze: October 7, 2026 at 21:26:20 Asia/Tehran (17:56:20 UTC).

The source and frozen states are bit-identical; optimizer state is excluded.
Both reconstructed models' complete outputs, and the inference API waveform,
were verified exactly equal on a real subject19 validation window. Evidence
is saved in `inference_smoke.json`.

`rppg_lab.waveform_inference.FrozenWaveformExtractor` validates frozen status,
manifest/model hash, config and ROI/prior/channel order. `prepare_window()`
reuses bounded Phase 1 RGB resampling, the training dataset's exact source
support/quality rule and `build_window_priors()` on only that window.
`infer_window()` returns the signed waveform, original video timestamps,
attention, prior weights, masks, source-valid fractions, residuals and model
identity. The unsupervised model quality head is omitted. Missing ROIs are
masked identically to training. Continuous overlap reconstruction is deferred;
do not concatenate overlapping independent windows. See README for usage.

Detailed review: `checkpoints/waveform_v1_official_ubfc/waveform_v1_review.json`
and `.md`. The pre-test review remains in `waveform_v1_validation_review.json`.
Validation/test artifacts live in `outputs/waveform_v1_official_validation.json`
and `outputs/waveform_v1_official_LOCKED_TEST.json`. Large local checkpoint,
cache and output artifacts remain ignored by Git.

## Final verification

All **138 tests** passed, including all original 113 plus 25 finalization and
inference regressions. `python -m compileall -q rppg_lab scripts tests` and
`git diff --check` passed. The full 30-epoch log, validation baseline artifact,
accepted pre-test review, single consumed locked test, freeze manifest and
real-window equality smoke artifact are preserved. Model/loss definitions,
trainer, dataset, Phase 1 extraction/preprocessing, BP modules and timing path
are unchanged from the training commit. Machine-readable final evidence is
in `checkpoints/waveform_v1_official_ubfc/final_verification.json`.

## Scientific boundaries

Frozen Waveform v1, if accepted, is intended for facial morphology extraction.
It is NOT approved as a source for face-hand inter-site timing. The BP timing
path remains conservative matched face/hand GREEN until learned-waveform
phase behavior is separately validated. BP code is unchanged in this task.

Neither acceptance nor freezing establishes clinical BP validity, clinical
PTT, exact facial/fingertip anatomical morphology equivalence, validated
dicrotic-notch recovery, reflection index, APG, hand-waveform generalization,
or BP morphology accuracy. A freeze establishes only a fixed, reproducible
facial rPPG extractor for the next BP feasibility stage.
