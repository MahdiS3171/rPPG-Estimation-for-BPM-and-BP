# Gate 09: frozen V1 versus every fixed classical condition

Date: 2026-11-08. Branch: `feature/waveform-v2-frozen-comparison-gate09`.
Stacked on Gate 08 / PR #5 at `bcd3f98dd65088d41c447a2651e145d70cfc8e05`.

## Decision and scientific interpretation

**GO for a separately designed, restricted training proof of concept.** This is
permission to design a new experiment, not authorization to execute training.
Matching/provenance checks pass, classical and frozen neural outputs contain
recoverable synthetic pulse information, and the frozen comparison gives a
concrete failure baseline. It does not demonstrate a generalization advantage.
The tiny exposed sample can justify a Train-only gradient/memorization/plumbing
investigation; it cannot validate Waveform v2 efficacy or real-world transfer.

Frozen V1 is not uniformly better than the fixed classical conditions. Primary
Train recording-balanced HR MAE is **15.575 bpm**, RMSE **27.233 bpm**, signed error
**+5.244 bpm**, native signed r **0.364**, and lag-optimized diagnostic r **0.548**.
It has lower MAE than 19/24 conditions and higher native r than 17/24; these are
correlated descriptive comparisons, not 24 independent tests or significance
claims. All 24 comparators remain below in fixed order. Five have lower MAE and
seven have higher native correlation. Forehead CHROM has native r=0.715 and
MAE=22.308; forehead GREEN has MAE=9.541 but r=-0.602, while forehead OMIT has
MAE=8.662 and r=-0.217. These show why native waveform recovery and spectral HR
must remain separate. No per-recording or overall selected baseline is proposed.

The already exposed Test recording is a substantial frozen-V1 failure:
MAE=46.518 bpm despite several fixed cheek/fusion conditions having small HR
error. The single Val recording has V1 MAE=0.210 bpm. Those two-recording outcomes
are descriptive appendices, not model-selection or independent test evidence.

## Frozen design, identity and exact matching

The [protocol](WAVEFORM_V2_FROZEN_COMPARISON_GATE09_PROTOCOL.md) was committed at
`3c5efa8` before any Gate 09 neural forward, including manufactured tests.
Implementation/tests were committed at `87a9e81` before actual inference.
`git fetch origin --prune` confirmed the reviewed parent is still current.
PRs [1](https://github.com/MahdiS3171/rPPG-Estimation-for-BPM-and-BP/pull/1),
[2](https://github.com/MahdiS3171/rPPG-Estimation-for-BPM-and-BP/pull/2),
[3](https://github.com/MahdiS3171/rPPG-Estimation-for-BPM-and-BP/pull/3),
[4](https://github.com/MahdiS3171/rPPG-Estimation-for-BPM-and-BP/pull/4) and
[5](https://github.com/MahdiS3171/rPPG-Estimation-for-BPM-and-BP/pull/5) remain
open/draft/unmerged, with no review/discussion comments at inspection.
Their local implementations, protocols and reports were available in the
inherited stack and reviewed for the applicable input/loading/scoring contracts.

Official checkpoint SHA256:
`d6b2d1eeebc0153197a2a12da736b957b32d72ae00b22631e9afd0b2097131ff`.
Companion manifest SHA256:
`da5ba397b3be389deec8821f8ea16944b5f5b93e2834bb79c3c147eb3d1f89dc`.
Both were verified before loading; the existing `FrozenWaveformExtractor` loader
validates frozen status, provenance/configuration and strict state reconstruction.
Architecture: `MultiROIPriorResidualWaveformNet`, width 32, three encoder blocks,
dropout 0.2 (disabled by eval), residual scale 0.1. ROI order is forehead,
left_cheek, right_cheek. Nine channels are RGB_R/G/B then GREEN, CHROM, CHROM_WIN,
PBV, POS_WIN, OMIT; sample rate 30 Hz, window length 300. Every forward uses CPU,
one Torch thread, deterministic algorithms, eval and `torch.inference_mode()`.
No wrapper/adaptation, optimizer, backward, calibration or GPU execution.

Actual input tensors came directly from Gate 08's pinned prediction archive;
all 20 match the original Gate 06 tensor hashes. Original quality/validity/prior
masks were restored directly from Gate 06 metadata, never inferred from labels.
No actual ROI extraction, landmark processing, prior rebuilding or timestamp
construction was run. Windows retain original indices [0,300), [300,600) and
the documented nominal index/30 axis, which is not a verified physical clock.
The ten source .mat files were read only for before/after checksums; no HDF5
video, segmentation, raw PPG or other physiological variable was decoded.
No private Dataset/, collected subjects, reserved S007-S011, UBFC or rPPG-10
recording/cache was read. Historical conclusions use saved reports only.

Exactly **20 actual neural forward calls** produce 20 original window waveforms.
All are finite, nonflat and numerically scorable: Train 16/16, Val 2/2, Test 2/2.
There are no actual prediction/reference failures. All 480 fixed classical cells
retain exact prediction hashes and reproduce every original Gate 08 metric to
1e-12 tolerance. Every raw-reference hash agrees across its 24 comparisons.
All neural outputs, allocation values, masks and indices were saved/hashed before
the reference archive was opened for values. Checkpoint/model state, source files,
all parent tracked files, Gate 08 artifacts and prediction locks are unchanged.
See [identity](waveform_v2/frozen_comparison_gate09/model_identity.json),
[prediction lock](waveform_v2/frozen_comparison_gate09/prediction_lock.json) and
[full provenance/hashes](waveform_v2/frozen_comparison_gate09/provenance_and_validation.json).

## Complete fixed comparison tables

Each recording contributes its two original windows, then receives equal weight.
RMSE is sqrt(mean recording MSE), not mean recording RMSE. All conditions have
16/16 Train windows and 8/8 recordings, or 2/2 windows and 1/1 recording in each
Val/Test table. The finite counts, intended counts, failure reasons and per-record
means remain explicit in the [condition JSON](waveform_v2/frozen_comparison_gate09/condition_metrics.json).

HR uses unchanged Gate 08 Welch defaults, 0.7-3.5 Hz and raw same-index d_ppg.
Signed error is prediction minus reference spectral HR. Native r0 is signed
Pearson correlation against per-window standardized raw d_ppg. No prediction
inversion/shift/extra filtering is performed. Classical-prior filtering and raw
reference filtering differ. The matched-band-reference r is secondary only.
Aligned r maximizes signed correlation over integer +/-15 samples; selected lags
and boundary flags are diagnostic, never applied to a saved or plotted output.

### Train: primary, eight exposed recordings

| Fixed condition | MAE bpm | RMSE bpm | Bias bpm | Native r0 | Lag diagnostic r | Matched-band ref r0 |
|---|---:|---:|---:|---:|---:|---:|
| frozen_v1 | 15.575 | 27.233 | 5.244 | 0.364 | 0.548 | 0.360 |
| forehead/GREEN | 9.541 | 23.331 | -7.065 | -0.602 | 0.458 | -0.610 |
| forehead/CHROM | 22.308 | 46.254 | 20.512 | 0.715 | 0.715 | 0.721 |
| forehead/CHROM_WIN | 22.338 | 46.322 | 20.604 | 0.709 | 0.709 | 0.713 |
| forehead/PBV | 34.638 | 56.319 | 27.539 | -0.445 | 0.361 | -0.447 |
| forehead/POS_WIN | 19.219 | 34.978 | -15.660 | 0.365 | 0.447 | 0.377 |
| forehead/OMIT | 8.662 | 16.630 | 4.398 | -0.217 | 0.526 | -0.223 |
| left_cheek/GREEN | 29.346 | 42.661 | -18.667 | -0.406 | 0.375 | -0.418 |
| left_cheek/CHROM | 15.408 | 25.854 | 2.755 | 0.054 | 0.604 | 0.051 |
| left_cheek/CHROM_WIN | 15.390 | 25.877 | 2.751 | 0.052 | 0.605 | 0.048 |
| left_cheek/PBV | 11.476 | 22.016 | 4.895 | -0.110 | 0.528 | -0.105 |
| left_cheek/POS_WIN | 23.996 | 35.424 | -7.308 | 0.182 | 0.385 | 0.199 |
| left_cheek/OMIT | 31.590 | 44.134 | 10.130 | -0.182 | 0.517 | -0.191 |
| right_cheek/GREEN | 26.270 | 44.800 | 0.688 | -0.352 | 0.384 | -0.356 |
| right_cheek/CHROM | 19.560 | 37.471 | 0.937 | 0.532 | 0.681 | 0.533 |
| right_cheek/CHROM_WIN | 18.935 | 36.443 | 1.639 | 0.531 | 0.673 | 0.530 |
| right_cheek/PBV | 21.984 | 37.574 | 0.867 | -0.447 | 0.473 | -0.447 |
| right_cheek/POS_WIN | 32.511 | 39.379 | -23.924 | 0.078 | 0.312 | 0.080 |
| right_cheek/OMIT | 21.809 | 32.264 | 10.634 | 0.261 | 0.556 | 0.271 |
| equal_roi_average/GREEN | 19.574 | 43.471 | 2.456 | -0.612 | 0.442 | -0.623 |
| equal_roi_average/CHROM | 19.160 | 32.770 | -1.181 | 0.556 | 0.572 | 0.552 |
| equal_roi_average/CHROM_WIN | 20.874 | 33.820 | -3.689 | 0.553 | 0.567 | 0.547 |
| equal_roi_average/PBV | 24.363 | 38.679 | 1.248 | -0.477 | 0.337 | -0.471 |
| equal_roi_average/POS_WIN | 23.548 | 36.668 | -16.175 | 0.350 | 0.438 | 0.367 |
| equal_roi_average/OMIT | 25.730 | 40.191 | 5.592 | -0.075 | 0.519 | -0.078 |

### Val: descriptive only, one exposed recording

| Fixed condition | MAE bpm | RMSE bpm | Bias bpm | Native r0 | Lag diagnostic r | Matched-band ref r0 |
|---|---:|---:|---:|---:|---:|---:|
| frozen_v1 | 0.210 | 0.210 | -0.006 | 0.716 | 0.716 | 0.737 |
| forehead/GREEN | 42.424 | 59.632 | -42.424 | -0.318 | 0.384 | -0.325 |
| forehead/CHROM | 0.212 | 0.219 | -0.212 | 0.903 | 0.903 | 0.929 |
| forehead/CHROM_WIN | 0.175 | 0.175 | -0.175 | 0.911 | 0.911 | 0.933 |
| forehead/PBV | 0.395 | 0.415 | -0.395 | -0.784 | 0.694 | -0.806 |
| forehead/POS_WIN | 0.443 | 0.447 | 0.057 | 0.589 | 0.589 | 0.611 |
| forehead/OMIT | 0.211 | 0.249 | 0.211 | -0.732 | 0.662 | -0.754 |
| left_cheek/GREEN | 0.206 | 0.206 | 0.004 | -0.648 | 0.583 | -0.641 |
| left_cheek/CHROM | 0.210 | 0.221 | 0.070 | -0.882 | 0.774 | -0.935 |
| left_cheek/CHROM_WIN | 0.192 | 0.219 | 0.106 | -0.881 | 0.776 | -0.935 |
| left_cheek/PBV | 0.473 | 0.483 | 0.098 | 0.759 | 0.759 | 0.820 |
| left_cheek/POS_WIN | 0.216 | 0.217 | -0.216 | 0.737 | 0.737 | 0.753 |
| left_cheek/OMIT | 0.111 | 0.141 | -0.087 | -0.901 | 0.803 | -0.941 |
| right_cheek/GREEN | 37.056 | 51.764 | -37.056 | -0.464 | 0.427 | -0.475 |
| right_cheek/CHROM | 0.534 | 0.542 | 0.534 | -0.686 | 0.670 | -0.730 |
| right_cheek/CHROM_WIN | 0.331 | 0.359 | 0.331 | -0.643 | 0.662 | -0.690 |
| right_cheek/PBV | 0.483 | 0.483 | 0.483 | 0.732 | 0.732 | 0.776 |
| right_cheek/POS_WIN | 0.311 | 0.337 | 0.129 | 0.829 | 0.829 | 0.847 |
| right_cheek/OMIT | 0.312 | 0.413 | 0.271 | 0.829 | 0.829 | 0.853 |
| equal_roi_average/GREEN | 0.418 | 0.431 | -0.418 | -0.778 | 0.656 | -0.784 |
| equal_roi_average/CHROM | 0.858 | 0.913 | 0.858 | -0.591 | 0.617 | -0.655 |
| equal_roi_average/CHROM_WIN | 0.676 | 0.817 | 0.676 | -0.547 | 0.593 | -0.616 |
| equal_roi_average/PBV | 1.001 | 1.081 | 1.001 | 0.550 | 0.562 | 0.613 |
| equal_roi_average/POS_WIN | 0.238 | 0.239 | -0.024 | 0.874 | 0.874 | 0.897 |
| equal_roi_average/OMIT | 0.579 | 0.647 | -0.287 | -0.636 | 0.593 | -0.665 |

### Test: descriptive only, one exposed recording

| Fixed condition | MAE bpm | RMSE bpm | Bias bpm | Native r0 | Lag diagnostic r | Matched-band ref r0 |
|---|---:|---:|---:|---:|---:|---:|
| frozen_v1 | 46.518 | 49.984 | -46.518 | 0.258 | 0.372 | 0.221 |
| forehead/GREEN | 59.109 | 64.344 | -59.109 | -0.321 | 0.369 | -0.342 |
| forehead/CHROM | 68.835 | 69.485 | -68.835 | 0.109 | 0.265 | 0.100 |
| forehead/CHROM_WIN | 14.110 | 18.821 | -14.110 | 0.257 | 0.375 | 0.237 |
| forehead/PBV | 68.835 | 69.470 | -68.835 | -0.103 | 0.245 | -0.094 |
| forehead/POS_WIN | 35.692 | 50.280 | -35.692 | -0.288 | 0.228 | -0.299 |
| forehead/OMIT | 35.689 | 50.441 | -35.689 | 0.405 | 0.423 | 0.431 |
| left_cheek/GREEN | 0.503 | 0.516 | -0.116 | -0.879 | 0.744 | -0.911 |
| left_cheek/CHROM | 0.287 | 0.295 | 0.066 | 0.754 | 0.754 | 0.768 |
| left_cheek/CHROM_WIN | 0.277 | 0.309 | 0.137 | 0.741 | 0.741 | 0.754 |
| left_cheek/PBV | 0.383 | 0.392 | 0.083 | -0.676 | 0.605 | -0.688 |
| left_cheek/POS_WIN | 70.850 | 71.259 | -70.850 | -0.372 | 0.412 | -0.376 |
| left_cheek/OMIT | 0.409 | 0.439 | -0.158 | 0.878 | 0.878 | 0.901 |
| right_cheek/GREEN | 16.437 | 23.102 | -16.234 | -0.623 | 0.583 | -0.640 |
| right_cheek/CHROM | 0.157 | 0.189 | 0.157 | 0.824 | 0.824 | 0.827 |
| right_cheek/CHROM_WIN | 0.128 | 0.140 | 0.058 | 0.819 | 0.819 | 0.821 |
| right_cheek/PBV | 0.230 | 0.289 | 0.230 | -0.800 | 0.633 | -0.800 |
| right_cheek/POS_WIN | 70.358 | 74.133 | -70.358 | -0.284 | 0.362 | -0.297 |
| right_cheek/OMIT | 0.076 | 0.076 | -0.076 | 0.864 | 0.864 | 0.880 |
| equal_roi_average/GREEN | 0.280 | 0.355 | -0.219 | -0.822 | 0.711 | -0.854 |
| equal_roi_average/CHROM | 0.232 | 0.235 | -0.040 | 0.784 | 0.784 | 0.786 |
| equal_roi_average/CHROM_WIN | 0.204 | 0.252 | -0.148 | 0.804 | 0.804 | 0.801 |
| equal_roi_average/PBV | 0.237 | 0.238 | -0.020 | -0.758 | 0.637 | -0.757 |
| equal_roi_average/POS_WIN | 0.313 | 0.358 | 0.175 | -0.585 | 0.456 | -0.602 |
| equal_roi_average/OMIT | 0.043 | 0.049 | -0.023 | 0.872 | 0.872 | 0.898 |

### All paired Train differences, V1 minus each fixed comparator

Negative MAE/RMSE difference and positive native-r difference favor V1 only
in this descriptive sample. Every pair below has all 16 matched windows/eight
recordings. Full bias/MSE/lag/reference-band differences, metric-specific pair
counts and separate Val/Test comparisons are in the
[paired CSV](waveform_v2/frozen_comparison_gate09/paired_comparisons.csv) and
[paired JSON](waveform_v2/frozen_comparison_gate09/paired_comparisons.json).

| Fixed comparator | Delta MAE bpm | Delta RMSE bpm | Delta native r0 | Delta diagnostic aligned r |
|---|---:|---:|---:|---:|
| forehead/GREEN | 6.033 | 3.902 | 0.965 | 0.090 |
| forehead/CHROM | -6.734 | -19.021 | -0.352 | -0.167 |
| forehead/CHROM_WIN | -6.764 | -19.088 | -0.345 | -0.161 |
| forehead/PBV | -19.063 | -29.086 | 0.809 | 0.187 |
| forehead/POS_WIN | -3.644 | -7.745 | -0.002 | 0.101 |
| forehead/OMIT | 6.912 | 10.603 | 0.581 | 0.022 |
| left_cheek/GREEN | -13.772 | -15.428 | 0.769 | 0.173 |
| left_cheek/CHROM | 0.166 | 1.379 | 0.309 | -0.056 |
| left_cheek/CHROM_WIN | 0.185 | 1.356 | 0.311 | -0.057 |
| left_cheek/PBV | 4.099 | 5.217 | 0.474 | 0.020 |
| left_cheek/POS_WIN | -8.421 | -8.191 | 0.181 | 0.163 |
| left_cheek/OMIT | -16.015 | -16.901 | 0.546 | 0.031 |
| right_cheek/GREEN | -10.695 | -17.567 | 0.716 | 0.164 |
| right_cheek/CHROM | -3.986 | -10.238 | -0.169 | -0.133 |
| right_cheek/CHROM_WIN | -3.360 | -9.210 | -0.168 | -0.125 |
| right_cheek/PBV | -6.409 | -10.341 | 0.810 | 0.075 |
| right_cheek/POS_WIN | -16.937 | -12.146 | 0.286 | 0.236 |
| right_cheek/OMIT | -6.235 | -5.031 | 0.103 | -0.008 |
| equal_roi_average/GREEN | -3.999 | -16.238 | 0.976 | 0.106 |
| equal_roi_average/CHROM | -3.585 | -5.537 | -0.192 | -0.024 |
| equal_roi_average/CHROM_WIN | -5.299 | -6.587 | -0.190 | -0.019 |
| equal_roi_average/PBV | -8.789 | -11.445 | 0.840 | 0.211 |
| equal_roi_average/POS_WIN | -7.973 | -9.435 | 0.014 | 0.110 |
| equal_roi_average/OMIT | -10.155 | -12.958 | 0.438 | 0.029 |

## Recording heterogeneity and all 20 intended neural windows

These are ten recordings with correlated two-window measurements, not 20
independent participants. No dataset participant-identity independence is inferred.
The complete [250 recording-condition rows](waveform_v2/frozen_comparison_gate09/per_recording_metrics.csv)
and [500 window-condition rows](waveform_v2/frozen_comparison_gate09/per_window_metrics.csv)
include all fixed baselines and explicit failure/finite counts.

| Recording | Split | V1 MAE bpm | V1 RMSE bpm | V1 bias bpm | Native r0 | Diagnostic aligned r |
|---|---|---:|---:|---:|---:|---:|
| P000001.mat | Train | 0.200 | 0.273 | 0.186 | 0.743 | 0.743 |
| P000002.mat | Train | 32.820 | 46.274 | 32.621 | -0.155 | 0.338 |
| P000003.mat | Train | 10.516 | 10.731 | -10.516 | 0.048 | 0.175 |
| P000006.mat | Train | 49.786 | 49.791 | 49.786 | 0.484 | 0.484 |
| P000007.mat | Train | 29.920 | 34.592 | -29.920 | 0.112 | 0.215 |
| P000008.mat | Train | 0.637 | 0.775 | -0.441 | -0.168 | 0.583 |
| P000009.mat | Train | 0.493 | 0.493 | 0.010 | 0.921 | 0.921 |
| P000010.mat | Train | 0.225 | 0.261 | 0.225 | 0.925 | 0.925 |
| P000004.mat | Val | 0.210 | 0.210 | -0.006 | 0.716 | 0.716 |
| P000005.mat | Test | 46.518 | 49.984 | -46.518 | 0.258 | 0.372 |

| Recording | Split | Start | Ref HR | V1 HR | Signed error bpm | Native r0 | Aligned r | Lag ms | Boundary | Harmonic indicator |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| P000001.mat | Train | 0 | 130.053 | 130.438 | 0.386 | 0.696 | 0.696 | 0.000 | False | none |
| P000001.mat | Train | 300 | 129.017 | 129.003 | -0.014 | 0.790 | 0.790 | 0.000 | False | none |
| P000002.mat | Train | 0 | 54.123 | 53.925 | -0.198 | -0.192 | 0.492 | 500.000 | True | none |
| P000002.mat | Train | 300 | 53.155 | 118.597 | 65.441 | -0.117 | 0.185 | 500.000 | True | other discrepancy |
| P000003.mat | Train | 0 | 122.530 | 109.876 | -12.654 | 0.182 | 0.213 | -500.000 | True | other discrepancy |
| P000003.mat | Train | 300 | 121.985 | 113.607 | -8.379 | -0.087 | 0.137 | 400.000 | False | other discrepancy |
| P000006.mat | Train | 0 | 49.297 | 99.789 | 50.492 | 0.471 | 0.471 | 0.000 | False | suspected double |
| P000006.mat | Train | 300 | 49.656 | 98.735 | 49.079 | 0.496 | 0.496 | 0.000 | False | suspected double |
| P000007.mat | Train | 0 | 131.165 | 83.884 | -47.280 | 0.160 | 0.273 | 500.000 | True | other discrepancy |
| P000007.mat | Train | 300 | 132.091 | 119.532 | -12.559 | 0.064 | 0.157 | -400.000 | False | other discrepancy |
| P000008.mat | Train | 0 | 96.407 | 95.329 | -1.078 | -0.221 | 0.596 | 233.333 | False | none |
| P000008.mat | Train | 300 | 97.217 | 97.413 | 0.196 | -0.115 | 0.570 | -433.333 | False | none |
| P000009.mat | Train | 0 | 124.855 | 124.372 | -0.483 | 0.925 | 0.925 | 0.000 | False | none |
| P000009.mat | Train | 300 | 126.022 | 126.526 | 0.504 | 0.916 | 0.916 | 0.000 | False | none |
| P000010.mat | Train | 0 | 82.035 | 82.393 | 0.358 | 0.923 | 0.923 | 0.000 | False | none |
| P000010.mat | Train | 300 | 81.888 | 81.980 | 0.091 | 0.927 | 0.927 | 0.000 | False | none |
| P000004.mat | Val | 0 | 136.581 | 136.785 | 0.204 | 0.702 | 0.702 | 0.000 | False | none |
| P000004.mat | Val | 300 | 135.422 | 135.206 | -0.216 | 0.729 | 0.729 | 0.000 | False | none |
| P000005.mat | Test | 0 | 142.377 | 114.150 | -28.227 | 0.087 | 0.272 | 500.000 | True | other discrepancy |
| P000005.mat | Test | 300 | 141.079 | 76.271 | -64.808 | 0.428 | 0.472 | 33.333 | False | suspected half |

P000001, P000009 and P000010 show good native recovery with small HR errors.
P000002/P000003/P000007 have weak or negative native correlations; P000006 has
positive native r around 0.48 but selects approximately twice the reference HR
in both windows. The two predeclared double-peak flags occur there, with errors
+50.49/+49.08 bpm. Five other Train windows have >6 bpm peak discrepancies.
Test P000005 second window has a suspected half-frequency selection. No harmonic
peak was repaired. The 2x/0.5x +/-6-bpm flags are hypotheses relative to d_ppg's
selected spectral peak, not verified fundamental labels. Other discrepancies
can include higher harmonics; raw-reference power fractions are retained.
Ten-second support gives approximately 6-bpm spectral resolution; padding does
not create independent frequency resolution. Reference HR spans 49.30-142.38 bpm.

Five of 16 Train V1 windows have negative native r. Diagnostic alignment raises
Train r from 0.364 to 0.548 (mean gain 0.184); four optima hit +/-500 ms. In
P000008, recording-native r=-0.168 becomes aligned r=0.583 while HR MAE is only
0.637 bpm. That apparent waveform improvement depends on reference-optimized
lag and can compensate phase/polarity, not demonstrate timing accuracy. Val has
zero alignment gain; Test gains 0.115 and has one of two optima at the boundary.
All original waveforms remain unshifted.

## Descriptive ROI/prior allocation and component behavior

Allocation is not a validated explanation, physiological confidence or semantic
skin detector. Train mean ROI attention is forehead 0.3380, left cheek 0.3291,
right cheek 0.3329. Effective prior weights sum ROI-attention times prior-weight;
Train means in fixed prior order are:

| GREEN | CHROM | CHROM_WIN | PBV | POS_WIN | OMIT |
|---:|---:|---:|---:|---:|---:|
| 0.1851 | 0.2728 | 0.2610 | 0.2362 | 0.0290 | 0.0158 |

Thus POS/OMIT receive little allocation and GREEN/PBV remain substantial, even
though the latter have negative native correlations in many conditions. This
may permit cancellation; weights alone cannot adjudicate physiological quality.
[600 component rows](waveform_v2/frozen_comparison_gate09/allocation_components.csv)
retain each input prior's metrics/weight, the equal-six-prior diagnostic mixture,
learned prior mixture, ROI prior-plus-residual, and residual-only diagnostic.
The latter four are descriptive decompositions, never added selected comparators.

| Train ROI | Equal-six prior r0 | Learned prior r0 | Prior plus residual r0 | Equal-six MAE | Learned-prior MAE | Prior plus residual MAE |
|---|---:|---:|---:|---:|---:|---:|
| forehead | 0.283 | 0.595 | 0.583 | 34.918 | 13.360 | 12.576 |
| left_cheek | -0.152 | -0.115 | -0.100 | 18.212 | 26.151 | 14.110 |
| right_cheek | 0.321 | 0.349 | 0.334 | 24.968 | 23.727 | 15.417 |

The changes differ by ROI/metric. Learned mixtures improve forehead native
correlation over the equal-six mixture while residuals slightly reduce native r
there; left-cheek native mixture r remains negative. Residuals lower Train cheek
HR error while slightly reducing right-cheek native correlation. These are
fixed-output arithmetic comparisons, not causal feature attribution or ablations.

P000005's first window illustrates failure to reject unsuitable components:
attention is 0.307/0.384/0.309. Left/right CHROM_WIN native r is 0.734/0.769 with
HR error 0.141/0.186 bpm, while forehead is weak. The left learned mixture has
MAE=0.893 bpm, but its prior-plus-residual output has MAE=45.820; right mixture
already has MAE=94.458. The final model error is -28.227 bpm. Moderate forehead
weight and added residuals do not rescue the good cheek priors. The forehead's
known hair concern does not become a confirmed skin label; skin remains UNKNOWN.
In contrast, Val's first-window forehead attention 0.421 accompanies positive
forehead morphology while the cheek learned mixtures are negative. V1 attention
is not uniformly adaptive to waveform suitability across these examples.

## Negative controls: declared expectations and observed limitations

All eight unchanged Gate 08 fixtures/seeds are retained, two windows each.
15/16 have a finite computed model output; 13/16 meet numerical scoring
eligibility. Constant RGB produces small finite edge/residual activity but is
withheld from pulse metrics; all-missing has no forward and explicit failure.
The `pulse_estimate_eligible` field is numerical eligibility only, never proof
that a pulse is physiological. Noise, drift, permutation and illumination remain
unvalidated even when their computed values are finite. All 16 rows and explicit
expectations appear in the [control CSV](waveform_v2/frozen_comparison_gate09/negative_control_metrics.csv).

| Fixed control | Numerical scorable / intended | Mean HR error bpm | Mean native r0 | Observed interpretation |
|---|---:|---:|---:|---|
| sinusoidal_positive | 2/2 | 0.023 | -0.977 | Correct spectral rate but native inversion; no sign repair. |
| constant_nonpulsatile | 0/2 | NA | NA | 0.0070 output std from edges/residual; FLAT_SOURCE_RGB both windows. |
| slow_illumination_drift | 2/2 | -2.110 | 0.044 | Nonphysiological input produces a spectral estimate; not success. |
| independent_noise | 2/2 | 33.921 | -0.206 | Finite false-pulse estimates; no physiological validation. |
| coherent_pulse_band_illumination | 2/2 | 0.009 | 0.911 | High manufactured correlation demonstrates confounding. |
| clipped_impulses | 2/2 | 0.306 | -0.930 | Native r remains negative and less close to -1 than clean positive. |
| missing_windows | 1/2 | -0.017 | -0.976 | Missing forehead gets zero weight; all missing has NO_USABLE_ROI. |
| time_permuted | 2/2 | 47.164 | 0.056 | Temporal structure destroyed but finite spectral peaks remain. |

The clean positive fixture has native r=-0.977, although HR error is about
0.02-0.06 bpm; this is spectral success with a native-waveform polarity failure.
Coherent 1.2-Hz illumination has r=0.910/0.911 and approximately 0.009-bpm HR
error against its manufactured sinusoid: that does not make it a physiological
pulse. Temporal permutation yields HR errors +61.66/+32.67 bpm. No learned
physiological rejection or validated confidence threshold is demonstrated.

## Synthetic timing preservation

The [timing JSON](waveform_v2/frozen_comparison_gate09/synthetic_timing.json)
contains all 16 audits: four injected scenarios, two windows, two reference
anchors. The nominal anchor is compared separately with the actual first-ROI
source (known relative delay 45 ms, or 130 ms for the differential-input case).
The heterogeneous mixture has no single anatomical source phase. No independent
window outputs were concatenated or shifted.

| Synthetic case | Window start | Nominal-anchor audit | Actual-first-ROI audit |
|---|---:|---|---|
| baseline | 0 | ABSTAIN_UNVERIFIED | ABSTAIN_UNVERIFIED |
| baseline | 300 | ABSTAIN_UNVERIFIED | ABSTAIN_UNVERIFIED |
| common_input_delay_100ms | 0 | ABSTAIN_UNVERIFIED | ABSTAIN_UNVERIFIED |
| common_input_delay_100ms | 300 | ABSTAIN_UNVERIFIED | ABSTAIN_UNVERIFIED |
| differential_second_stream_85ms | 0 | ABSTAIN_UNVERIFIED | ABSTAIN_UNVERIFIED |
| differential_second_stream_85ms | 300 | ABSTAIN_UNVERIFIED | ABSTAIN_UNVERIFIED |
| heterogeneous_roi_phases | 0 | ABSTAIN_UNVERIFIED | ABSTAIN_UNVERIFIED |
| heterogeneous_roi_phases | 300 | FAIL | FAIL |

There are **zero passing neural timing audits**, 14 ABSTAIN_UNVERIFIED and two
FAIL results; anchor variants reuse the same outputs and are not independent
replicates. Abstaining component estimators report 24 polarity-ambiguous and two
half-window-unstable estimates. In heterogeneous phases, second-window estimates
show about -333/-337 ms source-anchor shifts, despite differential-delay error
about -2.445 ms. This is a failure of the predefined camera-anchor criterion;
it is not an anatomical measurement, and the native polarity/periodic ambiguity
limits interpreting any selected lag as a unique signal delay.

The six analytical guard controls yield identity PASS, common 100-ms latency
FAIL for camera anchoring, differential 85-ms extra latency FAIL, polarity/noise
ABSTAIN, and morphology-change FAIL. They confirm that preserving a relative
lag can hide a common time shift. All retained subframe values have at least
33.33-ms source-sample uncertainty. V1's lag-tolerant loss and a high aligned r
do not establish camera-time timing accuracy. SCAMPS lags are not face-hand PTT;
no BP estimate or physiological timing conclusion is made.

## Historical results remain separate

The [saved UBFC finalization report](WAVEFORM_V1_FINALIZATION.md) records
subject-balanced locked-test HR MAE=1.096377 bpm and aligned waveform r=0.799999
(238 windows from eight participants). This supported that original dataset's
frozen morphology evaluation. Gate 09 does not revoke or overwrite it. Its
aligned face/contact-PPG metrics, selected dataset population and overlapping
window structure differ from SCAMPS native-zero-lag nominal-grid metrics.
Their numerical values are not interchangeable estimates of one population.

The [saved rPPG-10 report](WAVEFORM_V1_RPPG10_EXTERNAL_VALIDATION.md) already
showed limited transfer: subject-balanced V1 HR MAE=15.780 bpm versus fixed
multi-ROI GREEN=7.168, 26 subjects and 2,830 valid ECG pairs out of 3,094 scheduled
windows. Its ECG-derived HR, acquisition/synchronization QC and mean-subject
RMSE convention differ from this study. ECG is not an optical morphology target.
The historical evidence plus SCAMPS supports dataset-dependent frozen behavior,
not a universal neural advantage. Neither historical dataset was rerun/read.

## Verification, artifacts and reproducibility

The inference-only profile passes **37 tests**, and the inherited reviewed
Gate 08 profile passes **175 tests**: zero failures/errors/runtime skips/protected
I/O attempts. The earlier 36-test run also passed before adding the explicit
source/checkpoint mutation test. Backward, autograd and optimizer steps, and CUDA
entry points are actively blocked. Checkpoint identity, strict loading,
input/mask/index integrity, no gradients, deterministic fixture inference,
reference isolation, sign, harmonics, failure denominators, RMSE balancing,
source/state immutability, output safety and timing limitations are covered.

A separate saved-result verifier disables all neural calls/checkpoint loads,
checks all 32 artifact hashes and all 15 PNGs, reproduces all 500 scoring rows,
75 condition summaries and 72 paired comparisons, and confirms original inputs,
masks, indices and locks. No repeat actual-data inference was needed. Code syntax
and `git diff --check` pass. CLI refuses an existing output before model loading.
All added paths and final hash evidence are indexed in
[verification](waveform_v2/frozen_comparison_gate09/verification_summary.json) and
[exact changed-file inventory](waveform_v2/frozen_comparison_gate09/changed_files.txt).

Commands from repository root:

```powershell
.\.venv\Scripts\python.exe -m scripts.verify_waveform_v2_frozen_safe
.\.venv\Scripts\python.exe -m scripts.verify_waveform_v2_classical_safe
.\.venv\Scripts\python.exe -m scripts.audit_waveform_v2_frozen_comparison --out-dir outputs/waveform_v2_frozen_comparison_gate09_20261108_01
.\.venv\Scripts\python.exe -m scripts.verify_waveform_v2_frozen_results outputs/waveform_v2_frozen_comparison_gate09_20261108_01
git diff bcd3f98 --check
```

The 20 forwards total **0.1918 s**
(mean 9.59 ms/window, including
first-call overhead). Inference/scoring/controls/plots and post-run source checks
total **89.24 s**
after initial input/source verification; interpreter startup/model loading and
initial checks are excluded. Tracemalloc peak is
**42.12 MiB**; process peak working set
**924.46 MiB**, final RSS
**911.04 MiB**. Process memory includes
Torch/runtime/plotting, not isolated model tensor storage. The installed Torch
build supports CUDA but no GPU path was used. Versions are preserved in provenance.

Local arrays and 15 diagnostic PNGs remain in the ignored directory
`outputs/waveform_v2_frozen_comparison_gate09_20261108_01/`. Git contains only new
code/tests, protocol/report and small metrics/provenance metadata; no parent
tracked path was edited. The local artifact manifest pins every array/image.
The plots include all ten recordings' native windows, fixed first-window
P000001/P000003 comparisons against every baseline, full Train table, HR/native/
lag diagnostics and all eight controls. Display-only z scaling preserves sign
and original axes and never enters scoring. Three overview/control plots and
both predetermined baseline examples were visually checked for layout and axes.
SCAMPS attribution and research-use license are retained from the prior gates;
UNKNOWN skin labels and UNVERIFIED segmentation/timebase claims are unchanged.

## Conditions for a separate proof-of-concept design

A new proposal should fix its Train-only objective/polarity policy, 16-window/
eight-recording scope, CPU/GPU compute cap and stopping criteria before any
training authorization. Include the full frozen V1/classical table, constant/
noise/illumination/permutation controls, failed-window denominators and native
zero-lag scoring. Make HR/harmonic and residual limitations explicit; do not
optimize reference lags or reinterpret weights as validated quality. Decide
missingness/quality handling without target leakage. This sample can test
learning mechanics and memorization only. Efficacy/generalization requires
additional permitted, properly grouped data and an independently specified
held-out evaluation; clinical/timing claims require separate evidence.

**GO for a separately designed, restricted training proof of concept.**
Training remains unauthorized and disabled by this task.
