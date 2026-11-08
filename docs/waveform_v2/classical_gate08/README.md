# Gate 08 evidence and changed-file index

Read the [fixed protocol](../../WAVEFORM_V2_CLASSICAL_FEASIBILITY_GATE08_PROTOCOL.md)
and [scientific report](../../WAVEFORM_V2_CLASSICAL_FEASIBILITY_GATE08_REPORT.md).
This is exploratory evidence from ten exposed synthetic examples, with Train
primary and Val/Test separate. Training remains disabled.

## Committed metadata

| File | Purpose |
|---|---|
| [prediction_lock.json](prediction_lock.json) | Hashes/commit recorded before any actual reference-value read. |
| [provenance_and_validation.json](provenance_and_validation.json) | Ten unchanged source/cache identities, 20 input hashes, splits, fusion weights, code hashes, versions, runtime and memory. |
| [per_window_metrics.csv](per_window_metrics.csv) | All 480 intended cells: native/lag-diagnostic metrics, signed HR errors, selected lags/overlap, quality and failures. Undefined numeric metrics are empty fields. |
| [per_recording_metrics.csv](per_recording_metrics.csv) | All 240 recording/method/ROI conditions, each with two intended windows; MAE, RMSE, signed errors, correlations and counts. |
| [condition_metrics.json](condition_metrics.json) | All 72 split/method/ROI conditions, recording-balanced and pooled summaries, per-recording values and metric-specific finite denominators. |
| [quality_associations.json](quality_associations.json) | 384 Train-only descriptive Spearman associations using recording means; undefined ties/support retained. |
| [negative_control_metrics.csv](negative_control_metrics.csv) | All 384 cells from eight fixed synthetic controls, including failed/missing estimates. |
| [negative_control_summary.json](negative_control_summary.json) | Per-control and per-condition summaries with complete denominators; controls are manufactured, not participant data. |
| [synthetic_timing.json](synthetic_timing.json) | Seven Gate 03 transfer/boundary/fusion controls with passes, abstentions and an expected phase failure. |
| [exploratory_summary.json](exploratory_summary.json) | Intended counts, observed coverage/failures, native sign/boundary counts and research decision. |
| [validation_summary.json](validation_summary.json) | 175 guarded tests, syntax/CLI guards, post-run hash/aggregate checks and layout-only plot revision provenance. |
| [local_artifact_manifest.json](local_artifact_manifest.json) | Local files, sizes and SHA256; final plots are in plots/. Arrays/images are not committed. |

The quality and metrics tables are generated under the fixed protocol; no score
is a confirmed physiological-confidence or skin/hair label. Positive lag means
prediction follows reference. Lags maximize signed overlap correlation only as
a disclosed diagnostic and never modify saved estimates, HR or quality inputs.

## Local artifacts

Root: `outputs/waveform_v2_classical_feasibility_gate08_20261105_02/`.
Use the six `plots/*.png` files: two predeclared waveform examples,
all-Train HR pairs, native-correlation heatmap, quality/error scatter and negative
controls. The first run remains intact in `_01`; the final fresh run includes
the legend layout and unknown-clipping correction for missing controls. Actual
results/arrays are byte-identical across runs. `locked_predictions.npz` retains
all predictions and x tensors; `reference_windows.npz` retains original d_ppg
windows; `label_blind_predictions.json` preserves all pre-reference quality/hash
observations. The local packet includes the existing license and attribution.

Do not share SCAMPS-derived artifacts without retaining their R-UDA terms and
attribution. No source `.mat`, diagnostic image, array or model is added to Git.

## Exact changed-file inventory

All **21 files are additions** relative to Gate 07 parent
`4b3d01a3a8e7ef0354b9c5b5f73ad23df9f64675`: the twelve metadata files above,
this index, the two linked protocol/report documents, and six Python files:

| File | Role |
|---|---|
| `rppg_lab/waveform_v2_classical_study.py` | Unchanged-prior/fusion reuse, label-free descriptors, isolated scoring, balanced aggregation and synthetic controls. |
| `rppg_lab/waveform_v2_classical_cache.py` | Hash-verified Baseline A reuse and separate bounded d_ppg reads. |
| `rppg_lab/waveform_v2_classical_plots.py` | Fixed waveform examples and complete Train/control scientific figures. |
| `scripts/audit_waveform_v2_classical_feasibility.py` | Commit/provenance verification, all-recording prediction lock before scoring, fresh local artifacts and metrics. |
| `scripts/verify_waveform_v2_classical_safe.py` | Reviewed regression profile with neural/checkpoint/backward and private-I/O protection. |
| `tests/test_waveform_v2_classical_study.py` | Twenty-six deterministic tests of leakage, prior/preprocessing equality, signs, frequencies, missingness, controls and balanced metrics. |

No earlier tracked code, report, official split, checkpoint, loss, training file
or `.gitignore` changes. The two pre-existing untracked handoff/ZIP files remain
excluded. Preregistration and implementation commits precede actual label
scoring. Later commits adjust legend placement and retain unknown clipping for
missing control RGB, followed by tests and a fresh complete run. Their scope
and byte-equality checks are explicit in validation_summary.json.
