# Gate 09 frozen comparison evidence

Start with the [scientific report](../../WAVEFORM_V2_FROZEN_COMPARISON_GATE09_REPORT.md)
and [fixed protocol](../../WAVEFORM_V2_FROZEN_COMPARISON_GATE09_PROTOCOL.md).

- `per_window_metrics.csv`: 500 rows = 20 V1 + 480 unchanged fixed baseline cells.
- `per_recording_metrics.csv`: 250 rows; `condition_metrics.json`: 75 split/condition groups with all intended and finite counts.
- `paired_comparisons.csv/json`: every comparator, split and primary/diagnostic metric with matched counts.
- `allocation_components.csv`: 600 descriptive component rows, never validated explanations.
- `negative_control_metrics.csv`: 16 intended neural control windows and explicit expectations; summary preserves failures.
- `synthetic_timing.json`: 16 neural audits and six analytical controls, including all abstentions/failures.
- `model_identity.json`, `prediction_lock.json`, `provenance_and_validation.json`: strict frozen identity, before-reference lock and full source/code/state hashes.
- `verification_summary.json` and `changed_files.txt`: tests, independent output verification and exact added-file list.
- `local_artifact_manifest.json`: checksums/paths for ignored arrays and all 15 PNGs, rooted at `outputs/waveform_v2_frozen_comparison_gate09_20261108_01` from the repository root.

No neural training, calibration, adaptation, GPU work, BP, source changes or private/reserved subject access. Train is primary; exposed Val/Test are separate and descriptive. GO concerns designing a separately authorized restricted proof of concept only. SCAMPS license/attribution from `../scamps/` continues to apply.
