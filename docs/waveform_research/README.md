# Current waveform research

Start with the [master roadmap](../../PROJECT_MASTER_PLAN.md),
[decisions](../../DECISION_LOG.md) and [evidence ledger](../../EVIDENCE_LEDGER.md).
The project-level audit is [here](../PROJECT_REASSESSMENT_2026-11-08.md).
The [campaign report](../WAVEFORM_DEVELOPMENT_CAMPAIGN_REPORT.md) consolidates
current conclusions and negative results.
The leading local model's [manifest](development_candidate_20261108.json) records
its configuration and checkpoint identity; the report includes the reproduction
command and preprocessing contract. It is a development artifact without a new
validation score.

| Investigation | Fixed design | Evidence |
|---|---|---|
| Two-source linear baseline | [Protocol](../TWO_SOURCE_WAVEFORM_DEVELOPMENT_PROTOCOL.md) | [Report](../TWO_SOURCE_WAVEFORM_DEVELOPMENT_REPORT.md), [receipt](two_source_development_20261108.json) |
| Adaptive hybrids, models and losses | [Screen protocol](../ADAPTIVE_WAVEFORM_SCREEN_PROTOCOL.md) | [Screen results](adaptive_screen_20261108.csv), [receipt](adaptive_screen_20261108.json) |
| Subject folds, source mixtures, context, capacity and seeds | [Confirmation protocol](../ADAPTIVE_WAVEFORM_CONFIRMATION_PROTOCOL.md) | [Confirmation results](adaptive_confirmation_20261108.csv), [receipt](adaptive_confirmation_20261108.json) |
| Adaptive versus fixed/global/quality gates; shuffled labels | [Matched ablations](../ADAPTIVE_WAVEFORM_ABLATION_PROTOCOL.md) | [Results](adaptive_ablations_20261108.csv), [receipt](adaptive_ablations_20261108.json) |
| Longer context, attention, spectral and time-varying gates | [Extension protocol](../WAVEFORM_ARCHITECTURE_EXTENSION_PROTOCOL.md) | [Screen](architecture_screen_20261108.csv), [confirmation](architecture_confirmation_20261108.csv), [receipt](architecture_confirmation_20261108.json) |
| Known shifts, missing inputs and equivalent 30/60-Hz grids | [Robustness protocol](../WAVEFORM_ROBUSTNESS_PROTOCOL.md) | [Results](waveform_robustness_20261108.csv), [receipt](waveform_robustness_20261108.json); no native-60 ground truth. |
| Region dropout and output consistency | [Follow-up/export protocol](../WAVEFORM_ROBUSTNESS_PROTOCOL.md) | [Training](roi_training_20261108.csv), [receipt](roi_training_20261108.json), [robustness](roi_robustness_20261108.csv) |
| Native 60-fps strategy | [Strategy and transfer audit](../NATIVE_60FPS_WAVEFORM_STRATEGY.md) | [Sampling transfer](../project_audit/sample_rate_results_20261108.json) |
| Physical acquisition timing | [Protocol](../CAMERA_TIMING_IDENTIFIABILITY_PROTOCOL.md) | [Report](../CAMERA_TIMING_IDENTIFIABILITY_REPORT.md) |

Reusable commands are listed in their protocols. The current pipeline is:

```text
allowed historical Train caches
  -> prepare_waveform_screen.py (one immutable derived input archive)
  -> run_waveform_screen.py (model/loss hypotheses)
  -> confirm_waveform_candidates.py (subject folds and seeds)
  -> ablate_waveform_hybrid.py (matched mechanism controls)
  -> verify_waveform_screen_results.py / analyze_waveform_candidates.py
```

Raw data, prepared arrays, checkpoints, complete predictions and participant
waveform figures remain local under ignored directories. This folder contains
aggregate research evidence. None of these development studies is an independent
third-dataset evaluation, physiological timing validation or BP validation.

Historical Gate artifacts remain at their original paths in `docs/waveform_v2/`
because saved experiments depend on those paths and byte identities. See
[repository organization](../REPOSITORY_ORGANIZATION.md) before moving old files.
