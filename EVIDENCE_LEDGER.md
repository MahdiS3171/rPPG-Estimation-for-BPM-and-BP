# Research evidence ledger

Updated 2026-11-08. Software correctness, synthetic controls, exposed real data
and independent physiological validation are different evidence classes.
Repeated test execution does not add participants or independent observations.

| ID | Class / exposure | Evidence and location | What it establishes / does not establish |
|---|---|---|---|
| E001 | Historical real UBFC, consumed locked test | [V1 freeze](docs/WAVEFORM_V1_FINALIZATION.md): 41 participants; 26/7/8 split, epoch 25, locked-test HR MAE ~1.096 bpm, aligned r ~0.800. | Fixed useful UBFC morphology baseline; no native local timing or universal transfer claim. Scores reviewed from saved report, not newly rerun. |
| E002 | Historical external real, consumed | [rPPG-10 report](docs/WAVEFORM_V1_RPPG10_EXTERNAL_VALIDATION.md): V1 15.780 vs fixed GREEN fusion 7.168 bpm subject-balanced HR MAE. | Cross-domain fragility. ECG HR is not an optical waveform/morphology reference. No fresh evaluation this cycle. |
| E003 | Historical private S001–S006, exposed | [Pilot report](docs/PILOT6_PROTOCOL_AWARE_DELAY_FEASIBILITY.md): 422/599 finite delay windows; two strict overlapping windows, one independent; median ~2.94 ms vs declared 16.67-ms floor; zero timing-verified cuff links. | No repeatable resolved delay or BP result. Only tracked report read; participant media and workbooks not opened. |
| E004 | Synthetic SCAMPS, all ten examples exposed | [Admission](docs/WAVEFORM_V2_SCAMPS_ADMISSION_GATE05.md), [streaming](docs/WAVEFORM_V2_SCAMPS_STREAMING_GATE06.md). 600 video/reference vs 599 masks. | Loader/geometry/input support; mask temporal mapping unverified and no skin/physiological validation. No raw SCAMPS decoded this cycle. |
| E005 | Synthetic optical/processing controls | [Quality](docs/WAVEFORM_V2_OPTICAL_QUALITY_GATE07_REPORT.md), [classical](docs/WAVEFORM_V2_CLASSICAL_FEASIBILITY_GATE08_REPORT.md). Hair passes chroma; candidate trusted output abstains; patch/fusion phase shifts exist. | Signal quality/phase hazards. Historical reports and metrics, not newly annotated skin truth. |
| E006 | Synthetic frozen inference, exposed | [Gate 09](docs/WAVEFORM_V2_FROZEN_COMPARISON_GATE09_REPORT.md): Train V1 MAE 15.575 bpm, native signed r .364, aligned r .548; Val .210/Test 46.518 bpm. | Heterogeneous failure; no V2 training, independent held-out generalization or timing qualification. |
| E007 | New software replay, 2026-11-08 | [Audit receipt](docs/project_audit/20261108.json). 175 classical + 37 frozen-profile tests pass; saved verifier reproduces all 500 rows, 32 artifacts, 15 image-file integrity checks; no neural forward in replay. Broad profile: 326 pass, 43 exclusions. | Confirms reported metrics/software identities. Test executions overlap between profiles. PNG integrity is not a new visual review. Zero blocked Python I/O attempts; this is not an OS-level forensic guarantee. |
| E008 | New synthetic acquisition model | [Fixed protocol](docs/CAMERA_TIMING_IDENTIFIABILITY_PROTOCOL.md), [result report](docs/CAMERA_TIMING_IDENTIFIABILITY_REPORT.md), [machine results](docs/project_audit/camera_timing_results_20261108.json). 1,536 attempts, 1,533 finite diagnostic estimates, 3 abstentions. | Unknown row skew is observationally confounded with physiological delay under matched pulse shapes. Morphology can reverse lag sign; coherent light can yield finite estimates. Not a measured phone limitation or physiological validation. |
| E009 | Literature/access audit only | [Real-data plan](docs/REAL_WAVEFORM_BENCHMARK.md). PURE pending; MSPM multisite reference lead, agreement required; iBVP supervisor EULA and ~400-GB compressed distribution. | Candidate fitness and access constraints; none downloaded, admitted or consumed. |
| E010 | Integrity, checked this cycle | Checkpoint `d6b2d1eeebc0153197a2a12da736b957b32d72ae00b22631e9afd0b2097131ff`; manifest `da5ba397b3be389deec8821f8ea16944b5f5b93e2834bb79c3c147eb3d1f89dc`. | Exact frozen bytes match. Does not extend the model's scientific validity. |
| E011 | New public development sample; no efficacy evaluation | [MPU admission](docs/project_audit/mpu_development_admission_20261108.json), [timebase](docs/project_audit/mpu_development_timebase_20261108.json), [duplicate screen](docs/project_audit/mpu_frame_duplicates_20261108.json): one official 106.82-MB container-60-fps MP4 plus 0.54-MB CSV, both 34,496 samples. CC BY 4.0 and research-use notice. | PPG schema and regular container grid; all 34,496 frames decoded for hash-only duplicate screening, zero exact adjacent duplicates. No verified pairing, sensor latency, native sensor sampling or participant identity; no inference/training. Sample is exposed development material, never an untouched test. |
| E012 | New synthetic sampling/filter transfer | [Native-60 strategy](docs/NATIVE_60FPS_WAVEFORM_STRATEGY.md), [results](docs/project_audit/sample_rate_results_20261108.json). Existing 60-to-30 adapter preserves clean low band but attenuates higher content; default GREEN suppresses 8 Hz even at 60 Hz. | Motivates testing native-60 inputs plus appropriate targets/bandwidth. Does not demonstrate real waveform detail or a training improvement. |
| E013 | Remote clean-runner software validation | [Successful CI run](https://github.com/MahdiS3171/rPPG-Estimation-for-BPM-and-BP/actions/runs/42478042383) for implementation/documentation head `014ef78fb00a46ddb77468dd0d44cbe48f90e055`, in [integration PR #7](https://github.com/MahdiS3171/rPPG-Estimation-for-BPM-and-BP/pull/7). | CPU synthetic profiles, acquisition controls and compilation pass on GitHub's Ubuntu runner. No source data/checkpoint is provisioned there; this validates software portability, not physiology. Later ledger-only commits retain this result's exact revision. |

## Closed / unavailable evidence

S007–S011: no enumeration, inspection, hashing, decoding, inference or evaluation.
Original UBFC test and rPPG-10 external evidence are consumed; SCAMPS Val/Test
are exposed diagnostics. No newly untouched test, BP fit, clinical validation,
real camera calibration or new participant study was performed. Dataset access
and future splits must be recorded here before any evidence-consuming action.

## Reproduction receipts

Base audited tip: `bb5fb59683615489db0da451c52d9ff9de2fa49d`.
The local logs are in ignored `outputs/project_reassessment_20261108/`.
Original Gate 09 saved outputs remain in their manifest-declared directory;
replay uses `scripts.verify_waveform_v2_frozen_results` without model execution.
Acquisition results originated at clean commit `5edac8d` in
`outputs/camera_timing_identifiability/20261108_01/`. The committed summary
retains source hashes; the complete trial CSV remains local and is checksummed.
