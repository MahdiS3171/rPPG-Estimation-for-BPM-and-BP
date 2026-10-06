# Implementation and verification record

This is an experimental research system and is not a medical device.

## Outcomes and reasons

| Change | Why | Preserved behavior | Verification |
| --- | --- | --- | --- |
| Shared video reader and typed data | Face/hand delay needs one explicit clock; anonymous arrays hid units/support. | Flat package and legacy face extractor return signature. | Monotonic/shape assertions, external-clock count tests, original HR CLI. |
| Detector adapters and region masks | Installed MediaPipe lacks legacy solutions; hand extraction was absent. | Original forehead/cheek polygons and equal-weight face aggregation in legacy HR; classic backend remains supported. | Actual Tasks model loading, frame overlays, hand loss/identity tests, synthetic video runs. |
| Gap-aware resampling and segment filters | Dropped ROIs and unrestricted interpolation could create false timing. | Existing classical formulas and methods; existing HR path remains available. | Known delays at both signs and after irregular-clock resampling; filter length/phase/gap tests. |
| Shared legacy Butterworth helper | Filtering was duplicated with equivalent valid-input mathematics. | Legacy float32 output, cutoffs, short-input fallback. | Regression against the previous explicit Butterworth/filtfilt formula. |
| Exact UBFC reference interpolation | Matching independently initialized grids by array truncation introduced a fractional time offset. | Same window/model/target HR interfaces. | Ramp reference regression with a 23 ms grid-origin mismatch. |
| Raw quality indicators and exclusion records | A waveform can exist but be unusable; missing/poor regions must remain visible. | Original heuristic/model quality remains in legacy experiments. | Missing-site/gap tests; accepted/excluded CSV rows and strict JSON nulls inspected. |
| Label-free BP features and cuff schemas | Cuff readings are intermittent, optical lag is not established PTT. | Existing exploratory array features and BP MLP retained. | Target-free feature schema; association containment, invalid-reference and timezone tests. |
| Mean/linear/ridge/forest baselines | Interpretable models should precede a large BP neural pipeline. | Existing neural architecture/checkpoint behavior unchanged. | Training-only schema/scaler tests, model save/load, nine synthetic-subject CLI run. |
| Subject and duplicate safeguards | Subject leakage or duplicated videos can invalidate BP experiments. | Existing participant-wise HR split seed behavior for valid unique subjects. | Disjoint/deterministic partitions; duplicate recording/hash/window rejection. |
| Reproducible artifacts and configs | Existing fixed filenames and scattered constants limited traceability. | Existing experiment CLIs; no raw data overwritten. | Exclusive run directories; input/config/model/source hashes, versions, git state, NPZ/CSV/JSON/debug artifacts. |

## Architecture

The original flat package is retained; no public imports were moved. Detectors
return `RegionObservation` geometry/confidence provenance. Mask selection and
RGB extraction are independent of the detector. `RGBTrace` keeps every decoded
timestamp and validity flag. `PhysiologicalSignal` asserts declared uniform
sample rate and matching lengths. The dual pipeline resamples once to a common
grid. Individual-site extraction supports HR when the other site is absent;
paired extraction uses identical segment boundaries for timing. The result
exposes face/hand RGB, rPPG, quality, shared clocks, HR, delay and feature windows.

Timing features: bounded cross-correlation lag with fractional peak interpolation,
phase delay at HR, multi-segment coherence, one-to-one beat matching, median/SD of
beat delay and matched-beat fraction. Morphology: native/normalized amplitude,
rise/fall duration, half-prominence width, normalized area, IBI, derivatives,
skewness and kurtosis. Quality and HR agreement remain separate raw fields.
Cross-correlation maxima at the search boundary are treated as unresolved.

No alignment rewrites the stored clock. Debug plots may shift one waveform for
display only. Filtering is offline zero phase and removes edge samples explicitly.
Large gaps stay NaN; interpolation flags and segment failure reasons are saved.
Material downsampling currently raises instead of silently aliasing RGB traces.

`Recording`/`BPReference`/`ReferenceAssociation` preserve repeated cuff readings,
timezone-aware measurement intervals, before/during/after relation, measurement
indices, devices and explicit nearby-window approximations. Unassociated windows
retain features but no BP labels. The BP trainer takes a fixed allowed feature
schema; training-empty columns and all imputation/scaling use training only.
Subject partitions are saved and can be reused with `--split-manifest`.

## Verification (2026-10-06)

- Existing baseline: 5 tests passed before edits.
- Final suite: 32 tests passed, including the original five, timing/gap/video
  regressions, leakage checks and a pickle-free legacy cache round trip.
- Syntax compilation and `git diff --check` succeeded.
- All ten classical methods return predictable lengths; synthetic GREEN HR is
  approximately 72 bpm. Existing temporal/video model smoke tests still pass.
- A 30-second encoded dual-color video with injected 120 ms offset and injected
  geometric detectors recovers delay within one acquired frame. Nominal clock
  mode excludes all timing windows by default.
- Actual Tasks adapters loaded the downloaded version-1 face and hand bundles;
  blank frames returned missing observations without crashing.
- Actual detector end-to-end smoke: 480 decoded frames at 30 FPS, a composite of
  two public test images with a 1.2 Hz green-channel sinusoid injected separately
  into each panel; hand panel offset 120 ms. Correct-hand selection produced
  100% face/hand RGB validity, 3 accepted timing windows and 1 excluded edge window.
  HR face 71.979 bpm; hand 71.998 bpm; median optical lag 116.491 ms, window SD
  2.002 ms. This is a software fixture, **not physiological data or BP evidence**.
- Original CLI on the same fixture: GREEN HR 71.98 bpm, 479 uniformly resampled
  samples (legacy resampler excludes its final endpoint). Cached rerun also works.
- BP mean/linear/ridge/forest CLI completed on nine synthetic subjects with
  artificial cuff labels. It selected using validation and wrote separate locked
  SBP/DBP metrics and a reloadable model. Synthetic error values are not study
  performance and are deliberately not quoted here.
- Saved ROI overlays and timing plots were visually inspected. The first
  `expected_hand: any` run selected a hand within the portrait panel and recovered
  nearly zero delay. Choosing the intended anatomical Right hand selected the
  separate panel and recovered the injected delay. This is evidence that initial
  hand selection must be reviewed using overlays, even with stable tracking.
- Initial unavailable MediaPipe asset URLs returned 403; the successful fixture
  used public images from the official samples repository. A Lena-based fixture
  had no detected face and correctly generated excluded windows. These early
  runs are preserved separately; they are not reported as successful dual delay
  recovery.

Local smoke artifacts (ignored by git) are in `outputs/smoke_bp_right/`,
`outputs/smoke_inputs/`, `outputs/smoke_bp_models/`, and earlier diagnostic runs.
Test models are in `models/` with versioned URLs and hashes. Face model SHA256:
`64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff`.
Hand model SHA256:
`fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1`.

Fixture source images: [MediaPipe face test image](https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/face_landmarker/ios/FaceLandmarkerTests/business-person.png)
and [MediaPipe hand test image](https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/gesture_recognizer/android/app/src/androidTest/assets/hand_thumb_up.jpg).
These images remain local smoke assets, not distributed study recordings.

## Files changed

Modified: `.gitignore`, `README.md`, `rppg_lab/__init__.py`, `datasets.py`,
`metrics.py`, `roi.py`, `signals.py`, `scripts/train_bp_baseline.py`.

Added core: `rppg_lab/types.py`, `config.py`, `video.py`, `detection.py`,
`region_masks.py`, `processing.py`, `quality.py`, `pipeline.py`, `artifacts.py`,
`bp.py`, `study.py`, `splits.py`, `bp_models.py`.

Added commands: `scripts/setup_landmarkers.py`, `run_dual_roi.py`,
`run_bp_pilot.py`, `train_bp_models.py`.

Added configuration: `configs/default.json`, `hr_experiment.json`,
`bp_pilot.json`, `recording.example.json`.

Added tests/docs: `tests/test_dual_roi.py`, `docs/DUAL_ROI_AUDIT.md`,
`docs/DUAL_ROI_IMPLEMENTATION.md`. No new package dependency was required.

## Intentionally preserved and remaining work

Classical formulas, neural architectures, ablation scripts, original datasets,
old documentation and the ZIP archive remain. Rewriting them all would risk
invalidating checkpoint compatibility and prior experiments. The legacy HR ROI
resampler still drops unusable detections and permits unbounded interpolation;
it is documented as unsuitable for timing research. Legacy video training still
has nominal-FPS seeking, center-crop fallback, stale-cache risks and permissive
clip-read handling. Legacy phase coherence can be degenerate on one Welch segment;
the new BP path reports it only with multiple segments. ROI cache version 2
creates new keys without deleting old cache files. Tasks can produce landmarks
different from legacy FaceMesh, and corrected UBFC target timing may change
retrained numbers. No old HR results are claimed reproduced on original subjects.

Ranked next steps:

1. **Acquire and validate real pilot recordings.** Confirm camera exposure clock,
   rolling-shutter effects, intended hand/skin mask, lighting and tracking stability;
   record repeated cuff timings and protocol metadata. Real recordings and a
   synchronized contact reference were not supplied, so physiological feasibility
   remains unresolved.
2. **Lock study/evaluation design.** Reserve subjects before model iteration,
   collect enough independent participants/BP variation, compare face-only,
   hand-only and dual-site baselines, and aggregate uncertainty by subject or
   recording rather than treating overlapping windows as independent.
3. **Characterize timing fidelity.** Compare polarity, phase, filters, boundaries
   and adaptive methods across sites; quantify native-frame resolution and temporal
   uncertainty. Add verified external clock readers and explicit antialiasing for
   irregular-clock downsampling. Do not rename optical delay to true PTT/PWV.
4. **Calibrate quality/ROI strategies using development data.** Check fixed skin
   chroma thresholds across skin tone/illumination and hand posture; investigate
   richer masks, multiple simultaneous faces/hands and identity resets. Handedness
   confidence is not ROI detection confidence. No universal thresholds established.
5. **Complete metadata ablations and subgroup reporting.** Signal-only is
   implemented; metadata-enhanced models remain separate future experiments.
   The metrics API accepts strata, but lighting/motion/skin-tone evaluation needs
   acquired metadata. Add recording/subject bootstrap intervals and independent
   physiological verification of beat morphology.
6. **Incrementally migrate legacy experiments.** Fix rPPG-10 ECG clock inference,
   video/trace window matching and video-cache identities with dataset-backed
   regression tests. Introduce locked HR tests and strict cross-dataset evaluation.
   These cannot be validated fully without raw UBFC/rPPG-10 data.

Exact setup/extraction/pilot/training/test commands are in the root README.
