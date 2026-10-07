# rPPG Lab — HR and synchronized face/hand BP feasibility research

This is an experimental research system and is not a medical device.

The research question is whether face and hand rPPG, recorded in the **same RGB
video**, provide reproducible timing/morphology information useful for later
cuff-referenced SBP/DBP experiments. A BP model is not yet scientifically validated.

```text
                         +--> HR (each site)
RGB video --> one clock -+--> face detector --> face mask --> RGB --> rPPG --+
                         +--> hand detector --> hand mask --> RGB --> rPPG --+
                                                                            |
                           paired timing + morphology + raw quality features
                                                                            |
                        explicit nearby-cuff association --> BP baselines
                                                        --> SBP and DBP
```

## Setup and runnable commands

From the repository root, use Python 3.10+ and the project environment:

```powershell
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe scripts/setup_landmarkers.py
```

The setup command downloads Google's versioned face/hand Tasks model bundles to
`models/` and records their SHA256 hashes. Extraction does not download models.
The adapters also support legacy MediaPipe installations. Backend interfaces follow
the official [Face Landmarker](https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker/python)
and [Hand Landmarker](https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker/python)
guides. Their VIDEO mode processes each supplied frame synchronously.

Replace `video.mp4` with your recording. Each `--out` must be a **new** directory.
The first command is the original single-video HR workflow, with its existing CLI:

```powershell
.\.venv\Scripts\python.exe scripts/extract_rppg_from_video.py video.mp4 --roi face --method POS_WIN --fs 30 --cache-dir cache_roi
```

New timing-aware face extraction, hand extraction, and synchronized extraction:

```powershell
.\.venv\Scripts\python.exe scripts/run_dual_roi.py video.mp4 --regions face --config configs/hr_experiment.json --out outputs/face_run01 --debug
.\.venv\Scripts\python.exe scripts/run_dual_roi.py video.mp4 --regions hand --config configs/bp_pilot.json --out outputs/hand_run01 --debug
.\.venv\Scripts\python.exe scripts/run_dual_roi.py video.mp4 --regions both --config configs/bp_pilot.json --out outputs/dual_run01 --debug
```

Create a recording manifest following `configs/recording.example.json`; replace
every illustrative value with acquisition information. Video paths in a manifest
resolve relative to the manifest. Config/model/output paths resolve from the
working directory. BP pilot features and simple baseline comparison:

```powershell
.\.venv\Scripts\python.exe scripts/run_bp_pilot.py data_sessions/S001/rest_01/recording.json --config configs/bp_pilot.json --out outputs/bp_S001_run01 --debug
.\.venv\Scripts\python.exe scripts/train_bp_models.py outputs/bp_S001_run01/bp_table.csv outputs/bp_S002_run01/bp_table.csv outputs/bp_S003_run01/bp_table.csv --out outputs/bp_baselines_run01 --seed 42
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m compileall -q rppg_lab scripts tests
```

At least three participants with accepted, cuff-associated intervals are needed
for train/validation/test; that is a software minimum, not a sufficient study size.
Mean, linear, ridge and random forest models fit training only; validation selects
the model before one locked-test evaluation. Features are signal-only. Subject
metadata stays in manifests and is not automatically a model input.

## Clock, gaps and configuration

`configs/default.json`, `hr_experiment.json`, and `bp_pilot.json` use one validated
dataclass/JSON mechanism. Missing fields use documented defaults in `config.py`;
unknown fields raise. Change `sample_rate` to the actual capture rate for a high-FPS
recording. Material downsampling is currently refused because antialiasing has not
been implemented for irregular RGB traces. Upsampling does not improve acquired
time resolution.

Supply genuine video-relative timestamps as an optional NPY vector, one per
decoded frame:

```powershell
.\.venv\Scripts\python.exe scripts/run_dual_roi.py video.mp4 --timestamps frame_times.npy --out outputs/dual_with_clock01 --debug
```

Otherwise the reader uses container presentation timestamps, with explicit
nominal-FPS fallback. Container timestamps are not independently verified sensor
exposure times. Windows involving a nominal clock are excluded from timing by
default; `allow_nominal_timing: true` is an explicit exploratory override.
Original/backend timestamps and per-frame timestamp sources are saved. A reported
container-frame-count shortfall is recorded separately from missing ROI frames;
capture drops cannot be inferred definitively from this metadata alone.

The new pipeline keeps NaNs/validity for absent ROIs. Linear interpolation only
bridges gaps within `max_gap_sec`; it never extrapolates. Face and hand share the
same uniform grid. Algorithms run with their internal filters disabled, followed
by an explicit zero-phase SOS bandpass per contiguous segment. Edge guards discard
unreliable boundary samples without shifting times. Timing uses matching paired
segment boundaries, while individual HR remains available when the other ROI is
missing. Windows with gaps, low quality, inconsistent HR or unresolved lag remain
in outputs with exclusion reasons. HR in the run summary uses the longest usable
contiguous segment, whose time support is saved.

GREEN is the conservative default for timing experiments. Existing CHROM/POS,
windowed methods, PCA/ICA and approximations remain available, but adaptive
projections, component polarity, tracking and filtering can change morphology
and apparent phase. Their timing fidelity must be investigated, not assumed.

## ROIs and inspectable outputs

Face strategies: `combined` forehead/cheek mask, `forehead`, `left_cheek`,
`right_cheek`, or `full_skin` (face oval with eye/mouth holes plus chroma gate).
Hand strategies: `palm`, `back_of_hand`, `landmark_polygon`, `bounding_box`, or
`skin_masked_bbox`. The central-hand polygon excludes much of the background.
`back_of_hand` shares that geometry and relies on acquisition posture; automatic
palm/back orientation is not established. Landmark hulls are geometric masks,
not semantic segmentation. Fixed chroma skin gates are experimental and can bias
coverage by skin tone/lighting; they are disabled for the default polygon ROIs.

Set `expected_hand` to `Left` or `Right` when the protocol fixes anatomical side,
and set `input_mirrored` to match the stored video. `any` locks the first selected
hand by handedness and then nearest position; large jumps are rejected even after
loss. If several hands are visible, check the overlay to ensure the intended hand
was selected. The tracker cannot guarantee identity for two hands of the same
side belonging to different people. Face detection assumes one participant.

`process_recording(video, config)` returns `result.face.rgb`, `.rppg`, `.quality`,
the corresponding hand objects, `result.shared.timestamps` and
`.original_timestamps`, `result.hr`, `result.delay`, and `.bp_features`.
`result.hr` is a dictionary with `face` and `hand` keys. RGB is `(T,3)` in RGB
order, times are seconds, HR is bpm, cuff targets are mmHg. Original RGB/ROI
observations use the decoded clock; rPPG uses the uniform clock.

Facial extraction now retains `forehead`, `left_cheek`, `right_cheek`, and
`combined` simultaneously in `result.face_rois: dict[str, RegionResult]`.
One decoded frame and one face-landmark detection produce all masks; landmark
groups remain defined in `roi.FACE_ROIS`. `combined` is their pixel union,
counting overlapping pixels once. Keeping regions separate preserves information
for future window-level ROI quality/attention experiments: a region can retain
the HR frequency while corrupting pulse morphology.

All facial RGB traces share the exact original frame timestamps, with independent
NaNs and validity masks; all rPPG outputs use `result.shared.timestamps`.
Bounded interpolation, segment filtering and edge guards apply separately to
each ROI. A failed cheek does not invalidate the forehead or hand. Per-ROI
diagnostics include valid fractions, pixel count, frame coverage, brightness,
illumination spread, shared face motion, rejection reasons and filter segments.
`RegionObservation.valid` describes detection geometry; `rgb.validity_mask` and
`rgb.quality["extraction_valid"]` describe measurement success.

Existing JSON configs and CLIs work without changes. `face_roi` still selects
`result.face` for legacy HR and face–hand timing/BP features; that object also
appears in `result.face_rois[config.face_roi]`. Its legacy `roi_name`/source label
remains `face`; the mapping key identifies the actual facial ROI. By default,
old configs gain all four traces while retaining their selected face output.
An explicit list can limit extraction or add `full_skin`; the selected `face_roi`
is always included. For example:

```json
{
  "face_roi": "combined",
  "face_rois": ["forehead", "left_cheek", "right_cheek", "combined"]
}
```

```python
from rppg_lab.config import PipelineConfig
from rppg_lab.pipeline import process_recording
from rppg_lab.processing import build_classical_priors

config = PipelineConfig()
result = process_recording("video.mp4", config)
forehead_rgb = result.face_rois["forehead"].rgb.values
left_cheek_rgb = result.face_rois["left_cheek"].rgb.values
right_cheek_rgb = result.face_rois["right_cheek"].rgb.values
combined_rgb = result.face_rois["combined"].rgb.values
forehead_valid = result.face_rois["forehead"].rgb.validity_mask
priors = build_classical_priors(result.face_rois["forehead"].rgb,
                                result.shared.timestamps, config,
                                methods=("GREEN", "CHROM"))
green_forehead = priors["GREEN"].values
```

Saved NPZ artifacts add `face_roi_names` and `face_roi_<name>_rgb`, `_valid`,
`_rppg`, `_rgb_uniform`, `_interpolated`, `_reason` and diagnostic arrays;
`summary.json` adds per-ROI quality and preprocessing under `face_rois`.
Existing artifact keys remain available. Hand-only runs return an empty
`face_rois` mapping. See [multi-ROI extraction details](docs/MULTI_FACE_ROI.md)
for the exact schema and Phase 2 boundaries. This infrastructure prepares the
upcoming MultiROI waveform model; BP accuracy or morphology improvement has
not been demonstrated by this extraction change.

Each new run saves configuration, source/input/model hashes, package versions,
git revision/dirty state, timestamp sources, frame counts, per-frame geometry and
quality, original/uniform RGB, interpolation flags, individual and paired rPPG,
feature windows and exclusions. JSON uses null for nonfinite values; NPZ retains
NaNs and loads without pickle. CSV feature columns are fixed. Debug mode adds ROI
overlays, RGB/rPPG/validity plots, and the first accepted window's cross-correlation
and display-only lag alignment. Original waveforms are never shifted in storage.

Raw indicators include handedness classification score (not detection confidence),
ROI pixel count, clipping fraction, brightness, motion speed, illumination spread,
valid/missing/interpolated fractions, pulse-band SNR, spectral prominence and
periodicity. Face per-frame confidence is null because the adapter API does not
expose it. Quality thresholds are experimental heuristics, not medical confidence.

## Dataset and leakage rules

```text
data_sessions/S001/rest_01/
    video.mp4                 # immutable acquired data
    recording.json            # subject/session/recording IDs + acquisition metadata
    frame_times.npy            # optional actual capture clock
outputs/bp_S001_run01/
    features.csv              # labels never enter feature generation
    bp_table.csv              # explicit cuff association + separate SBP/DBP
    recording.json            # input manifest snapshot
    signals.npz
    config.json / provenance.json / summary.json / bp_exclusions.json
```

Record each repeated cuff reading with measurement index, start/end times including
timezone, device, and before/during/after relation. Associations name one reference
and a video interval, use `nearby_cuff_pilot`, and explain the approximation in
notes. Only fully contained windows receive labels. Missing associations are
excluded from BP training while signal-only features remain available. Acquisition
metadata can hold phone/settings, lighting, posture, motion, distance, hand used,
protocol version and optional subject characteristics; do not invent absent data.

All windows/beats/sessions for a subject belong to exactly one split. Shared split
validation rejects overlap, duplicate recordings assigned to different subjects,
duplicate file content and duplicate derived windows. Overlap inside a partition
is permitted and must not be interpreted as independent observations. New models
use a fixed signal feature schema, drop training-empty columns using train only,
and fit imputation/scaling on train only. Reuse an existing `splits.json` with
`train_bp_models.py --split-manifest` when
comparing later experiment variants; avoid repeatedly tuning against test results.

BP reports separate bias, SD, MAE, RMSE,
correlation, Bland–Altman points, per-subject errors, and error-vs-BP/quality data.
The metrics API also supports explicit subgroup labels for lighting/motion/skin
tone; those analyses require collected metadata.

## Structure, migration and scientific limits

The existing flat `rppg_lab/` package remains import compatible. `types.py`,
`config.py`, `video.py`, `detection.py`, `region_masks.py`, `processing.py`,
`quality.py`, `pipeline.py` and `artifacts.py` implement the new extraction path.
`bp.py`, `study.py`, `splits.py`, `bp_models.py` implement pilot research.
Original classical methods, neural architectures and experiment scripts remain.

The original HR face aggregation and filtering behavior are retained. Face
extraction now uses the shared video reader and compatible detector adapters.
Cache version 2 intentionally invalidates old ROI cache keys; old cache files are
preserved. Tasks and legacy landmark outputs may differ, so reproducing old
experiment numbers requires the original backend/environment. UBFC targets now
interpolate onto the exact retained video grid, fixing a fractional timestamp
offset; target alignment may therefore change old training results. Legacy
`extract_rgb_trace` still drops invalid faces and can bridge long gaps; use the new
pipeline for timing research. The old BP MLP JSON workflow is retained with explicit
warning that cuff timing is unchecked; use the new pilot/table workflow for studies.

Face–hand delay is an **experimental optical inter-site delay**, not established
PTT/PAT or clinical PWV. Camera exposure/readout, rolling shutter, compression,
tracking, illumination and different pulse morphology can confound it. Zero-phase
filtering prevents filter group delay but does not preserve every morphology
feature. Normalized amplitudes/derivatives are in relative algorithm units. A cuff
reading is intermittent and its nearby-window association is not continuous BP
ground truth. No clinical accuracy or standards compliance is claimed.

The audit, implementation record, changed-file inventory, test evidence and ranked
remaining work are in [docs/DUAL_ROI_AUDIT.md](docs/DUAL_ROI_AUDIT.md) and
[docs/DUAL_ROI_IMPLEMENTATION.md](docs/DUAL_ROI_IMPLEMENTATION.md).

---

## Preserved HR experiment context

This repository contains the current research code for estimating heart rate
from facial video and preparing a future cuff-referenced blood-pressure study.
It is an academic prototype, not a medical device.

## Evidence-backed current path

The present results support a conservative hybrid strategy:

```text
face video -> RGB traces -> classical rPPG priors -> prior-preserving model -> HR
```

On the stored UBFC validation split, `CHROM_WIN` is already very strong. The
recommended waveform model, `PriorResidualWaveformNet`, starts from a weighted
fusion of classical priors and learns only a small residual correction. Its
residual head is zero-initialized, so the untrained model does not destroy a
strong prior.

The six default prior channels are:

```text
GREEN, CHROM, CHROM_WIN, PBV, POS_WIN, OMIT
```

`PBV`, `LGI`, and `OMIT` in this repository are transparent baseline
approximations; they must not be described as exact reproductions of every
published variant.

## Datasets and evaluation status

- **UBFC-rPPG:** used for participant-wise model development and validation.
- **rPPG-10:** used for an external classical-method benchmark and, separately,
  for multi-dataset training of `HRQualityNet`.

The current multi-dataset experiment is **not** a strict cross-dataset test:
UBFC-rPPG and rPPG-10 are both represented in training and validation after
independent participant-wise splits. A strict train-on-one/test-on-the-other
experiment remains future work.

No locked final test set has yet been reported for the HR models. Stored "best"
checkpoint metrics are validation results used for model selection.

## Recommended UBFC waveform command

```powershell
python scripts/train_waveform_ubfc.py --ubfc-root UBFCData --cache-dir cache_roi_oldpoints --epochs 20 --batch-size 32 --fs 30 --win-sec 10 --stride-sec 2 --roi face --out checkpoints/prior_residual_ubfc.pt
```

RGB-only ablation:

```powershell
python scripts/train_waveform_ubfc.py --ubfc-root UBFCData --cache-dir cache_roi_oldpoints --epochs 20 --batch-size 32 --fs 30 --win-sec 10 --stride-sec 2 --roi face --no-priors --model waveform --out checkpoints/rgb_waveform_ubfc.pt
```

## Visual waveform inspection

```powershell
python scripts/plot_waveform_predictions.py --ubfc-root UBFCData --cache-dir cache_roi_oldpoints --checkpoint checkpoints/prior_residual_ubfc.pt --mode worst --num-plots 12 --show-priors --out-dir outputs/waveform_plots
```

The waveform target is fingertip/contact PPG while the input is facial rPPG.
A small physiological and acquisition delay can exist, so both zero-lag and
lag-aligned correlation are reported. HR is estimated from the predicted
waveform with Welch spectral analysis.

## Blood-pressure code status

The BP path is a scaffold awaiting synchronized, participant-level data. The
reviewed baseline now provides:

- participant-wise train/validation/locked-test splits;
- training-only feature imputation and standardization;
- training-only target standardization;
- a population-mean BP reference baseline;
- separate SBP/DBP metrics;
- checkpointed preprocessing and split metadata.

Run only after valid pilot recordings exist:

```powershell
python scripts/train_bp_baseline.py --data-dir data_sessions --out checkpoints/bp_feature_mlp.pt
```

Face-to-hand optical delay is stored as an **inter-site peripheral delay**, not
true PTT. Neither camera signal marks cardiac ejection, and an apparent velocity
must not be reported as clinical PWV.

## Reproducibility audit

Create a manifest from stored checkpoints and CSV outputs:

```powershell
python scripts/audit_results.py
```

The generated `outputs/results_audit.json` summarizes existing artifacts; it
does not rerun training.

## Validation commands used in the review

```powershell
python -m compileall -q rppg_lab scripts tests
python -m unittest discover -s tests -v
```

The datasets were not included in the uploaded archive, so full extraction and
retraining were not rerun during this review. Existing checkpoints and result
CSVs were audited, and model/signal/BP-preprocessing paths were smoke-tested.
