# Pilot-6 dual-ROI rPPG and BP feasibility

This experiment uses six local smartphone videos and the user-supplied metadata
workbook. No contact PPG, ECG, or continuous BP recording is available. Frozen
Waveform V1 is unchanged. No neural BP model is trained.

## Implementation and repository audit

The pilot lives in new modules and uses a new ignored output namespace. Existing
training, architecture, checkpoint, preprocessing, splits, and benchmark files
were not edited. The historical Part 4.5A pipeline remains intact.

| Existing component | Reuse / verified contract |
|---|---|
| `waveform_inference.FrozenWaveformExtractor` | Mandatory original freeze manifest, checkpoint hash/order/config checks, evaluation/inference mode, signed independent windows |
| `waveform_training.checkpoint_dataset_config` and `reconstruct_waveform_model` | Validate architecture, strict state dict, input order, extraction configuration, split identity, sampling and window size |
| `detection` | MediaPipe face and anatomical hand identity/jump logic; unavailable face confidence stays missing |
| `region_masks`, `roi.FACE_ROIS` | Existing forehead/cheek polygons and central-hand hull; means are RGB, not BGR |
| `processing` | Bounded common-clock interpolation, valid segments, unchanged classical formulas, explicit zero-phase filtering and edge rejection |
| `quality`, `signals.estimate_hr_welch` | Existing spectral HR and engineering signal indicators |
| `bp.estimate_delay` | Signed cross-correlation, phase and beat matching, positive hand-lags-face convention |
| Part 4.5A `waveform_visualization` | Reliable-beat gates, coarse morphology diagnostics, normalized Welch PSD, strict CSV serialization |
| `bp_models.BPModel` | Train-only imputation/scaling and fixed-alpha ridge / training mean baselines |

New implementation files:

- `rppg_lab/pilot_metadata.py`: local CSV/XLSX and authorized read-only Google
  import, explicit headers/mappings, raw preservation, exact recording matching,
  and measurement-level cuff associations.
- `rppg_lab/pilot_video.py`: native MP4 display timestamps and synchronized raw
  extraction of all seven ROI variants on every decoded frame.
- `rppg_lab/pilot_detection.py`: one-visible-hand adapter using the existing
  identity logic; historical two-hand detector defaults remain unchanged.
- `rppg_lab/pilot_analysis.py`: classical methods, quality/morphology/HR,
  timing validation, delay ablations, figures, and grouped exploratory BP fits.
- `rppg_lab/pilot6.py`: provenance, compatibility gates, orchestration, reports,
  failure retention, and identity-checked reuse.
- `scripts/evaluate_pilot6.py`, `configs/pilot6_metadata.json`, and
  `tests/test_pilot6.py` provide the entry point, verified mappings, and fixtures.

`sheets/` was added to `.gitignore` because it contains source physiological and
health metadata. Existing `Dataset/`, `outputs/`, checkpoints and model assets
remain ignored. No original Google Sheet or local source workbook is modified.

## Frozen checkpoint: verified identity, blocked inference

Actual local file: `checkpoints/waveform_v1_frozen.pt`.

```text
SHA256 d6b2d1eeebc0153197a2a12da736b957b32d72ae00b22631e9afd0b2097131ff
```

This is the exact official hash documented by the existing finalization and
Part 4.5A code, rather than a selection based on the filename. Its state loads
strictly as `MultiROIPriorResidualWaveformNet`, selected epoch 25. Input is
three facial ROIs in this order: forehead, left cheek, right cheek. Each has
nine channels: `RGB_R, RGB_G, RGB_B, GREEN, CHROM, CHROM_WIN, PBV, POS_WIN, OMIT`.
The frozen model uses 30 Hz, 300 samples / 10 seconds, per-window channel
standardization and window-local priors, with the existing source-support gate.
The saved training stride is 2 seconds; a pilot selection hop of 5 seconds does
not change any input window's processing. Outputs retain native signed polarity;
overlapping independently normalized outputs are not stitched into one waveform.

The original `waveform_v1_freeze_manifest.json` is absent from this workspace.
The existing inference API requires it. The pilot does not synthesize a replacement
manifest or bypass that gate. Face inference and the synthetic learned-model phase
probe are therefore blocked. There are two additional acquisition constraints:
these videos have about 60 Hz native sampling, and the pilot's classical run uses
spatial resizing. No validated 60-to-30 Hz antialiasing adaptation is present;
the existing preprocessing explicitly rejects that downsampling. Native spatial
processing is required before claiming the frozen input contract is met.

Hand input is unsupported by the existing named-three-facial-ROI contract. No
hand feature mapping is disguised as facial data and no architecture is changed.
The hand experiment uses classical GREEN/CHROM/POS. No learned hand prediction
is claimed, and no learned-model inter-site timing is reported.

## Metadata actually discovered

Both direct read-only public exports of spreadsheet
`1613BKOf1fJsOweixkMVrLQpImywScl4cd-q-0Xq-ZMQ` returned
**HTTP 401 Unauthorized** (`export?format=xlsx` and the CSV export for GID
`1483909197`). The browser tool also could not access its edit URL. Remote tabs
could not be enumerated. The user then supplied a subject CSV and subsequently
the full local `sheets/rPPG BP Data Collection final.xlsx`, which is the source
used for the full experiment. The CSV alone does not contain BP references.

Actual local workbook worksheets:

1. `00_Protocol`
2. `01_Subjects`
3. `02_Sessions`
4. `03_Review`
5. `04_Windows`
6. `05_Field_Guide`
7. `90_Source_Subjects`
8. `91_Source_Sessions`
9. `92_Migration_Audit`

Exact headers for every worksheet, source hash, and header positions are preserved
in [PILOT6_METADATA_SCHEMA.json](PILOT6_METADATA_SCHEMA.json). Observation-table
headers are on row 3 (zero-based index 2), after title and grouping rows.
`00_Protocol` has a primary table header on row 4 and additional guidance tables;
all of its cells are preserved in the local raw copy. Template formula rows are
not subjects or sessions. There are six actual subject IDs and six session IDs:
`S001`–`S006` and `S001_V01`–`S006_V01`.

Relevant verified source columns include:

- Identity and recording: `Subject_ID`, `Session_ID`,
  `Recording_Date_Jalali`, `Start_Local_HHMM`, `Video_File_or_Link`.
- Acquisition: `Phone_Device_ID`, `Video_Duration_s`, `Actual_FPS`,
  `Resolution`, `Camera_Distance_cm`, `Light_Type`, `Lighting_Lux`,
  `Posture`, `Subject_Motion_Level`, `Lighting_Changed`,
  `Original_Protocol_Condition`.
- Setup: `Hand_ROI_Side`, `Hand_Supported`,
  `Hand_Height_Relative_Heart_cm`, `Cuff_Arm`, `Cuff_Device_ID`,
  `Cuff_Size`, `Cuff_Arm_at_Heart_Level`.
- Three cuff groups: `BP1_Start_s`, `BP1_End_s`, `SBP1_mmHg`, `DBP1_mmHg`,
  `CuffHR1_bpm`, and corresponding numbered fields for readings 2 and 3.
- Event provenance: `Event_Time_Source`, `Manual_Exclude`, `Exclude_Reason`,
  `Operator_Notes`.
- Anatomy: `Suprasternal_to_FaceROI_cm`,
  `Suprasternal_to_IndexFingertip_cm`, `Arm_Span_cm_Optional`.

`Phone_Device_ID` is `P01`; a physical phone model, exposure, rolling-shutter
readout, and calibrated camera timing are not supplied. Actual FPS, resolution,
and video links are empty in the active session table; measured file properties
are saved separately. The cuff device ID is recorded as `0`, which the migration
audit explicitly says needs verification rather than interpreting as a model.
Jalali dates and local wall times remain raw; they are not silently treated as
Gregorian timestamps or precise timezone-aware synchronization.

The two suprasternal distance fields are populated for three subjects. Field
Guide row 9 defines them as tape measurements on the body to the face ROI and
index fingertip in cm, explicitly not actual arterial path lengths. Protocol row
22 repeats that boundary. The current hand masks cover the central hand, rather
than the recorded index-fingertip endpoint. No PWV calculation is justified.

Session IDs match video stems exactly under the existing naming convention;
matching does not use row order. Archive and derived sheets are retained raw,
but do not create duplicate participants, references or signal observations.
Raw cell values and normalized records are saved together under the ignored
`private_metadata/` directory, along with a byte copy of the original workbook.

## Video audit

All six files are H.264, nominally about 60 FPS, 1080×1920 in OpenCV's oriented
output with 90-degree rotation metadata. The native nonfragmented MP4 sample
tables provide actual display PTS (`stts` plus signed/unsigned `ctts`). Simple
movie edits are supported; complex edits/fragments fail explicitly. Empty movie
leads of 5.8 ms and 9 ms are preserved for S004 and S006.

| File | Reported/sample frames | Last PTS (s) | Long intervals >1.5× median | Largest interval (ms) |
|---|---:|---:|---:|---:|
| S001_V01.mp4 | 18047 | 301.176489 | 19 | 133.322 |
| S002_V01.mp4 | 18035 | 301.026500 | 28 | 50.000 |
| S003_V01.mp4 | 17994 | 300.409889 | 30 | 66.667 |
| S004_V01.mp4 | 18053 | 301.598922 | 42 | 50.000 |
| S005_V01.mp4 | 18014 | 300.943178 | 42 | 66.667 |
| S006_V01.mp4 | 17972 | 300.252233 | 44 | 50.000 |

These are container audit measurements, not assertions that all those frames
were successfully tracked. Median acquired interval is about 16.667 ms. All
videos have nonuniform intervals. Long intervals indicate timing irregularity;
the original capture-drop count cannot be recovered without a capture log.
Exact repeated decoded analysis frames are counted separately during extraction.
The full run also cross-checks normalized decoder times against the MP4 clock.

## Processing, interpretation, and outputs

Classical processing spatially resizes to a maximum dimension of 640 pixels,
using area interpolation, while decoding and measuring **every temporal frame**.
Face and hand detector graphs execute concurrently but each graph receives
strictly chronological frames. This is a new pilot configuration, not a change
to Frozen V1 preprocessing. Motion values are in analysis pixels/second.

ROIs are forehead, each cheek, their pixel-union combined face, cheeks only,
central-hand hull, and a contracted central-hand hull. All-frame original-pixel
landmarks, validity, RGB, coverage, clipping, brightness, motion and available
classification scores are retained. The geometry avoids eye/mouth/fingertip
areas but is not validated semantic skin segmentation. Existing chroma gating
is disabled; `skin_pixel_count` is the inherited name for geometric mask pixel
support. Face tracking confidence is unavailable, and the hand score measures
handedness classification rather than calibrated tracking confidence.

The workbook records Right hand; first-frame MediaPipe results report Left,
including at native resolution. Mirroring is unverified. The pilot locks the
detected hand identity and records this disagreement, instead of rejecting the
entire hand signal by an unverified side convention.

Classical GREEN, CHROM and POS use the existing formulas without their internal
filters, followed by identical 0.7–4 Hz order-3 Butterworth SOS forward/backward
filtering on each contiguous valid segment. There is no causal filter group lag;
edge distortion and adaptive projection phase can remain. One second at each
segment edge is excluded, minimum segment duration is four seconds, and linear
interpolation is bounded to 0.15-second gaps with explicit flags. Face/hand delay
signals are rebuilt on identical paired segment boundaries. A common uniform
grid at the measured native rate does not invent new acquired timing resolution.

Quality/HR uses 10-second windows / 5-second hops and separate nonoverlapping
30-second assessments. Every complete attempted window is retained, including
invalid windows. Original validity below 90%, missing filtered samples and flat
signals are invalid. SNR below 0 dB, periodicity below 0.5, clipping above 10%,
or illumination CV above 0.05 produces an uncertain label. These are engineering
screens, not calibrated biomedical confidence. HR search is 0.7–3.5 Hz / 42–210
bpm. Welch zero-padding refines the displayed frequency grid without acquiring
new information. Part 4.5A reliable-beat gates control coarse rise, pulse-width
and IBI descriptors; its historical additional 0.7–4 Hz morphology filtering is
retained and documented separately from the single-pass delay path. Rejected
features remain missing. Adjacent nonoverlapping intervals have a descriptive
median pulse-template correlation when at least three reliable beats support
each template. Beats use real elapsed time from candidate troughs, with no
duration warping or per-beat amplitude fit; an invalid intervening primary
window breaks adjacency. Signed algorithm-waveform correlations and HR ranges
also measure agreement between algorithms sharing the same RGB input, not
independent sensor validation. No dicrotic-notch or
calibrated blood-volume interpretation is made.

Delay estimators are first tested on 30 and 60 Hz analytical signals at both
signs, including one-frame delays, noise and irregular common clocks, using
5/10/30-second analysis intervals with extra filtering support. The synthetic
pulse fundamental is 1.2 Hz (72 bpm); this is not a validation over all possible
HRs, noise, motion or illumination conditions. Maximum absolute numerical
recovery errors from the 60 tested cases are:

| Synthetic sampling | 5s | 10s | 30s |
|---|---:|---:|---:|
| 30 Hz | 3.193 ms | 1.493 ms | 0.614 ms |
| 60 Hz | 3.283 ms | 1.626 ms | 0.457 ms |

The largest filter-versus-unfiltered estimator change in this simulation is
0.126 ms. These are numerical simulation results, not physiological error bounds.
Positive delay means hand lags face. Actual acquisition intervals, half-window
disagreement and phase-versus-correlation disagreement set a conservative
diagnostic uncertainty floor. Long acquisition intervals and unconfirmed
cross-method agreement reject physiological interpretation. Search-boundary
maxima are unresolved. Overlapping windows are not independent participants or
independent cuff measurements. Primary summaries select chronological
nonoverlapping windows before quality filtering.

Timing error bars in the diagnostic plots show this lower bound, not a
statistical or physiological confidence interval. The same camera provides a
common frame clock, but unknown rolling-shutter
readout, exposure, compression, ROI position and extraction phase can bias
relative timing. An apparent face-to-hand lag is an **experimental optical
inter-site delay**, not validated PTT. The learned model's phase has not been
tested here because its inference gate is blocked.

Full-run artifacts are under `outputs/pilot6_dual_roi/full03/`:

```text
dataset_audit.json                   metadata_validation.json
video_session_manifest.csv           resolved_metadata_mapping.json
checkpoint_provenance.json           experiment_manifest.json
synthetic_timing_validation.json     model_phase_probe.json
private_metadata/                   # raw workbook/cells and resolved records
S001_V01/ ... S006_V01/
  raw_rgb.npz                       # original PTS; all ROI RGB/quality/landmarks
  extraction.json                   analysis_config.json
  mask_<ROI>_<frame>.png             roi_geometry_<frame>.png
  classical_waveforms.npz            # uniform RGB and GREEN/CHROM/POS waveforms
  quality_hr_morphology.csv          method_agreement.csv
  delay_feasibility.csv              delay_summary.json
  signal_summary.json
  bp_reference_alignment.csv         pilot_summary.json or failure.json
  interpretability.png              delay_correlation.png
bp_reference_alignment.csv           bp_session_features.csv
exploratory_bp_predictions.csv       exploratory_bp_metrics.csv
bp_feasibility.json                  cross_recording_summary.json
research_report.md                   invocations/ # previous manifests on reuse
```

Default ROI PNGs contain masks/geometry only, without participant pixels. The
optional `--private-overlays` flag explicitly enables local participant overlays
inside ignored outputs. No video or physiological data is uploaded by this
workflow. Google import uses GET requests only and does not persist a bearer token.

## BP alignment and feasibility protocol

There are **18 complete cuff SBP/DBP readings and 18 cuff HR values**. All six
sessions have three readings. Five sessions have no cuff end times. S004 has
logged ends 133, 209 and 292 seconds, recovered from operator notes, with event
source `Legacy_operator_notes_time`. The other sessions are marked
`Legacy_needs_video_confirmation`. These legacy sources are not promoted to
verified timed alignment. No window receives cuff BP as continuous ground truth.

Protocol rows 9 and 18–20 explicitly prohibit substituting the planned 50-second
duration for an unknown actual end. They define the last-two cuff-reading mean
as a descriptive session summary, not instantaneous BP. The pipeline uses that
session summary only, retains each observed cuff row separately, and distinguishes
observations from exploratory estimates in column names and prediction kinds.
No BP value is substituted for a failed prediction.

Session cuff-HR means and differences from video-derived HR are descriptive
comparisons of incompletely aligned measurements. Cuff HR is not used as a BP
predictor, a continuous HR target, or waveform ground truth.

Exploratory baselines require full recordings, resolved sessions, references,
and at least three nonoverlapping good face and hand windows. They use session
features, not overlapping windows as samples. Leave-one-subject-out evaluation
keeps every session and reference of a subject together. Training subjects are
aggregated equally; imputation, scaling and train-empty feature removal are fit
within each training fold. A training-mean predictor is compared with ridge
(`alpha=10`, fixed without tuning). Prespecified ablations are HR/quality,
HR/quality/morphology, and those plus stable delay. Delay remains missing unless
at least three nonoverlapping candidate intervals agree and their variability
does not exceed the timing floor. Selected and dropped features are recorded.

With six subjects and uncertain cuff timing, even completed held-out exploratory
fits cannot establish BP accuracy or clinical generalization. Within-subject
calibration is not attempted because repeated cuff readings do not provide
verified window-level physiological labels. No neural BP training is performed.

## Reproduction on Windows

Run from the repository root. The existing `.venv` uses Python 3.13.9 on Windows.
`openpyxl==3.1.5` was the missing declared dependency and was installed locally.
Direct dependency versions, source hashes, code revision/dirty status, seed 42,
checkpoint SHA256, configuration and invocation are recorded in the experiment
manifest. No ffprobe or PyAV is required for these six nonfragmented MP4 files.

Audit, including schema, video clocks, checkpoint and synthetic timing:

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_pilot6.py `
  --dataset-root Dataset `
  --metadata-file "sheets\rPPG BP Data Collection final.xlsx" `
  --checkpoint checkpoints\waveform_v1_frozen.pt `
  --out-dir outputs\pilot6_dual_roi\audit_reproduction01 `
  --all-recordings --dry-run --frozen
```

Full experiment (the actual execution used `full03`):

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_pilot6.py `
  --dataset-root Dataset `
  --metadata-file "sheets\rPPG BP Data Collection final.xlsx" `
  --checkpoint checkpoints\waveform_v1_frozen.pt `
  --out-dir outputs\pilot6_dual_roi\reproduction01 `
  --all-recordings --frozen --workers 6
```

Regenerate analysis/reports from identity-checked intermediates:

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_pilot6.py `
  --metadata-file "sheets\rPPG BP Data Collection final.xlsx" `
  --out-dir outputs\pilot6_dual_roi\full03 `
  --all-recordings --frozen --workers 2 --reuse --reports
```

Use `--recording S001_V01` for one recording, `--metadata-only` for metadata/audit,
`--classical-only` to disable frozen inference attempts, and `--max-frames` for
an explicitly partial run that is ineligible for full-session BP fits. Existing
run directories are refused unless `--reuse` is explicit; incompatible raw
caches, extraction code, model assets or extraction library versions are rejected.
Use `--max-dimension 0` for native spatial processing; this does not resolve the
separate frozen manifest and 60-to-30 Hz compatibility blockers.

Read-only Google import supports `--google-sheet-id` and an authorized bearer
token in `PILOT6_GOOGLE_ACCESS_TOKEN`, or public XLSX export without a token.
All worksheets are inspected; the target GID is not assumed to contain all data.
CSV exports require the exact configured sheet name and header-row mapping.

Validation commands:

```powershell
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m compileall -q rppg_lab scripts tests
git diff --check
```

Tests use synthetic/non-identifiable CSV, Excel, MP4, RGB, signal and model
fixtures. They cover headers/mappings/missing BP, ambiguous video matching,
subject grouping, strict timestamps, signed MP4 composition offsets and variable
intervals/edits, tracking failures, RGB masks, cache identity, checkpoint gates,
filter/HR/quality behavior, signed delay recovery, cuff alignment, leakage, and
report generation. Historical Frozen V1 finalization, waveform, split and
visualization regression tests are included unchanged.

## Measured full-run results and next step

The complete run and the final cache-based analysis both succeeded for **all six
recordings**, with **108,115 decoded frames**, no recording-level failures, and
100% original-frame RGB support for the selected combined-face and central-hand
ROIs. MP4 and normalized decoder timestamps agree to floating-point roundoff.
That verifies software clock consistency, not physical camera timing accuracy.
There were no exact repeated analysis frames in these recordings.

There are 59 overlapping 10s windows and 30 chronological nonoverlapping primary
10s intervals per recording. There are also ten 30s assessments per ROI/method.
The final artifacts retain **8,694 quality/HR/morphology rows** across seven
ROIs, three methods and two window lengths, and **10,620 delay diagnostic rows**
across 30 ROI/method combinations. These are dependent diagnostics from six
participants, not thousands of independent observations.

Combined-face and central-hand GREEN results:

| Recording | Face good / uncertain / invalid | Hand good / uncertain / invalid | Primary good face / hand |
|---|---|---|---|
| S001_V01 | 3 / 55 / 1 | 41 / 17 / 1 | 2 / 21 |
| S002_V01 | 2 / 56 / 1 | 35 / 23 / 1 | 2 / 17 |
| S003_V01 | 32 / 25 / 2 | 7 / 50 / 2 | 13 / 4 |
| S004_V01 | 18 / 40 / 1 | 38 / 20 / 1 | 9 / 19 |
| S005_V01 | 1 / 56 / 2 | 3 / 54 / 2 | 1 / 2 |
| S006_V01 | 0 / 57 / 2 | 1 / 56 / 2 | 0 / 1 |

Only **56/354 combined-face GREEN windows** and **125/354 hand GREEN windows**
pass the uncalibrated engineering screens. Most exclusions involve weak
periodicity and/or low spectral SNR. The nine invalid windows per site are
missing filtered support at recording edges, not failed original-frame RGB
extraction. Successful landmark tracking does not establish a usable pulse.

Median Welch HR among primary good windows is shown below. The session cuff-HR
column is the observed last-two-reading mean, not a continuous reference and
not a prediction. Windows are incompletely aligned to those cuff observations;
differences are descriptive, not HR accuracy measurements.

| Recording | Face HR (bpm) | Hand HR (bpm) | Observed session cuff HR (bpm) |
|---|---:|---:|---:|
| S001_V01 | 67.65 | 79.91 | 80.5 |
| S002_V01 | 67.77 | 72.27 | 72.5 |
| S003_V01 | 71.26 | 71.73 | 72.5 |
| S004_V01 | 79.87 | 80.02 | 82.0 |
| S005_V01 | 72.09 | 65.70 | 89.0 |
| S006_V01 | unavailable | 54.85 | 95.5 |

The large differences in S005/S006 are a further reason not to equate the
engineering good label with verified physiological HR. Some session pulse-rate
estimates appear compatible with cuff HR, but the lack of continuous contact
reference prevents waveform or exact timing validation.

ROI/method ablations change coverage substantially. Counts below are good
**primary nonoverlapping intervals out of 180** for each column, pooled only
descriptively over the same six subjects:

| Method | Forehead | Left cheek | Right cheek | Combined face | Cheeks | Hand hull | Hand center |
|---|---:|---:|---:|---:|---:|---:|---:|
| GREEN | 16 | 51 | 13 | 27 | 48 | 64 | 78 |
| CHROM | 10 | 4 | 4 | 34 | 18 | 18 | 3 |
| POS | 1 | 5 | 1 | 6 | 6 | 48 | 22 |

These coverage differences support inspecting facial ROI choice and acquisition
quality. They do not justify selecting a BP model after viewing this cohort.
Reliable adjacent primary pulse templates were sparse: the selected GREEN path
has three face-template comparisons in S003 (median r=0.858) and one hand-template
comparison in S004 (r=0.963); other sessions lack supported adjacent pairs.
That is insufficient evidence of stable physiological morphology.

**No delay candidate passed all configured checks**, including across alternate
ROIs/methods: 270 delay rows were invalid and 10,350 uncertain. The selected
combined-face/hand GREEN path has zero qualified primary delay intervals in
every recording. Numerical lags still exist and are retained for inspection:

| Recording | Unqualified primary lag median (ms) | Lag SD (ms) | Median diagnostic uncertainty floor (ms) |
|---|---:|---:|---:|
| S001_V01 | -29.9 | 111.0 | 143.4 |
| S002_V01 | -5.2 | 59.0 | 67.4 |
| S003_V01 | 20.3 | 63.5 | 38.4 |
| S004_V01 | 15.6 | 47.1 | 44.0 |
| S005_V01 | 66.5 | 44.2 | 54.0 |
| S006_V01 | 11.3 | 131.9 | 101.0 |

These summaries include uncertain numerical estimates and exclude unresolved
boundary maxima; they are not accepted transit-time estimates. Temporal
variability, limited pulse quality, phase/correlation disagreement, algorithm
disagreement and acquisition irregularity prevent a reliable physiological
interpretation. No PTT or PWV result is reported.

All **18 cuff readings** are associated with their correct session IDs.
**Zero readings have verified temporal alignment** under the source-provenance
rules; the three S004 end-time notes remain explicitly legacy/unconfirmed.
Only S003 and S004 meet the prespecified minimum dual-signal coverage for
session features. Consequently the three-subject minimum for exploratory
subject-held-out baselines is not met: **no actual BP model was fitted or
evaluated, and no BP predictions or BP accuracy metrics were generated**.
Implemented mean/ridge baselines and leakage controls are tested on synthetic
fixtures; the final real-data `bp_feasibility.json` says `not_evaluated`, with
`eligible_subjects: 2` and `bp_accuracy_validated: false`.

The final full regression run passed **229 tests**. After final reporting
changes, all **32 pilot-specific tests** passed again. Compilation and whitespace
checks passed. `artifact_verification.json` additionally checks array shapes,
all frame counts, monotonic clocks, raw artifact hashes, current source hashes,
18 unique cuff observations, and unchanged checkpoint/workbook hashes. Exact
environment pins are in `environment_freeze.txt`; the final manifest includes
Python/platform/library versions and the code revision. Figures were inspected
locally without publishing participant images.

Recommended next steps, in priority order:

1. **Prioritize signal acquisition and physical timing validation.** Repeat a
   small instrumented study with stronger facial pulse visibility, stable ROI
   positioning, documented camera settings and synchronized contact pulse
   references. Test camera/readout timing at the actual ROI positions with a
   known optical timing signal. Synthetic recovery alone does not validate the
   camera-plus-extraction system.
2. Recover the original frozen manifest and use recordings that meet the frozen
   30 Hz/native-spatial preprocessing contract for direct model comparison.
   Keep any future sampling adapter separate, versioned and independently
   validated. Do not alter Frozen V1 weights, architecture or its contract.
3. Review actual cuff starts/ends against video, document synchronization
   uncertainty and verify cuff device identification, hand orientation/support
   and reference geometry. Preserve unknown events as unknown.
4. Expand subject and repeated-session BP coverage after pulse quality and
   timing controls are adequate. Maintain subject grouping and compare simple
   predictors before considering a neural BP model.

The present pilot establishes that synchronized face/hand RGB can be extracted
reproducibly and that some intervals contain plausible pulse-rate structure.
It does not yet demonstrate reliable face-to-hand physiological timing,
validated waveform morphology on these videos, or usable BP estimation.
