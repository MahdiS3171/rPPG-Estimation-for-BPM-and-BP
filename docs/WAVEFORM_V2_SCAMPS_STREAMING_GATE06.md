# Gate 06: SCAMPS streaming multi-ROI inputs

Date: 2026-11-05. Outcome: the mask-independent input pipeline passes engineering
checks and constructs all 20 intended windows. **Training remains disabled.**
Finite geometric coverage is not validated skin quality: visual review found a
forehead polygon over hair in P000005. Mask correspondence remains **UNVERIFIED**.

## Revisions and protection

Fetched heads before editing:

- `main`: `27e54556025e2eff4e1b4091001a2eaf28667cc7`.
- PR #1, draft/unmerged: `243d41b729e028dad90182e14927a5c499dfd530`.
- PR #2, draft/unmerged: `2264fbe2aa2b3b0474674fe8f6193c5155a31dcb`.

Work branch: `feature/waveform-v2-scamps-gate06`, based on PR #2's current head.
Implementation commit: `1fc3c6f222b24f8138cf62c058d9af1e5a6fa53e`.
The new draft PR targets `feature/waveform-v2-scamps-gate05`. No predecessor PR
or `main` is changed or automatically merged. Exact final commits/PR URL are
provided in the handoff and PR metadata.

All Gate 06 source/report files are additive. V1 architecture, losses, training,
checkpoint, official splits, prior algorithms, dataset classes, PURE and Gate 05
reader remain unchanged. The frozen checkpoint was not loaded or invoked.
Only its named companion JSON manifest was read for configuration verification;
its SHA-256 matches the source-pinned official value
`da5ba397b3be389deec8821f8ea16944b5f5b93e2834bb79c3c147eb3d1f89dc`.

Private `Dataset/` contents and actual S007-S011 recordings were never opened,
enumerated or processed. No collected-subject video, training, BP regression,
neural waveform inference, model selection or classical performance benchmark
was run. CPU facial landmark inference is the explicitly requested geometry
step; it does not evaluate the frozen waveform model.

Original SCAMPS source checksums were verified against Gate 05 before and after
each recording's diagnostics. Raw recordings, generated images and arrays are
excluded from Git. Existing historical outputs/reports were not modified; each
actual diagnostic run used a fresh ignored output directory.

## Verified V1 contract and reuse

Source review covered `roi.py`, `region_masks.py`, `detection.py`, `datasets.py`,
`processing.py`, `signals.py`, `window_priors.py`, `classical.py`, architecture
documentation and their tests. The official freeze-manifest configuration
confirms 30 Hz, 300 samples, the following order, min_pixels=50, min_valid_fraction
0.9, max_gap_sec=0.15, skin_mask=false and confidence thresholds 0.5.

| Contract | Implementation |
|---|---|
| ROI order | forehead, left_cheek, right_cheek |
| RGB channels | RGB_R, RGB_G, RGB_B |
| Prior order | GREEN, CHROM, CHROM_WIN, PBV, POS_WIN, OMIT |
| Input | float32 `[B,3,9,300]`, with bool ROI/prior masks and float32 quality |
| ROI geometry | Existing `roi.FACE_ROIS` indices and `region_masks.get_face_masks` rounded/clipped convex hulls |
| RGB measurement | Existing `extract_frame_rgb_by_roi`; finite-pixel mean, insufficient support stays NaN |
| Gap handling | Existing `processing.resample_trace`, bounded 0.15 s on the nominal grid, no extrapolation |
| RGB standardization | Existing `signals.standardize_channels` per current window |
| Priors | Existing `build_window_priors`, only the current 300-sample RGB window |
| Internal prior filtering | Unchanged method defaults: order-3 0.7-4 Hz Butterworth/filtfilt; CHROM_WIN/POS_WIN retain 1.6 s overlap-add |
| Target | Original d_ppg indices/values retained; separate `standardize_1d` output and target-validity flag |
| Window quality | Original valid source-frame fraction, compared in float64 and reported in float32 |

The official configuration says detector_backend=`auto`; its historically
resolved backend was not recorded, as noted in existing `pilot_frozen.py`.
Gate 06 explicitly uses the available **Tasks CPU** backend and the existing
face_landmarker.task, SHA-256
`64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff`.
This freezes the present extraction implementation without pretending that the
historical UBFC detector backend is known. Preprocessing equality is established
on identical RGB traces, not by claiming identical acquisition across datasets.

The complete machine-readable contract, inspected freeze configuration,
software versions and code hashes are in
[preprocessing_contract.json](waveform_v2/scamps_gate06/preprocessing_contract.json)
and each recording's preparation provenance.

## Streaming extraction and interface

`waveform_v2_scamps_roi.py` streams stored Xsub in 32-frame chunks. It makes one
face-detection call per original frame and keeps only O(N*R) RGB/quality/label
traces. The previous chunk is released before the next read; a weak-reference
test verifies this. RawFrames videos are not decoded by this pathway.

The landmarker gets a rounded uint8 copy of Xsub*255. ROI color measurements use
the original floating Xsub*255 without rounding. Stored Xsub values and source
files are never altered. This explicitly supplies V1 intensity units without
quantizing the measured optical trace. No SCAMPS segmentation mask or chroma
skin gate is used for the admitted path.

Missing detections, nonfinite/out-of-range frames, malformed landmarks and
insufficient per-ROI pixels retain original frame positions, NaN RGB and failure
reasons. No previous polygon, other ROI or geometric fallback is substituted.
Invalid frame/channel/index contracts raise. The caller owns and resets the
detector per recording; the CLI creates and closes a fresh instance each time.

`waveform_v2_scamps_inputs.py` exposes:

- `prepare_scamps_v1_inputs(traces, starts=(0,300))`: NumPy inputs, targets,
  original frame/label indices, raw RGB/PPG, original-valid/imputation masks,
  quality and explicit per-window status.
- `as_torch_inputs(batch)`: CPU conversion of x, roi_quality, roi_valid and
  prior_valid, without constructing or calling a network.

The fixed half-open windows are `[0,300)` and `[300,600)`. There is no reference
HR/BP computation or calibration/context selection. This is a forward-input
contract, not a drop-in training Dataset: no participant identity or y_hr is
invented. Unusable intended windows remain in the denominator with false masks,
zero model channels and status. Original missing RGB remains separate from any
bounded acquisition interpolation; invalid raw targets remain available with
an explicit invalid-target flag and a zero standardized placeholder.

V1's acquisition helper can interpolate from neighbouring source frames just
outside a window, within its fixed gap limit. Only the retained window reaches
prior construction and normalization. This offline convention is preserved,
not advertised as strictly causal processing or measured timing precision.

## Mask alignment investigation

The mask-only diagnostic reads RGB and the two original mask streams; it never
opens or scores d_ppg. For each recording it compares mask indices
`1,149,299,449,597` with RGB indices `i-1,i,i+1`: 50 mask locations, with three
candidate frames each. Stored fractional/negative/over-one mask values are
preserved. No threshold, padding, duplication, truncation, resize or offset is
applied to obtain matching lengths.

The exploratory descriptor is the cosine agreement of continuous spatial edge
magnitudes. It is not an independently validated segmentation/alignment test.
The side-by-side visual montages retain separate original indices. Results:

| Stream | Highest exploratory score at frame i-1 / i / i+1 | Candidate spread range |
|---|---|---|
| skin_mask | 14 / 19 / 17 | 0.00007359-0.00556537 |
| combined_mask | 14 / 17 / 19 | 0.00022250-0.00453305 |

Preferences vary by mask/frame/location and the largely static/similar geometry
does not identify a reproducible global offset. A controlled tiny spatial-edge
fixture shows that the descriptor responds to geometry, while identical static
candidates are indistinguishable. Neither control certifies actual temporal
mapping. Decision: **UNVERIFIED**; extraction remains mask-independent.

See [mask_alignment_diagnostic.json](waveform_v2/scamps_gate06/mask_alignment_diagnostic.json).
No PPG prediction performance was used to select masks, landmarks or thresholds.

## Actual ROI coverage and input construction

All ten already-admitted examples were inspected with fixed settings. Each row
has 600 intended frames, 600 detections and 600 valid finite measurements for
each ROI. There were zero missing detections or insufficient-support failures
in this specific example set; the tests exercise those failure paths separately.

| File | Forehead / left-cheek / right-cheek valid frames | Constructed windows |
|---|---|---|
| P000001 | 600 / 600 / 600 | 2 / 2 |
| P000002 | 600 / 600 / 600 | 2 / 2 |
| P000003 | 600 / 600 / 600 | 2 / 2 |
| P000004 | 600 / 600 / 600 | 2 / 2 |
| P000005 | 600 / 600 / 600 | 2 / 2 |
| P000006 | 600 / 600 / 600 | 2 / 2 |
| P000007 | 600 / 600 / 600 | 2 / 2 |
| P000008 | 600 / 600 / 600 | 2 / 2 |
| P000009 | 600 / 600 / 600 | 2 / 2 |
| P000010 | 600 / 600 / 600 | 2 / 2 |

Totals: **6,000 detections / 6,000 intended frames**, **18,000 / 18,000 supported
ROI measurements**, **20 / 20 constructed windows** and **360 / 360 finite prior
channels**. Each input window has shape `[1,3,9,300]`; a CPU Torch tensor of that
shape was constructed without a waveform-network call. Per-window hashes,
source checksums, failure denominators, masks and min/max pixel counts are in
[roi_coverage_and_inputs.json](waveform_v2/scamps_gate06/roi_coverage_and_inputs.json).
No HR error, waveform correlation or classical-model ranking was calculated.

**Coverage measures finite landmark-polygon support, not anatomical correctness,
skin purity or pulse quality.** The P000005 forehead overlay visibly overlaps
bangs/hair. Its quality=1.0 is still the V1 source-valid fraction; it must not be
described as 100% optical quality. This was recorded without changing polygons,
thresholds or reclassifying coverage after looking at labels. No fallback was
needed or evaluated. See
[visual_roi_review.json](waveform_v2/scamps_gate06/visual_roi_review.json).

## Memory, repeatability and timebase integrity

The declared streaming array budget is **128 MiB**. The conservative workspace
estimate is **109,411,200 bytes (104.34 MiB)**; traced Python/NumPy extraction
peaks ranged from **89,295,338 to 89,330,068 bytes** (about 85.2 MiB). A 32-frame
float64 Xsub chunk contains 44,236,800 RGB bytes; no 600-frame video or collection
was materialized. Persistent compact traces are separate from frames.

An RSS guard samples every 2 ms and checks before/after each chunk. The total
process limit is **1 GiB**, with observed peak **724,062,208 bytes (690.52 MiB)**,
including library/model allocations and final CPU Torch import/conversion.
The RSS limit was not exceeded. Extraction took approximately 14.8-16.9 seconds
per recording in the final run. HDF5 time chunks span 75 frames, so bounded reads
still incur decompression overhead. These measurements/estimates are software
instrumentation, not an OS allocation sandbox; very brief native peaks can be
missed and tracemalloc omits some native memory. See
[memory_timebase_integrity.json](waveform_v2/scamps_gate06/memory_timebase_integrity.json).

Final-review fixes preserve float64 threshold comparison and reject overflowing
target variance. The fresh final-code run produced **identical input hashes for
all 20 windows and identical six ROI overlay images** to the earlier diagnostic
run. A separate two-detector CPU check on P000001 frames `[0,8)` also produced
bit-identical ROI RGB/validity (maximum RGB difference 0). This is local bounded
repeatability evidence, not a cross-version/hardware guarantee. See
[native_repeatability_check.json](waveform_v2/scamps_gate06/native_repeatability_check.json).

Every original frame/label index is retained. Nominal seconds are explicitly
`frame_index/30` on the documented synthetic frame grid. MediaPipe's rounded
internal millisecond coordinate is a processing API requirement, not acquired
timestamps. Capture timestamps are null, physical clock verification is false,
and no source/PPG shift or mask offset is applied. An unchanged classical prior
can still have method-dependent phase/edge behavior. Index preservation is not
proof of biological pulse-arrival timing, face-hand delay or clinical BP validity.

## Calibration and a separate future context proposal

The admitted engineering interface requires adaptation=false and has an empty
calibration-index list. The existing V2 30-second calibration contract is not
modified, instantiated or silently shortened.

A separately reviewed future experiment could predeclare calibration `[0,150)`
(nominal 0-5 s), guard `[150,180)`, and prediction `[180,480)` (300 samples,
nominal 6-16 s), with `[480,600)` unused. Context must be label-free, disjoint
from prediction, fixed before outcomes are inspected and compared against a
no-context ablation. This is a proposal only. File IDs identify recordings, not
stable people or avatar identity. Official split provenance is retained; these
ten examples, including their Val/Test entries, remain engineering-exposed and
are not an independent generalization benchmark.

## Tests, checks and exact changed files

Final command:

```powershell
.\.venv\Scripts\python.exe -m scripts.verify_waveform_v2_scamps_inputs_safe
```

```text
Ran 119 tests in 5.583s
OK
GATE06_SAFE: run=119 failures=0 errors=0 skipped=0 blocked_io_attempts=0 neural_execution_allowed=False
```

This includes **35 new Gate 06 tests** and **84 existing non-neural tests**.
The input-only allowlist selects Gate 02/03 contracts, Gate 05 fixture tests,
V1 window priors, facial masks and synthetic timebase/pixel contracts. Torch
Module execution, torch.load and backward are actively blocked. The other 285
declared tests are not selected; the broader Gate 05 suite includes random
network forwards and is intentionally not rerun in this gate. There are zero
runtime skips. See [verification_results.json](waveform_v2/scamps_gate06/verification_results.json).

Tests cover exact equality with V1 Dataset.__getitem__ on analytical RGB,
helper/filter/order consistency, bounded gap masks, missing detections, malformed
frames/landmarks/masks, wrong channel/index/timebase contracts, NaNs/overflow,
quality threshold precision, source immutability, chunk release/determinism,
failed priors and retained denominators. The initial fixture run caught and
fixed a source-file/code-hash field collision; regression assertions preserve
both provenance namespaces. Syntax checks on all six added Python files pass;
CLI help succeeds and existing/source-tree output directories are refused.

All changed paths are **new** relative to PR #2:

```text
rppg_lab/waveform_v2_scamps_roi.py
rppg_lab/waveform_v2_scamps_inputs.py
rppg_lab/waveform_v2_scamps_mask_audit.py
scripts/prepare_waveform_v2_scamps_inputs.py
scripts/verify_waveform_v2_scamps_inputs_safe.py
tests/test_waveform_v2_scamps_inputs.py
docs/WAVEFORM_V2_SCAMPS_STREAMING_GATE06.md
docs/waveform_v2/scamps_gate06/preprocessing_contract.json
docs/waveform_v2/scamps_gate06/roi_coverage_and_inputs.json
docs/waveform_v2/scamps_gate06/mask_alignment_diagnostic.json
docs/waveform_v2/scamps_gate06/memory_timebase_integrity.json
docs/waveform_v2/scamps_gate06/native_repeatability_check.json
docs/waveform_v2/scamps_gate06/visual_roi_review.json
docs/waveform_v2/scamps_gate06/verification_results.json
```

Reproduce bounded diagnostics with the existing local dependencies/model and a
fresh output directory:

```powershell
.\.venv\Scripts\python.exe -m scripts.prepare_waveform_v2_scamps_inputs --scamps-root SCAMPSData --out-dir outputs/scamps_gate06_NEW
```

Final local visual artifacts, not in Git:

```text
outputs/waveform_v2_scamps_gate06_20261105_02/
  roi_overlay_review.png
  roi_P000001_frame0000.png  (also 0300, 0599)
  roi_P000005_frame0000.png  (also 0300, 0599)
  mask_candidates_P000001.png
  mask_candidates_P000005.png
```

## Gate 07 recommendation

Review the input contract and the forehead/hair finding first. Predeclare an
independent optical/skin quality admission test or a separately labeled
mask-independent extraction ablation; do not optimize it against target PPG
performance or quietly change V1 geometry. Retain the present pathway as the
explicit V1-polygon baseline and keep SCAMPS masks unverified until their mapping
and semantics have independent evidence.

Then validate the corresponding input contracts, lawful access, clocks and
reference pairing for permitted real public datasets, and freeze recording/
participant/domain splits and synthetic/real weighting. Review a small,
separately authorized adaptation-disabled development experiment only after
those gates. A shorter-context experiment needs its own explicit specification.
No training or real-participant evaluation is authorized or performed here.
S001-S006 remain historically inspected; S007-S011 stay reserved and inaccessible.

SCAMPS attribution/license remains at
[ATTRIBUTION.md](waveform_v2/scamps/ATTRIBUTION.md). This work establishes input
engineering behavior, without clinical BP or physiological timing claims.
