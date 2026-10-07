# Multi-ROI facial trace extraction (Phase 1)

The timing-aware recording path now preserves facial ROI identity before signal
processing. This enables later waveform experiments to compare or weight regions
whose pulse morphology may differ in quality despite similar HR spectra. No new
neural model, learned attention, pixel encoder, BP architecture or morphology
feature is introduced. Extraction alone establishes no BP accuracy improvement.

## Acquisition and masks

`process_recording` uses one `VideoReader`, one face detector and the existing
hand detector. Each selected detector runs once per decoded frame. The original
frame index and timestamp must be returned unchanged by injected detectors.

`get_face_masks(frame_rgb, observation, strategies=DEFAULT_FACE_ROIS, ...)`
constructs all requested masks from the same observation. It never detects a
face. `roi.FACE_ROIS` remains the sole definition of the forehead and cheek
vertex groups. The existing rounding, clipping and convex-hull rasterization
are preserved. `combined` is exactly the union of the three component hulls;
overlapping pixels contribute once. Its RGB is a pixel-weighted union mean,
not an equal-weight average of three ROI means. Experimental `full_skin` and
chroma gating remain available; the gate is evaluated once per face/frame.

`extract_frame_rgb_by_roi` calls the same RGB aggregation primitive for every
mask. RGB is in RGB channel order and acquisition intensity units (normally
0–255). Finite pixels determine support. A mask with fewer than `min_pixels`
returns three NaNs and `extraction_valid=False`; valid black pixels return zero
RGB with `extraction_valid=True`. Clipped pixels remain included, as before;
their clipping fraction is diagnostic rather than a new rejection rule.

## Returned schema

No additional result class is required; the existing types are reused:

```text
DualROIResult
  face_rois: dict[str, RegionResult]
    forehead / left_cheek / right_cheek / combined (default keys)
      rgb: RGBTrace
        timestamps: float64[N]       original decoded frame clock
        values: float64[N, 3]        NaN RGB for failed measurements
        validity_mask: bool[N]       measurement validity for this ROI
        roi_name: str                ROI key, or legacy "face" for selected ROI
        quality: dict[str, ndarray[N]]
          skin_pixel_count          finite pixels inside the applied mask
          mask_pixel_count          applied mask area after optional skin gate
          mask_coverage             mask area / frame area
          extraction_valid          measurement succeeded
          brightness                mean RGB intensity / 255
          clipping_fraction         fraction with any channel <=1 or >=254
          detector_confidence       existing confidence provenance (face: NaN)
          motion_px_per_sec         existing face-center motion shared by ROIs
      rppg: PhysiologicalSignal
        timestamps: float64[U]       result.shared.timestamps
        values: float64[U]           configured classical method; missing/edges NaN
        preprocessing               segment boundaries, failures, filter/gap rules
        quality                     same dictionary as RegionResult.quality
      quality: dict[str, Any]        existing region_quality indicators and reasons
      observations: list[RegionObservation] of length N
        frame_index / timestamp     original source identity
        landmarks                   shared detection geometry; no extra detection
        valid                       geometry validity, not RGB validity
        reason                      local pixel-support failure or detection reason
  face: RegionResult                aliases face_rois[config.face_roi]
  hand: RegionResult                existing hand result
  shared.original_timestamps        common original clock
  shared.timestamps                 common uniform clock
  hr / delay / bp_features / exclusions / video / config
                                    existing recording result fields
```

`region_quality` is applied identically to each ROI. It exposes
`valid_frame_fraction`, `missing_frame_fraction`, `signal_valid_fraction`,
`interpolated_grid_fraction`, diagnostic `<metric>_mean`, `illumination_std`,
HR/SNR and other existing signal indicators, `usable`, and `reasons`.
Supplemental ROI failures appear in their own quality/preprocessing and
observations; the recording's legacy exclusions and BP decisions remain tied
to the selected face/hand signals. A bad supplemental cheek cannot exclude a
usable selected combined signal.

## Processing and classical priors

One uniform grid is constructed from the full decoded source clock before any
ROI processing. `process_rgb_trace` resamples, extracts a signal and computes
quality against that caller-provided grid. Each trace retains its own validity.
`max_gap_sec` limits interpolation between supported samples, with explicit
interpolation flags. There is no extrapolation. Long gaps split filtering and
classical projections into separate segments; short segments and edge guards
remain excluded. Raw RGB and raw validity are never overwritten by interpolation.

`build_classical_priors(trace, grid, config, methods=DEFAULT_PRIORS)` returns
`dict[str, PhysiologicalSignal]` with one waveform per named existing method.
It resamples the trace once and reuses `extract_signal` for each method.
Callers must pass the recording's shared grid. The default prior list is the
existing GREEN, CHROM, CHROM_WIN, PBV, POS_WIN and OMIT list. Priors are computed
on demand; `RegionResult.rppg` already contains the configured method for each
ROI. No algorithm formulas or normalization/polarity conventions change.

Face–hand timing still uses the configured `face_roi` and matching paired
segment boundaries, independent of the supplemental ROI waveforms.

## Configuration, artifacts and compatibility

The additive `PipelineConfig.face_rois` field defaults to all four masks. Old
JSON configs load unchanged; `face_roi` still selects the legacy face result.
Explicit lists must be nonempty, unique and contain known strategies. The
`requested_face_rois` property appends `face_roi` if necessary. For example,
`face_rois=["forehead"]` with `face_roi="combined"` extracts those two masks;
`face_rois=["combined"]` restores combined-only extraction/processing cost.
`full_skin` can be selected or added without changing canonical geometry.
Hand-only runs expose `face_rois={}` and keep the old missing-face placeholder.
The added result field defaults to an empty dictionary for old constructors.

Use the existing CLI with `--config configs/multi_face_roi.json`. `save_result`
keeps every existing NPZ/JSON key and adds:

- `signals.npz`: `face_roi_names` (Unicode strings) and, per ROI,
  `face_roi_<name>_rgb`, `_valid`, `_rppg`, `_rgb_uniform`, `_interpolated`,
  `_reason` (Unicode strings; empty means no reason), and each raw diagnostic.
- `summary.json`: `face_rois[name].quality` and `.preprocessing`.

All raw arrays align with `timestamps_original`; uniform arrays align with
`timestamps_uniform`. Face geometry is stored once under the existing face
landmarks/bbox keys. Arrays load with `allow_pickle=False`. The debug callback
retains `face`/`hand` masks and observations; additive `face_<name>` masks allow
custom callbacks to inspect each ROI. The built-in overlay continues displaying
the configured legacy face ROI and hand.

The older `FaceROIExtractor`/`extract_rgb_trace`, their caches and HR training
scripts remain unchanged. Their historical `"face"` trace averages ROI means
equally and uses a different missing-data path; it must not be confused with
the timing-aware pipeline's union mask or used as a Phase 2 timing source.

## Verification and Phase 2 decisions

`tests/test_multi_face_roi.py` adds deterministic tests for distinct masks,
union equality including overlaps/clipping, previous combined-mask numeric
compatibility with/without gating, known RGB means and black pixels, independent
missingness, one gate/detection/decode pass, shared original/uniform timestamp
identity, owned detector lifecycle, config selection, per-ROI priors and gaps,
pickle-free artifact round trips, and unchanged selected face/hand BP outputs.
The encoded synthetic dual-site fixture injects a 120 ms delay and cheek-only
loss; it checks recovered delay within one source frame. Existing HR, model,
BP, timestamp and hand regressions remain in the complete suite.

Verification on 2026-10-07: all 49 tests passed (32 existing + 17 new), syntax
compilation and `git diff --check` passed. A separate 600-frame encoded synthetic
run made exactly 600 face and 600 hand detector calls, retained 100% valid
forehead/right-cheek/combined samples and 88% valid left-cheek samples, and
recovered the injected 120 ms delay as 124.17 ms. NPZ/JSON artifacts, per-ROI
GREEN/CHROM priors and legacy debug plots were checked. This uses injected
geometry and synthetic color variation, not real physiological measurements
or a fresh validation of MediaPipe detection. Local ignored artifacts are in
`outputs/smoke_multi_face_roi_20261007/run/`.

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m compileall -q rppg_lab scripts tests
git diff --check
```

For Phase 2, use the new recording traces rather than legacy cached RGB. Default
multi-ROI extraction adds mask/statistic and classical-processing cost, but no
extra decoding or detection. Per-ROI motion remains the existing shared face
motion indicator. ROI quality thresholds are existing heuristics, and fixed
chroma gating remains experimental. Classical projections/filtering and their
polarity/phase need morphology validation against synchronized references;
this infrastructure does not establish waveform or BP feasibility.
