# Repository audit and implementation plan (2026-10-06)

## Scope and baseline

Inspected the package, all script entry points, tests, dependency declarations,
documentation, and the ZIP inventory before implementation. No AGENTS.md was
found in the workspace or its project parents. The working tree was clean.
The ZIP contains older code, checkpoints and CSVs, but no raw videos (the UBFC
and rPPG-10 directories are empty archive entries). It is preserved as an archive,
not used as a second source tree. The five existing unittest tests passed.

## Current architecture

| Concern | Existing implementation / finding |
| --- | --- |
| Entry points | Classical UBFC/rPPG-10/consensus benchmarks; single-video extraction; waveform, HR-quality, video, video-prior and BP MLP training; plotting and artifact audit. Scripts mostly have main guards. |
| Video/clock | OpenCV in roi.py, video_datasets.py and the rPPG-10 benchmark. POS_MSEC with nominal-FPS fallback; one path treats frame indices as seconds when FPS is unavailable. Video clips seek by nominal FPS. |
| Face/ROI/RGB | MediaPipe legacy FaceMesh; polygons for forehead and cheeks; face trace is an equal average of three ROI means, not a full-face skin segmentation. Missing faces are dropped. |
| rPPG | GREEN, CHROM, CHROM_WIN, POS, POS_WIN, PCA, ICA, PBV, LGI, OMIT in classical.py. PBV/LGI/OMIT explicitly approximate. ICA silently falls back to PCA. |
| Filtering | Duplicate Butterworth implementations in signals.py and classical.py; forward/backward filtering is zero phase, but failures silently return unfiltered signals. ECG has a separate appropriate band. |
| HR/quality | Welch HR with log-parabolic spectral peak refinement; heuristic confidence/SNR. Frame quality mixes global brightness, sharpness, pixel count and motion with fixed weights. Learned quality targets use reference errors and are supervision, not inference-time measurements. |
| Datasets | UBFC subject folders, rPPG-10 cropped ROI videos/ECG, feature/label session JSON. Tensor windows and record dictionaries retain some IDs. |
| Splits | HR scripts split participants before windows. BP MLP already uses train/validation/test subjects and train-only imputation/scaling. HR has no locked final test or strict cross-dataset test. |
| Models | Several PyTorch temporal/video models; prior-preserving waveform and HR models. No validated BP data or trained dual-region BP model. |
| Outputs | NPZ ROI caches, Torch checkpoints, CSVs, plots; fixed output names can overwrite runs. Configuration is argparse, with many method constants in source. |
| Dependencies | Flat rppg_lab package; no evident package import cycle. Some script-to-script imports couple datasets and training. Installed MediaPipe exposes Tasks but no solutions, so current FaceMesh extraction fails in this environment. |
| Maintenance | models.py (808 lines), train_hr_quality.py (639), train_video_prior_hr.py (609) mix research concerns. Duplicate window/split/metric helpers. No confidently removable dead experiment: preserve ablations and checkpoint compatibility. Core paths are generally relative/CLI paths rather than machine-specific absolute paths. |

## Scientific and technical risks

1. Dropping undetected faces and quality-filtered frames, followed by unrestricted
   interpolation/extrapolation, hides gaps. min_quality can silently revert to
   accepting every frame. Do not reuse this path for timing research.
2. UBFC target resampling starts at common-support t0 while retained video samples
   start at the next actual grid point; truncating arrays does not fix a fractional
   grid offset. Fix target interpolation onto the exact video timestamps with a test.
3. Zero-phase filtering removes filter group delay, not edge distortion or
   morphology distortion. Independent segment boundaries, adaptive projections,
   overlap-add weighting, component polarity and landmark tracking can still
   change apparent face-hand phase. A 30 FPS video cannot acquire sub-frame
   information simply by upsampling or FFT zero-padding.
4. phase_delay_at_hr can return coherence ~1 for a single Welch segment, which
   is degenerate evidence. The new path must require multiple segments for
   reporting coherence. Periodic lag ambiguity and polarity must remain explicit.
5. rPPG-10 infers ECG sampling rate from video duration, assuming exact matching
   support. Combining cropped videos by minimum array length assumes synchronized
   starts. These assumptions need independent acquisition metadata.
6. Video crop failures silently use center crops; unknown FPS defaults to 30;
   failed clip reads repeat the last frame. These are HR ablation behavior,
   inappropriate for BP timing. Crop/clip cache names omit source/config details.
7. HR video-prior window matching uses s/fs rather than recording timestamps.
   Lag-tolerant training loss does not establish physiological timing fidelity.
8. Subject splits need explicit duplicate-ID and overlap checks. BP feature-key
   discovery over the full dataset is schema peeking; new baselines must use a
   fixed feature schema or discover keys from training only. Legacy BP JSON lacks
   cuff timing/provenance and silently skips missing labels.
9. Normalized pulse amplitude/area are relative units, not calibrated blood
   volume. Face/hand delay is an optical inter-site surrogate, not established PTT.
   A cuff reading is intermittent and cannot label arbitrary frames continuously.
10. Dependencies advertise NumPy >=1.23 while pulse features use np.trapezoid
    (NumPy 2 API). Shape mismatches in regression metrics can broadcast silently.

## Plan before refactoring

Preserve flat package imports, classical formulas, training models/checkpoints
and original script CLIs. Add cohesive modules inside rppg_lab rather than moving
everything into src. Use stdlib JSON dataclass configuration. Add structured
signals/video/ROI observations; one-pass video reading with source timestamps;
pluggable face/hand detectors (Tasks and legacy adapters); configurable landmark
masks and an explicitly experimental skin gate; gap-limited common-grid resampling,
segment filtering and raw quality diagnostics. Add label-free timing/morphology
features, cuff association schemas, strict subject partitions, simple sklearn BP
baselines and evaluation. Save raw traces, provenance, exclusions and debug plots
in new run directories. Make only tested correctness fixes to shared legacy code.
Use synthetic delay/gap/video tests plus the existing model smoke tests. Real
recording validation remains necessary because no subject video was supplied.
