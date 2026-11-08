# Gate 07 evidence index and changed-file inventory

See the [report](../../WAVEFORM_V2_OPTICAL_QUALITY_GATE07_REPORT.md),
[locked specification](../../WAVEFORM_V2_OPTICAL_QUALITY_GATE07_SPEC.md) and
[annotation guide](../../WAVEFORM_V2_OPTICAL_QUALITY_ANNOTATION_GUIDE.md).
This packet contains metadata, not independent skin/hair labels. Training is disabled.

## Metadata files

| File | Contents |
|---|---|
| [optical_quality_comparison.json](optical_quality_comparison.json) | Ten source hashes, 20 baseline input hashes, fixed config/code hashes; per-ROI indicators, coverage, weights, reasons and missingness. |
| [documented_forehead_comparison.json](documented_forehead_comparison.json) | Fixed P000001/P000005 comparison; color-proxy failure and whole-ROI abstention. |
| [contact_review_index.json](contact_review_index.json) | Coordinator-only filename/opaque-ID mapping, fixed frames and contact-sheet hashes. Withhold during initial blinded review. |
| [human_annotation_template.json](human_annotation_template.json) | 150 pending items with two null reviewer slots and no adjudication. |
| [synthetic_timing.json](synthetic_timing.json) | Twelve Gate 03 synthetic mixture controls, including expected failures/abstentions. |
| [memory_runtime.json](memory_runtime.json) | Per-recording observational runtimes and measured memory peaks/limits. |
| [local_artifact_manifest.json](local_artifact_manifest.json) | SHA256, sizes and relative paths of local PNG/NPZ/license artifacts; no artifact payloads. |
| [retained_array_validation.json](retained_array_validation.json) | Ten retained-array checks, 20 rebuilt input hashes, planned weight changes and per-frame provisional losses. |
| [software_and_validation.json](software_and_validation.json) | Software/source hashes, preregistration and guarded test results. |

Local artifacts are under `outputs/waveform_v2_optical_quality_gate07_20261105_01/`.
Use its ten `contact_sheets/*.png` plus the guide/template for review. The source
images, contact PNGs and retained NPZ arrays stay ignored. The layout-QA mosaic is
an assistant inspection aid, not a human assessment. Preserve SCAMPS attribution
and the included license with any shared packet.

## Changed files relative to the Gate 06 parent

All 20 files are additions relative to `65ac812e69cc7d39e98b1643b7e25df997eef0f5`:
the nine JSON files above, this index, the three linked specification/guide/report
files, and these seven Python files:

| File | Purpose |
|---|---|
| `rppg_lab/waveform_v2_optical_quality.py` | Separate non-physiological descriptors and fixed engineering config. |
| `rppg_lab/waveform_v2_micro_rois.py` | Exact polygon partition, frozen bounded mixtures and whole-ROI confidence/rejection gates. |
| `rppg_lab/waveform_v2_optical_audit.py` | Same-frame streaming collection, unchanged A, reference-free Xsub reads. |
| `rppg_lab/waveform_v2_quality_timing.py` | Gate 03 transfer controls for spatial mixing. |
| `scripts/audit_waveform_v2_optical_quality.py` | Fixed ten-recording audit and local review/array packet generation. |
| `scripts/verify_waveform_v2_optical_quality_safe.py` | Guarded regression runner with optical/timing additions. |
| `tests/test_waveform_v2_optical_quality.py` | Thirty deterministic tests of occlusion, ambiguity, missingness, numerical and timing behavior. |

Earlier tracked files and `.gitignore` are unchanged. The specification was
committed before implementation; its later correction only removes a typographic
`+` and wraps the review-blinding sentence. Pre-existing untracked handoff/ZIP
files are excluded from these commits.
