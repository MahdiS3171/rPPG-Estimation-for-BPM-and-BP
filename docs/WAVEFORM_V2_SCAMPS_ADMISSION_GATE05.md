# Waveform v2 Gate 05: SCAMPS admission audit

Audit date: 2026-11-05. Decision: **admit the ten examples for engineering only;
do not enable Waveform v2 training or physiological timing validation.**

## Repository and scope

PR #1 was inspected and remains open, draft and unmerged at
`243d41b729e028dad90182e14927a5c499dfd530`.
Fetched `origin/main` remains `27e54556025e2eff4e1b4091001a2eaf28667cc7`.
Work is isolated on `feature/waveform-v2-scamps-gate05`, based on PR #1's branch.
The new draft PR targets `feature/waveform-v2-gates02-03` so its diff contains
Gate 05 only; retarget it after the prerequisite PR is reviewed and merged.
Neither `main` nor PR #1 is modified by this work.

Your pre-existing `.gitignore` addition for `SCAMPSData/` was preserved and is
included with `cache_scamps*/` and `data/scamps/` exclusions. All ten actual raw
paths pass `git check-ignore`. No original `.mat`, frame, video, mask tensor,
preview PNG or waveform plot enters Git. The public split CSV is small metadata,
not a recording download. Snapshot line endings are pinned to LF to keep the
split/license checksums stable on Windows.

Only the explicit `SCAMPSData/` root and its ten direct P000001-P000010 files
were inspected. No `Dataset/` contents, S007-S011 recordings or other participant
data were opened, enumerated or processed. No training, model inference,
model selection, full benchmark, BP fitting or GPU experiment was performed.
V1 checkpoint, architecture, losses, splits, training and historical artifacts
remain unchanged. New local diagnostic files live under a newly created ignored
output directory; existing reports/outputs were not overwritten.

## Evidence boundaries

| Evidence | What it supports | What it does not establish |
|---|---|---|
| Actual local HDF5 data and bounded full-value scans | Keys, shapes, dtypes, layouts, finite/range counts, source hashes, index counts | Authenticity against unpublished publisher checksums, physical clocks or measured physiology |
| Producer `Python.Shape` attributes plus non-square RawFrames and visual review | Explicit native C,W,H,T to T,H,W,C RGB decoding; W,H,T to T,H,W masks | Mask-to-frame temporal mapping when counts differ |
| Official README/datasheet | RGB channel names and intended same-frame synthetic waveform labels; avatar-independent split claim | Independent confirmation of release-specific mask semantics or avatar groups |
| Official split CSV snapshot | Recording-level Train/Val/Test assignments | Identity/texture/lighting-factor independence |
| Paper: 20-second generated sequences and datasheet: 600 frames | An inferred nominal 30 Hz render grid | Embedded/acquired timestamps or real camera precision |

The [official repository](https://github.com/danmcduff/scampsdataset),
[datasheet](https://github.com/danmcduff/scampsdataset/blob/0d3e1fd9915ea52a66af87caa1508b292555f765/Datasheet%20-%20Synthetics%20for%20Camera%20Measurement%20of%20Physiological%20Signals.pdf)
and [paper](https://arxiv.org/abs/2206.04197) were inspected. Documentation is
pinned at `0d3e1fd9915ea52a66af87caa1508b292555f765`; a separate dataset-release
version is not embedded in these files. MATLAB container headers identify the
MATLAB 7.3/HDF5 schema and CPython producer, including file-creation dates; those
dates are not frame acquisition timestamps. The header is retained in inventory.

## Verified actual file structure

All ten contain these 21 root keys, with no root attributes:

```text
RawFrames Xsub au12 au15 au17 au18 au2 au26 au27 au4 au45 au7
combined_mask d_br d_ekg d_pitch d_ppg d_roll d_yaw numerical_mask skin_mask
```

All required fields `RawFrames`, `Xsub`, `skin_mask`, `combined_mask` and `d_ppg`
are present. No variable is nonfinite in any file. The mask-length anomaly is
retained as a warning/admission blocker rather than hidden as a successful match.

| Variable | Native HDF5 shape | dtype | Compression / chunks |
|---|---|---|---|
| RawFrames | `[3,320,240,600]` | float64 | gzip / `[1,20,15,75]` |
| Xsub | `[3,240,240,600]` | float64 | gzip / `[1,15,15,75]` |
| skin_mask | `[240,240,599]` | float32 | gzip / `[15,15,75]` |
| combined_mask | `[240,240,599]` | float32 | gzip / `[15,15,75]` |
| numerical_mask | `[240,240,599]` | float64 | gzip / `[15,15,38]` |
| d_ppg and all other auxiliary waveform/AU fields | `[600,1]` | float64 | none / contiguous |

Full per-file attributes and layouts are in
[dataset_inventory.json](waveform_v2/scamps/dataset_inventory.json).
The native HDF5 axes agree with the README's C,W,H,T description. Their producer
attributes describe the original Python arrays as T,H,W,C, so direct HDF5 reads
must be explicitly transposed, not interpreted as already NumPy video layout.
Channel indices 0,1,2 follow documented R,G,B order; no BGR swap is made.

Every stored RGB value is finite and lies in [0,1]; values remain float64 and are
not rescaled or resized. In P000001 at indices 0,300,599, Xsub differs from the
central RawFrames crop by up to `0.003921568859368563` (about one 8-bit level), so
the reader uses the requested stored stream rather than fabricating one from
the other. No interpretation of that small difference is asserted.

PPG values are finite, with maxima ranging approximately 3.09-3.67 across the
ten files. They are preserved as stored: a unit-amplitude waveform is not assumed.
The paper's upstream 300 Hz waveform-generation rate is not assigned to the
released 600-sample frame-level `d_ppg`. No target interpolation, differencing,
lag fitting, reference HR selection or sample shift is applied.

## Source identities and official sample splits

| Filename | SHA-256 | Official split |
|---|---|---|
| P000001.mat | `66134a63723035ae5da6abc53785abe64c87295473e9d7e20d3752d2088a52ae` | Train |
| P000002.mat | `1d66ec0b5fdef03ca083c24ca87b0db852799e1c3ee7335ec56d371e09a11b4f` | Train |
| P000003.mat | `6411b40f99958a3b6cf0fc61ac460e41c947c1feca80dbb801fc2d31c74ebeef` | Train |
| P000004.mat | `aed82a87b3451379debcb6745158805b0c5ba8ae82f829dd4b2ef37d49b27178` | Val |
| P000005.mat | `f83788832701b85a82239c35626a4a68a194b537c6947a8b94403fe767c7a37d` | Test |
| P000006.mat | `18310c736ac51303b6e4b783633b897c9a72ee7690d11c7c3b2b1f1fa3f684cc` | Train |
| P000007.mat | `3786c4d0d59b8d175fbca9fd8de347f8bee821d3adef4de33649180ebbbe7515` | Train |
| P000008.mat | `f564249af83319fb00ab2f00c27c4a5efde65766bcb6496139e6c21eefed1887` | Train |
| P000009.mat | `c97a1b50047df6ec7e7aa646514b99040396634a903c820a5dc3cdab4fe11d22` | Train |
| P000010.mat | `847d0942cf21e4cf00e40ab121298becdaa2a34813361f8ef7644e054f0284df` | Train |

Each checksum was reverified during subsequent lazy access; source bytes stayed
unchanged. No publisher checksum list was used, so these are reproducible local
identities, not an independent certification of the download's provenance.

## Mask and timing admission decision

All mask streams have **599 samples versus 600 video frames/PPG samples**. There
is no metadata explaining which frame is omitted or whether masks describe
differences/transitions. Neither a first-frame nor last-frame omission is assumed.

Skin masks contain fractional values; P000005, P000008 and P000010 also have
small negative values (down to about -0.00392163). Every combined mask stream
reaches `1.003921627998352`. These are not silently cast to booleans, thresholded,
clipped, padded, repeated, shifted or resized. Anti-aliasing or export arithmetic
could be explanations, but that remains unverified. Numerical-mask class IDs
and any fractional boundary values are inventoried without guessing semantics.

The adapter returns RGB and `d_ppg` using identical preserved indices. With the
actual short mask streams it returns no masks attached to RGB, marks segmentation
validity false and rejects `require_segmentation=True`. Separate mask diagnostic
windows retain their own indices and have `frame_indices=None`. Equal-length
fixture masks can be decoded directly, with finite flags but no training approval.

Actual files contain no capture timestamps, sample-rate fields, clock units,
independent synchronization evidence or avatar/appearance-asset IDs. The intended
RGB/PPG pairing is supported by synthetic provenance and equal index counts,
not a separately measured timing experiment. No SCAMPS result is reported as
face-hand PTT, BP evidence, physiological delay or real camera timing precision.
See [alignment_timing_audit.json](waveform_v2/scamps/alignment_timing_audit.json).

| Eligibility | Decision |
|---|---|
| Read-only engineering inspection and synthetic index plumbing | Admitted for these ten files |
| Frame/PPG development-window decoding | Available with explicit finite/range flags and documented index pairing |
| Waveform v2 multi-ROI or subject-context training | Disabled; unresolved preprocessing/context protocol |
| Mask-supervised or mask-weighted processing | Blocked pending mask temporal/semantic clarification |
| Independent generalization benchmark on these ten | Not admitted; all are engineering-exposed |
| Absolute/physiological timing or BP validation | Not admitted |

## Split and leakage protocol

The public split snapshot has 2,800 unique recording IDs: 2,000 Train, 400 Val,
400 Test. It contains the literal two-field header `nk,Split` over three-field
rows (recording ID, URL, split). The dedicated parser verifies the full snapshot
SHA-256 and this exact profile; generic `DictReader` column shifting is avoided.
The ten examples have 8 Train / 1 Val / 1 Test assignments, retained unchanged.

The datasheet calls the official partitions avatar/person independent. That is
a documentation claim, not a fact we can independently establish from local
files/CSV: no identity, texture, hair, clothing or illumination asset IDs are
available. We do not infer identity from appearance. Actual auxiliary traces
show shared rendering factors: all ten have the same zero d_pitch/d_roll/d_yaw
traces and nine zero AU traces. This establishes limited confounder diversity,
not shared avatar identity. The full hash-group evidence is in
[split_audit.json](waveform_v2/scamps/split_audit.json).

Keep every recording's context and target windows in the same partition. Never
randomly divide overlapping windows between training and evaluation. Preserve
the official assignments for provenance; the inspected sample Val/Test files
must not be advertised as a new untouched benchmark. Seek full asset/group
metadata before claiming identity/domain independence. The sample is an
engineering fixture, not a representative generalization study.

For later separately authorized multi-dataset training, use explicit domain and
dataset sampling weights rather than allowing 2,800 synthetic recordings to
outnumber real participants automatically. A development proposal could cap
synthetic exposure at a predeclared minority of batches (for example 25%), then
balance real datasets/participants within the remaining batches. That number is
an illustration for review, not a selected training setting. Keep synthetic and
real validation metrics separate, select models using a declared real-development
protocol, and reserve a genuinely independent real domain for final evaluation.
S001-S006 are historically exposed; S007-S011 stay reserved and inaccessible.

## Adapter and memory behavior

`rppg_lab/waveform_v2_scamps.py` uses explicit-root nonrecursive discovery,
read-only HDF5 handles and a strict layout profile. It rejects unexpected shapes,
conflicting producer axis annotations, indirect/external/virtual HDF5 sources,
unsupported numeric types, private/reserved paths and symlink paths. It preserves
file checksums, frame/label indices and untransformed stored values.

`read_window` defaults to at most eight frames and a decoded payload limit of
64 MiB. It reads only the requested hyperslabs, returns T,H,W,C RGB, raw `d_ppg`,
frame/label/segmentation/pixel validity flags and descriptive finite/range
fractions. These are engineering validity flags, not calibrated optical quality.
No model, landmark detector, prior computation or training loader is introduced.

The full scan streamed all numeric variables of all ten files. Compressed source
files total 1,344,960,263 bytes; their uncompressed arrays total 24,874,752,000
bytes. The largest audit read was **180,000 bytes**; no entire video/collection
was allocated. Lazy memory measurements for P000001, Xsub at index 300:

| Window | Returned RGB+PPG bytes | Traced Python/NumPy peak | Sampled native RSS increase |
|---|---:|---:|---:|
| 1 frame | 1,382,408 | 2,769,740 | 11,386,880 |
| 8 frames | 11,059,264 | 22,123,731 | 18,382,848 |

These are in-process measurements, with native RSS sampled every 2 ms, not
allocator-independent upper bounds. Tracemalloc does not cover all native HDF5
allocations. RGB transposition creates a temporary copy. The native cache is
1 MiB; the 75-frame storage chunks imply decompression overhead for short reads.
See [memory_usage.json](waveform_v2/scamps/memory_usage.json).

## Validation and local artifacts

Environment: Windows, Python 3.11.0, NumPy 2.4.6, h5py 3.16.0, psutil 7.2.2.
Use the optional extra `pip install -e ".[scamps]"` to add admission dependencies
without changing V1's required packages. PDF inspection additionally used pypdf
and pypdfium2 locally; these are not adapter dependencies.

- 29 SCAMPS checks: tiny HDF5 fixtures plus one pinned public split-metadata
  snapshot test. Coverage includes axes, indices, short masks, fractional values,
  missing/malformed arrays, auxiliary length mismatches, read-only/lazy bounded
  access, nonfinite/range flags, checksums, determinism and safe path rejection.
- Expanded reviewed engineering suite: **326 passed, 0 failures, 0 errors**;
  **43 explicitly excluded, 0 runtime skips, 0 protected Python I/O attempts**.
  The same existing exclusions cover official checkpoint access, training/gradient
  plumbing, BP fitting and one conservative source-inspection match. Existing
  V1/PURE tests are not rewritten. SCAMPS root/cache paths join the runner's I/O guard.
- Actual sample diagnostics: 10/10 structures accepted for engineering, 10/10
  full finite scans passed, 10/10 two-frame/label/independent-mask window checks
  passed, 10/10 source hashes unchanged. No training or timing admission enabled.
- Syntax checks, JSON/TOML parsing, pinned snapshot hashes and the complete
  Gate 05 commit-range whitespace check passed. CLI help and existing-output
  refusal were checked without repeating the full value scan.

Final reviewed runner output:

```text
Ran 326 tests in 47.510s
OK
SAFE_VERIFICATION: run=326 failures=0 errors=0 skipped=0 excluded=43 blocked_io_attempts=0
```

Local generated previews (ignored, not in Git):

```text
outputs/waveform_v2_scamps_gate05_20261105_01/diagnostics/
  synthetic_frames_masks_preview.png
  synthetic_ppg_preview.png
```

The first shows three Xsub frames and separately labelled skin/combined-mask
indices; no temporal alignment is implied by the side-by-side layout. The second
plots one raw PPG vector against synthetic index, without fabricated timestamps.
Both were visually checked and retain SCAMPS attribution. Inventory/timing/split/
memory JSON copies are committed under `docs/waveform_v2/scamps/`; raw media is not.

Exact Gate 05 file inventory (relative to the PR #1 head): modified `.gitignore`,
`pyproject.toml`, and `scripts/verify_waveform_v2_safe.py`; added:

```text
.gitattributes
rppg_lab/waveform_v2_scamps.py
scripts/audit_waveform_v2_scamps.py
tests/test_waveform_v2_scamps.py
docs/WAVEFORM_V2_SCAMPS_ADMISSION_GATE05.md
docs/waveform_v2/scamps/ATTRIBUTION.md
docs/waveform_v2/scamps/LICENSE_SCAMPS.txt
docs/waveform_v2/scamps/official_sources.json
docs/waveform_v2/scamps/official_split.csv
docs/waveform_v2/scamps/dataset_inventory.json
docs/waveform_v2/scamps/alignment_timing_audit.json
docs/waveform_v2/scamps/split_audit.json
docs/waveform_v2/scamps/memory_usage.json
docs/waveform_v2/scamps/summary.json
```

Reproduction from the project root (use a fresh output directory):

```powershell
.\.venv\Scripts\python.exe -m unittest -v tests.test_waveform_v2_scamps
.\.venv\Scripts\python.exe -m scripts.verify_waveform_v2_safe
.\.venv\Scripts\python.exe -m scripts.audit_waveform_v2_scamps --scamps-root SCAMPSData --out-dir outputs/scamps_gate05_NEW --official-split docs/waveform_v2/scamps/official_split.csv --source-manifest docs/waveform_v2/scamps/official_sources.json --scan-values
```

The audit CLI is restricted to P000001-P000010. The full-value pass decompresses
about 24.9 GB sequentially, bounded in memory; omit `--scan-values` for metadata
and small-window diagnostics only. Neither mode downloads data or trains a model.

## Recommendation and next decision gate

SCAMPS is useful and admitted **now for engineering development**, but the
current adapter/sample are **not ready to enable a Waveform v2 training run**.
Resolve the 599/600 mask correspondence and value semantics, or predeclare and
validate a mask-independent alternative; build the actual multi-ROI RGB/prior/
target pipeline; and freeze the split, loss and domain-weighting protocol first.
The nominal 20-second clips cannot contain the current default 30-second subject
calibration plus a later target. A shorter disjoint calibration policy or an
explicit no-context SCAMPS ablation needs separate development review.

Then request a separately authorized, non-commercial synthetic/real development
experiment with real-domain controls. Do not reinterpret synthetic same-index
labels as clinical validation, face-hand optical delay or BP evidence. The
[SCAMPS license and attribution](waveform_v2/scamps/ATTRIBUTION.md) remain part of
the admission contract; no additional dataset download or training is authorized
by this engineering audit.
