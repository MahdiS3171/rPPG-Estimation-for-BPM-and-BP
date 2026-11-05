# Waveform v2 — Gate 02: PURE inventory + matched benchmark scaffold

**Implementation state:** local, additive code package; NOT merged into GitHub.
**Pinned source reviewed:** `MahdiS3171/rPPG-Estimation-for-BPM-and-BP@27e54556025e2eff4e1b4091001a2eaf28667cc7`.
**Date:** November 3, 2026.
**Scope:** dataset metadata and reproducible *draft* protocols only; no learned model modified, no training or actual-data inference.

## The precise outcome

Introduced four files without changing any pre-existing project files:

- `rppg_lab/waveform_v2_registry.py`: recognizes PURE's `NN-NN/NN-NN/*.png` + `NN-NN/NN-NN.json` layout, audits the `/FullPackage` waveform and reported timestamps, records file identities, and produces a fail-closed inventory. Optional full PNG content hashing is explicitly opt-in.
- `rppg_lab/waveform_v2_benchmark.py`: draft benchmark protocol and strict per-window baseline-pairing contract. Counts missing/invalid predictions against every intended window; DOES NOT score actual signals or run baselines.
- `scripts/prepare_waveform_v2_pure.py`: offline command-line inventory; does not read the project's private `Dataset` folder and does not overwrite outputs. Omit `--out-dir` for a read-only summary.
- `tests/test_waveform_v2_infrastructure.py`: 12 unit tests using generated synthetic JSON/PNG-name fixtures (not real physiological data).

### Source layout supported

```
PURE_ROOT/
  01-01/
    01-01/
      00000.png
      ...
    01-01.json
  01-02/
    01-02/
      *.png
    01-02.json
```

Frames are **not decoded**. There is no ROI extraction, model invocation or actual-data benchmark at Gate 02. The adapter refuses to invent acquisition timestamps from the image index and a nominal 30-fps rate. PURE source documentation reports a 30-Hz camera and a parallel 60-Hz finger pulse-oximeter reference; independently verified clock alignment is still unknown.

Source evidence:
- PURE official description: https://www.tu-ilmenau.de/en/university/departments/department-of-computer-science-and-automation/profile/institutes-and-groups/institute-of-computer-and-systems-engineering/group-for-neuroinformatics-and-cognitive-robotics/data-sets-code/pulse-rate-detection-dataset-pure
- rPPG-Toolbox PURE folder/parser example: https://github.com/ubicomplab/rPPG-Toolbox/blob/main/dataset/data_loader/PURELoader.py

## Research safety contracts

1. **Reserved subjects:** S007–S011 are prohibited from model development and matched benchmark registration. No videos from S001–S011 are opened by this adapter.
2. **Frozen Waveform v1:** no pre-existing source file, checkpoint, result artifact, or split changed. `FrozenWaveformV1` is included only as a mandatory future comparator name.
3. **Unknown is not approval:** waveform/HR/absolute timing eligibility remains `false` for every discovered PURE recording, even if the JSON contains sample values. Admission requires a separate data-quality and timing review.
4. **No guessed clock:** the `Timestamp` field is audited for ordering, but its physical units default to `unknown`; the video capture clock remains `unknown`. Even an explicitly supplied reference timestamp unit does not independently prove synchronization.
5. **Participant integrity:** `PURE:01` remains in the same partition for all its sequences; split status is `DRAFT_NOT_FROZEN`, not a new pristine test.
6. **No silent failure drop:** benchmark records require an independent intended-window entry, common reference/input identities, and explicit failure reasons. Matched baseline coverage counts remain visible.
7. **No overwrite:** outputs require a fresh directory outside the dataset tree; no default output file is written.

## Example commands — *do not run until PURE is obtained*

```powershell
# Using the project's existing Python environment and a locally downloaded PURE dataset
.\.venv\Scripts\python.exe scripts/prepare_waveform_v2_pure.py --pure-root PURE\RawData

# Create a new, draft-only output directory (will refuse an existing directory)
.\.venv\Scripts\python.exe scripts/prepare_waveform_v2_pure.py --pure-root PURE\RawData --out-dir outputs\waveform_v2_pure_registry_draft01

# Run the synthetic-fixture engineering tests (does not open any real dataset videos)
.\.venv\Scripts\python.exe -m unittest -v tests.test_waveform_v2_infrastructure
```

If a PURE download's verified timestamp unit and explicit access evidence become available, pass `--reference-timestamp-unit milliseconds` or the proven unit and `--access-status verified --license-evidence ... --license-scope ...`. Do not infer units from numeric magnitude without documentation.

Outputs (with `--out-dir`):
- `dataset_manifest.json`: one row per recording, unknowns and blockers included;
- `split_manifest.json`: deterministic draft participant splits, not yet frozen;
- `benchmark_protocol.json`: canonical required comparator names and matched-window contract.

## Test coverage

- Strict PURE directory discovery and provenance;
- No private video opening;
- Waveform value validity, missing/duplicate/nonmonotonic reference timestamps;
- Optional deterministic image hashing;
- Explicit access-evidence requirement;
- Grouped deterministic subjects and multiple recordings;
- Refuse overwriting an existing manifest;
- Require registered evaluation intervals and matching source/reference contracts;
- Explicit missing baseline and failure accounting;
- Reserved-subject and duplicate-prediction rejection.

**Execution completed here:** `python -m unittest -v tests.test_waveform_v2_infrastructure` → 12/12 pass; `python -m compileall` → pass. No full project test suite, no real-data preprocessing, no model evaluation performed.

## Not yet delivered / next gate

- No PURE files were downloaded; access/licensing and actual format/timing have not been verified against a real recording.
- A metadata adapter is **not** a ready-to-train PURE tensor loader. ROI frame extraction, validated reference interpolation and sample rate handling must be added after a real-data admission audit.
- Splits are provisional; dataset/domain holdouts must be chosen before using outcome labels.
- The benchmark interface does not yet produce classical rPPG predictions. It is a scaffold for future matched execution.
- The full existing project test suite and integration in a Python 3.10 environment have not been run.

**Next decision gate:** obtain PURE (or a single permitted sample), inspect actual timestamp and reference JSON values, confirm license scope, then implement timebase-aware ROI/reference adapters and a non-training matched classical baseline smoke test. Do not yet use reserved S007–S011 or claim physiological timing or BP accuracy.
