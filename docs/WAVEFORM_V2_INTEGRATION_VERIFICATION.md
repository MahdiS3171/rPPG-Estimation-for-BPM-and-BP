# Waveform v2 integration verification — 2026-11-05

Repository: https://github.com/MahdiS3171/rPPG-Estimation-for-BPM-and-BP

## Revisions and scope

- Fetched `origin/main` before editing; both local `main` and fetched `origin/main`
  were `27e54556025e2eff4e1b4091001a2eaf28667cc7`, matching the earlier audit.
- Work branch: `feature/waveform-v2-gates02-03`; no commit was made on `main`.
- Integration commit: `9a6b465`.
- Prototype commit: `42321217ca5ddf361df1330f6a7b0b2c0a2810a9`.
- This report and the safe verification runner are added in a subsequent commit.
  The exact final commit and draft PR URL are provided in the handoff/PR.
- No training, BP regression, GPU job, participant inference, real-video benchmark,
  dataset download or real PURE inventory was run.

## Exact changed-file inventory

Only `.gitignore` modifies a file that existed at the base commit. It adds
`PUREData/`, `PURE/`, `data/pure/`, `cache_pure*/`, and `incoming/`. Existing private
dataset/checkpoint/output exclusions remain in place. `git check-ignore` confirms
all five new namespaces using nonexistent synthetic probe paths.

New files from the supplied archive (manifests relocated as indicated):

```text
rppg_lab/waveform_v2_registry.py
rppg_lab/waveform_v2_benchmark.py
rppg_lab/waveform_v2_timing_contract.py
scripts/prepare_waveform_v2_pure.py
scripts/audit_waveform_v2_timing_synthetic.py
tests/test_waveform_v2_infrastructure.py
tests/test_waveform_v2_timing_contract.py
docs/WAVEFORM_V2_INFRASTRUCTURE_GATE02.md
docs/WAVEFORM_V2_TIMING_CONTRACT_GATE03.md
docs/WAVEFORM_V2_SUBJECT_ADAPTATION_DESIGN.md
docs/waveform_v2/GATE02_PATCH_MANIFEST.json
docs/waveform_v2/GATE03_PATCH_MANIFEST.json
```

Additional new files:

```text
rppg_lab/waveform_v2_adaptation.py
scripts/verify_waveform_v2_safe.py
tests/test_waveform_v2_integration.py
tests/test_waveform_v2_adaptation.py
docs/WAVEFORM_V2_ADAPTATION_PROTOTYPE_GATE04.md
docs/WAVEFORM_V2_INTEGRATION_VERIFICATION.md
```

The pre-existing untracked ZIP and
`Waveform_v2_Codex_GitHub_Handoff_2026-11-03.md` were retained locally and are not
included in commits. No existing experimental document was edited.

## Patch integrity and integration fixes

Verified ZIP SHA-256:

```text
9a8c11f5213473c47f7209f78503ef74420300099c584a65d44f900a2169f038
```

It matches the supplied expected digest. All 12 ZIP entries passed CRC checks,
relative-path/duplicate-name/symlink checks, and destination nonexistence checks
before extraction. Gate 02's five size/hash entries and Gate 03's eleven size/hash
entries all matched; the latter covers every other ZIP entry. Both JSON manifests
are preserved byte-for-byte under `docs/waveform_v2/`. Their original paths still
describe archive-relative provenance, not a mutable installed-file manifest.

Nine extracted files remain byte-identical to the archive, including both
manifests. Two new Markdown documents (`WAVEFORM_V2_SUBJECT_ADAPTATION_DESIGN.md`
and `WAVEFORM_V2_TIMING_CONTRACT_GATE03.md`) have trailing-space hard breaks
removed so that the final commit-range whitespace check passes. The registry
has the following reviewed post-extraction fixes:

1. A malformed `/FullPackage` row such as JSON `null` previously raised an
   uncaught `AttributeError` while reading timestamps. It now becomes an explicit
   invalid-reference audit row, with scoring still disabled.
2. Private/reserved path components are rejected before filesystem checks;
   symlink PURE roots are rejected before enumeration. These reinforce the
   explicit-input contract and do not authorize any dataset access.

Registry original SHA-256:
`e3ab2fcb394bb890088617f230fea0a487f242ce2fe8ec0448c48a51b1d9ef4d`.
Registry installed SHA-256:
`5dea95692ccc51ff3516034b899fd648eed14a921dad666dd7d51e32399402eb`.
Three synthetic integration regressions cover these fixes. The original 26 tests
were not rewritten.

## Environment and executed checks

Windows, existing `.venv/Scripts/python.exe`; Python 3.11.0, NumPy 2.4.6,
SciPy 1.17.1, PyTorch 2.12.1+cu132. All neural checks used CPU tensors;
the reviewed runner hides CUDA devices and sets Torch threads to one.
Python 3.10 and other dependency combinations were not independently tested.

| Check | Result |
|---|---|
| Original two Gate 02/03 modules, before fixes | 26 passed, 0 failed |
| Gate 02/03 plus integration regressions | 29 passed, 0 failed |
| Subject-context prototype module | 21 passed, 0 failed |
| Reviewed runner: 247 existing + all 50 new tests | 297 passed, 0 failures, 0 errors |
| Runner exclusions / unittest skips | 43 explicitly excluded / 0 runtime skips |
| Runner protected-path Python I/O attempts | 0 |
| `py_compile` on all 11 added Python files | Passed |
| Synthetic timing CLI, temporary output directory only | 12 scenarios; expected outcomes below |
| Synthetic CLI refuses an existing output directory | Passed; expected exit code 2 |
| PURE CLI `--help` only | Passed; no real inventory |
| `git diff --check` | Passed |
| `git diff --check <base>...HEAD` | Passed after normalizing four supplied Markdown hard breaks |
| Protected tracked-file diff against fetched base | Empty |

The prototype initially had one incorrect test expectation: a uniformly slower
calibration clock was expected to be rejected even though it was regular and
covered the declared interval. The test was corrected to inject an actual gap.
All final prototype checks pass; no implementation change was needed for that
test correction.

Reproduction from the repository root:

```powershell
.\.venv\Scripts\python.exe -m unittest -v tests.test_waveform_v2_infrastructure tests.test_waveform_v2_timing_contract
.\.venv\Scripts\python.exe -m unittest -v tests.test_waveform_v2_integration tests.test_waveform_v2_adaptation
.\.venv\Scripts\python.exe -m scripts.verify_waveform_v2_safe
git diff --check
```

Reviewed runner's final output:

```text
Ran 297 tests in 53.761s
OK
SAFE_VERIFICATION: run=297 failures=0 errors=0 skipped=0 excluded=43 blocked_io_attempts=0
```

The timing CLI generated **2 PASS_SYNTHETIC_ONLY, 6 FAIL, 4 ABSTAIN_UNVERIFIED**
at 30 and 60 Hz. FAIL/ABSTAIN are the expected negative-control outcomes, not
unit-test failures. Generated JSON and synthetic test media lived in new OS
temporary directories, without writes to historical `outputs/`.

## Explicit exclusions from existing tests

The runner prints every excluded test ID and reason. Its reviewed selection is:

| Exclusion | Count | Reason |
|---|---:|---|
| `test_frozen_timebase.OfficialFrozenTests` | 8 | Opens the official checkpoint/manifest |
| `test_pilot6_delay_feasibility...test_official_frozen_identity_when_available` | 1 | Reads official checkpoint/manifest |
| `test_waveform_v1.WaveformTrainingIntegrationTests` | 7 | Runs a synthetic training epoch |
| `test_waveform_finalization.ReviewArtifactTests` | 17 | Uses the training fixture in class setup |
| `test_waveform_v1.InvalidROIBatchNormTests` | 1 | Optimizer steps |
| `test_waveform_multi_roi` training CLI and optimizer tests | 2 | Training plumbing/optimizer steps |
| `test_dual_roi` BP roundtrip and baseline selection | 2 | Fits BP regression/baselines |
| `test_pilot6` grouped BP-baseline test | 1 | Fits exploratory BP models |
| `test_waveform_v1.WaveformV1LossTests` gradient-executing tests | 3 | Conservative exclusion of backward calls |
| `test_rppg10_external...test_no_training_shape_reference_or_physiological_timing_code` | 1 | Conservative source-string filter also matches its forbidden-call list |

The last test is a read-only inspection check; excluding it is conservative, not
a claim that it trains. The runner is a reviewed allowlist, not a general-purpose
test sandbox. Re-review it when existing tests change. Its Python I/O audit hook
blocks reads/writes/enumeration in known private data, cache, checkpoint and
output directories; native-library I/O is not covered by Python's audit hook.
Existing native video tests use generated temporary media and fake detectors.

## V1 and data preservation evidence

- The only base-tracked file changed is `.gitignore`. `git diff` against the
  fetched base shows no changes to V1 model/loss/training/inference/split code,
  existing tests, old reports, configs, checkpoints or data.
- The new adapter calls the original V1 forward without hooks or mutation.
  Synthetic tests check every V1 state tensor remains equal and strict loading
  of an in-memory V1 state dictionary has no missing/unexpected keys.
- The official frozen checkpoint was not loaded, converted or rewritten.
  Tests requiring its contents were excluded. Historical output directories were
  not written; the reviewed runner blocks their Python I/O.
- `Dataset/` contents and actual S007-S011 recordings were never opened,
  enumerated, decoded or used. No hashes of reserved recordings were computed:
  hashing would itself violate the no-access restriction. Synthetic tests may
  use dummy reserved identifiers to test rejection; those are temporary fixtures,
  not participant recordings.
- This is source-diff, test-state and operation-scope evidence, not an OS-level
  forensic proof for untracked data. Avoiding data access takes precedence over
  creating a new private-data hash inventory.

## Prototype result and remaining limitations

Subject-session adaptation **was implemented**, separate from V1 and disabled by
default. It uses bounded corrections to the already window-dependent V1 gate,
with zero initialization, fixed past calibration, explicit target-clock checks,
mask preservation, missing-context identity and per-window phase-guard fallback.
See `WAVEFORM_V2_ADAPTATION_PROTOTYPE_GATE04.md` for the API, checkpoint keys,
coverage counters and synthetic example.

No accuracy/generalization gain has been measured. PURE is still inventory-only
and unadmitted: access/license, acquisition clocks, reference alignment, real ROI
loading and benchmark execution remain unresolved. The adaptation wrapper
requires truthful same-session clock/provenance and untainted upstream features;
it cannot authenticate caller-supplied tensors. Underflowed V1 weights cannot be
revived, and the CPU phase guard may conservatively reject many candidate changes.
Passing that guard only constrains added changes relative to V1 and cannot
validate V1's own absolute phase, physiological PTT, or BP accuracy.

Recommended next phase: review the draft PR and context/timing contract; obtain
explicit dataset authorization and validate public-data timebases/references;
freeze participant/domain splits and a matched classical benchmark protocol
before considering separately authorized multi-dataset training. Keep S007-S011
reserved. S001-S006 and previously inspected UBFC/rPPG-10 evidence remain
historically exposed, not independent tests. Do not advance to participant
evaluation or BP regression from these engineering checks.
