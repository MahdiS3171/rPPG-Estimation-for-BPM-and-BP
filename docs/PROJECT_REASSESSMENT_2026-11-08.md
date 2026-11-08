# Project reassessment — 8 November 2026

## Strategic brief

**We have a reproducible baseline, not a validated waveform/timing/BP system.**
V1's UBFC result is real but domain-specific: external HR MAE rises to 15.780 bpm,
versus 7.168 for fixed GREEN fusion. The face–hand pilot supports only one strict
independent window, with a reported delay below its declared uncertainty floor.
No timing-verified cuff association exists. More synthetic optimization would
not resolve these scientific uncertainties.

The six PRs are intact, open, draft and unmerged. They form one linear 19-commit
stack above main, with no reviews, comments or CI results. Their implementation
is predominantly additive and their saved SCAMPS comparison reproduces. The
important result is heterogeneous waveform/polarity/harmonic failure, not test
counts or geometric coverage. The handoff's main quantitative findings agree
with code and saved evidence; no V2 training took place.

The user has now specified **existing phones and cuff only**, prioritized public
waveform data and requested a model that uses the 60-fps recordings. We will
preserve V1 as a 30-Hz comparator and make native 60-Hz waveform reconstruction
the next candidate direction. The current 60-to-30 adapter deliberately discards
some bandwidth; the prior filters impose an additional limitation. This is a
testable model/input issue, not proof that capture angle is the main pilot defect.

New work already completed: acquisition-confounding simulation; exact adapter/
filter transfer audit; safe regression/result replay; a bounded official-source
MPU-rPPG development download. Its MP4 has 34,496 samples at container 60 fps and
the candidate CSV has 34,496 indexed PPG rows. All decoded frame hashes are unique,
but capture originality, pairing and sensor latency are
still unverified. No training or efficacy scoring is justified from that match
alone. Tricam and VicarPPG-2 are relevant 60-fps access leads; MSPM remains a
multisite timing lead. More suitable real subjects and labels are needed.

Independent workstreams: admit available real data carefully; define matched
native waveform and beat metrics; design the 60-Hz candidate and a paired
60-versus-30 ablation; prepare phone-only clock diagnostics; consolidate the
repository. Keep BP fitting and new volunteer collection deferred until a
specific discriminating study exists. Equipment uncertainty and domain shift
are competing explanations; neither has been proven primary.

The user has now approved history-preserving integration after green CI and
closure of the old drafts, and will pursue Tricam access with their supervisor.
No paid compute or sensor purchase is proposed. The roadmap, decisions and evidence register are maintained
at the repository root. This assessment is an evidence checkpoint, not a claim
that the full research objective is complete.

## Six-PR audit

Fetched origin and checked GitHub's read-only API; API access initially failed
intermittently, then all six metadata/review/check endpoints were retrieved.
Main remains `27e54556025e2eff4e1b4091001a2eaf28667cc7`.

| PR | Audited head | Incremental commits | Incremental changed files | Main contribution / finding |
|---|---|---:|---:|---|
| [1](https://github.com/MahdiS3171/rPPG-Estimation-for-BPM-and-BP/pull/1) | 243d41b | 4 | 19 | PURE inventory, matched interface, timing contract, optional context prototype. No admitted real benchmark. |
| [2](https://github.com/MahdiS3171/rPPG-Estimation-for-BPM-and-BP/pull/2) | 2264fbe | 2 | 17 | Safe HDF5 SCAMPS admission; 600/599 frame/mask ambiguity remains. |
| [3](https://github.com/MahdiS3171/rPPG-Estimation-for-BPM-and-BP/pull/3) | 65ac812 | 2 | 14 | Bounded streaming and V1-compatible tensors; valid geometry is not pulse purity. |
| [4](https://github.com/MahdiS3171/rPPG-Estimation-for-BPM-and-BP/pull/4) | 4b3d01a | 3 | 20 | Optical proxy fails hair case; trusted candidate abstains; patch mixture shifts timing. |
| [5](https://github.com/MahdiS3171/rPPG-Estimation-for-BPM-and-BP/pull/5) | bcd3f98 | 5 | 21 | 480 fixed classical cells; waveform sign/HR/illumination hazards. |
| [6](https://github.com/MahdiS3171/rPPG-Estimation-for-BPM-and-BP/pull/6) | bb5fb59 | 3 | 25 | Frozen V1 inference, 20 windows; heterogeneous performance and failed/abstaining timing controls. |

Each head equals the handoff SHA and the fetched remote ref. Each previous head
is an ancestor of the next; PR bases match that chain. All six are open/draft/
unmerged, with zero reviews, issue comments, review comments, check runs and
commit statuses. This is absence of CI evidence, not a failing CI run.

Cumulative stack: **114 files, 89,469 insertions**, with only `.gitignore` and
`pyproject.toml` modifying main's existing files. The volume is largely retained
reports/metric artifacts. V1 model, loss, inference, training, splits and historical
reports are unchanged. The latest tip already integrates every PR; creating six
new merge operations or cherry-picking them is unnecessary.

Recommended policy: one integration PR directly against main, with a merge commit
retaining the full ancestry. After review and green CI, request user approval
to merge and close superseded drafts while retaining branches. Avoid squash,
force-push, rewriting published history or deleting evidence. A local descendant
branch and additive CI definition are published as
[draft PR #7](https://github.com/MahdiS3171/rPPG-Estimation-for-BPM-and-BP/pull/7),
directly against main. User approval to merge after final checks and close the old
drafts was subsequently received; execution status is on the PR. Remote CI
[passed at head 014ef78](https://github.com/MahdiS3171/rPPG-Estimation-for-BPM-and-BP/actions/runs/42478042383);
the evidence ledger pins that revision separately from subsequent report updates.

## Verification and reconciliation

The [machine receipt](project_audit/20261108.json) records exact SHAs, profile
logs/hashes and replay counts. The two reported Gate 09 profiles reproduce:
175 + 37 tests, no failures/errors/skips or blocked Python I/O attempts. The broader
profile additionally passes 326 tests with 43 documented exclusions; these
profiles overlap. The broader count differs from PR #1's old 297 because the
same runner now includes added SCAMPS tests. Neither number is physiological
evidence or a count of independent observations.

Saved-result verification checks 32 files and 15 PNG file integrities, recomputes
all 500 rows and paired/balanced summaries, and confirms unchanged inputs, masks,
indices, checkpoint and prediction locks. It performs zero neural forwards or
checkpoint loads. The frozen-profile tests load the official identity on CPU
and use synthetic inputs; they do not evaluate participant recordings.
Checkpoint and manifest SHA256 match the handoff exactly.

Review scope: branch ancestry/diff classification; focused loader, input,
preprocessing, timing, adaptation, scoring and safe-verifier code; tracked
scientific reports; independent saved-result replay. We did not repeat actual
SCAMPS extraction, review participant media, reproduce the historic UBFC/rPPG-10
runs, or conduct an exhaustive line-by-line security review.

Material reconciliations:

- Gate 09's GO permits considering a separate plumbing experiment; it does not
  authorize or prioritize Gate 10. The project-level request supersedes it.
- The inherited benchmark interface still prioritizes HR and is explicitly
  draft; the new scientific protocol prioritizes waveform and timing separately.
- Existing 60-to-30 processing is intentional and guarded. It does not make V1
  native-60, and its old reports already acknowledge transfer limitations.
- Contact PPG at a different site is a valuable waveform target, not proof of
  local facial phase. A naive zero-lag loss against finger PPG may encode real
  propagation/sensor latency as model phase; target definition must precede loss.
- No novel independent test evidence was created by reusing exposed SCAMPS or
  replaying historical results. MPU sample files are designated development only.

## Boundaries and next actions

S007–S011 and private Dataset/ and sheets/ contents were not enumerated, opened,
hashed or evaluated. No old tracked file was edited in this reassessment.
The newly authorized online-data work downloaded only two public MPU development
files (~107.36 MB total) into ignored incoming/, with byte count/MD5/SHA256 checks.
No identifiable public sample image or raw physiology is redistributed.

Continue with the [native-60 strategy](NATIVE_60FPS_WAVEFORM_STRATEGY.md),
[real waveform benchmark](REAL_WAVEFORM_BENCHMARK.md), and
[phone-only plan](PHONE_CAMERA_FEASIBILITY_PLAN.md). These steps need usable
reference correspondence and subject identities before training; substantial
compute and any new held-out evaluation require a concrete later decision.
