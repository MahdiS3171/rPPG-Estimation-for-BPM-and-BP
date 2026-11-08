# Waveform and face–hand timing research

Updated 2026-11-08. Lead: Codex, within the active runtime. Current branch:
`research/native-waveform-development`. This is the project roadmap; historical
Gate documents remain evidence, not current work assignments.

## Strategic assessment

We have a reproducible facial waveform baseline and a substantial diagnostic
pipeline. We do not yet have a generalizable waveform estimator, a reproducible
physiological face–hand delay, or evidence that delay adds BP information.
The project objective is broader than the recent synthetic training proposals.
Training more parameters on the same ten exposed synthetic clips is not the
critical path. Gate 10 is not started.

Two investigations proceed together: real-video waveform reconstruction and
camera timing feasibility. BP modelling depends on credible timing or on an
explicitly approved, different morphology hypothesis. HR is a secondary sanity
outcome. A negative timing result is a valid research outcome.

The user confirmed **existing phones and cuff only** on 2026-11-08. Prioritize
camera calibration and waveform research; assume no sensor purchase or loan.
Known equipment does not currently provide independent two-site timing truth.
The user subsequently prioritized available online data and a model that uses
native 60-fps recordings. Adopt a native-60 V2 candidate and paired FPS ablation
as the next model direction. Do not assume camera angle is the main failure or
repeat volunteer collection without a discriminating protocol.

## Hypotheses and present status

| Question | Present evidence | Next discriminating observation |
|---|---|---|
| Can facial waveforms generalize? | V1 is useful on UBFC; external HR deteriorates substantially. | Matched native-waveform comparisons on newly admitted real data. |
| Do reconstructed timestamps preserve local pulse timing? | Processing controls fail or abstain; sensor/reference clock calibration is missing. | Known acquisition timing controls and appropriate site references. |
| Is simultaneous face–hand optical delay repeatable and resolved? | Pilot has one independent strictly accepted window; no repeatability established. | Calibrated phone acquisition, repeated observations, uncertainty smaller than the effect. |
| Does delay improve BP estimation? | No timing-verified cuff associations; no credible incremental test. | Only after the preceding evidence, subject/session-separated controlled comparison. |

## Workstreams and exit conditions

1. **Camera feasibility, equipment-limited parallel work.** The data-free acquisition experiment
   is complete: unmeasured row timing confounds physiological delay, and waveform
   mismatch can reverse its estimate. Next use the
   [phone-only calibration design](docs/PHONE_CAMERA_FEASIBILITY_PLAN.md).
   File timestamps and interpolation cannot certify exposure timing. If the
   phone-only setup cannot bound the uncertainty, retain a negative/inconclusive
   timing conclusion and continue waveform research; do not escalate BP training.
2. **Real-waveform evidence.** PURE access is pending. Its existing adapter is
   inventory-only. Follow the [benchmark design](docs/REAL_WAVEFORM_BENCHMARK.md),
   audit licence/clock/reference/site, then lock subject splits before tuning.
   MSPM is the most relevant additional timing-data lead; institutional access
   remains a user decision. iBVP is a lower-priority waveform alternative.
   A bounded official MPU-rPPG development sample was subsequently downloaded
   under the user's online-data instruction: one container-60-fps MP4 and one candidate
   PPG CSV (~107.36 MB total). Pairing and sensor timing remain unverified;
   no scoring or unseen evaluation occurred. Tricam/VicarPPG-2 are native-60 leads.
3. **One native-60 V2 candidate after label admission.** Start with fixed GREEN/CHROM/POS and
   frozen V1 on matched windows. Determine polarity, harmonic and domain failures
   before fitting a compact 600-sample/10-second model. The
   [native-60 strategy](docs/NATIVE_60FPS_WAVEFORM_STRATEGY.md) defines the matched
   60-versus-30 ablation and characterizes adapter/filter bandwidth. A finger reference is not the
   same physiological site as the face: forcing zero lag can teach the wrong
   local arrival time. Detailed architecture and substantial training budget are
   decided after data admission and baseline evidence. Personalization and
   semantic patch refinement remain optional later ablations.
4. **Integration complete.** The six original PRs formed a linear 19-commit
   stack. [PR #7](https://github.com/MahdiS3171/rPPG-Estimation-for-BPM-and-BP/pull/7)
   integrated it with history preserved after user approval and green CI; the
   six old drafts are closed and branches retained. Current development stays
   on one branch. No force-push, raw data release or branch deletion is authorized.

## Active development campaign

The approved integration is complete on main at `50cb264`; PR #7 was merged with
history preserved, the six superseded drafts are closed, and all branches remain.
No new experiment PR stack is being created.

The user subsequently authorized broad experimentation on current datasets,
including adaptive classical selection/fusion, model expansion and loss variants.
The first Train-only, subject-fold linear study is complete: 50:50 SCAMPS pooling
worsened its real-source metrics; fixed CHROM_WIN remained stronger. That finding
motivated explicit source-mixture controls rather than assuming more data helps.

The [adaptive screen](docs/ADAPTIVE_WAVEFORM_SCREEN_PROTOCOL.md) trained 21 compact
configurations on the local GPU in 157.42 seconds, using one immutable 168-window
archive. It tests signed learned gating, residual correction, raw-RGB temporal
models, dual-band heads, four loss schedules, past-only context, dropout and
shuffled targets. Preliminary aligned shape/derivative gains from the hybrid
morphology model do not establish improved native phase or generalization.

The completed [confirmation study](docs/ADAPTIVE_WAVEFORM_CONFIRMATION_PROTOCOL.md) expands
to all 26 original UBFC Train subjects across five folds, with source-mixture,
loss, context, width and multi-head comparisons, fixed ensembles, and additional
seeds. Local time caps are 15 minutes per compact screen. No cloud, new equipment,
original Val/Test use, private-data access or large video-model campaign is
included. Native-60 model operation has engineering tests; current labels remain
30 Hz and cannot validate extra physiological detail at 60 Hz.

## Ranked risk register

Current findings and negative results are consolidated in the
[campaign report](docs/WAVEFORM_DEVELOPMENT_CAMPAIGN_REPORT.md). Matched controls
favor learned waveform correction; input-adaptive gating adds little. All 123
neural development fits and 20 linear solves are complete. Longer TCN context
improves shape; attention adds no clear increment. ROI dropout/consistency
improves robustness; retain the TCN consistency variant as the leading local
candidate, with fixed classical and simple hybrids as comparators. Its current
OOF shape r is .811 vs CHROM_WIN .729; native-r gains remain uncertain and HR
does not clearly improve. A one-time full-Train fit exports a separate local
checkpoint without a new validation score. These are 30-Hz labels, not native-60
supervision. No timing or BP qualification follows.

Next execution priorities: preserve this candidate and its software contracts;
complete documented public-reference/permission checks; admit a second real
waveform domain when available; then lock a third domain before evaluation.
Do not continue broad model selection on the same 26 people indefinitely.

| Priority | Risk / consequence | Mitigation and stop criterion |
|---|---|---|
| P0 | Optical delay is smaller than unknown acquisition/fiducial error. | Establish row timing, exposure, estimator and repeatability error separately. No resolved timing claim when the effect is indistinguishable from those errors. |
| P0 | No new admitted real waveform and clock evidence. | PURE permission; assess MSPM access. Keep morphology eligibility separate from physical timing eligibility. |
| P1 | Cross-domain failure; coherent illumination looks like pulse. | Strong fixed baselines, negative controls, full failure denominator and independent domains. A spectral peak is not a pulse-source label. |
| P1 | Reference target differs anatomically or in filtering/latency. | State the estimand and sensor processing. Native contact correlation is descriptive until local timing is independently established. |
| P1 | Exposed splits or repeated windows masquerade as independent evidence. | Record all consumption; group by subject/session/domain; no new locked test use without specific approval. |
| P1 | Integration changes frozen evidence or private data. | Additive branch, pinned hashes, protected-data exclusion, review before main merge. |
| P2 | Sparse cuff events and confounding produce spurious BP accuracy. | No BP fitting now; require audited cuff intervals and an incremental-value study first. |
| P2 | Growing infrastructure substitutes for research. | Reuse current loaders/scorers; one roadmap, one evidence ledger; stop redundant synthetic model campaigns. |

## User decisions and autonomy envelope

- **Repository resolved:** user approved this one reviewed integration PR into
  main after final checks, preserving ancestry and closing old drafts without
  deleting branches. Future major main changes still require their decision.
- **Data route selected:** user will pursue the native-60 Tricam access process
  with their supervisor, alongside existing PURE. MSPM is a separate timing lead.
  No agent contract acceptance, external email or controlled-data download is implied.
  The open MPU sample needs correspondence documentation, rather than a signed
  application for the bounded research use already performed.
  Storage needs must be confirmed before any substantial transfer.
- **Equipment resolved:** phones and cuff only; no repeat purchase question.
- **Compute deferred:** local CPU diagnostics and synthetic checks proceed.
  No paid compute, substantial training or bulk data download is proposed now.

S007–S011, private Dataset/ and sheets/ remain closed. Frozen V1 and all historical
sources/results remain untouched. No one should infer clinical BP validity from
this project. See [decisions](DECISION_LOG.md), [evidence](EVIDENCE_LEDGER.md), and
the [full audit](docs/PROJECT_REASSESSMENT_2026-11-08.md).

Use the [current research index](docs/waveform_research/README.md) for protocols
and results. The [organization review](docs/REPOSITORY_ORGANIZATION.md) records
reversible cleanup and historical paths that must stay stable.
