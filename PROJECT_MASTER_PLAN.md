# Waveform and face–hand timing research

Updated 2026-11-08. Lead: Codex, within the active runtime. Current branch:
`research/project-reassessment-2026-11-08`. This is the project roadmap; historical
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

1. **Camera feasibility, highest priority.** The data-free acquisition experiment
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
4. **Integration.** The six draft PRs are a linear 19-commit stack. Their latest
   tip already contains all six. A local descendant branch preserves the stack;
   existing tracked files remain unchanged. A synthetic-only CI workflow and
   an [integration/review packet](docs/PROJECT_REASSESSMENT_2026-11-08.md)
   are prepared. Request one main-merge policy; no stack merge, force-push,
   public result upload, or branch deletion is authorized by this plan.

## Ranked risk register

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

## Decisions pending and autonomy envelope

- **Repository:** recommend one reviewed integration PR into main, retaining all
  ancestral commits; close superseded drafts after merge, retain branches.
  Main merges still require user approval. Alternative: keep the stack for
  separate review, with greater review/maintenance burden.
- **Data access:** recommend a focused native-60 access request (Tricam or
  VicarPPG-2), alongside existing PURE; MSPM is a separate multisite timing lead.
  No contract acceptance, external email or controlled-data download is implied.
  The open MPU sample needs correspondence documentation, rather than a signed
  application for the bounded research use already performed. Alternative:
  available samples and PURE only, with less high-FPS/multisite evidence.
  Storage needs must be confirmed before any substantial transfer.
- **Equipment resolved:** phones and cuff only; no repeat purchase question.
- **Compute deferred:** local CPU diagnostics and synthetic checks proceed.
  No paid compute, substantial training or bulk data download is proposed now.

S007–S011, private Dataset/ and sheets/ remain closed. Frozen V1 and all historical
sources/results remain untouched. No one should infer clinical BP validity from
this project. See [decisions](DECISION_LOG.md), [evidence](EVIDENCE_LEDGER.md), and
the [full audit](docs/PROJECT_REASSESSMENT_2026-11-08.md).
