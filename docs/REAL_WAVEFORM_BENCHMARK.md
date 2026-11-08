# Real waveform benchmark and data admission

Design dated 2026-11-08. A bounded MPU public development sample was subsequently
downloaded under the user's online-data instruction; it is not yet admitted for
training/scoring. No new real efficacy evaluation or split was run. Final protocol and unseen split use require the
corresponding permission and user decision. Reuse current loaders and scoring
functions; avoid building adapters for inaccessible data speculatively.

## Estimands first

1. **Reference waveform reconstruction:** native-index signed waveform agreement,
   shape/derivatives, polarity and spectral HR against the stated contact target.
   Face and finger/ear/wrist PPG differ anatomically. Disagreement combines real
   propagation, optical morphology, clock offset, sensor filtering and model error.
2. **Local optical arrival preservation:** estimate a site's own physiological
   waveform without unmeasured processing latency. It cannot be established by
   forcing a facial estimate into zero-lag agreement with a different-site PPG.
3. **Face–hand optical delay:** compare simultaneous sites using predeclared
   fiducials, calibrated camera row/exposure timing, and appropriate independent
   site references. An HR-labelled or facial-only dataset cannot establish this.

Native signed correlation remains useful and must be reported alongside
lag-aligned diagnostics. Neither score alone separates the above error sources.
Optimizing a per-window lag or polarity against evaluation labels is prohibited.
No single HR-based score will select the next waveform system. The inherited
`waveform_v2_benchmark.benchmark_protocol()` still declares HR as primary and is
explicitly a draft interface; it is not the final scientific protocol.

## Focused dataset decision

| Candidate | Specific value | Access and scale | Eligibility decision now |
|---|---|---|---|
| PURE | Priority real facial video/contact waveform path with existing inventory adapter. | Supervisor access process already pending per handoff; no local root traversal attempted. | No scoring until licence and actual timestamp/reference audit; not two-site timing truth. |
| MSPM | Strongest additional lead for camera/multisite timing. | Institutional agreement and Globus; transfer size/subsetting not confirmed. | Assess application, then audit synchronization and sensor/ROI location mismatch. No automatic download. |
| iBVP | Facial waveform and independent reference-quality labels. | Supervisor-signed academic EULA; ~400 GB compressed. | Lower priority because storage is substantial and supplied BVP is processed/downsampled; not physical timing truth by default. |
| MPU sample | Open development video actually probed at container 60 fps with candidate PPG CSV. | Two official sample files downloaded (~107.36 MB), verified hashes; research use. | Verify correspondence, sensor processing and identity; no timing/training admission from equal counts alone. |
| Tricam / VicarPPG-2 | Native-60 waveform candidates matching the user's new direction. | Controlled research access; licence/supervisor steps. | Request minimal RGB/reference subset and timing documentation; see native-60 strategy. |

[PURE's official page](https://www.tu-ilmenau.de/en/university/departments/department-of-computer-science-and-automation/profile/institutes-and-groups/institute-of-computer-and-systems-engineering/group-for-neuroinformatics-and-cognitive-robotics/data-sets-code/pulse-rate-detection-dataset-pure)
is the authoritative access entry; pending signature status comes from the user
handoff rather than a newly checked email.

The [MSPM primary paper](https://arxiv.org/html/2402.02224v1) describes 90-fps
overhead video, a raised-hand segment, and multisite contact measurements.
Contact sensors include neck and limb locations, so they are not automatically
collocated face/palm truth. Its reported arm-site discrepancy merits scrutiny;
failure to reject a statistical difference is not an equivalence demonstration.
Our recommendation is to audit raw timing, channels and spatial correspondence
before selecting this dataset for delay validation.
[Official access instructions](https://cvrl-web.crc.nd.edu/projects/data/) require
an agreement executed by an authorized institutional signatory and institutional
email; a student's assent is insufficient. Review the
[dataset agreement](https://cvrl.nd.edu/media/django-summernote/2023-07-16/2b230d9b-e2e0-4bdc-97ae-1c08f9ecfeae.pdf)
with the institution, including permitted processing and publication scope.

The [iBVP authors' README](https://github.com/PhysiologicAILab/iBVP-Dataset)
specifies ear PPG, 124 three-minute sessions, 30-fps RGB, a supervisor-signed EULA,
and approximately 400 GB compressed. Distributed BVP is filtered and reduced to
the video rate. Some participant media have additional consent restrictions.
Those properties support waveform research only after permission/processing
audit; they do not provide a known unfiltered local facial arrival time.

## Admission receipt required before evaluation

- Written access scope and storage/consent limitations; source/version hashes.
- Participant/session/device grouping and all previous exposure, including any
  training overlap from pretrained checkpoints. Identity uncertainty stays explicit.
- Original video indices and timestamps, unit/epoch, dropped/duplicate frames,
  actual cadence, exposure and readout metadata where available.
- Raw reference modality/site/rate, timestamps, clock relation, filtering and
  latency; distinguish documented synchrony from independently calibrated error.
- Separate fields for HR eligibility, waveform-target eligibility, and physical
  timing eligibility. Do not use a generic admitted=true to erase uncertainty.
- Define calibration/development/validation/unseen groups before label inspection.
  Dataset schema inspection must be bounded to a designated development sample;
  opening all reference files for inventory is itself recorded as exposure.

## Minimal matched experiment when admitted

Use fixed anatomical forehead/cheek ROIs with unknown optical purity explicitly
labelled, original window indices, identical source support and 10-second windows
for the initial V1-compatible comparison. Predeclare window stride and edge guards
from development data. Preserve separate ROIs for timing; do not select the best
ROI or method using per-window reference HR/correlation.

Compare fixed GREEN, CHROM and POS_WIN, frozen V1, then at most one justified V2
candidate. State all sign conventions before reference evaluation; retain signed
results even when an algorithm produces inverted pulses. Report native signed r,
normalized waveform error, derivative agreement, polarity/harmonic failures,
HR error and intended/accepted coverage. Include all failures and subject-balanced
summaries; paired comparisons need the same intended windows and separate
common-valid coverage. Resample reference only under the audited clock contract.

Beat timing is enabled only where fiducial and reference validity are supported.
Freeze the detector/matching tolerance on development signals, report missed and
extra beats, and avoid reporting matched-beat error alone. Do not call an
unverified face–finger offset model latency. Lag-aligned correlation is diagnostic
and cannot repair the stored output or certify absolute timing.

Use participant-level intervals for uncertainty and subject/session/device/domain
separation. Lock development choices and the final evaluation before any new
held-out use. Substantial training gets a single compute/admissibility proposal
after the matched baseline establishes a tractable failure and a suitable target.
