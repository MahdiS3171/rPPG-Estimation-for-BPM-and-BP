# Gate 07: optical quality and whole-ROI abstention

Date: 2026-11-05. Branch: `feature/waveform-v2-optical-quality-gate07`.
Parent: PR #3, `65ac812e69cc7d39e98b1643b7e25df997eef0f5`.

## Result

**The color proxy fails to distinguish the documented hair-covered P000005
forehead from P000001.** Both have 600/600 acquisition-valid forehead frames,
median chroma-compatible fraction 1.0, and six equally weighted provisional
patches. This is a negative engineering result, not successful hair detection.
The separate trusted output **abstains on the entire forehead for all 600 frames
of both recordings**, because independent skin confidence is UNKNOWN. A few
acceptable patches cannot override this whole-ROI decision.

The implementation supports whole-ROI rejection for low whole-polygon color
compatibility or independent occlusion evidence, as well as abstention for
insufficient independent skin evidence. Controlled tests exercise all three
paths, including a geometrically valid polygon that is 75% dark hair-like
material while two patches remain locally compatible. A skin-colored-hair
counterexample demonstrates why the color proxy alone cannot establish skin.

**Gate 08: GO for independent annotation and further quality/timing research;
NO-GO for waveform training.** No human labels were collected here. Independent
skin/hair accuracy, physiological signal quality and real-domain generalization
cannot be established from the available engineering evidence.

## Review, preregistration and protected contracts

The latest remote heads were fetched, PRs #1/#2/#3 and their reports reviewed,
and the V1 ROI definitions, mask construction, RGB extraction, quality interfaces
and Gate 03 timing contracts inspected. Final fetch still places PR #3's branch
at the reviewed parent above and main at
`27e54556025e2eff4e1b4091001a2eaf28667cc7`. The preceding draft PRs remain unmerged.

The [fixed specification](WAVEFORM_V2_OPTICAL_QUALITY_GATE07_SPEC.md) was committed
as `0d0fb28ffb27ad0abf2d478367f0027114de1ae6` before implementation and comparison.
A subsequent editorial correction removes an accidental `+` and wraps a sentence
about review blinding; sampling, parameters and decision rules are unchanged.
No thresholds were selected or adjusted using these examples, reference PPG,
HR/BP labels or waveform-model performance.

Baseline A reuses Gate 06's unchanged `_measure_frame` implementation with the
original V1 landmark polygons and continuous RGB*255 measurements. The old
`quality.signal_quality` pulse/HR routine is not used for optical selection.
The new comparison decodes **only Xsub values**. Its legacy trace object's
reference slot contains explicitly NOT LOADED NaNs, solely to call the unchanged
input helper and compare **x** hashes; its target outputs are not evaluated.

All ten source checksums match Gate 06 before and after extraction. All **20
original V1 input hashes** match, and were independently rebuilt again from the
retained baseline RGB arrays during packaging. No earlier tracked implementation,
report, input hash, preprocessing contract, V1 architecture/checkpoint/loss or
official split changes in this branch. No private `Dataset/`, reserved S007-S011
or collected-subject data was accessed. No waveform network/checkpoint call,
training, HR error/model ranking, BP regression or GPU evaluation was run. Only
the existing CPU face landmarker was used for geometry, without new downloads.

## Same-frame methods and confidence schema

A retains original per-pixel polygon means and acquisition validity. B partitions
each polygon into six disjoint 2x3 bounding-box cells whose exact union is the
original mask. Both see identical source frames. B uses the preregistered defaults:
16 finite pixels per patch, at least 95% block support, median clipping <=5%,
median chroma compatibility >=50%, at least two eligible patches, and bounded
normalized scores with a maximum nonzero ratio of 2:1.

Weights are selected from each 300-frame block's optical descriptors and held
fixed within that block. This uses **offline within-block lookahead**, not causal
or subject-session adaptation. At frame 300 the new block may change weights.
Moving landmark/grid boundaries can change sampled pixels even with fixed weights.
Uniform patch weights also differ from a per-pixel polygon mean when patch areas
differ. Candidate B is therefore a separate extraction method.

The whole-polygon median proxy is checked before patch selection: below 0.5
rejects all patches. Independent non-skin/occlusion evidence also rejects the
whole ROI. UNKNOWN independent skin evidence always withholds trusted output.
Provisional RGB is retained for inspection separately. If any positively weighted
patch is unavailable or exceeds clipping limits on a frame, the whole ROI frame
abstains; there is no dynamic rescue or renormalization. Missing outputs are NaN,
validity is false, and applied weights are zero.

Four concepts remain separate in the machine-readable results:

| Concept | Meaning and current status |
|---|---|
| Acquisition | Finite pixels, landmarks and polygon support; measured for A and patches. |
| Anatomy/skin confidence | Independent evidence UNKNOWN on every actual sample; human review pending. |
| Optical reliability | Separate unvalidated color, clipping, brightness, dispersion, change/flicker and movement descriptors. |
| Physiological pulse quality | UNVERIFIED; no pulse-based selection or quality claim. |

The existing YCrCb gate (Cr 133-173, Cb 77-127) is reported as
**chroma-compatible fraction**, an estimated skin-like color fraction. Its
complement is not measured hair/background contamination. Brightness, spatial
chromaticity MAD and temporal change/flicker are descriptive, not direct rejection
criteria. Darkness, heterogeneous color or shadow alone does not establish poor
skin or pulsatility. Optical scores do not create independent skin labels.

## P000001 versus P000005

| Forehead outcome | P000001 | P000005 |
|---|---:|---:|
| A acquisition-valid frames | 600/600 | 600/600 |
| Median chroma-compatible fraction | 1.0 | 1.0 |
| Median clipped-pixel fraction | 0.0 | 0.0 |
| Eligible patches in each block | 6, 6 | 6, 6 |
| Planned weights in each block | six x 1/6 | six x 1/6 |
| B provisional frames | 600/600 | 600/600 |
| B trusted frames | 0/600 | 0/600 |
| Whole-ROI trusted reason | ABSTAIN_SKIN_UNVERIFIED | ABSTAIN_SKIN_UNVERIFIED |

Fixed contact sheets at frames 0,149,299,449,599 show the original polygons,
full image context and ROI insets. Assistant visual QA of the pair is consistent
with the documented P000005 forehead polygon lying over hair, while P000001's
forehead is exposed. This is an assistant observation, not a completed human
assessment or segmentation ground truth; no exact hair fraction is asserted.
The color proxy accepts the documented hair case, so changing only patch weights
does not solve semantic contamination. Withholding both trusted outputs preserves
uncertainty but does not demonstrate useful hair-versus-skin specificity.

See [pair metrics](waveform_v2/optical_quality_gate07/documented_forehead_comparison.json).
The local sheets are `contact_sheets/Rec7cd2f7.png` (P000001) and
`contact_sheets/R8929b290.png` (P000005) under the artifact root below. Their
coordinator mapping must be withheld from initial blinded reviewers.

## Full denominator and per-ROI results

Each entry below is available frames out of 600; F/L/R means forehead,
left cheek and right cheek. A has 600/600 for every ROI. Trusted B has 0/600
for every ROI, preserving whole-ROI abstention in the absence of independent skin
evidence. Values in the proxy column are medians over all 600 original frames.

| Recording | A F/L/R | B provisional F/L/R | A chroma proxy F/L/R |
|---|---|---|---|
| P000001 | 600/600/600 | 600/600/600 | 1.0000/1.0000/1.0000 |
| P000002 | 600/600/600 | 600/600/600 | 1.0000/1.0000/1.0000 |
| P000003 | 600/600/600 | 600/600/600 | 0.8781/0.7487/0.9129 |
| P000004 | 600/600/600 | 584/0/600 | 1.0000/0.9798/1.0000 |
| P000005 | 600/600/600 | 600/600/600 | 1.0000/1.0000/1.0000 |
| P000006 | 600/600/600 | 600/600/600 | 1.0000/1.0000/1.0000 |
| P000007 | 600/600/600 | 600/600/600 | 1.0000/1.0000/1.0000 |
| P000008 | 600/600/600 | 600/600/600 | 1.0000/1.0000/0.9440 |
| P000009 | 600/600/600 | 0/0/0 | 0.0000/0.0000/0.0000 |
| P000010 | 600/600/600 | 600/0/600 | 0.8691/0.0136/1.0000 |

Across **18,000 intended ROI-frames**, A acquisition coverage is 18,000 (100%),
B provisional availability is 14,984 (83.244%), and B trusted availability is
0 (0%; 100% abstention). Provisional unavailability is 3,016 (16.756%). A's
100% acquisition coverage must not be interpreted as 100% optical or pulse quality.

The **60 intended ROI-blocks** retain these primary reasons:

| Trusted block reason | Blocks | Corresponding ROI-frames |
|---|---:|---:|
| ABSTAIN_SKIN_UNVERIFIED | 50 | 15,000 |
| ABSTAIN_WHOLE_ROI_LOW_SKIN_PROXY | 8 | 2,400 |
| ABSTAIN_TOO_FEW_ELIGIBLE_PATCHES | 2 | 600 |

The eight low-proxy blocks are P000009's three ROIs and P000010's left cheek.
Those are color-proxy decisions, not confirmed hair/non-skin labels. P000004's
left cheek has median clipping 0.4615 and too few eligible patches in both
blocks. Its forehead loses 16 provisional frames because a selected patch exceeds
the per-frame clipping limit; no alternate patch is substituted. Its median
whole-forehead clipping is 0.0294. The 50 UNKNOWN-skin blocks include those 16
frames, so the table of block reasons is not a count of provisional availability.

The [full per-ROI report](waveform_v2/optical_quality_gate07/optical_quality_comparison.json)
contains finite/missing observation counts and distributions for every indicator,
both methods' temporal descriptors, failure reasons, weights and coverage. No
confirmed hair/background fraction, sensitivity, specificity or accuracy is
computed because independent labels do not exist. The ten previously exposed
examples constitute exploratory engineering evidence only.

## Numerical, weight and timing behavior

Retained arrays preserve all 600 original indices, A RGB, patch RGB/validity,
optical descriptors, planned/provisional/trusted weights and whole-ROI reasons.
Post-run checks confirm finite A RGB, no patch infinities, explicit missingness,
bounded ratios, exact frozen planned weights on available frames, and zero trusted
weights/all-NaN trusted RGB under UNKNOWN confidence. No missing frame is dropped.

Six of 30 ROI traces change planned weights at the block boundary; the maximum
L1 change is 0.01409. All planned weights stay fixed inside blocks. P000004's
16 transient provisional abstentions cause transitions to/from zero applied
weights, with maximum L1 step 1. The reported `delta_weight * current_patch_rgb`
term reaches 183.47 intensity units there. This is bookkeeping at a missing-output
boundary, **not an observed waveform spike**: that frame's output is NaN. Treating
missing values as zero or interpolating across them would need separate validation.
See [retained-array checks](waveform_v2/optical_quality_gate07/retained_array_validation.json).

Existing Gate 03 transfer tests are applied to controlled mixtures at 30 and
60 Hz, without waveform networks or reference-based correction:

| Control, at each rate | Outcome |
|---|---|
| Identical-phase frozen mixture | PASS_SYNTHETIC_ONLY |
| Symmetric spatial phases, uniform mixture | PASS_SYNTHETIC_ONLY |
| Same spatial phases, bounded unequal 2:1 mixture | FAIL: camera_time_anchor_shift |
| Unsafe rapid patch switching, negative control | ABSTAIN_UNVERIFIED |
| Block-boundary weight change with different spatial baselines | ABSTAIN_UNVERIFIED |
| Missing positively weighted patch, no rescue | ABSTAIN_UNVERIFIED |

Totals are **4 PASS, 2 FAIL, 6 ABSTAIN**. Expected negative outcomes are verified
by the passing regression tests. Unequal mixtures shift the source anchor by
about -53.26 ms at 30 Hz and -53.22 ms at 60 Hz despite preserving differential
delay; bounded frozen weights alone do not guarantee phase preservation. Switching
and block controls induce a 13.33-unit RGB mixing step. These constructed spatial
lags are test conditions, not measurements of actual SCAMPS physiology. Synthetic
passes do not validate physiological timing. See [timing evidence](waveform_v2/optical_quality_gate07/synthetic_timing.json).

## Contact packet, labels and segmentation ambiguity

The fixed sample includes all ten recordings and five temporal positions: 50
frames, 150 ROI annotation items, and 300 intended ratings from two reviewers.
The [annotation guide](WAVEFORM_V2_OPTICAL_QUALITY_ANNOTATION_GUIDE.md) defines
material-fraction bins, shadow/glare, anatomical confidence, uncertainty, blinding,
agreement and adjudication. All human fields remain null/PENDING, including reviewer
IDs and adjudication. No human agreement statistic is available.

All ten sheets received assistant layout QA; the pair was inspected at full
sheet resolution. Titles, frames, outlines and insets are visible without overlap.
No assistant observation was inserted into the human template. Full context and
12-pixel crop margins support material interpretation; screenshots cannot create
physical pulse truth. Original source images and source files remain unchanged.

**The 599-mask/600-frame temporal mapping remains UNVERIFIED.** Stored SCAMPS
segmentation values were not decoded, shifted, padded, duplicated or truncated.
There is no independent skin-label mapping. No third-party segmentation candidate
or new weights were introduced; provenance, terms and costs would need separate
review if one is proposed later.

Local artifact root:
`outputs/waveform_v2_optical_quality_gate07_20261105_01/`.
It contains ten full contact sheets, ten compact retained-array NPZ files, a
layout-QA mosaic, license and run JSON files. The [artifact manifest](waveform_v2/optical_quality_gate07/local_artifact_manifest.json)
records PNG/NPZ/license sizes and SHA256 values. Source-derived images and arrays
remain ignored and are not included in Git. The Git metadata packet is indexed
[here](waveform_v2/optical_quality_gate07/README.md).

## Runtime, validation and reproducibility

One streaming pass processed 6,000 frames using chunks of 32. The array budget was
192 MiB and RSS limit 1 GiB. Observed RSS peak was **439,713,792 bytes
(419.34 MiB)**. Traced collection peaks ranged from 96,499,640 to 96,607,772 bytes
(92.03-92.13 MiB). RSS was sampled every 2 ms and can miss short native peaks;
tracemalloc excludes some native allocations.

Across ten recordings, measured geometry plus A extraction took 89.42 s,
A optical descriptors 18.50 s, B patch descriptors 97.25 s, and B selection/mixing
0.0256 s. Total recording wall time, including I/O/contact/array writing, was
278.90 s (26.69-29.23 s per recording). This is an instrumented same-frame run,
not a matched standalone runtime benchmark or a throughput/generalization claim.
See [memory/runtime details](waveform_v2/optical_quality_gate07/memory_runtime.json).

Validation command:

```powershell
.\.venv\Scripts\python.exe -u -m scripts.verify_waveform_v2_optical_quality_safe
```

**149 tests passed: 0 failures, 0 errors, 0 runtime skips, 0 protected-I/O attempts.**
This includes 30 new optical/timing tests and 119 retained regression tests.
The runner blocks waveform module calls, checkpoint loading and backward calls.
Controlled fixtures cover whole-ROI hair-like occlusion, compatible-color hair,
independent synthetic evidence, saturation, missing geometry, changing boundaries,
NaNs/extremes, determinism, frozen/bounded weights and no patch rescue. A guarded
HDF5 fixture rejects any value read except Xsub. Real reviewers are not simulated.
Seven new Python files compile. CLI help and refusal of existing or source-child
output paths pass. All metadata JSON parses and all 150 human items remain blank.

To reproduce in a **fresh** local output directory using the existing environment:

```powershell
.\.venv\Scripts\python.exe -u -m scripts.audit_waveform_v2_optical_quality --scamps-root SCAMPSData --out-dir outputs/waveform_v2_optical_quality_gate07_repeat --chunk-frames 32
```

The [software/validation manifest](waveform_v2/optical_quality_gate07/software_and_validation.json)
records Python 3.11.0, package versions and source/protocol SHA256 values. The
full comparison records all ten source hashes and 20 unchanged input hashes.
The reused FaceLandmarker task SHA256 remains
`64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff`.

The [official SCAMPS license](https://github.com/danmcduff/scampsdataset/blob/main/LICENSE.txt)
was rechecked: R-UDA v1.0, upstream blob
`381ad325da802fc3f45b4591ffebf2ba3f238491`, vendored SHA256
`4ca5e434d1faf7e964b14e4a2508ae33ef5447ecd84a9a9f6e25767a64498595`.
Its non-commercial computational-research restrictions, commercial-use/result
restrictions, attribution and redistribution terms remain applicable. The
[existing license and attribution](waveform_v2/scamps/) are retained; the local
review packet includes the license. No new dataset/model downloads occurred.

## Gate 08 decision

Proceed with actual independent two-reviewer annotation of the fixed 150-item
packet, retaining uncertainty and obtaining 300 genuine individual ratings before
agreement/adjudication. Lock labels before examining automated scores. Predeclare
a separate permitted quality-validation cohort/protocol; these exposed examples
must remain development evidence. Investigate material-aware evidence that can
reject skin-colored hair while preserving abstention, and evaluate phase safety
under spatial lag, moving boundaries, missingness and block changes. Do not tune
on PPG or convert pending reviews into verified-skin block states automatically.

| Readiness | Decision |
|---|---|
| Engineering inspection and controlled experimentation | GO: streaming, provenance, whole-ROI abstention, retained denominators and review packet are available. Semantic hair discrimination remains unresolved. |
| Synthetic waveform training | NO-GO: independent optical validation is absent and phase-transfer controls expose failures. `allow_training` remains false. |
| Real-domain generalization | NOT ESTABLISHED / NO-GO for claims: no collected-subject evaluation or independent real-domain evidence. |

Gate 07 does not enable training. Review and research can proceed through the
stacked draft PR; existing PRs are not merged by this work.
