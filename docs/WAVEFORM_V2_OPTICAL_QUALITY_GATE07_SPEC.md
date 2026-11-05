# Gate 07 research specification (locked before extraction changes)

Date: 2026-11-05. Parent: PR #3 at
`65ac812e69cc7d39e98b1643b7e25df997eef0f5`.
Purpose: independent-of-pulse-label optical engineering, not waveform training.

## Fixed evidence and sampling plan

Only existing SCAMPS examples P000001-P000010 are permitted. Source checksums
must match Gate 05. PRs #1/#2/#3, their reports, V1 polygons/extraction/quality,
and Gate 03 timing contracts were reviewed. Existing inputs and contracts are
retained. No private Dataset, actual S007-S011, collected subjects, training,
waveform inference, HR error, BP regression or model ranking is permitted.

For contact sheets use **all ten recordings**, **original frame indices
0,149,299,449,599**, and **all three V1 ROIs**: 50 frames and 150 ROI annotations.
This balances recordings, positions and ROI types; it is not demographic or
motion-domain stratification. There is no pulse/PPG-based frame selection.
P000001 and P000005 are predeclared documented engineering comparisons, already
exposed in Gate 06. Neither is an independent test or a source of tuned cutoffs.

Use one full 600-frame streaming pass per recording, chunk size 32, array budget
192 MiB and RSS limit 1 GiB. Original floating Xsub samples remain unchanged;
existing uint8 detector copies and continuous RGB*255 measurement conventions
are retained. Existing CPU face-landmarker weights are reused, not downloaded.
No additional face/skin segmentation model or weights are introduced.

## Baseline A and independent quality schema

Baseline A is the unmodified Gate 06 V1 landmark-polygon path. Its outputs must
reproduce all 20 prior input hashes. The new comparison reuses its frame-level
measurement implementation on identical frames, not a rewritten ROI algorithm.

Keep four separate concepts:

1. **Acquisition validity**: finite in-range RGB, available landmarks, polygon
   support and original frame indices. This is not physiological quality.
2. **Anatomical/skin confidence**: independent evidence status remains UNKNOWN
   unless explicitly supplied by a separate qualified review/validated source.
   Automated color compatibility is never promoted to confirmed skin labels.
3. **Optical indicators**: chroma compatibility, clipped-pixel fraction,
   brightness, spatial chromaticity dispersion, temporal change/flicker proxies,
   polygon/patch movement, availability and failure reasons.
4. **Physiological pulse quality**: UNVERIFIED; no estimate is inferred from
   coverage, colors, PPG labels or prediction performance.

The estimated skin-like pixel fraction is the existing experimental V1 YCrCb
gate (Cr 133-173, Cb 77-127), explicitly named **chroma-compatible fraction**.
Its complement is a non-compatible color fraction, not a confirmed hair or
background fraction. The gate may accept skin-colored hair or reject skin under
different pigmentation/illumination. Human-review fields identify hair,
background, shadow, glare and uncertainty; they start PENDING, not filled by an
algorithm or represented as human assessments.

Clipping follows existing intensity endpoints (any channel <=1 or >=254 in
RGB*255 units). Spatial chromaticity MAD, brightness and temporal RGB changes
are descriptive only. Color heterogeneity, darkness and flicker alone do not
prove bad skin or poor pulsatility and are not automatic rejection criteria.

## Separate experimental Candidate B

Split each original V1 polygon's bounding box into a deterministic **2x3 grid**,
intersecting each cell with that polygon. The union is exactly the baseline
polygon, with no overlapping patch membership and no extra semantic mask.
Moving landmark boundaries are tracked and reported. No stored SCAMPS mask is
used, thresholded or temporally reassigned.

Fixed engineering defaults, chosen before this comparison and not tuned to any
SCAMPS/PPG scores:

- minimum finite pixels per patch: **16**;
- patch original acquisition support per 300-frame block: **>=0.95**;
- median patch clipped fraction: **<=0.05**;
- median patch chroma-compatible fraction: **>=0.5**;
- at least **two** eligible patches per ROI/block;
- bounded scores in [0.5,1] among eligible patches, normalized, so nonzero
  weight ratio is at most **2:1**;
- block boundaries: **[0,300), [300,600)**.

Weights are computed from label-free optical descriptors of that same block
and then frozen for the block. This is **offline, within-block lookahead**, not
a causal algorithm or subject-session calibration. No future block informs the
current one. No temporal filtering or frame-by-frame patch switching is used.
If a positively weighted patch becomes unavailable or excessively clipped,
the entire ROI frame abstains; remaining weights are not silently renormalized.
Each boundary is explicitly reported as a possible discontinuity, not stitched
into a supposedly continuous waveform.

### Whole-ROI hair/skin-confidence gate

The candidate must support **rejecting the entire ROI**, even if a few patches
look acceptable. Before patch mixing:

- median whole-polygon chroma-compatible fraction <0.5 causes whole-ROI
  ABSTAIN_LOW_SKIN_PROXY (a hypothesis-based rejection, not a confirmed hair label);
- independent evidence of occlusion/non-skin rejects the whole ROI;
- UNKNOWN independent skin confidence causes **ABSTAIN_SKIN_UNVERIFIED** for
  the trusted output; acceptable patches cannot override that outcome.

Provisional engineering patch RGB/weights may still be retained under UNKNOWN,
clearly separated from trusted/applied output. Actual SCAMPS human review is
pending; no ROI is positively admitted as confirmed skin by this gate. This
avoids replacing one unvalidated validity score with another. A skin-colored
hair counterexample must test the color proxy's limitation explicitly.

Compare P000001/P000005 under these unchanged defaults. Report honestly whether
the optical proxy specifically flags the documented hair case. Failure to do so
is a negative result; do not adjust thresholds to force separation. Retain the
whole-ROI UNKNOWN-confidence abstention even if the proxy accepts the hair.

## Comparison and retained evidence

Retain Baseline A RGB, raw patch RGB, original patch validity, optical metrics,
planned block weights, provisional weights/output, trusted applied weights,
whole-ROI reason codes and all 18,000 intended ROI-frame observations. Local
compact NPZ arrays and PNG contact sheets remain ignored; Git stores metadata,
specification, tests and reports only. Preserve source hashes and software/code
versions. Human annotation templates contain PENDING fields for all 150 ROIs.

Report acquisition coverage, provisional/trusted coverage and abstention
separately. Report each optical indicator separately, without an unsupported
single physiological-confidence score. Hair/background labels are unavailable
until review; algorithmic outputs must not be called confirmed contamination.

## Synthetic timing and numerical tests

Controlled fixtures cover dark hair-like occlusions, skin-colored hair,
saturation, color changes, dropped landmarks, moving polygon boundaries,
all-patch failure, tiny masks, NaNs/overflow and determinism. Verify whole-ROI
rejection when only a minority of patches remain compatible; test positive and
negative independent-evidence states with known synthetic truth only.

Use the existing Gate 03 transfer audit at 30 and 60 Hz on controlled patch
signals: identical-phase fixed mixtures, heterogeneous spatial phases with
bounded unequal weights, unsafe rapid switching as a negative control, and
block-boundary weight changes with spatial baseline differences. Report the
mixing term caused by changing weights and do not repair candidate signals by
reference alignment. Passing synthetic controls does not validate physiology.

## Independent review and readiness

Prepare a guide for blinded, independent human review of all contact sheets.
Use at least two reviewers, explicit uncertainty, inter-reviewer agreement and
adjudication. Keep original-image context and native-resolution insets. Scores,+candidate weights and PPG traces should be hidden during initial annotation.
No human assessments will be invented. Assistant visual notes, if any, are
separate non-human exploratory observations and not validation labels.

SCAMPS mask mapping remains UNVERIFIED (599 vs 600), as does skin-label truth.
The upstream R-UDA v1.0 was rechecked; its Git blob remains
`381ad325da802fc3f45b4591ffebf2ba3f238491` and the vendored license remains
applicable to non-commercial computational research and attribution/terms.

Gate 08 GO: independent optical/skin annotation and held-out engineering
validation under a separately fixed protocol. Training is NO-GO until quality
and timing questions are reviewed and training is separately authorized.
Real-domain generalization remains unestablished. No earlier contract or input
hash is modified by this experiment.
