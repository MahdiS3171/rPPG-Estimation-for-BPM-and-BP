# Independent optical/skin review guide

This is a **protocol and blank template**, not a completed human assessment.
No human labels, skin-confidence validation or inter-reviewer agreement has
been obtained in Gate 07. Automated color/optical scores are hypotheses.

## Reviewer packet and blinding

Use all ten fixed contact sheets and all five frames per sheet (indices
0,149,299,449,599). Every frame has three original V1 ROI annotations, giving
150 ROI items and 300 intended individual ratings with two reviewers. The
upper contact-sheet row shows full Xsub context; lower rows show the original
forehead/left-cheek/right-cheek polygons with 12-pixel context margins.
Outlines are display-only copies; stored source images remain unchanged.

The sheet identifier is an opaque source-derived ID. A coordinator retains
`contact_review_index.json`, which maps IDs to source filenames/checksums.
Withhold that mapping, prior reports naming example failures, automated scores,
Candidate B weights and all PPG traces during initial independent annotation.
This is a review procedure, not an access-control mechanism. Do not search the
coordinator mapping while blinded. Do not infer a person's identity from an
avatar or filename.

Use two reviewers familiar with facial anatomy and imaging artifacts. Each
reviews the images independently at native resolution or greater, without
discussing cases. Record actual reviewer IDs, timestamps and qualifications.
Only those people may fill their reviewer slots. Assistant visual observations
must be recorded separately with `reviewer_type=assistant`, never entered as
human ratings or used to calculate human agreement.

## Fields and anchors

For the visible area **inside each original polygon**, record approximate
fractions in bins: `0`, `(0,10%]`, `(10,25%]`, `(25,50%]`, `(50,75%]`,
`(75,90%]`, `(90,100%]`, or `UNASSESSABLE`.

| Field | Question / caveat |
|---|---|
| skin_fraction_bin | How much appears to be facial skin? Color alone is insufficient. Use boundaries, hair strands and context. |
| hair_fraction_bin | How much is visibly covered by hair/bangs/beard? Mark uncertainty for skin-colored or poorly resolved strands. |
| background_fraction_bin | How much lies outside the visible face? Context is essential. |
| shadow_fraction_bin | How much skin is shadowed? Shadow can coexist with skin and does not establish poor pulsatility. |
| glare_fraction_bin | How much is dominated by obvious specular highlight/glare? It may coexist with skin. |
| occlusion_severity | NONE / MINOR / MODERATE / MAJOR / UNASSESSABLE. Explain the material/context. |
| anatomical_confidence | HIGH / MEDIUM / LOW / UNASSESSABLE confidence in the visible region/material assessment, not confidence in pulse measurement. |
| uncertainty_reason | Resolution, illumination, similar hair/skin color, ambiguous boundary, missing geometry or another explicit reason. |
| notes | Brief visual evidence; no PPG or model-performance interpretation. |

Skin, hair and background are approximate material fractions; they should be
roughly consistent, but allow an unassessable remainder. Shadow and glare overlap
material categories and should not be forced to sum with them. A finite polygon
or bright skin-colored patch is not proof of skin, optical reliability or
physiological signal quality.

If the polygon is almost entirely hair, mark the whole-ROI assessment accordingly;
do not rescue it because a small subpatch looks like skin. If material confidence
is insufficient, mark UNASSESSABLE and preserve whole-ROI abstention. Do not turn
uncertainty into a positive skin label. The automated proxy cannot resolve all
skin-colored hair cases.

## Locking, agreement and adjudication

1. Retain the blank template and write each review to a new versioned file.
   No overwrite of original source images, templates or another reviewer's data.
2. Lock each review before unblinding. Record missing/unassessable items against
   the full 150-item denominator rather than dropping them.
3. Report exact agreement and a predeclared ordinal agreement statistic for
   material-fraction bins (e.g. linearly weighted kappa), plus per-category
   counts and unassessable rates. Do not calculate statistics until real ratings
   exist. With few examples, report uncertainty and avoid generalization claims.
4. Have a third reviewer adjudicate disagreements while retaining both original
   ratings. An adjudication cannot create physical pulse-quality truth.
5. Only after labels are locked, unblind optical proxy scores and methods.
   Compare failures, including hair accepted by the color proxy and legitimate
   skin rejected under lighting changes. Do not tune thresholds and report the
   same ten examples as independent validation.

These already engineering-exposed SCAMPS samples are a development audit. A
future quality-validation cohort and thresholds require a separate fixed
protocol and permitted data. SCAMPS's 599-sample segmentation masks remain
temporally unverified and cannot supply the missing skin labels here.

## Current status and data terms

Every template item is `PENDING_HUMAN_REVIEW`; reviewer IDs/ratings and
adjudication fields are null. Independent skin evidence remains UNKNOWN.
The candidate's trusted output abstains under that state. The separate
provisional engineering traces do not earn training admission.

SCAMPS: McDuff et al. (2022), [official repository](https://github.com/danmcduff/scampsdataset).
Use is subject to the retained [R-UDA v1.0](waveform_v2/scamps/LICENSE_SCAMPS.txt)
for non-commercial research. Retain attribution/terms with any shared review
packet. Contact sheets and compact arrays remain local ignored artifacts;
no source/derived images are committed to Git.
