# Pilot-6 protocol-aware Face–Hand Optical Delay feasibility

This isolated study tests acquisition support, numerical delay estimation,
repeatability, and exploratory cuff associations independently. The quantity is
**Face–Hand Optical Delay**, positive when the hand lags the face. It is not
validated physiological PTT, and no clinical PWV or optical BP accuracy is claimed.

## Inputs and protected contracts

The default source is the explicit local export
`sheets/rPPG_BP_Data_Collection_2026-10-24.xlsx`, declared by the user as an export
of Google Sheet `1613BKOf1fJsOweixkMVrLQpImywScl4cd-q-0Xq-ZMQ`.
Local source bytes, SHA256, every sheet/header, formulas, cached cells, and changes
from `sheets/rPPG BP Data Collection final.xlsx` are audited. The local export
lineage is user declared; no live Google identity or visual timing verification
is implied. `--metadata-path` permits another explicitly selected export; an
absent current source never triggers fallback to an old workbook.

The update supplies 15 formerly missing cuff end times. S001 has 60–110,
145–195 and 220–270 seconds. S004 has 60–133, 145–209 and 225–292 seconds.
All six sessions retain legacy unverified timing provenance. Review/Windows
formulas and provisional candidate intervals are not independent evidence.
Duration/FPS headers are now `Video_Duration_Reported_s` / `FPS_Reported`.
The update also populates reported acquisition/support fields and additional
geometry cells; SBP/DBP values are unchanged. The protocol preamble still says
S005 geometry is pending despite positive numeric cells in the subject table.
That annotation/cell discrepancy is retained as unresolved provenance; geometry
is preserved raw and never used to infer arterial path length or BP.

All original tracked source, checkpoint/manifest files, source spreadsheets,
and files beneath the two historical Pilot-6 output namespaces are hashed
before and after execution. Nothing in those artifacts is rewritten. New
outputs must live beneath `outputs/pilot6_protocol_aware_delay_v1/`; an existing
run requires explicit matching-input resume.

The official Frozen V1 checkpoint SHA256 remains
`d6b2d1eeebc0153197a2a12da736b957b32d72ae00b22631e9afd0b2097131ff`;
the original manifest SHA256 remains
`da5ba397b3be389deec8821f8ea16944b5f5b93e2834bb79c3c147eb3d1f89dc`.
The unchanged original extractor verifies its full contract when these files
are available. Optional `--frozen-face` inference uses that API and the accepted
external V2 timebase adapter separately. It is never used for hand inference or
face–hand timing. Missing optional assets block that optional branch, without
inventing a replacement checkpoint or manifest.

## Implementation and component reuse

New modules:

- `pilot6_protocol_timeline.py`: complete read-only workbook audit, explicit
  identity matching, source-event validation, provenance-aware masks and intervals.
- `pilot6_delay_acquisition.py`: native-pixel central palm/contracted palm
  extraction and identity-checked, read-only native facial cache reuse.
- `pilot6_delay_validation.py`: original signed delay estimators with added
  ambiguity/robustness diagnostics and seeded synthetic RGB controls.
- `pilot6_clock_controls.py`: separate jitter-only and single-frame-loss controls,
  without the deliberately large gap used in the primary failure control.
- `pilot6_delay_feasibility.py`: independent waveform quality, paired contiguous
  processing, agreement/stability, unique cuff-event associations and descriptive BP.
- `pilot6_delay_reporting.py`: complete traces, unfavorable examples, plots and
  a self-contained report with quantitative tables and evidence classes.
- `pilot6_delay_runner.py`: orchestration, immutable input identity, provenance,
  protected-artifact inventories and resumable acquisition/analysis stages.

Reused existing components include MP4 `stts`/`ctts` plus supported edit-list
presentation clocks, `VideoReader`, `PilotHandDetector`, the original native face
extractor/cache checks, geometric ROI hulls and pixel metrics, `RGBTrace`, bounded
resampling, contiguous zero-phase filtering, GREEN/CHROM/POS, `window_quality`,
`reliable_beats`, `bp.estimate_delay`, artifact hashes/serialization and the original
Frozen V1 API/accepted external adapter. Their original semantics are unchanged.
Bounded native-pixel mask measurement has an exact full-frame equivalence test.
No subject image pixels or crops are saved.

## Analysis rules and engineering refinements

`configs/pilot6_delay_v1.json` declares all quality/timing thresholds, seeds,
synthetic delays/heart rates and guard policies before BP analysis; thresholds
remain unchanged throughout execution. Policies A/B/C use 0/5/10-second guards. These margins are
sensitivity choices, not calibrated error bounds on legacy event times.

Masks separately retain cuff activity, boundary guards, candidate cuff-free
support, verified cuff-free support and timing uncertainty on every original
video PTS. Unverified events never create verified cuff-free time. Missing
endpoints block uncertain time without filling an assumed duration.

Six native ROI variants are assessed independently: forehead, both cheeks,
their pixel union, palm and contracted central palm. Face RGB from the previous
native run can be reused only after the original full identity check. Hand RGB
and combined face RGB are measured on every decoded native-resolution frame.
Protocol right-hand identity and detector classification/mirroring uncertainty
remain separate. Geometric coverage is not semantic skin or pulse validation.

Signals use the native-rate common clock, bounded linear interpolation across
at most 50.001ms, the existing classical projections, 0.7–4Hz order-3 zero-phase
filtering and one-second edge exclusion. Every phase and every unbridgeable
gap receives independent filter support. No discontinuous intervals or neural
windows are concatenated. The unchanged single-window POS implementation can
cancel common pulse components; simulation failures are retained.

10-second windows advance five seconds, and all attempted windows have rows.
Primary windows are anchored one filter-edge margin inside each contiguous
protocol phase/camera-support segment, then advance ten seconds without overlap.
Supplemental global-grid and overlapping attempts are retained, including
failures. This avoids missing a short usable phase due solely to grid alignment.
Scheduling uses protocol/camera support before any ROI quality or BP values.
The initial engineering pass used only global-grid windows. Phase anchors were
added after detecting missed short intervals caused by grid/edge alignment;
the preliminary pass is retained in explicitly named diagnostic logs. This is
an engineering refinement, not a prospectively registered window protocol.
The primary pairing is combined face / palm GREEN, independent of BP labels.
All other ROIs and algorithms provide explicit sensitivity diagnostics.
Spectral HR, SNR, acquisition support, reliable-beat screening, numerical
computability and strict timing status are distinct quantities.

The existing signed normalized correlation and fundamental phase/coherence
estimators are augmented with search-boundary, polarity and full-cycle ambiguity,
first/second-half disagreement, method/ROI agreement and reliable matched extrema.
No absolute-correlation polarity selection occurs. Subframe parabolic estimates
are model-based interpolation. The uncertainty floor includes native source
steps and observed disagreements; it is not a calibrated physiological CI.

Synthetic controls span 0 and ±20/40/60/80/120ms, 54/78/108bpm, 30/60Hz,
all six original camera clocks, noise/drift, unequal amplitudes, morphology
mismatch, jitter and a deliberately unbridgeable loss. RGB projection,
interpolation, filtering and all estimators are exercised. Both signed errors
and numerical recovery failures beyond one source step are reported; rejecting
a window crossing the known large gap is the intended result. Clean synthetic
success is not clinical timing validation.

Each cuff label appears once in the primary association table. Only preceding,
verified strict candidate support can create a primary feature. Repeated-label
sensitivity tables explicitly are not independent additional observations;
they compare legacy timing, guard policies, quality groups and 15/30/60-second
separation limits. Exact selected windows, temporal distance, beat counts,
uncertainty, relevant measured surface geometry and feature reuse are retained.
BP is an intermittent later cuff measurement, never a continuous window label.

Descriptive analyses separate pooled from within-subject-centered relations.
Where verified features are unavailable, optical BP regression/MAE and the
HR-only versus HR-plus-delay comparison remain unavailable. An equal-subject,
training-only cuff-mean LOSO baseline is descriptive only and cannot establish
optical BP accuracy. There is no BP-informed quality selection or tuning.

## Reproduction and evidence

Use a fresh run directory:

```powershell
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
.\.venv\Scripts\python.exe scripts/evaluate_pilot6_delay.py `
  --workers 3 --out-dir outputs/pilot6_protocol_aware_delay_v1/reproduction01

# Long runs may separate stages without changing input identities.
.\.venv\Scripts\python.exe scripts/evaluate_pilot6_delay.py `
  --stage acquire --workers 3 `
  --out-dir outputs/pilot6_protocol_aware_delay_v1/reproduction02
.\.venv\Scripts\python.exe scripts/evaluate_pilot6_delay.py `
  --stage analyze --resume `
  --out-dir outputs/pilot6_protocol_aware_delay_v1/reproduction02

.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe scripts/verify_pilot6_delay.py `
  --run-dir outputs/pilot6_protocol_aware_delay_v1/run01 `
  --test-log outputs/pilot6_protocol_acceptance_tests.log --thread-limit 1
```

The run report is `outputs/pilot6_protocol_aware_delay_v1/run01/research_report.md`.
`results_summary.json`, `cohort_summary.csv`, `experiment_manifest.json` and
`verification.json` provide compact findings, input/source identities and checks.
The report indexes every CSV, mask, native RGB cache and visual artifact.
Source health metadata and all generated health results remain ignored/local.

Evidence is explicitly distinguished as local verified measurement, engineering
diagnostic, unverified physiological interpretation, exploratory statistical
finding, or unsupported claim. The run report gives the complete results and
whether the six recordings justify further collection.

## Measured run01 outcome

All six recordings completed: 108,115 native frames and approximately 1,805.39s.
Face and palm have 100% finite geometric RGB support, which is acquisition
support rather than verified pulse quality. No source, model, workbook or
historical output artifact was changed.

For policy B and the declared combined-face / palm GREEN pairing, 422/599
attempted windows yield finite delays, covering 1,364.15s by interval union.
Candidate cuff-free computable coverage is 499.15s. Only two overlapping S003
windows pass every strict engineering screen: 11s union, and only one primary
nonoverlapping window. Their median delay is about 2.94ms against a 16.67ms
uncertainty floor. Verified cuff-free coverage and verified strict cuff
associations are both zero. These data do not establish delay repeatability.

Numerical simulation artifacts contain 15,840 broad controls plus 1,188 isolated
clock controls. Clean filtered GREEN/CHROM mean absolute errors are approximately
0.92–1.10ms, with maximum error 2.49ms; these are interpolated numerical errors,
not measured physiological accuracy. Morphology mismatch raises mean errors to
about 11.5–12.4ms. POS has 30Hz clean-case errors as large as 262.5ms. Unbridgeable
gap controls fail explicitly, and isolated frame losses expose the expected
30Hz limitation of the configured interpolation allowance.

All 18 cuff labels are unique. The 15 available unverified candidate sensitivity
associations have pooled delay/SBP r≈−0.14 and delay/DBP r≈0.18. Within-subject
values are approximately 0.06 and −0.34 using 14 events from five repeated
subjects. Dispersion associations have too little repeated-subject support for
within-subject correlation. There is no justified optical BP MAE or validated
HR-only versus delay-plus-HR comparison.

The full regression suite passes 290 tests (254 existing plus 36 new). Development
checks caught and corrected integer centering and nondeterministic last-digit
rounding in the descriptive baseline; logs and source snapshots are retained.
Real-artifact verification checks original clocks, unique labels, gap/boundary
handling, exact final statistical replay and all protected artifact hashes.

The evidence supports a targeted acquisition/timing-reference pilot before a
larger BP prediction collection: video-review cuff events, characterize exposure
and rolling shutter, add synchronized contact waveform references, and repeat
sessions per subject. Validated PTT, PWV, clinical BP accuracy and standards
compliance remain unsupported.
