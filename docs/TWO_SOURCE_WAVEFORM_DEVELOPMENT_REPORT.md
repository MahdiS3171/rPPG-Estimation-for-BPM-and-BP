# Two-source development result — 8 November 2026

## Decision

Adopt the user's design: training/selection within two sources, followed by a
frozen third-real-dataset evaluation. Keep a single-source control. **Do not
assume mixing datasets improves generalization.** The first small experiment
shows degradation when adding the exposed SCAMPS examples to UBFC at a fixed
50:50 source weight. This is not a result about a final neural model or a third
domain, and it does not establish that larger, diverse real datasets would hurt.

The [protocol](TWO_SOURCE_WAVEFORM_DEVELOPMENT_PROTOCOL.md) and fitter were
committed at `f135bbb` before execution. Twenty small linear systems were fitted
on local CPU; total preparation/fitting/scoring took 217.54 seconds. No neural
training, GPU, paid job or new held-out dataset was used.

## Matched development evidence

Original UBFC Train only: 26 subjects, 152/152 nonoverlapping ten-second windows.
Five folds hold out subjects from the fitter. Additional source: eight original
SCAMPS Train recordings, 16 windows. The same UBFC folds and reference targets
are used in both training conditions; lambda and source weight were fixed.

| Fixed method | Native signed r | Standardized RMSE | Derivative r | HR MAE bpm |
|---|---:|---:|---:|---:|
| GREEN | -0.247 | 1.560 | -0.182 | 6.365 |
| Negative GREEN | 0.247 | 1.191 | 0.182 | 6.365 |
| CHROM | 0.259 | 1.171 | 0.185 | 1.529 |
| CHROM_WIN | 0.267 | 1.163 | 0.189 | 1.151 |
| POS_WIN | 0.171 | 1.265 | 0.126 | 11.519 |
| Prior fusion, UBFC only | 0.257 | 1.175 | 0.181 | 1.600 |
| Prior fusion, UBFC+SCAMPS | 0.236 | 1.195 | 0.171 | 2.252 |
| RGB+prior fusion, UBFC only | 0.254 | 1.179 | 0.178 | 1.987 |
| RGB+prior fusion, UBFC+SCAMPS | 0.229 | 1.205 | 0.159 | 2.250 |

All values are subject-balanced; all nine methods produce finite, nonflat
predictions on all 152 intended windows. The linear candidates do not improve
these development metrics over fixed CHROM_WIN. Negative GREEN was a separately
predeclared fixed comparator, never a reference-chosen per-window inversion.
No hyperparameter, source weight or model was retuned after these outcomes.

These are fresh out-of-fold fits of the small baseline, but the underlying UBFC
subjects were all used in historic V1 training. This is development evidence,
not an independent confirmation of a generalizable estimator. The tiny synthetic
source has unknown avatar dependence and very little diversity; equal source
weight makes its contribution large despite only 16 windows.

## Timing diagnosis and implication for the next loss

A separate **post-hoc diagnostic** reconstructed references from the same Train
contact files and reproduced all native correlations within 2e-6. Integer
lag-optimized correlation within +/-0.5 s rises from 0.267 to 0.712 for CHROM_WIN,
and from 0.257 to 0.690 for UBFC-only prior fusion. Fifteen CHROM_WIN windows hit
the search boundary. Predictions were not shifted, repaired or selected with
these lags; the diagnostic was not a fitting objective.

This gap is important but does not identify a camera clock fault. Real
face-to-finger propagation, sensor clock/processing latency, waveform shape and
filtering can all contribute. The data do not separate them. It explains why
low HR error does not certify native waveform phase, even within 30-fps UBFC.
The 30-versus-60 mismatch alone cannot explain this particular within-UBFC result.

Next model design must distinguish a contact-waveform reconstruction target from
a timing-preserving local facial pulse. Do not force zero-lag finger agreement
and then claim local facial arrival truth. Retain signed native metrics and
lag-aligned shape diagnostics separately; audit contact/reference timing before
using an absolute-phase loss as a physiological objective.

## Preservation, validation and next work

No original UBFC Val/Test or SCAMPS Val/Test waveform entered fitting or
evaluation. No private pilot or rPPG-10 signal was used. Only cache metadata was
scanned to locate the allowed historical Train traces; no non-Train trace was
decoded. Current extraction identities differ because a Tasks model bundle is
now present. The study explicitly reuses the original full-length, legacy-
detector traces, with identical extraction config/code/library/source metadata;
it does not silently assign the new detector to them or rebuild old caches.
All selected cache, reference and SCAMPS archive hashes are unchanged afterward.

Two numerical contract tests pass: fixed signed fitting can recover an inverted
channel, and duplicating windows in one source does not increase its total
weight. A separate reference replay verifies the stored predictions' native
correlations. Full predictions, coefficients and per-window/subject results stay
in ignored `outputs/two_source_waveform/20261108_01/`; only the compact
[receipt](waveform_research/two_source_development_20261108.json) is tracked.

Use this result to define a controlled neural comparison, not another blind
synthetic-data scaling exercise. Prefer two complementary real waveform sources
when permission permits, keep source-balanced sampling and a single-source
ablation, and lock the third dataset before model selection. The native-60
comparison also needs genuine high-FPS reference support; interpolating these
30-Hz training sources cannot validate newly reconstructed detail. Tricam access
is being pursued by the user; the public MPU example remains development-only
with unverified pairing. No final third-domain result or improved neural model
is claimed.
