# Bounded architecture extensions

Fixed 2026-11-08 before fitting. The residual hybrid shows development shape
gains; extra width/context/heads did not clearly add value. The user asked for
creative, broader model tests, so screen four distinct mechanisms while keeping
the same data, split, prior base, optimizer, seed and 120-step budget:

- Longer physical-context TCN with four dilations, rather than only a short
  temporal neighborhood.
- Attention with a distance bias expressed in seconds (two temporal scales),
  allowing input-dependent long-context interactions.
- Learned spectral residuals on fixed 0–8-Hz modes of the ten-second window.
  This introduces a periodic-window assumption, not proven physical phase.
- Bounded, smoothed time-varying prior gates rather than one gate per window.
  Its flexibility can distort fiducials; it remains a morphology hypothesis.

Use native and lag-tolerant morphology losses for each: eight runs on the original
screen's development split. Native-60 input is supported, but no native-60 labels
are invented. These models can learn phase shifts; no timing safety is claimed.
No global best-of-all-models generalization claim follows from this small screen.
Expand a family only when its measured gain warrants more compute/data.

## Follow-up fixed before five-fold fitting

The eight-run exploratory screen completed. All four morphology architectures
will now use the existing five subject folds, 300 steps, seed and 20% synthetic
sampling of the first confirmation: 20 fits, capped at 15 minutes locally.
Attention and the longer TCN improved shape/native metrics on the small initial
split; the spectral alternative harmed HR. Retaining the spectral alternative
in this bounded confirmation checks whether that failure repeats. Report every
configuration, compare with the earlier compact residual and fixed baselines,
and replay saved metrics independently. This is repeated development selection,
not a new generalization test. No final test or private source will be accessed.
