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
