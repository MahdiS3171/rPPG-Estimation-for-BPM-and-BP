# Three-part feature model: fixed task and comparison definitions

Fixed 2026-11-08 before contact-feature extraction/fitting. The user requested
independent task specialists alongside the new public subject split. Use its
846 Train and 173 Validation windows; no Test scoring during development.

Three groups form a ten-component feature vector:

1. Rhythm: Welch HR, median peak-to-peak interval, interval coefficient of
   variation. Three outputs, with a robust standardized regression loss.
2. Pulse geometry: foot-to-peak duration as a fraction of beat duration, widths
   at 50%/75% height as duration fractions, and normalized pulse area. Four
   outputs, with standardized squared error.
3. Derivatives: maximum rising slope, maximum falling-slope magnitude and RMS
   second derivative of a unit-height beat. Three outputs, robust regression.

These are computational summaries of the contact reference at 30 Hz, not
verified facial fiducials, clinical HRV, dicrotic notches or BP labels. No model
can infer missing physiological timing truth by specializing its loss.

Reference extraction: third-order zero-phase 0.5–8-Hz Butterworth; Welch peak
0.7–3.5 Hz, 4096 FFT, ten-second segment. Peaks require 0.5-SD prominence and
0.55 estimated periods of separation; omit 0.5 seconds at either edge. Interpeak
minima define feet. Retain full beats of 0.5–1.8 estimated periods, at least
five samples and 0.2-SD height. Subtract the line between feet, normalize by
positive peak height and use interpolated level crossings. At least three
accepted beats are required for geometry/derivative targets. Missing features
are explicitly masked, never replaced with invented labels. Report coverage.
Filtering and differentiation define the estimand; they do not recover detail
absent from the original reference or certify millisecond precision.

Compare three from-scratch systems with the same inputs/sampling/targets:
one joint feature network, a shared encoder with three heads, and three fully
independent encoder/head experts. Independent experts share no trainable
parameters; each receives only its own task's gradients. Concatenate their
feature estimates. Parameter-match joint/shared widths to the three width-16
experts before fitting; disclose actual parameter count and runtime. All use
the same physical-time TCN family, input-only descriptors and normalized prior
spectrum. Extra spectrum features enter every direct-feature model equally.

Feature centering/scaling and constant/ridge baselines use Train only, weighted
80% real/20% synthetic and equally by subject within source. Use fixed standard-
deviation floors for stable scaling. AdamW .002, weight decay 1e-4, clip 1,
batch 16, 600 steps, seeds 20261108/20261125/20261147. Group losses receive equal
weight. No validation-best checkpoint or parameter search using Test.

Report per-feature absolute error in physical units and Train-scale units,
averaging within subject then subjects, separately for UBFC and SCAMPS. Retain
all windows in the intended denominator; enumerate missing reference features,
abstentions and invalid predicted values. Compare with Train means, descriptor-
ridge estimates, and features extracted from fixed CHROM and the predeclared
from-scratch TCN ROI-consistency waveform model. No contact alignment is used
as an input. Reuse the same feature extractor for predicted and target waves.

The system estimates features. It does not yet reconstruct one mutually
consistent waveform from them; better individual feature estimates must be
demonstrated before adding a constrained reconstruction stage. Frozen V1 and
the previous waveform checkpoint remain unchanged. Internal Test remains
separate until the waveform/specialist validation comparison is finalized.
