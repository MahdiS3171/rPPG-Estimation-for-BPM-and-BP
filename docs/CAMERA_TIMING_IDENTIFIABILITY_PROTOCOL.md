# Camera timing identifiability experiment

Fixed design, 2026-11-08; synthetic acquisition model only. This replaces no
historical protocol and accesses no recording, checkpoint, or reference dataset.

Question: can a stable subframe cross-correlation estimate distinguish a true
inter-site delay from an uncalibrated difference in row exposure times?

Let the face be sampled at frame time t, the hand at t+s, and the hand's latent
pulse lag be d. With matched pulse shape and exposure integration, observed
hand(t) = pulse(t+s-d). Thus the apparent delay is d-s. Two worlds whose d and s
increase equally generate identical observations. Sampling faster can reduce s
for some cameras; it cannot identify unknown s from these two traces alone.
The sign convention is hand later than face = positive. All exposure times here
refer to midpoints; unequal exposure integration and nonlinear camera processing
are additional limitations outside this minimal model.

Predeclared design:

- Native rates: 30, 60, 120, 240 Hz; 12-second nonoverlapping simulated trials.
- Hypothetical full sensor readout: 60% of frame interval; hand is 60% of image
  height below face. Differential row timing s is 36% of a frame, **not a phone
  specification or a literature estimate**. Integration is 25% of a frame, with
  nine midpoint quadrature samples and identical integration at each site.
- Reuse the existing deterministic nonstationary harmonic pulse and conservative
  signed-lag estimator. Do not train, tune thresholds, invert, or correct outputs.
- Six fixed cases: 8-ms physical delay with zero scan skew; an observationally
  equivalent delay of 8+s ms with s-ms skew; complete cancellation d=s; no
  physiological delay with s-ms skew; 8-ms delay with unequal pulse morphology;
  coherent illumination with no physiological source or estimable true delay.
- Noise SD: 0 and 0.15 times the face pulse SD, independently at the two sites;
  32 seeds per case/rate/noise level, fixed initial seed 20261108. Noise level is
  a numerical stress condition, not a measured sensor SNR. Per-sample noise is
  fixed across rates, so rate comparisons are not a photon-budget comparison.
- Intended denominator: 6 x 4 x 2 x 32 = 1,536 trials. Zero-noise repeats are
  deterministic duplicates retained explicitly, not independent observations.
- Report attempts, finite diagnostic estimates, abstentions by reason, median
  estimate, bias and 95th percentile absolute error against both physical and
  apparent delay where defined. Percentiles describe simulations, not a patient
  confidence interval. Never remove estimator uncertainty floors.

Falsification and interpretation fixed before execution:

1. The paired equivalent worlds must differ by <5e-12 in noiseless arrays. If
   they do, their biological delays cannot be identified from these observations.
2. Positive and negative clean grid-delay controls must recover the expected
   sign and remain within 2 ms; otherwise stop and investigate the implementation.
3. A stable/finite delay from coherent illumination is a false physiological
   interpretation if promoted as a pulse measurement, even when numerically exact.
4. Higher FPS or clean subframe recovery is insufficient evidence of physical
   timing precision. No simulated outcome admits a real dataset or device.

Command: `python -m scripts.audit_camera_timing_identifiability --out NEW_DIRECTORY`.
Output refuses an existing directory. Results include source hashes and complete
trial-level rows. No threshold or scenario will be selected using outcome labels.

The acquisition model is motivated by the primary study
[Mironenko et al. (2020)](https://arxiv.org/html/2004.12695v1), which experimentally
examines row-dependent phase artifacts. Our numerical assumptions are separate
from their measurements. This is a sensitivity and identifiability analysis,
not a claim that our phones have a particular shutter artifact.
