# Phone-only camera timing feasibility

Draft 2026-11-08, incorporating the user's equipment decision: existing phones
and cuff only. This is a proposed measurement design, not authorization to open
reserved sessions or collect participant health data. No camera was calibrated
in the present Codex session.

## What we can establish without new physiological sensors

First obtain phone model, recording app and original native mode. A newly captured
**nonparticipant** diagnostic clip can establish container cadence, duplicates,
visible exposure/brightness instability and whether advertised high FPS survives
export. Record original resolution, orientation, stabilization/HDR, exposure,
ISO, white balance, codec and manual-control availability; retain original files.
Do not apply a slow-motion playback factor as if it were acquisition timing.
An app's advertised frame rate is not evidence of sensor exposure precision.

Use three short repeats per available native mode with a stationary diffuse
surface and fixed illumination. Inspect PTS increments, repeated frames, spatial
brightness and pulse-band illumination. A suggested 30-second clip is adequate
for this engineering diagnostic, not for a physiological repeatability result.
Estimate resource needs from one clip before choosing the complete set of modes.
No recommendation to collect more volunteer BP data follows from this step.

## Row timing is the harder measurement

A spatially uniform optical signal with known simultaneous modulation can test
relative row timing. Under the simple model, apparent hand lag = true lag minus
the hand/face exposure-midpoint difference. ROI row coordinates must refer to
sensor readout, accounting for rotation, crop and stabilization.

A second phone screen is readily available but has its own refresh scan/PWM.
Filming it therefore measures a combination of display and camera timing.
An orientation-reversal/crossover test may help diagnose these contributions
under stable manual settings; it is an exploratory sensitivity experiment until
display behavior and sensor-row mapping are independently bounded. Do not use
a screen recording, flashing app or subframe fit as certified timing truth.

If existing illumination offers a common modulated source, document and test
that assumption; do not invent a known light waveform. A genuine calibration
would need a characterized common optical modulation source or equivalent timing
evidence. None is established as available. Under the phones-only constraint,
this may leave the camera's absolute/differential uncertainty unresolved.

## Conditions for any subsequent physiological study

Fix capture mode and processing from engineering diagnostics, with no reference-
selected lags, ROI switching or polarity repair. Start with matched fixed GREEN
processing and separate face/palm ROIs; learned fusion stays outside timing.
Document the sign of delay, edge guards, failure rules, fiducial estimators,
repeated sessions, cuff-free intervals and the uncertainty budget before data use.
Separate camera clock/readout error, estimator error, fiducial variation and
repeatability. A sample-spacing floor is a conservative policy, not a universal
proof that subframe estimation is impossible. Conversely, successful synthetic
subframe estimation cannot validate a physical timing claim.

No particular face–hand delay magnitude is established for our posture. Candidate
engineering sensitivity tests can span 0–100 ms with both signs; these are design
values rather than asserted physiological ranges. For a future protocol,
choose the measurement target and uncertainty limits from calibration first.
Require uncertainty small enough to distinguish the observed effect from zero
and artifacts, stable sign/repeatability across independent recordings, and
acceptable full-denominator coverage. Set numerical criteria before recordings.

Even a repeatable phone-only optical offset cannot by itself establish arterial
PTT, local wavefront truth or incremental BP value. If uncertainty cannot be
bounded with the existing apparatus, report timing as unresolved and continue
the waveform objective. An authorized external multisite dataset could investigate
timing physiology separately, while transfer back to the phones remains unproven.

The later user-authorized [Pilot11 protocol](PILOT11_NEW_MODELS_PROTOCOL.md)
opens S001–S011 for fixed local evaluation. Their original recordings are being
re-extracted consistently. This does not calibrate the phones; legacy cuff
times remain unverified until actual video intervals are reviewed.
