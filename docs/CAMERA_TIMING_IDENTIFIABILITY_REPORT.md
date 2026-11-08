# Camera acquisition identifiability result

Date: 2026-11-08. Protocol and simulator were committed at `5edac8d` before the
full experiment. Run: `python -m scripts.audit_camera_timing_identifiability --out
outputs/camera_timing_identifiability/20261108_01`.

## Result

The matched-shape acquisition model identifies **d-s**, where d is latent physical
delay and s is differential row exposure timing. It cannot identify d and s
separately. At 30 fps, an 8-ms delay with zero skew and a 20-ms delay with 12-ms
skew produce sampled arrays differing by only 4.11e-15. The clean estimates for
both are 7.931 ms. More flexible model fitting cannot distinguish these two
worlds using only those input traces under the declared assumptions.

This does not measure our phones' row timing. The 12-ms skew is a hypothetical
stress condition, not an observed value. Physiology has not been falsified by
this simulation; an unqualified timing interpretation has been shown ambiguous.

There were 1,536 attempted numerical trials: 1,533 finite diagnostics and three
abstentions, all for half-window inconsistency at 240 Hz with noise. All signed
grid-delay controls passed. Zero-noise repeats are duplicates, and equivalent
worlds use paired noise, so trial count is not independent sample size.

| 30-Hz case, clean | Latent physical delay | Row skew | Estimated lag | Error relative to latent delay |
|---|---:|---:|---:|---:|
| Physical delay | 8 ms | 0 ms | 7.931 ms | -0.069 ms |
| Equivalent acquisition | 20 ms | 12 ms | 7.931 ms | -12.069 ms |
| Cancellation | 12 ms | 12 ms | 0.000 ms | -12.000 ms |
| Row timing only | 0 ms | 12 ms | -11.945 ms | -11.945 ms |
| Different morphology | 8 ms | 0 ms | -8.445 ms | -16.445 ms |
| Coherent illumination only | No physiological delay defined | 0 ms | 0.000 ms | Undefined |

At noise SD 0.15 of pulse SD, the 30-Hz physical-only case has 95th percentile
absolute numerical error 2.454 ms, versus 14.088 ms for the equivalent 20-ms
world. Both have the same median estimate, 7.521 ms. This is numerical precision
under a simple model, not a calibrated physiological uncertainty interval.

Higher native FPS reduces the imposed skew under our chosen assumption that
readout scales with frame interval. It does not eliminate ambiguity about unknown
skew; waveform-shape bias also persists. Real readout, exposure, compression,
photon noise and physiology need measurement. The existing estimator's declared
source-sample floor remains unchanged; no subframe success licenses reducing it.

## Decision

Calibrate acquisition and identify signal origin before interpreting face–hand
lags. Retain simple fixed processing for the timing investigation. With existing
phones and cuff only, waveform research proceeds while physical timing may remain
unresolved. Do not use a larger learned model to declare that uncertainty solved.

Complete aggregate results and source hashes are in
[camera_timing_results_20261108.json](project_audit/camera_timing_results_20261108.json).
All trial rows remain in the ignored output directory, with their SHA256 in that
receipt. This report adds no real participant, BP fit or independent efficacy test.
