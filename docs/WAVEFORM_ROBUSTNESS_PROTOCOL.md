# Leading-model numerical robustness checks

Fixed before execution, 2026-11-08. Use the compact residual, longer TCN and
attention five-fold checkpoints already created, hash-checking each and
reproducing saved predictions on CPU. No fitting or original-source access.

Use all 152 real development windows, with their own out-of-fold checkpoint:

- Shift all input traces by +/-2 and +/-4 samples with edge extension. Compare
  output with the same shift of the original output, after discarding one second
  at both edges. This checks equivariance, not accuracy against local physiology.
- Remove each ROI separately using its validity mask; recompute input-only
  descriptors. Report the full failure denominator and output changes.
- Remove every classical prior; hybrid models should abstain. Flat input should
  also abstain. Finite zeros are not successful waveforms.
- Bandlimited-resample the existing features from 30 to 60 Hz, recompute
  descriptors with the correct rate, infer at 60 and compare every second output
  sample with the 30-Hz prediction. This is equivalent-grid consistency using
  interpolated data, never a claim about recovered native-60 detail.

Report zero-lag agreement and best additional lag on a +/-5-sample search with
the original 33.33-ms bins. Zero additional bins cannot demonstrate millisecond
timing. ROI removal occurs after feature extraction and does not simulate every
camera failure. No reference-based calibration, final evaluation or BP fitting
is implied. Evidence from these checks guides whether the models need
equivariance/rate-consistency regularization before any deployment.
