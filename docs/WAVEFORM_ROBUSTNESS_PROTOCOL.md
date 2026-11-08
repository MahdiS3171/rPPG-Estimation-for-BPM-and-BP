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

## Follow-up fixed after initial robustness results

The leading models preserve tested whole-input shifts and equivalent grids,
but removing an ROI changes estimated relative output lag by one 33.33-ms bin
in some windows. Test two controls on the same five folds, 300 steps, seed and
20% synthetic mixture: longer TCN with region dropout; the same with dropout
and a 0.2-weight consistency loss. Drop one uniformly selected ROI in half the
training examples, zeroing input features, masks and corresponding descriptors.
Use an independent augmentation RNG so source/subject sampling remains matched.
The consistency target is the current full-input output with stopped gradients;
standardized native-waveform MSE penalizes changes under region removal. It is
an engineering regularizer, not physiological timing supervision. Repeat the
same robustness checks and report waveform/HR tradeoffs for both controls.
Ten fits maximum, capped at 15 minutes; no search over consistency weights.

## Candidate export fixed after follow-up

The ten fits completed. Dropout improves shape/derivatives; the 0.2 consistency
variant has similar shape, higher mean native correlation and stronger ROI-output
stability. Retain the latter as the development candidate, not a universally
best model. Fit it once on all 168 permitted original-Train windows, same seed,
300 steps and source-balanced 20% synthetic sampling. Export a local checkpoint
and hash manifest, test exact serialization replay, flat abstention and 30/60-Hz
input shapes. Do not report a validation score for this full-development fit.
Earlier OOF scores describe the selection procedure, not this final checkpoint.
Frozen V1 is untouched. The candidate remains separate from production and is
not qualified for physiological timing or BP.
