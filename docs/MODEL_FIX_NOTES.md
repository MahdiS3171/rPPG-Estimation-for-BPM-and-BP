# Model fix notes

## What was wrong

The older `WaveformNet` training reduced training loss but did not improve validation waveform correlation or HR reliably. The main reasons were:

1. The model rebuilt the waveform from scratch even when strong classical priors were available.
2. The loss emphasized pointwise waveform similarity, but face rPPG and finger PPG can be phase-shifted and morphologically different.
3. The first HR-distribution loss used raw PSD magnitudes as logits, producing huge losses and unstable training.
4. Zero-lag correlation underestimated waveform similarity when a physiological lag was present.

## What changed

1. Added `PriorResidualWaveformNet`.
   - It starts from a weighted fusion of classical priors.
   - It learns only a small residual correction.
   - The residual head is zero-initialized.

2. Replaced the unstable frequency loss.
   - The new frequency loss uses log-PSD logits, making the scale stable.

3. Added aligned waveform correlation.
   - `waveform_corr_aligned` reports the best correlation within a small lag window.

4. Updated `train_waveform_ubfc.py`.
   - Default is residual-prior model when prior channels exist.
   - It prints input-channel baselines before training.
   - It evaluates epoch 0 before training.
   - It saves the untrained prior-fusion model if it is already the best.
   - It supports early stopping.

## Expected behavior

For UBFC with priors, epoch 0 should already be close to the best classical prior. Training should improve it only if the residual correction is useful. If training makes validation worse, early stopping keeps the prior-fusion checkpoint.

This behavior is intentional: a hybrid model must not be worse than its best input prior on the same validation windows.
