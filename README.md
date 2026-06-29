# rPPG Lab – cleaned project

This is the working codebase for the face-video rPPG / HR project.

## Current stable path

1. Classical baselines remain mandatory.
   - UBFC-rPPG: CHROM / CHROM_WIN are strong.
   - rPPG-10: GREEN-face was strongest in the current tests.
2. The deep model should not rebuild a waveform from scratch when a classical prior is already strong.
3. The default hybrid model is now `PriorResidualWaveformNet`:

```text
RGB + classical priors -> weighted prior fusion + small learned residual -> rPPG waveform
```

The residual head is zero-initialized. At epoch 0 the model behaves like a conservative prior fusion, so it should not destroy CHROM/CHROM_WIN before learning anything.

## Recommended UBFC training command

```powershell
python scripts/train_waveform_ubfc.py --ubfc-root UBFCData --cache-dir cache_roi_oldpoints --epochs 20 --batch-size 32 --fs 30 --win-sec 10 --stride-sec 2 --roi face --out checkpoints/prior_residual_ubfc.pt
```

For an RGB-only control:

```powershell
python scripts/train_waveform_ubfc.py --ubfc-root UBFCData --cache-dir cache_roi_oldpoints --epochs 20 --batch-size 32 --fs 30 --win-sec 10 --stride-sec 2 --roi face --no-priors --model waveform --out checkpoints/rgb_waveform_ubfc.pt
```

## Visual waveform inspection

After training:

```powershell
python scripts/plot_waveform_predictions.py --ubfc-root UBFCData --cache-dir cache_roi_oldpoints --checkpoint checkpoints/prior_residual_ubfc.pt --mode worst --num-plots 12 --show-priors --out-dir outputs/waveform_plots
```

This saves plots with:

- target PPG waveform,
- model output,
- optional prior channels.

Use these plots before trusting HR metrics.

## Important evaluation notes

- Zero-lag waveform correlation is not always the right metric because facial rPPG and finger PPG can have a physiological delay.
- The training script prints both `val_wave_corr` and `val_wave_corr_aligned`; the aligned one is more informative for waveform recovery.
- HR metrics are computed from the predicted waveform using Welch.
- The dataset now defines window-level `y_hr` from the same reference PPG window used as the waveform target, avoiding the earlier mismatch between UBFC's HR row and window-level Welch estimates.

## rPPG-10 notes

`Subject_4` in the downloaded rPPG-10 copy has empty video files and should be excluded automatically by the benchmark script. ECG HR reference is extracted using R-peak detection, not Welch on downsampled ECG.

## HRQualityNet

A direct HR-distribution + quality/confidence model is available in:

```text
scripts/train_hr_quality.py
rppg_lab.models.HRQualityNet
```

See `docs/HR_QUALITY_MODEL.md` for commands and interpretation.
