# HRQualityNet

`HRQualityNet` is the direct HR + confidence model added after the waveform experiments.

Why this model exists:

- Contact finger PPG is not a perfect pointwise waveform target for facial rPPG.
- A waveform model can have low raw correlation while still estimating HR correctly.
- For the practical system, HR accuracy and confidence/rejection are more important than copying finger PPG morphology.

Model output:

- `hr_logits`: probability distribution over HR bins, default 40--180 bpm.
- `hr_bpm`: expected HR from that distribution.
- `quality`: predicted confidence that the current window contains reliable pulse information.

Recommended first tests:

```powershell
python scripts/train_hr_quality.py --ubfc-root UBFCData --cache-dir cache_roi_oldpoints --epochs 30 --batch-size 64 --fs 30 --win-sec 10 --stride-sec 2 --roi face --out checkpoints/hr_quality_ubfc.pt
```

For rPPG-10 only:

```powershell
python scripts/train_hr_quality.py --rppg10-root rPPG-10 --epochs 30 --batch-size 64 --fs 30 --rppg10-win-sec 30 --rppg10-stride-sec 10 --out checkpoints/hr_quality_rppg10.pt
```

For combined UBFC + rPPG-10:

```powershell
python scripts/train_hr_quality.py --ubfc-root UBFCData --rppg10-root rPPG-10 --cache-dir cache_roi_oldpoints --epochs 50 --batch-size 64 --fs 30 --win-sec 10 --stride-sec 2 --rppg10-win-sec 30 --rppg10-stride-sec 10 --out checkpoints/hr_quality_ubfc_rppg10.pt
```

Interpretation:

- Compare `val_HR_MAE` against the printed input-channel baselines.
- Check the top-quality coverage table. A useful model should reduce MAE substantially on high-confidence windows.
- Do not expect this model to improve waveform correlation; it is not trained for waveform reconstruction.
