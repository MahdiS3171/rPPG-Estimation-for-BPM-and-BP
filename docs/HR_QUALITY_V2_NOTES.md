# HRQualityNetV2 notes

This version adds an explicit candidate/prior-selection head to close the gap
between the best single input channel and the input-channel oracle.

## New idea

For every window the dataset computes fixed HR candidates from:

- RGB_G
- GREEN
- CHROM
- CHROM_WIN
- PBV
- POS_WIN
- OMIT

The model predicts:

1. an HR distribution from temporal features;
2. selection weights over the fixed HR candidates;
3. a final HR as a blend of learned HR and candidate-fused HR;
4. a quality score.

The auxiliary selection loss teaches the model which candidate is closest to the
reference HR in each training window.

## Recommended combined training command

```powershell
python scripts/train_hr_quality.py --ubfc-root UBFCData --rppg10-root rPPG-10 --cache-dir cache_roi_oldpoints --epochs 50 --batch-size 64 --fs 30 --win-sec 30 --stride-sec 2 --rppg10-win-sec 30 --rppg10-stride-sec 10 --model-version v2 --selection-weight 0.5 --hr-reg-weight 0.1 --quality-weight 0.2 --out checkpoints/hr_quality_v2_ubfc_rppg10_30s.pt
```

## Ablation: old model

```powershell
python scripts/train_hr_quality.py --ubfc-root UBFCData --rppg10-root rPPG-10 --cache-dir cache_roi_oldpoints --epochs 50 --batch-size 64 --fs 30 --win-sec 30 --stride-sec 2 --rppg10-win-sec 30 --rppg10-stride-sec 10 --model-version v1 --out checkpoints/hr_quality_v1_ablation.pt
```
