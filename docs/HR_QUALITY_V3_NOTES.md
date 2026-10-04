# HRQualityNet V3 notes

This version keeps the V2 architecture but changes the supervision and checkpointing.

## Why V3?

V2 improved overall HR MAE compared with V1, but the hard candidate-selection accuracy stayed low. The likely reason is that many windows have several candidates with nearly equivalent HR error; forcing a single hard argmin target is noisy.

## Changes

1. **Soft candidate-selection target**
   Candidate priors receive probabilities proportional to their closeness to the ground-truth HR. This gives partial credit to near-optimal candidates.

2. **Dual checkpointing**
   The script now saves:
   - the normal `--out` checkpoint: best overall validation MAE;
   - `*_quality60.pt` by default: best MAE among the top-quality 60% windows.

3. **Additional diagnostics**
   Validation now reports soft-selection cross entropy and expected candidate error under the model selection distribution.

## Recommended command

```powershell
python scripts/train_hr_quality.py --ubfc-root UBFCData --rppg10-root rPPG-10 --cache-dir cache_roi_oldpoints --epochs 50 --batch-size 64 --fs 30 --win-sec 30 --stride-sec 2 --rppg10-win-sec 30 --rppg10-stride-sec 10 --model-version v3 --selection-weight 0.2 --soft-selection-weight 0.8 --hr-reg-weight 0.1 --quality-weight 0.2 --quality-save-coverage 0.60 --out checkpoints/hr_quality_v3_ubfc_rppg10_30s.pt
```

Use the best-overall checkpoint for normal evaluation and the quality checkpoint when the product behavior allows rejection/retry.
