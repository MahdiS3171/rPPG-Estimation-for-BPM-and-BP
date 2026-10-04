# VideoHRNet v1

This is the first spatio-temporal video model in the project.

## Why this model exists

The HRQualityNet v3 feature-level path is now a strong prior-fusion baseline:
- best overall validation MAE is around 3.7 bpm on UBFC+rPPG-10;
- best quality-aware 60% coverage MAE is around 2.1 bpm.

That path uses ROI-averaged RGB traces and classical priors, so it cannot learn spatial information from the face.  VideoHRNet v1 starts the video-based path.

## Input/output

Input tensor:

```text
(B, T, 3, H, W)
```

Default:

```text
T=160 frames, H=W=64, clip_sec=10
```

Output:
- HR probability distribution over bpm bins;
- expected HR bpm;
- quality/confidence score.

## Current scope

This first version trains on UBFC video clips only.  It is a sanity-check bridge before building a heavier PhysFormer-like model.

## Intended interpretation

Do not expect this v1 video model to beat HRQualityNet immediately.  The goals are:
1. verify that video loading/cropping/labeling is correct;
2. compare video-only learning with 1D CHROM/GREEN baselines on the same UBFC validation split;
3. inspect whether video features add signal beyond ROI-averaged traces.

If it learns stably and reaches a reasonable MAE, the next model should add a prior-guided branch:

```text
video clip + HRQualityNet/prior features -> HR distribution + quality
```
