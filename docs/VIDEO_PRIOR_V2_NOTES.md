# VideoPriorHRNet v2

This version is a targeted fix after the first VideoPriorHRNet run.

Observed in v1:
- Epoch-0 was already close to CHROM_WIN, which confirmed prior-safe initialization.
- Training often degraded the initial prior-preserving prediction.
- `gate_mean` collapsed toward ~0 and video/residual contribution was negligible.
- The best real checkpoint could be epoch 0, but v1 did not save epoch-0.

Changes in v2:
1. Epoch-0 checkpointing
   - Saves the initial conservative model as both best overall and quality-aware checkpoints before training.

2. Video auxiliary head
   - `VideoPriorHRNet` now includes `video_hr_head`, a HR distribution head from video features only.
   - Training includes video-only distribution and regression losses.
   - Evaluation prints `val_VIDEO_ONLY_MAE` so we can see whether the video branch is learning.

3. Candidate dropout/noise during training
   - Candidate HRs are randomly dropped and noised only during training.
   - This prevents the model from relying exclusively on CHROM_WIN and forces fallback/correction learning.
   - Validation uses clean candidates.

Expected behavior:
- Epoch 0 should still be close to CHROM_WIN.
- If the video branch learns, `val_VIDEO_ONLY_MAE` should decrease meaningfully over epochs.
- The fused HR should not be allowed to degrade far beyond epoch-0; epoch-0 is saved.

Recommended first command:

```powershell
python scripts/train_video_prior_hr.py --ubfc-root UBFCData --epochs 10 --batch-size 4 --clip-sec 10 --stride-sec 2 --clip-frames 96 --image-size 48 --video-cache-dir cache_video_prior --roi-cache-dir cache_roi_oldpoints --candidate-dropout-prob 0.35 --candidate-noise-std 2.0 --video-aux-reg-weight 0.5 --video-aux-dist-weight 0.25 --out checkpoints/video_prior_hrnet_v2_ubfc.pt
```

If fused MAE degrades but video-only MAE improves, we keep the video branch and later use it as an auxiliary residual/quality cue, not as the primary HR estimator.
