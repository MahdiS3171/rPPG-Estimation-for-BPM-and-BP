# VideoPriorHRNet notes

This version should be used after the raw `VideoHRNet` ablation. The raw video-only model was intentionally useful as a negative control: it learned slowly, was expensive to train, and stayed far behind validated 1D priors on UBFC.

`VideoPriorHRNet` is prior-guided:

- video branch: face clip `(B,T,3,H,W)`;
- trace branch: RGB + classical priors `(B,C,L)`;
- candidate HR branch: RGB_G + GREEN + CHROM + CHROM_WIN + PBV + POS_WIN + OMIT;
- output: HR distribution, final HR, candidate selection, residual correction, and quality.

The model is initialized conservatively. The final HR starts from a candidate-prior fusion biased toward strong priors such as `CHROM_WIN`, while the video/learned branch initially has very small influence. This avoids the failure mode where a video model destroys a good prior during early training.

Recommended smoke/first run:

```powershell
python scripts/train_video_prior_hr.py --ubfc-root UBFCData --epochs 15 --batch-size 4 --clip-sec 10 --stride-sec 2 --clip-frames 96 --image-size 48 --video-cache-dir cache_video_prior --roi-cache-dir cache_roi_oldpoints --out checkpoints/video_prior_hrnet_ubfc_v1.pt
```

If the first epoch is slow, let it finish once: clip tensors are cached to `cache_video_prior/clips`, so later epochs/reruns should be much faster.

Key success criteria:

- epoch-0 MAE should be close to the best candidate prior rather than random-video MAE;
- if training improves, it should improve over the conservative prior initialization;
- quality@60% should be checked separately from overall MAE.
