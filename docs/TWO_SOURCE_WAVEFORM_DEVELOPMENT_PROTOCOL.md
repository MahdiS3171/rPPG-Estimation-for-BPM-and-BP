# Two-source waveform development

Fixed 2026-11-08 before fitting. Implements the user's direction to use existing
data while pursuing access, and to train on two datasets before a third-domain
evaluation. This first bounded experiment tests whether adding synthetic data
helps a simple native-index waveform baseline. It is not the final neural model.

Sources: the 26 original UBFC **Train** participants and the eight original
SCAMPS **Train** example recordings. No original UBFC Val/Test, SCAMPS Val/Test,
rPPG-10, private pilot or MPU labels enter fitting/evaluation. Reuse read-only
historical raw RGB caches and pinned SCAMPS input/reference archives. The legacy
cache's absent Tasks-model bundle is retained as historical provenance, rather
than pretended to match a new extraction. Never rebuild or rewrite old caches.

Predeclared comparison:

- Five deterministic subject folds within UBFC Train, using seed 20261108.
  These are **development folds**, not a new independent locked test.
- Ten-second, nonoverlapping 30-Hz windows; existing input preprocessing and
  reference interpolation unchanged. All intended windows/failures are counted.
- Fit global signed ridge waveform fusion with lambda=0.01, no intercept or
  hyperparameter search. Two feature sets: 18 prior channels, and all 27 RGB/prior
  channels. Normalize covariance by subject, then source. Compare UBFC-only fit
  against a 50:50 source-balanced UBFC+SCAMPS fit on identical held-out UBFC folds.
- Fixed equal-ROI GREEN, negative GREEN, CHROM, CHROM_WIN and POS_WIN comparisons;
  report all, never per-window reference-selected polarity/method/lag.
- Primary: subject-balanced native signed correlation and standardized waveform
  RMSE. Secondary: native derivative correlation, spectral HR MAE and coverage.
  Reference-optimized lag is not applied to outputs or used to select a model.
- At most 20 solves of matrices no larger than 27x27; local CPU only. No neural
  optimizer, GPU, large learning campaign or paid resources. This bounded linear
  baseline can run under routine research authority.

An improvement here supports a development hypothesis, not unseen-domain efficacy.
A degradation is evidence against assuming synthetic pooling always helps. Keep
the single-source control in later neural experiments either way. Do not retune
lambda/source weight/folds after reading outcomes and call them fixed validation.

The eventual decisive study is: train and validate within two admitted sources,
freeze architecture/preprocessing/selection, then evaluate once on a third real
dataset whose labels remain outside development. Tricam is the selected access
route, not an already admitted final test. Its partition and first use will be
fixed separately. Compare native 60 versus properly antialiased 30 on the same
high-FPS recordings to separate sampling and domain effects. Current 30-Hz sources
cannot validate added physiological detail above their supported bandwidth.
