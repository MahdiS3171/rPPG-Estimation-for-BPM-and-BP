# rPPG Lab - reviewed research code

This repository contains the current research code for estimating heart rate
from facial video and preparing a future cuff-referenced blood-pressure study.
It is an academic prototype, not a medical device.

## Evidence-backed current path

The present results support a conservative hybrid strategy:

```text
face video -> RGB traces -> classical rPPG priors -> prior-preserving model -> HR
```

On the stored UBFC validation split, `CHROM_WIN` is already very strong. The
recommended waveform model, `PriorResidualWaveformNet`, starts from a weighted
fusion of classical priors and learns only a small residual correction. Its
residual head is zero-initialized, so the untrained model does not destroy a
strong prior.

The six default prior channels are:

```text
GREEN, CHROM, CHROM_WIN, PBV, POS_WIN, OMIT
```

`PBV`, `LGI`, and `OMIT` in this repository are transparent baseline
approximations; they must not be described as exact reproductions of every
published variant.

## Datasets and evaluation status

- **UBFC-rPPG:** used for participant-wise model development and validation.
- **rPPG-10:** used for an external classical-method benchmark and, separately,
  for multi-dataset training of `HRQualityNet`.

The current multi-dataset experiment is **not** a strict cross-dataset test:
UBFC-rPPG and rPPG-10 are both represented in training and validation after
independent participant-wise splits. A strict train-on-one/test-on-the-other
experiment remains future work.

No locked final test set has yet been reported for the HR models. Stored "best"
checkpoint metrics are validation results used for model selection.

## Recommended UBFC waveform command

```powershell
python scripts/train_waveform_ubfc.py --ubfc-root UBFCData --cache-dir cache_roi_oldpoints --epochs 20 --batch-size 32 --fs 30 --win-sec 10 --stride-sec 2 --roi face --out checkpoints/prior_residual_ubfc.pt
```

RGB-only ablation:

```powershell
python scripts/train_waveform_ubfc.py --ubfc-root UBFCData --cache-dir cache_roi_oldpoints --epochs 20 --batch-size 32 --fs 30 --win-sec 10 --stride-sec 2 --roi face --no-priors --model waveform --out checkpoints/rgb_waveform_ubfc.pt
```

## Visual waveform inspection

```powershell
python scripts/plot_waveform_predictions.py --ubfc-root UBFCData --cache-dir cache_roi_oldpoints --checkpoint checkpoints/prior_residual_ubfc.pt --mode worst --num-plots 12 --show-priors --out-dir outputs/waveform_plots
```

The waveform target is fingertip/contact PPG while the input is facial rPPG.
A small physiological and acquisition delay can exist, so both zero-lag and
lag-aligned correlation are reported. HR is estimated from the predicted
waveform with Welch spectral analysis.

## Blood-pressure code status

The BP path is a scaffold awaiting synchronized, participant-level data. The
reviewed baseline now provides:

- participant-wise train/validation/locked-test splits;
- training-only feature imputation and standardization;
- training-only target standardization;
- a population-mean BP reference baseline;
- separate SBP/DBP metrics;
- checkpointed preprocessing and split metadata.

Run only after valid pilot recordings exist:

```powershell
python scripts/train_bp_baseline.py --data-dir data_sessions --out checkpoints/bp_feature_mlp.pt
```

Face-to-hand optical delay is stored as an **inter-site peripheral delay**, not
true PTT. Neither camera signal marks cardiac ejection, and an apparent velocity
must not be reported as clinical PWV.

## Reproducibility audit

Create a manifest from stored checkpoints and CSV outputs:

```powershell
python scripts/audit_results.py
```

The generated `outputs/results_audit.json` summarizes existing artifacts; it
does not rerun training.

## Validation commands used in the review

```powershell
python -m compileall -q rppg_lab scripts tests
python -m unittest discover -s tests -v
```

The datasets were not included in the uploaded archive, so full extraction and
retraining were not rerun during this review. Existing checkpoints and result
CSVs were audited, and model/signal/BP-preprocessing paths were smoke-tested.
