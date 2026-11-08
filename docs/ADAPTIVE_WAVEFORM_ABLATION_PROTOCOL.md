# Matched hybrid ablations

Fixed 2026-11-08 after the screen/confirmation results, before these fits.
The learned residual improved aligned shape/derivatives, but replacing adaptive
descriptors by a constant changed little in a counterfactual inference check.
That motivates testing simpler and more transparent mechanisms directly.

Match the confirmation's residual/morphology candidate: same five real-subject
folds, all original SCAMPS Train clips, 20% synthetic draws, seed, optimizer,
batch size and 300-step compute budget. Four variants, five folds = 20 small fits:

- Global learned gate: replace descriptors with the fitting-input mean during
  both fitting and validation; keep the temporal residual model unchanged.
- Frozen classical gate: retain the initial equal-ROI CHROM_WIN gate, train only
  the residual correction. It must not be described as adaptive method selection.
- Quality-weighted classical base plus residual: input-only concentration weights
  and fit-only global prior polarity, followed by the same residual encoder.
- Matched shuffled-waveform control: permute fitting targets within source,
  preserve all validation targets, keep the same adaptive architecture/loss.

The quality residual has fewer trainable gate parameters; report that structural
difference. No reference-driven choice of prior, polarity, lag or threshold is
allowed during prediction. No evaluation target enters fitting or corruption.
Compare to the already completed matched true-target candidate. A good output
from shuffled-target training can reflect the classical base and generic learned
smoothing; it does not prove physiology or input-specific waveform recovery.

All results remain development-only. Original Val/Test and private sessions stay
closed. The purpose is to retain complexity only when it contributes measurable
value, while preserving the user's adaptive-hybrid idea as a testable hypothesis.
