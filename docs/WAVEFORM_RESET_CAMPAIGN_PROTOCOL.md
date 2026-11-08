# Public-data split reset and from-scratch model comparisons

Fixed 2026-11-08 before preparation/fitting. The user explicitly superseded the
old Train-only boundary: repartition all eligible UBFC/SCAMPS records for the
new research campaign. Historical results, split manifests and frozen V1 stay
unchanged. The existing candidate checkpoint is preserved.

All 41 public UBFC subjects are assigned once: 29 Train, six Validation, six
Test, PCG64 seed 20261110 on sorted identities. All ten admitted synthetic
SCAMPS clips are assigned 8/1/1 using seed +1. This uses every eligible local
recording in one role. It does not download the full 2,800-clip SCAMPS release.
Avatar independence cannot be verified from the admitted synthetic metadata.
The [immutable split](waveform_research/reset_split_20261108.json) is committed
before data preparation. Old public test subjects may enter new Train; this is
an explicitly authorized new internal benchmark, never an untouched cohort.
New weights do not erase prior analyst, architecture or dataset exposure.

Use full cached facial RGB traces, existing six classical priors and original
contact targets at 30 Hz. UBFC: all admissible ten-second windows at two-second
stride; overlapping windows remain grouped by person and are not independent
participants. SCAMPS: both saved ten-second windows of each admitted clip.
Past-only input context has the same availability rule as the previous study.
Reusing verified raw RGB extraction caches does not reuse learned model weights.

Train, Validation and Test arrays are stored separately. Model-development code
opens only Train/Validation. Every new model is initialized from scratch; old
V1 or the saved candidate would have seen some newly held-out people, so their
weights are not fair new-split controls. Keep them as archived artifacts.
No private pilot, sealed participant, unpaired MPU target or pending dataset is
admitted by this public-split reset. No test score is computed during model
selection. Final internal testing follows the waveform and three-part specialist
validation comparison and a locked selection rule.

Rerun the previous model/loss mechanisms under one fixed 300-step AdamW budget,
batch 16, learning rate .002, weight decay 1e-4, clipping 1. Source and person
sampling remain balanced, with specified 0/20/50% synthetic controls. Include
all four compact architectures x four losses; context, source, dropout and
shuffled controls; wider/context/curriculum confirmations; four extended
architectures with native/morphology losses; matched gate/quality/shuffle
ablations; and the two TCN ROI-robustness variants. Three fixed seeds for the
main registry: 20261108, 20261125, 20261147. All final-step results are reported;
no validation-best intermediate checkpoint or hidden failed run is discarded.
Local runtime cap 30 minutes; unfinished work is explicitly incomplete.

Fixed classical/quality and ridge controls are fitted using Train only and
scored separately for real and synthetic Validation. Select no winner using
Test. Report native correlation, fixed-support +/-0.5-second aligned shape and
same-lag derivative correlation, normalized RMSE, HR error and failure coverage.
Aligned morphology remains separate from local physiological timing.

For a controlled data-volume comparison, rerun three predeclared configurations
(compact morphology, raw temporal/native, TCN ROI-consistency) using 21 of the
29 new Train subjects at dense stride, and all 29 at nonoverlapping stride,
with all eight new synthetic Train clips unchanged. Use seeds 20261108/20261125.
Compare on the SAME new Validation subjects. This distinguishes additional real
participants and window overlap from a changed evaluation population. Comparing
old OOF scores with new Validation scores alone cannot establish a data-size
effect. These checks and the new three-part models use Validation for development.
