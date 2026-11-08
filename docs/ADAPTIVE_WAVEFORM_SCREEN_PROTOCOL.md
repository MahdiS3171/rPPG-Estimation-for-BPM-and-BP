# Adaptive waveform/model/loss screen

Fixed 2026-11-08 in response to the user's expanded mandate. Test a finite set
of competing ideas, record failures, and expand combinations only when evidence
supports them. No frozen V1 modification, private session access, final test use,
paid compute or controlled-data download is part of this campaign.

## Data and compute

Reuse the 152 original UBFC Train windows and 16 original SCAMPS Train windows.
Their read-only source/hash checks were established in the two-source study.
Prepare once, store a new immutable 168-window array archive, and reuse it for
every model. Six UBFC subjects (seed 20261108) and two SCAMPS recordings (seed
20261109) are development validation, disjoint from fitting. These are already
exposed development sources, not a new independent test. The original dataset
Val/Test groups stay outside this campaign. Synthetic avatar dependence is unknown.

Use one available local RTX 4060 8-GB GPU, 16 windows/batch, 120 fixed AdamW steps
per configuration, learning rate .002, weight decay .0001, gradient norm cap 1.
Initial seed is 20261108. First screen has 21 compact configurations with a
15-minute training-screen limit; report incomplete work as incomplete if reached.
No broad architecture/compute escalation is implicit. All weights stay ignored.

Sampling is subject/recording-balanced within source. Primary mixed runs use
20% synthetic draws; 0% and 50% controls test source contribution after the
previous 50:50 linear baseline deteriorated. Comparisons have equal compute,
not identical counts of real-source exposures. Record this distinction.

## Candidates and controls

The 4x4 core compares learned signed prior gating, gated temporal residual,
raw-RGB temporal reconstruction, and dual coarse/fine heads, under four losses:

1. Native signed correlation plus normalized MSE.
2. Lag-tolerant morphology plus same-lag derivatives and spectrum.
3. Native correlation plus native derivatives and spectrum.
4. A staged transition from lag-tolerant shape/spectrum toward native/derivative
   reconstruction over the fixed training steps.

The lag-tolerant training loss uses fixed central support and searches +/-0.5 s.
It does not shift stored outputs. Its native target is contact PPG; unmeasured
physiological/sensor offsets are not identifiable as model phase. Dual heads add
coarse (<=3.5 Hz) and complementary residual-target losses, without separately
normalizing the residual to amplify small high-frequency noise.

Five additional gated-residual/staged runs test past-only context, real-only
training, 50% synthetic training, shuffled fit targets, and 15% prior-mask dropout.
Shuffled targets are a negative control, not an efficacy candidate. Record epoch-0
predictions' metrics so classical initialization cannot masquerade as learning.

Fixed GREEN/CHROM_WIN and transparent quality hard-selection/weighted blending
are retained. Quality uses only video-derived descriptors. Global polarity for
quality baselines is fitted on real fitting windows only. No reference HR, target
waveform, dataset/subject identity or future calibration enters their prediction.
Spectral quality can still follow illumination; it is not physiological confidence.

## Adaptation and rate contracts

Window descriptors encode pulse-band concentration, harmonic concentration,
100-ms autocorrelation, skew and RGB variation. Context is the same recording's
initial complete 10-second **input-only** descriptor and is unavailable until that
interval ends. First target windows use the no-context path. This is a distinct
bounded quality-context design, not a claim that the old 30-second prototype was
validated on these 20-second examples. Context perturbation is bounded and zero-
initialized; target labels are never calibration inputs.

Temporal kernels use dilation 1 at 30 Hz and 2 at 60 Hz to retain physical span,
computing an output at every native sample. Tests compare matching sample times,
context identity, missing-input abstention and finite gradients for every loss.
Actual training is 30 Hz. Interpolating old sources cannot supply true 60-Hz
physiological detail or validate subframe pulse timing.

## Evaluation and follow-up

Report all final-step results, failed/flat predictions, native signed r, waveform
RMSE, native derivative correlation, spectral HR error, and lag-tolerant shape/
derivative diagnostics separately. Equal-weight subjects within each source.
Validation is evaluated at initialization and after the fixed training budget;
no best-of-many intermediate checkpoint selection. Keep native-target and
morphology rankings distinct. Any label-free quality scheme or learned hybrid
remains outside the physical face–hand timing track until independently tested.

Next evidence-driven stages may include seed replication, meaningful augmentation,
conservative ensembles, stronger residual/video models, or source scheduling.
Do not imply all conceivable ideas have been exhausted. A third real dataset
remains outside model development; its admission and first evaluation protocol
must be fixed before use. No development winner is a generalization claim.
