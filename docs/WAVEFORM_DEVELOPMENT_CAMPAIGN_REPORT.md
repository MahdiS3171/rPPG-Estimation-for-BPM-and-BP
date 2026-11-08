# Current waveform development campaign

Updated 2026-11-08. These are development experiments. Sources are original
UBFC Train (26 people, 152 ten-second windows) and original SCAMPS Train (eight
synthetic clips, 16 windows). Original Val/Test, private pilot media and sealed
participants remain unused. Two training sources do not establish transfer
across two real populations.

## Findings

The strongest repeatable gain so far is waveform shape correction using a
residual hybrid and a morphology loss. Native correlation improves less
consistently. Adding waveform complexity has not established a heart-rate gain.

| Method, same five development folds | Native r | Aligned shape r | Aligned derivative r | HR MAE, bpm |
|---|---:|---:|---:|---:|
| Fixed CHROM_WIN | .267 | .729 | .607 | 1.151 |
| Input-quality weighted classical | .267 | .728 | .609 | 1.126 |
| Adaptive residual, morphology loss, 20% synthetic | .304 | .788 | .733 | 1.163 |
| Same residual, real data only | .300 | .793 | .740 | 1.383 |
| Same residual, 50% synthetic | .286 | .784 | .727 | 1.137 |
| Constant learned gate plus residual | .294 | .786 | .731 | 1.411 |
| Frozen CHROM gate plus residual | .286 | .784 | .733 | 1.194 |
| Quality-weighted base plus residual | .300 | .785 | .734 | 1.330 |
| Adaptive residual, shuffled fitting targets | .167 | .468 | .395 | 8.530 |

Scores average windows within people, then people. Aligned metrics choose one
lag within +/-0.5 seconds against the contact target on fixed central support.
They describe shape after reference alignment. Physical phase, arrival-time
preservation and local face-to-hand timing are not established.

The adaptive residual improves aligned shape in 24/26 subjects and aligned
derivative correlation in 26/26. The mean native-r difference is +.038, with a
descriptive subject-bootstrap interval of [-.004, .082]. Development selection
and overlapping training cohorts limit inference from this interval. Additional
seeds reproduce the initial shape gain. The matched shuffled-label control is
substantially worse, supporting information learned from correct targets.

The adaptive component adds much less than the residual correction: full shape
.788, constant-gate .786, quality-based .785. Holding descriptors constant at
inference also does not worsen the initial-screen morphology score. Keep simple
hybrids as live comparators. Selecting the highest-quality individual trace was
weak (HR MAE 6.677 bpm); weighted classical fusion was stronger here.

## Decisions from negative results

- Do not pool synthetic and real data indiscriminately. A 50:50 linear mixture
  worsened metrics; neural mixture effects differ by endpoint.
- Wider networks, past-only calibration context, dual-band heads and fixed
  ensembles did not clearly improve on the compact residual. Preserve their
  evidence rather than accumulating mechanisms in one larger model.
- Native loss and schedules ending in native loss did not solve phase. Contact
  site, reference filtering and alignment remain possible confounders; a loss
  cannot identify an unmeasured sensor clock.
- Labels remain 30 Hz. Native-60 input support and physical-time kernels are
  engineering capabilities; interpolation cannot manufacture physiological
  supervision. Extra detail at 60 Hz remains unvalidated.

## Remaining bounded comparisons

Eight architecture/loss runs completed: longer TCN context, physical-time
attention, spectral residuals and bounded time-varying classical gates, with
native and morphology losses. Longer context and attention warrant five-fold
comparison; the spectral model worsened HR on the first split. All four
morphology variants enter a fixed 20-fit confirmation to report failures too.

Next value comes from robustness and data/reference admission, followed by a
locked third real domain. Current evidence does not require camera recapture
as its explanation. Phone calibration can measure acquisition behavior, but
phones and a cuff alone do not provide a local contact timing reference.

## Reproducibility

The [research index](waveform_research/README.md) links fixed protocols, result
tables and hash receipts. Checkpoints, per-window predictions and participant
waveform plots remain local and ignored. Independent numerical replay reproduced
552 first-screen aggregates, 366 confirmation/seed aggregates and 144 matched
control aggregates without neural inference or original-source access. Maximum
differences were below 6.1e-8. Saved-value replay verifies reporting; it does not
add independent scientific observations.
