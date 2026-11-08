# Gate 04: optional subject-session context prototype

Implemented on 2026-11-05 as an additive, inference-only engineering prototype.
The supplied `WAVEFORM_V2_SUBJECT_ADAPTATION_DESIGN.md` is preserved as the
original proposal. No model was trained and no participant was evaluated.

## Integration and checkpoint contract

`rppg_lab/waveform_v2_adaptation.py` wraps an existing eval-mode
`MultiROIPriorResidualWaveformNet`. Load its V1 checkpoint strictly using the
existing API **before** constructing the wrapper. The V1 class, checkpoint
format, source, losses, splits, training and forward implementation are unchanged.
There are no hooks, monkey patches or duplicated V1 encoder/forward code.

The wrapper defaults to `enabled=False`. The sole new parameter is
`context_head.weight`, initialized to exactly zero, with no bias. Wrapper state
keys are `v1.<original-key>` and `context_head.weight`; V1 checkpoint keys remain
their original names. Do not pass a V1 state dictionary to the wrapper's
`load_state_dict`. Restore V1 strictly and restore adaptation weights separately.
Compatibility is tested with an in-memory synthetic state dictionary, without
opening the frozen research checkpoint. Wrapper forward rejects training mode
and uses `torch.no_grad()`; training this prototype is outside this gate.

## Context and temporal boundary

Predeclare `CalibrationSpec(start_s=0, duration_s=30)` before inspecting outcomes.
Both this interval and `target_start_s` use the same session acquisition clock.
The default is an initial 30-second half-open interval; a different fixed duration
must be selected on development evidence, never reference HR/BP outcomes.

`summarize_calibration` accepts only derived RGB, its validity masks and timestamps.
It selects RGB samples inside that fixed interval, verifies regular/full interval
support and pools per ROI: mean RGB chromaticity, chromaticity standard deviation,
valid fraction and adjacent chromaticity change. All eight features are bounded
in [0,1]. Invalid/nonfinite, negative or zero-intensity observations remain
missing; fewer than two valid samples makes a context ROI unusable. The result
is a detached snapshot. There is no label, subject identifier, reference HR/BP,
reference waveform, automatic segment selection, video decoder or file I/O API.

Active context use requires a finite target start for every batch row and rejects
any target starting before calibration ends, even when the head is still zero.
Future RGB values cannot affect the context. Do not supply traces that were
already normalized or filtered using future target samples. Caller code remains
responsible for genuine timestamps, same-session pairing, and upstream provenance;
the tensor interface cannot authenticate those facts or detect falsified clocks.

## Correction and masks

For each batch/ROI/prior, compute:

```text
offset = max_logit_offset * tanh(context_head(features))
alpha_candidate = normalize(alpha_v1 * exp(offset))
```

The default logit bound is 0.25, constrained to (0,1]. This adds session context
to the existing window-dependent gate. The same context correction applies
throughout each window. No temporal shift, filter, resampling, residual adaptation,
ROI-attention adaptation or additional encoder is introduced. Reweighting uses
float64 intermediate arithmetic and returns V1's original dtype.

Disabled adaptation, absent context and exactly zero offsets return all original
V1 diagnostic tensors bit-for-bit. Missing context ROIs get zero offsets.
Invalid ROI/prior weights remain zero, masked NaNs stay masked, and valid prior
and ROI weights stay normalized. V1 errors for no valid ROI/prior remain errors.
Already-underflowed V1 probabilities stay zero; this conservative wrapper does
not recover logits from rounded probabilities.

## Phase guard and denominator

Bounded prior mixing alone cannot guarantee phase preservation. Every nonzero
candidate is compared with the native V1 output and each valid ROI output using
Gate 03's signed-lag estimator. The declared sample rate must match the actual
V1 input grid. The guard requires at least six seconds, correlation >=0.98,
stable half-window estimates and added shifts <=half a source sample. Failure or
abstention returns the original V1 tensors for the entire affected batch row.
It never realigns candidate samples. Successful comparisons describe only
processing-induced change relative to V1, whose absolute phase is not validated.

`return_audit=True` returns `(value, audit)` with every intended batch window,
accepted adaptation count, fallback/identity count and reason counts. No window
is silently omitted. This guard requires CPU comparisons and is not optimized
for production throughput. Its thresholds are engineering defaults, not tuned
clinical parameters or calibrated uncertainty. It does not establish sensor
synchronization, physiological pulse arrival, face-hand PTT, or BP validity.

## Synthetic example

```python
import torch
from rppg_lab.models import MultiROIPriorResidualWaveformNet
from rppg_lab.waveform_v2_adaptation import (
    CalibrationSpec, SubjectContextPriorAdapter, summarize_calibration,
)

v1 = MultiROIPriorResidualWaveformNet(5, ("GREEN", "CHROM")).eval()
spec = CalibrationSpec(duration_s=30)
adapter = SubjectContextPriorAdapter(v1, calibration=spec, enabled=True,
                                     sample_rate_hz=30)
context = summarize_calibration(
    torch.full((1, 3, 900, 3), 100.), torch.arange(900, dtype=torch.float64) / 30,
    torch.ones(1, 3, 900, dtype=torch.bool),
    roi_names=v1.roi_names, calibration=spec,
)
output, audit = adapter(torch.randn(1, 3, 5, 300), context=context,
                        target_start_s=[30.], return_audit=True)
# Zero initialization returns exactly V1; no training occurs.
```

Run `python -m unittest -v tests.test_waveform_v2_adaptation` for the 21 CPU
synthetic tests, including nonzero pre-existing V1 gates/residuals, identity,
missing contexts, masks, finite bounds, deterministic snapshots, target overlap,
future-sample independence, checkpoint-key accounting and phase rejection.
Equal-HR priors with different phases demonstrate why gate changes need timing
controls; short or unassessable changes fall back rather than earning timing
acceptance. Gate 03 separately checks common latency and differential distortion.

## Next decision gate

Review the wrapper, fixed context specification, phase guard and eventual
separate adaptation-state format before authorizing development-only training.
Then admit permitted public datasets through license/timebase/reference audits,
freeze grouped/domain splits and implement a matched classical benchmark runner.
PURE inventory remains locked against scoring. S001-S006 and previously inspected
UBFC/rPPG-10 evidence are not pristine tests; S007-S011 remain untouched.
No result here supports participant evaluation or BP claims.
