# Waveform v2 — Gate 03: Timing-transfer engineering contract

**Status:** Local add-on, not merged into GitHub; exclusively synthetic engineering controls.
**Pinned base:** `MahdiS3171/rPPG-Estimation-for-BPM-and-BP@27e54556025e2eff4e1b4091001a2eaf28667cc7`
**Date:** 2026-11-03
**Protected:** frozen Waveform v1, previously published outcomes, S007–S011 reserved recordings.

## Why this is needed

The current `waveform_v1_training_loss` uses `same_lag_waveform_derivative_loss` with `WaveformV1LossConfig.max_lag_sec=0.5`. The per-window lag is selected to minimize correlation and derivative losses, and the phase-insensitive spectral/HR losses do not anchor the output to camera time. This supports contact-PPG-to-facial-wave *morphology* learning with uncertain reference site offset, not absolute pulse-arrival estimation.

**Reproducible synthetic demonstration:** a 300-sample/30-Hz pulse shifted by 100 ms (three samples) has zero-lag Pearson correlation of about **0.639**, while the existing V1 lag-tolerant waveform correlation term gives a loss of **0.0** at selected lag **+3 samples**. These results were obtained with deterministic synthetic inputs only; no trained weights or video were used. This is a loss/objective property, not a measured v1 prediction error.

## Gate 03 addition

- `rppg_lab/waveform_v2_timing_contract.py` — strict native-time-grid signed-lag diagnostics, synthetic timing-transfer audit, and deterministic waveform fixtures.
- `scripts/audit_waveform_v2_timing_synthetic.py` — safe CLI that writes synthetic-only JSON reports to a *new* output directory.
- `tests/test_waveform_v2_timing_contract.py` — 14 pure synthetic unit tests.

Gate 03 reuses Gate 02 unchanged. No trained network, dataset loader, checkpoint, data manifest, or inference path is modified.

### Scientific conventions

- Positive delay means the second signal (hand in our example) **lags** the first (face).
- A `PASS_SYNTHETIC_ONLY` means both an injected differential delay and the original synthetic camera-time anchor were preserved within the **predeclared half-source-frame synthetic engineering threshold**. The stricter threshold is a test criterion, **not** an achievable physiological precision claim.
- `uncertainty_floor_ms` is a **conservative reporting-policy floor** of one observed source sampling step, not a statistical confidence interval, calibrated uncertainty, or physical bound.
- The signed delay estimator never inverts waveform polarity, shifts the candidate to match the reference, removes nonfinite time points, extrapolates a missing segment, or uses DTW. It refuses irregular clocks, unsupported shifts, short/flat signals, weak correlations, peak-at-boundary results and unstable first/second halves.
- The estimator is a **diagnostic proxy**, not a physiological timing estimator. Regular narrowband waves can falsely appear to have a stable lag due to shared pulse frequency; this test cannot independently prove pulse propagation.
- Processing-induced latency common to face and hand can cancel in their **difference**. We therefore report independent camera-anchor shifts for both sites as well as inter-site delay.
- A 100 ms common latency fails camera-time anchoring even if relative delay is preserved; an additional 85 ms hand-only latency fails differential delay. Site-specific morphology shifts are explicitly stressed.

### Deterministic synthetic scenarios

| Scenario | Intended result | Why it matters |
|---|---|---|
| Identity reconstruction | Pass, synthetic only | Positive control, signed-delay convention |
| Same +100 ms phase lag on both site outputs | Fail | Relative delay alone hides lost camera-time anchor |
| Extra +85 ms on hand reconstruction | Fail | Phase distortion mimics changed inter-site pulse transit |
| Hand polarity inversion | Abstain | Do not rescue output by flipping sign post hoc |
| Unrelated independent hand noise | Abstain | Reject clearly invalid delay input |
| Hand waveform morphology change | Fail | Different pulse shapes bias correlation-based lag |

Each scenario is evaluated at both nominal **30** and **60** Hz, 12 total cases. With the current fixed, deterministic synthetic signal: **2 PASS_SYNTHETIC_ONLY, 6 FAIL, 4 ABSTAIN_UNVERIFIED**. The module does not score real physiological data.

### Engineering test commands

From the project root after applying the additive patch (all dependencies installed):

```powershell
python -m unittest -v tests.test_waveform_v2_infrastructure tests.test_waveform_v2_timing_contract
python scripts/audit_waveform_v2_timing_synthetic.py --out-dir outputs/waveform_v2_gate03_synthetic_new
```

`--out-dir` must not exist; no default write is performed. Do not confuse the generated report with a BP or real-world waveform validation result.

## Next code gates and explicit blockers

1. **When allowed to make a change to a training prototype**, use this timing contract as an acceptance test for any proposed phase-aware model adaptation. Do not add a zero-lag penalty against an unknown face-to-finger contact PPG lag.
2. Build an input-to-output synthetic video pipeline under audited camera timestamps (no model weights/real recordings required to draft it). Gate 03 currently tests output signal streams, **not** video-to-signal transformations.
3. Once PURE access is approved, inspect actual frame/reference clock semantics before enabling any waveform shape/phase supervision. The fact that values have timestamps is not independent synchronization proof.
4. Later, collect a small synchronized two-site contact PPG/ECG study to determine whether an optical delay reflects true physiological timing. No current synthetic test can replace that evidence.
5. Keep five newly recorded participants S007–S011 untouched by exploratory model decisions.

**Conclusion:** Gate 03 establishes a fail-closed synthetic engineering contract and exposes a design blind spot. It does not establish timing accuracy of frozen Waveform v1 and does not validate a face-hand BP mechanism.
