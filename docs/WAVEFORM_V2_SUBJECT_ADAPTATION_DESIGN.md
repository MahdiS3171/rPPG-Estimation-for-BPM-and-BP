# Waveform v2 — Subject-context adaptive prior design (proposal only)

**Design state:** Architecture specification, not trained or implemented.  
**Base model:** `MultiROIPriorResidualWaveformNet` in `rppg_lab/models.py` (pinned `27e5455`); existing multi-ROI window-specific prior gate and residual must remain available as a baseline.

## What exists in Waveform v1

At each window the model receives `[B, R, C, L]`, where the face ROIs are forehead, left cheek and right cheek and each ROI has RGB plus six classical priors. The prior fusion uses static prior logits, ROI-specific bias, and a feature-dependent `gate_head`; `prior_weights` is `[B, R, K]`. ROI weighting is another gated softmax with validity and input quality. The final waveform is a weighted sum of each ROI's prior mixture plus a small learned residual. Quality-head outputs are not a validated acceptance score. The model is window-level; it does **not** maintain a subject-context vector.

## Proposed minimal Waveform v2 addition (A4)

**Context:** Use a fixed initial **30–45 second pre-cuff** segment *only for local data inference when available*. Public datasets without a matching pre-cuff segment should use a predeclared context window that is separate from scored prediction windows. Selection of duration must be performed on development data only. Never use a participant's HR, BP, identity or held-out target values to construct context.

**Encoder:** A small permutation-invariant or temporal pooling network summarizes robust per-ROI RGB chromaticity/covariance, dropout/missingness, motion proxies and classical-prior spectral consistency. It produces subject-session context `z_s`. The representation should be recalculated per session, not stored as a lifelong biometric profile.

**Gate:** Preserve existing per-window logits, adding a bounded, initially zero context-conditioned offset:

`prior_logits[b,r,k,w] = current_v1_logits[b,r,k,w] + scale * context_head(z_s)[b,r,k]`.

Similarly, allow an *optional separately ablated* context offset for ROI logits. Bias magnitudes and entropy regularization must be tuned only on development participants to prevent hard switching and false confidence. Initialize `context_head` to zero, allowing **exact v1-equivalent gating at epoch 0 if the compatible v1 weights are loaded**. Maintain invalid ROI/prior masks from v1; context never rescues a missing signal.

**Timing caution:** A soft mixture of two priors with different filter phase can change pulse-event timing. Context adaptation must be accepted only after Gate 03 tests measure relative **and** absolute phase shifts. Do not regard improving HR MAE as permission to use adapted waveforms for optical delay.

## Proposed ablation sequence

A0 frozen v1 and fixed classical controls; A2 multi-dataset v1 architecture (new training); A3 current-window quality-aware gate only; A4 add session-context gate at same encoder capacity; A5 add timing-preserving constraints after independently verified synchronization and synthetic tests; A6 examine larger backbones only if capacity bottleneck remains. Compute matched-window HR, shape, quality/coverage and synthetic timing metrics. Predeclare which experiments are descriptive vs locked.

## Dataset split and leakage precautions

- Entire person + their recordings are in **one** partition. No overlapping context or target windows between splits; do not train context on validation/test target labels.
- Original Waveform v1 UBFC locked test, rPPG-10 and pilot S001–S006 have been historically inspected. They may support diagnostic analysis but must not be quietly reused as pristine tests.
- S007–S011 remain **reserved** and cannot inform model, thresholds, ROI or analysis design.
- No BP supervision in A2–A5. Personalizing signal extraction with label-free context is *not* personalized BP calibration.
- Account for dataset-level camera/illumination correlation. Use leave-one-dataset-out exploratory design until a further independent external domain is available.

## Future scientific decisions (not blockers today)

- Whether context is taken from raw video patches or derived RGB/quality traces; start with derived traces to preserve v1 comparability.
- Whether personalization should adapt gate only (recommended first) or residual encoder weights (much larger overfitting risk).
- Whether separate face and hand session context is necessary for optical delay; revisit only after site-specific phase controls and physiological references exist.

This specification is intentionally conservative: it extends the system's existing adaptive weights rather than proposing hard RGB clusters or a new oversized network before multi-domain evidence is established.
