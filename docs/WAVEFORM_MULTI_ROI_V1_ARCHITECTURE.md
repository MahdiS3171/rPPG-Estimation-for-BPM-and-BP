# Multi-ROI prior-residual waveform architecture — Part 3 Waveform v1 candidate

Phase 2 supplied the architecture and synchronized UBFC data path. Part 3 adds
morphology-oriented loss, subject-balanced validation, HR-gated candidate
selection, and a locked participant test split. No final model is frozen.
Existing losses, waveform ablations, the old UBFC dataset/trainer, BP code, and
conservative face–hand timing retain their scientific behavior. See
[MULTI_FACE_ROI.md](MULTI_FACE_ROI.md) for Phase 1 acquisition and
[WAVEFORM_V1_TRAINING_AND_SELECTION.md](WAVEFORM_V1_TRAINING_AND_SELECTION.md)
for the full Part 3 objective, evaluation, selection, and scientific rationale.

## Input and exact model API

Default ROI order: `("forehead", "left_cheek", "right_cheek")`. Input is
`x: [B,R,C,L]`, channels `[RGB_R, RGB_G, RGB_B, prior_1, ..., prior_K]`.
Existing default priors: `GREEN, CHROM, CHROM_WIN, PBV, POS_WIN, OMIT`.
Defaults at 30 Hz/10 seconds give `R=3`, `K=6`, `C=9`, `L=300`.

`combined` is the deterministic pixel union of local masks, with overlaps counted
once. It remains a classical/single-ROI baseline, and adds redundant spatial
support as a default neural branch. Its RGB is a pixel-weighted union mean,
not an equal mean of local RGB means. Explicit ROI overrides can include it for
an ablation. ROI/prior/channel order is stored in dataset/checkpoint metadata.

```python
MultiROIPriorResidualWaveformNet(
    in_channels: int,
    prior_names: Sequence[str],
    roi_names: Sequence[str] = ("forehead", "left_cheek", "right_cheek"),
    prior_start: int = 3,
    base_channels: int = 32,
    num_blocks: int = 3,
    dropout: float = 0.20,
    residual_scale: float = 0.10,
    init_prior: str = "auto",
) -> None

forward(
    x: torch.Tensor,
    roi_quality: Optional[torch.Tensor] = None,
    roi_valid: Optional[torch.Tensor] = None,
    prior_valid: Optional[torch.Tensor] = None,
    return_dict: bool = False,
) -> torch.Tensor | Dict[str, torch.Tensor]
```

`in_channels=prior_start+K`, `prior_start>=3`; training uses offset 3. A custom
larger offset requires documenting additional prefix channels. ROI/prior names
must be nonempty and unique. `model.model_config` contains exact constructor
arguments for reconstructing the model before loading its state dictionary.

## Shared encoder and implemented fusion

One shared `TemporalEncoder1D` processes only the valid rows of flattened
`[B*R,C,L]` input. Differentiable `index_copy` reconstructs zero-filled
`F: [B,R,D,L]`, `D=base_channels`. Both gates pool across time before their
Linear layers. Prior and ROI weights are constant over each window, with no
sample-wise mixing that could itself change waveform morphology.

```text
F[b,r]         = shared_encoder(clean_x[b,r])
delta[b,r,k]   = Linear_prior(mean_t(F[b,r]))[k]
a[b,r,k]       = static_prior_logits[k] + roi_prior_bias[r,k] + delta[b,r,k]
alpha[b,r]     = masked_softmax_k(a[b,r])
fused_prior[b,r,t] = sum_k alpha[b,r,k] * P[b,r,k,t]
residual[b,r,t]    = shared_Conv1d_residual(F[b,r])[t]
roi_ppg[b,r,t]     = fused_prior[b,r,t] + residual_scale * residual[b,r,t]

s[b,r]        = static_roi_logits[r] + Linear_roi(mean_t(F[b,r]))
s[b,r]       += log(clamp(roi_quality[b,r], eps, 1))  # when provided
beta[b]       = masked_softmax_r(s[b])
ppg[b,t]      = sum_r beta[b,r] * roi_ppg[b,r,t]
features[b,d,t] = sum_r beta[b,r] * F[b,r,d,t]
quality[b]    = sigmoid(Linear_quality(mean_t(features[b])))
```

The residual head is one shared 7-sample convolution, matching the existing
prior-residual head and `residual_scale=0.10`. Quality uses the same beta-fused
latent features as waveform fusion. The trainer does not supervise or select
using this quality head; its output is uncalibrated.

Default forward returns only `ppg`. With `return_dict=True`:

| Key | Shape |
| --- | --- |
| `ppg` | `[B,L]` |
| `quality` | `[B]` |
| `features` | `[B,D,L]` |
| `roi_attention` | `[B,R]` |
| `roi_ppg` | `[B,R,L]` |
| `prior_weights` | `[B,R,K]` |
| `fused_priors` | `[B,R,L]` |
| `residuals` (unscaled) | `[B,R,L]` |

## Masks, quality and conservative initialization

`roi_valid: bool[B,R]`, `prior_valid: bool[B,R,K]`; missing masks mean all valid.
Failed prior logits are masked to negative infinity before softmax, giving zero
probability. Remaining prior weights sum to one for valid ROIs. A valid ROI
without any valid prior raises. Invalid ROI prior rows are handled without
`softmax(-inf,...,-inf)` NaNs and return all-zero prior weights.

Invalid ROI logits are masked before ROI softmax: attention equals zero and
remaining ROI weights sum to one. A batch sample with no valid ROI raises.
Invalid branch waveforms/residuals/features are zero. Masked numeric channels
are zeroed before encoding; masked NaNs cannot propagate. Other channels must
be finite. Only valid ROI rows enter shared BatchNorm during training. Invalid
zero placeholders cannot change valid representations or running statistics.
Valid features are scattered without detaching, preserving encoder gradients.
The correction retains all downstream masks and conservative epoch-0 behavior.

Optional finite floating `roi_quality: [B,R]` adds log quality. The clamp uses
`eps=max(1e-6, torch.finfo(x.dtype).tiny)`. Zero learned ROI logits give attention
proportional to clamped quality among valid ROIs. All-zero quality remains
finite and gives equal attention among valid ROIs. Quality never replaces the
validity mask. Attention/weights are diagnostics, not causal physiology.

Initialization uses `_initial_prior_logits(prior_names, init_prior)` directly.
ROI-specific prior biases, both dynamic Linear gates, the residual convolution,
and static ROI logits all start at zero. With all-valid inputs and no quality:

```text
alpha[b,r] = softmax(_initial_prior_logits(prior_names, init_prior))
residuals  = 0
beta[b,r]  = 1/R
ppg[b,t]   = mean_r(sum_k alpha[k] * P[b,r,k,t])
```

Prior masks renormalize the static preferences; quality may bias initial ROI
attention. The trainer evaluates epoch 0, logs all ROI/prior weights, and
explicitly checks zero residual contribution. A deterministic numerical test
verifies the exact conservative waveform independently of encoder randomness.

## New UBFC data path

`UBFCMultiROIRPPGDataset` calls `process_recording` once per video with
`regions=("face",)`, all requested ROIs, and the first requested ROI as legacy
selector. One decode/detection stream supplies all branches. Neither
`FaceROIExtractor` nor `extract_rgb_trace` is used. The old dataset's historical
aggregation/missingness and recording-wide priors remain available for prior
experiments, but are not the new model's data path.

Raw `RGBTrace` values are resampled onto exactly `result.shared.timestamps`
using Phase 1 `resample_trace`/`max_gap_sec`. Long gaps/unsupported endpoints
retain NaNs. No per-ROI grid is invented. Recording-wide rPPG/filter edges are
not used as training priors: priors are freshly computed within complete RGB
windows. Bounded acquisition interpolation can use neighboring source frames
at window edges; prior/statistic/filter computation only sees uniform RGB
samples retained inside the window.

Reference comes from `load_ubfc_ground_truth`. Keep only VIDEO grid points
within reference support and interpolate PPG/HR at those exact points, without
extrapolation or resetting the grid origin. Some local UBFC files contain
identical rounded contact timestamps. In this new dataset only, observations at
identical reference times are averaged, with a warning and recorded
`reference_diagnostics`. Backward/nonfinite reference clocks raise; no sorting
occurs. Nonfinite signal values remain nonfinite, and constant/nonfinite target
windows are excluded. The video clock and old reference loader remain unchanged.

For each ROI/window, the pure `build_window_priors` helper receives only
raw-intensity `RGB_r[start:end]` and calls the existing `METHOD_FUNCS` equations
with unchanged internal filter defaults. Each method gets its own copy. There
is no recording argument, target input, persistent state or prior cache.
Exception-producing, wrong-shaped or nonfinite results become invalid zero
channels. Finite constant output counts as a successful computation, not as a
physiological-quality assertion.

RGB channels and each valid prior use existing standardization helpers
independently within the current window. Invalid ROIs have all-zero input and
all-false prior masks. No normalization is fitted across recordings/subjects.
Standardize the target inside the same window. Compute `y_hr` from that window's
raw reference PPG using Welch; only a nonfinite estimate falls back to finite
reference HR values in that same window, never recording-average HR.

For `a=grid[start]`, `b=grid[end-1]+1/fs`, and original source frames `S` in
`[a,b)`, the exact transparent quality/validity rules are:

```text
roi_quality[r] = sum(validity_mask_r[S]) / len(S), or 0 if S is empty
roi_valid[r]   = len(S)>0
                 and roi_quality[r]>=min_valid_fraction
                 and all_finite(uniform_RGB_r[start:end])
                 and any_successful_window_prior
```

The threshold defaults to `PipelineConfig.min_valid_fraction=0.9` and is
overridable. Returned quality remains the measured source fraction even for an
unusable ROI; masking enforces rejection. Quality uses no reference PPG/HR.
Keep a sample if **any** ROI is usable. Construction screens local priors until
the first success per candidate ROI. Item retrieval computes all requested
priors. No prior waveforms are cached. If data/algorithms change after indexing
and all branches fail, retrieval raises rather than returning an unusable item.

Each default item has exactly this schema:

| Key | Type/shape |
| --- | --- |
| `x` | float32 tensor `[3,9,300]` |
| `y_ppg` | float32 tensor `[300]` |
| `y_hr` | float32 scalar tensor, bpm |
| `roi_valid` | bool tensor `[3]` |
| `roi_quality` | float32 tensor `[3]`, in `[0,1]` |
| `prior_valid` | bool tensor `[3,6]` |
| `subject_id` | string |
| `start_sec` | float64 scalar tensor on the video clock |

Constant names are attributes `roi_names`, `prior_names`, `channel_names`,
avoiding fragile per-item collation. `records` retain raw traces, uniform RGB,
timestamps and interpolation flags; `exclusions`/`reference_diagnostics` explain
decisions. Default window/stride are 10/2 seconds.

## Safe raw extraction cache

`extraction_cache.py` reuses `RGBTrace`, `SharedTimebase` and Phase 1 raw NPZ
keys: `timestamps_original`, `timestamps_uniform`, `face_roi_names`,
`face_roi_<name>_rgb`, `_valid`, and raw diagnostics. Unicode JSON metadata
records identity, diagnostic keys and trace labels. It stores no target,
classical prior, normalized window or learned feature. Reads reuse Phase 1
bounded interpolation from raw traces.

The separate default namespace is
`cache_roi_multi_phase1/phase1_face_<sha256>.npz`. Cache identity includes:

- resolved video path, size and nanosecond modification time;
- ROI names/order, complete extraction config, backend/confidences, sampling,
  timing/gap rules, skin mask settings and pixel thresholds;
- resolved face-model path and model-content SHA256 when available;
- `max_frames` and explicit `phase1_raw_face_rois_v1` schema version;
- extraction source hashes and MediaPipe/OpenCV/NumPy/SciPy versions.

Video identity uses metadata to avoid hashing large videos; deliberately
preserving size/mtime while replacing content requires manual cache invalidation.
Window/stride/prior changes cannot reuse global priors because none are cached.
Reading uses `allow_pickle=False`, checks clocks/rate/order, and rebuilds corrupt
artifacts with a warning. A sibling temporary file and atomic replacement protect
writes. Phase 1 provided a writer but no reusable artifact loader/cache; this
small raw subset avoids caching irrelevant global signals/geometry/BP outputs.
Legacy caches remain untouched. `cache_dir=None` disables caching.

## Part 3 training and checkpoint metadata

The default multi-ROI objective is now `waveform_v1_training_loss`:

```text
L = 1.00 * Lcorr + 0.25 * Ld1 + 0.10 * Lspectral + 0.10 * LHR + 0.02 * Lresidual
```

Waveform and first-difference losses share one discrete lag per sample selected
by minimizing `Lcorr + 0.25 * Ld1`, allowing up to 0.5 seconds. Signed Pearson
preserves polarity. The auxiliary normalized spectral comparison uses
Jensen-Shannon divergence / ln(2) over `[0.7, min(8.0, 0.45*fs)]` Hz. HR reuses
the stable label-distribution loss. Residual MSE observes valid ROIs only.
There is no default smoothing, second derivative/APG, or dicrotic/RI feature
supervision. Facial input and fingertip reference cannot be assumed anatomically
identical. Original losses remain unchanged; `--objective phase2_rppg` is the
explicit old-objective ablation.

```powershell
.\.venv\Scripts\python.exe scripts/train_waveform_multi_roi_ubfc.py --ubfc-root UBFCData --cache-dir cache_roi_multi_phase1 --out-dir checkpoints/waveform_multi_roi_v1_candidate --objective waveform_v1 --epochs 20 --batch-size 32 --fs 30 --win-sec 10 --stride-sec 2
.\.venv\Scripts\python.exe scripts/evaluate_waveform_multi_roi_ubfc.py --ubfc-root UBFCData --checkpoint checkpoints/waveform_multi_roi_v1_candidate/best_waveform_candidate.pt --split-manifest checkpoints/waveform_multi_roi_v1_candidate/waveform_v1_split.json --split val --out outputs/waveform_v1_validation.json
```

The reusable participant split reserves 20% test first and 20% of remaining
participants for validation, using sorted IDs and seeded PCG64 permutation.
`waveform_v1_split.json` records exact IDs, seed, fractions and algorithm version;
its content and canonical SHA256 are checkpointed. The trainer constructs only
train and validation datasets. Locked test requires separate explicit
`--split test` evaluation after selection, before a later freeze. Two-subject
legacy plumbing tests are explicitly smoke-only with no test set.

Evaluation reports zero/best-lag waveform correlation, lag samples/ms,
first-difference correlation at the waveform-selected lag, aligned z-score RMSE,
broad spectral distance/similarity, waveform-derived HR error, and coarse
width/upstroke/fall/area/IBI diagnostics. Missing metrics are NaN (JSON null).
Window summaries remain; finite means per subject are averaged equally for
subject-balanced metrics. ROI attention, validity, prior weights and residual
magnitude remain diagnostics. Coarse features and spectral metrics never rank
checkpoints.

```text
morphology_score = 0.5 * subject_balanced_metrics.wave_corr_aligned
                 + 0.5 * subject_balanced_metrics.d1_corr_same_lag
eligible: subject_balanced_HR_MAE(epoch) <= subject_balanced_HR_MAE(epoch0) + 1.0 bpm
```

The HR tolerance is an engineering guard, not a clinical threshold. Eligible
maximum morphology score selects `best_waveform_candidate.pt`. Score ties within
1e-8 prefer higher subject-balanced aligned correlation, lower HR MAE, then
earlier epoch. Epoch 0 is the finite conservative reference. Nonzero lag is not
penalized. Also save `best_morphology_unconstrained.pt`,
`best_hr_diagnostic.pt`, and `last.pt`. All carry
`waveform_v1_status="candidate_not_frozen"`; no final Waveform v1 is frozen.
Deprecated Phase 2 artifact aliases retain analysis/test compatibility with
explicit Part 3 metadata.

Checkpoints contain exact model config/state, ROI/prior/channel ordering,
extraction config/cache version, fs/window/stride/frame/quality limits,
objective/all weights/lag/spectral band, score formula/HR gate/tie rules,
manifest path/content/hash and all split IDs, epoch/validation/epoch-0 metrics,
selection criterion, seed and Git/package/source provenance. The separate
evaluator reconstructs this configuration and refuses incompatible ordering,
cache schemas or split identities. See the training/selection document for
complete CLI flags, manifest schema and validation policy.

## Small usage example

```python
from torch.utils.data import DataLoader
from rppg_lab.datasets import UBFCMultiROIRPPGDataset, find_ubfc_subjects
from rppg_lab.models import MultiROIPriorResidualWaveformNet

dataset = UBFCMultiROIRPPGDataset(find_ubfc_subjects("UBFCData"),
                                cache_dir="cache_roi_multi_phase1")
loader = DataLoader(dataset, batch_size=2)
batch = next(iter(loader))
model = MultiROIPriorResidualWaveformNet(
    len(dataset.channel_names), dataset.prior_names, dataset.roi_names)
out = model(
    batch["x"],                       # [2,3,9,300]
    roi_quality=batch["roi_quality"], # [2,3]
    roi_valid=batch["roi_valid"],     # [2,3]
    prior_valid=batch["prior_valid"], # [2,3,6]
    return_dict=True,
)
# ppg [2,300], quality [2], features [2,32,300]
# roi_attention [2,3], prior_weights [2,3,6]
# roi_ppg/fused_priors/residuals each [2,3,300]
```

## Validation and scientific limitations

`tests/test_waveform_multi_roi.py` retains Phase 2 model/data/cache numerical
expectations. `tests/test_waveform_v1.py` adds TRAIN-mode valid-only BatchNorm
statistics and gradients, same-lag losses/metrics, polarity, broad spectral
bands, component accounting and valid residual masks, subject-balanced metrics,
HR-gated morphology selection/ties, locked-test separation and metadata-driven
checkpoint reconstruction/evaluation. Existing tests and old losses are not
rewritten for the new objective.

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m compileall -q rppg_lab scripts tests
git diff --check
```

Software tests and small train/validation smokes establish implementation
behavior, not full-cohort performance. Prior polarity/phase/filter edges and
face/finger differences still require complete validation. There is no BP
accuracy claim, morphology-equivalence claim, validated dicrotic feature claim,
or hand-generalization claim. Pixel/video encoders, hand neural training,
learned face-hand timing, BP integration and HealthMultiTaskNet redesign remain
outside Part 3. The quality head remains uncalibrated, and the resulting model
remains a Waveform v1 candidate until a later explicit review and freeze.
