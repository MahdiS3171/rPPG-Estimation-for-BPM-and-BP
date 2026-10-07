# Multi-ROI prior-residual waveform architecture — Phase 2 candidate

This adds the architecture and UBFC training data path for a future Waveform
Model v1. Loss and checkpoint selection remain **provisional**. No final model
is frozen. Existing waveform ablations, the old UBFC dataset/trainer, BP code,
and conservative face–hand timing retain their scientific behavior. See
[MULTI_FACE_ROI.md](MULTI_FACE_ROI.md) for Phase 1 acquisition.

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

One shared `TemporalEncoder1D` processes flattened `[B*R,C,L]` input, producing
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
latent features as waveform fusion. The provisional trainer does not supervise
this quality head; its output is uncalibrated.

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
be finite. The shared encoder retains its existing BatchNorm behavior during
training, sharing weights and statistics across the branches.

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

## Training and checkpoint metadata

```powershell
.\.venv\Scripts\python.exe scripts/train_waveform_multi_roi_ubfc.py --ubfc-root UBFCData --cache-dir cache_roi_multi_phase1 --out-dir checkpoints/waveform_multi_roi_phase2 --epochs 20 --batch-size 32 --fs 30 --win-sec 10 --stride-sec 2
```

`--config` accepts Phase 1 settings; `sample_rate` must match `--fs`. Dataset
forces face-only extraction and `--rois` order. `--rois`/`--priors` are
comma-separated overrides. `--epochs 0` saves/evaluates initialization. A small
smoke run can add `--max-subjects 4 --max-frames 450 --epochs 1 --batch-size 4
--base-channels 8 --num-blocks 1 --stride-sec 10`.

The existing `subject_split(..., val_fraction=0.2, seed=42)` partitions whole
participants and exact IDs are printed. `--max-subjects` limits discovery before
splitting and is a smoke-run aid. The unchanged **PROVISIONAL**
`rppg_training_loss` supervises the final waveform and regularizes unscaled
residuals of valid ROI branches. No new morphology loss is added.

Validation logs HR MAE (including HR metric count), zero-lag correlation,
best correlation within +/-0.5 seconds, mean ROI attention, all ROI x prior
weight means, valid-window fractions and maximum absolute scaled residual.
Prior means are conditional on a valid ROI (absent ROIs report null); mean ROI
attention includes invalid windows as zero. These are descriptive diagnostics.

Every epoch, including epoch 0, saves `waveform_multi_roi_last.pt`. Optional
`waveform_multi_roi_best_hr_PROVISIONAL.pt` selects minimum validation HR MAE;
`--no-best-hr` disables it. HR selection is **not** the future final waveform
criterion. `training_log_PROVISIONAL.json` retains epoch-0/later diagnostics,
metadata and excluded windows. Checkpoint fields are:

```text
model_class, model_config                  exact constructor arguments
model_state, optimizer_state               tensor state dictionaries
phase                                      "Phase 2 PROVISIONAL"
roi_names, prior_names, channel_names/channel_ordering
extraction_config, extraction_cache_version
fs, win_sec, window_length_samples, stride_sec, min_valid_fraction
seed, train_subjects, val_subjects          exact participant IDs
reference_diagnostics                      duplicate-time policy/counts
loss_name                                  "rppg_training_loss (PROVISIONAL Phase 2)"
checkpoint_selection_criterion             LAST or provisional HR MAE
epoch, validation_metrics, args, provenance code/package/source identities
```

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

## Validation and Part 3 limitations

`tests/test_waveform_multi_roi.py` covers exact shapes and conservative
initialization, validity/quality/prior masks, explicit empty-support errors,
masked NaNs, shared feature fusion, finite backward/optimizer updates,
window-only method inputs and independent copies, outside-window perturbation
invariance in helper/dataset, synthetic Phase 1 data/loader schema, bad ROIs,
bounded gaps/source fractions, common support, duplicate reference handling,
ordering, target-independent quality/local normalization, prohibition of legacy
extraction, cache round trips/invalidation/corruption, epoch-0 evaluation,
training and checkpoint reload. Existing numerical expectations are unchanged.

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m compileall -q rppg_lab scripts tests
git diff --check
```

Verification on 2026-10-07: all **80 tests passed** (49 existing plus 31 new),
`compileall` and `git diff --check` passed. Synthetic forward/backward and an
optimizer step, mocked Phase 1 dataset construction, and checkpoint reload
passed. A local UBFC smoke run decoded at most 450 frames per subject: training
IDs `subject1, subject10, subject11`, validation ID `subject12`, three training
windows and one validation window, one epoch/optimizer step on CUDA. Epoch 0
had zero residual contribution, equal ROI attention and the static prior
weights. Local ignored artifacts are in
`outputs/smoke_waveform_multi_roi_20261007/`. This small run verifies plumbing,
not full-cohort performance or waveform morphology.

Part 3 must design morphology-oriented loss/evaluation and final checkpoint
selection. Classical polarity/phase/filter edges and contact-to-face alignment
still need scientific evaluation. HR/aligned correlation alone does not validate
morphology. There is **no BP accuracy claim, no morphology-validation claim,
and no hand-generalization claim**. No final Waveform v1 checkpoint is frozen.
Pixel/video encoders, hand neural training, learned face–hand timing, BP
integration and HealthMultiTaskNet redesign are outside this change. Software
tests and a small local training step establish software behavior only.
