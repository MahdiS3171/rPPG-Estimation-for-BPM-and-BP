#!/usr/bin/env python
"""Train prior-guided video HR model on UBFC.

V2 adds epoch-0 checkpointing, candidate dropout/noise, and a video-only
auxiliary HR head so the video branch is actually trained instead of staying
silent behind CHROM_WIN.

This is the second video-stage experiment after raw VideoHRNet.  The raw-video
model was too slow and far behind classical priors.  VideoPriorHRNet fixes the
main issue by giving the model both:

  video clip         -> spatial/appearance/motion cues
  1D RGB/prior trace -> validated GREEN/CHROM/CHROM_WIN/... pulse cues

It is initialized to preserve a strong candidate prior, so epoch-0 should be
close to CHROM_WIN instead of a random HR predictor.  Training then learns only
corrections, candidate weighting, and quality/confidence.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[1]))

from rppg_lab.classical import DEFAULT_PRIORS
from rppg_lab.datasets import find_ubfc_subjects, subject_split, UBFCRPPGDataset
from rppg_lab.metrics import regression_metrics
from rppg_lab.models import VideoPriorHRNet
from rppg_lab.signals import estimate_hr_welch
from rppg_lab.video_datasets import UBFCVideoHRDataset, read_video_clip_uniform


class UBFCVideoPriorDataset(Dataset):
    """Paired UBFC video clip + 1D trace/prior windows.

    The video part supplies raw face crops.  The trace part is exactly the
    validated UBFCRPPGDataset path, so candidate HR estimates match the strong
    1D baselines we already measured.
    """

    def __init__(
        self,
        subjects,
        fs: float = 30.0,
        clip_sec: float = 10.0,
        stride_sec: float = 2.0,
        clip_frames: int = 96,
        image_size: int = 48,
        video_cache_dir: str | Path = "cache_video_prior",
        roi_cache_dir: str | Path = "cache_roi_oldpoints",
        roi: str = "face",
        prior_methods: Sequence[str] = DEFAULT_PRIORS,
        quality_tau_bpm: float = 5.0,
        selection_tau_bpm: float = 3.0,
        clip_cache: bool = True,
    ) -> None:
        self.fs = float(fs)
        self.clip_sec = float(clip_sec)
        self.stride_sec = float(stride_sec)
        self.clip_frames = int(clip_frames)
        self.image_size = int(image_size)
        self.video_cache_dir = Path(video_cache_dir)
        self.roi_cache_dir = Path(roi_cache_dir)
        self.roi = roi
        self.prior_methods = list(prior_methods)
        self.quality_tau_bpm = float(quality_tau_bpm)
        self.selection_tau_bpm = float(selection_tau_bpm)
        self.clip_cache = bool(clip_cache)

        self.video_ds = UBFCVideoHRDataset(
            subjects,
            clip_sec=self.clip_sec,
            stride_sec=self.stride_sec,
            clip_frames=self.clip_frames,
            image_size=self.image_size,
            fs_target=self.fs,
            cache_dir=self.video_cache_dir,
        )
        self.trace_ds = UBFCRPPGDataset(
            subjects,
            fs_target=self.fs,
            win_sec=self.clip_sec,
            stride_sec=self.stride_sec,
            roi=self.roi,
            cache_dir=self.roi_cache_dir,
            prior_methods=self.prior_methods,
        )

        self.channel_names = ["RGB_R", "RGB_G", "RGB_B"] + self.prior_methods
        self.candidate_idx = ([1] if len(self.channel_names) > 1 else []) + list(range(3, len(self.channel_names)))
        self.candidate_names = [self.channel_names[i] for i in self.candidate_idx]
        self.pairs: List[Tuple[int, int]] = []
        self._match_windows()

    def _trace_keys(self) -> Dict[str, List[Tuple[float, int]]]:
        out: Dict[str, List[Tuple[float, int]]] = {}
        for idx, (rec_idx, s, e) in enumerate(self.trace_ds.samples):
            rec = self.trace_ds.records[rec_idx]
            sid = str(rec["subject_id"])
            start_sec = float(s) / self.fs
            out.setdefault(sid, []).append((start_sec, idx))
        for sid in out:
            out[sid].sort(key=lambda x: x[0])
        return out

    def _match_windows(self) -> None:
        trace_by_subject = self._trace_keys()
        tol = max(0.25, 0.51 * self.stride_sec)
        misses = 0
        for v_idx, win in enumerate(self.video_ds.windows):
            rec = self.video_ds.records[win.rec_idx]
            sid = str(rec["subject_id"])
            candidates = trace_by_subject.get(sid, [])
            if not candidates:
                misses += 1
                continue
            starts = np.asarray([c[0] for c in candidates], dtype=float)
            j = int(np.argmin(np.abs(starts - float(win.start_sec))))
            if abs(float(starts[j]) - float(win.start_sec)) <= tol:
                self.pairs.append((v_idx, candidates[j][1]))
            else:
                misses += 1
        if misses:
            print(f"Warning: unmatched video/trace windows: {misses}")

    def __len__(self) -> int:
        return len(self.pairs)

    def _clip_cache_path(self, rec: Dict, start_sec: float) -> Path:
        sid = str(rec["subject_id"])
        safe_start = int(round(float(start_sec) * 1000.0))
        return self.video_cache_dir / "clips" / f"{sid}_s{safe_start}_T{self.clip_frames}_S{self.image_size}.npy"

    def _read_clip(self, v_idx: int) -> torch.Tensor:
        win = self.video_ds.windows[v_idx]
        rec = self.video_ds.records[win.rec_idx]
        cache_path = self._clip_cache_path(rec, win.start_sec)
        if self.clip_cache and cache_path.exists():
            arr = np.load(cache_path).astype(np.float32)
        else:
            arr = read_video_clip_uniform(
                rec["video_path"],
                start_sec=win.start_sec,
                end_sec=win.end_sec,
                num_frames=self.clip_frames,
                crop_box=rec["crop_box"],
                image_size=self.image_size,
            )
            if self.clip_cache:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(cache_path, arr.astype(np.float16))
        return torch.from_numpy(arr.astype(np.float32))

    def _candidate_targets(self, x: torch.Tensor, y_hr: float) -> Dict[str, torch.Tensor]:
        cand_hrs: List[float] = []
        for ch in self.candidate_idx:
            try:
                cand_hrs.append(float(estimate_hr_welch(x[ch].numpy(), self.fs).hr_bpm))
            except Exception:
                cand_hrs.append(float("nan"))
        cand = np.asarray(cand_hrs, dtype=float)
        valid = np.isfinite(cand)
        if np.isfinite(y_hr) and valid.any():
            errs = np.abs(cand - y_hr)
            errs[~valid] = np.inf
            best_idx = int(np.argmin(errs))
            best_err = float(errs[best_idx])
            # Soft candidate target: partial credit to near-equivalent priors.
            logits = -0.5 * (errs / max(1e-6, self.selection_tau_bpm)) ** 2
            logits[~np.isfinite(logits)] = -1e9
            logits = logits - np.max(logits)
            soft = np.exp(logits)
            soft[~valid] = 0.0
            if float(soft.sum()) > 0:
                soft = soft / float(soft.sum())
        else:
            errs = np.full_like(cand, np.inf, dtype=float)
            best_idx = -1
            best_err = float("inf")
            soft = np.zeros_like(cand, dtype=float)
        q = float(np.exp(-((best_err / self.quality_tau_bpm) ** 2))) if np.isfinite(best_err) else 0.0
        return {
            "candidate_hrs": torch.from_numpy(np.nan_to_num(cand, nan=0.0).astype(np.float32)),
            "candidate_valid": torch.from_numpy(valid.astype(np.bool_)),
            "candidate_errs": torch.from_numpy(np.nan_to_num(errs, nan=999.0, posinf=999.0).astype(np.float32)),
            "selection_target": torch.from_numpy(soft.astype(np.float32)),
            "best_candidate_index": torch.tensor(best_idx, dtype=torch.long),
            "quality_target": torch.tensor(q, dtype=torch.float32),
            "best_input_err": torch.tensor(best_err if np.isfinite(best_err) else 999.0, dtype=torch.float32),
        }

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor | str]:
        v_idx, t_idx = self.pairs[idx]
        vwin = self.video_ds.windows[v_idx]
        vrec = self.video_ds.records[vwin.rec_idx]
        trace_item = self.trace_ds[t_idx]
        x = trace_item["x"]
        y_hr = float(trace_item["y_hr"])
        out = {
            "video": self._read_clip(v_idx),
            "x": x,
            "y_hr": trace_item["y_hr"],
            "subject_id": str(vrec["subject_id"]),
            "start_sec": torch.tensor(float(vwin.start_sec), dtype=torch.float32),
        }
        out.update(self._candidate_targets(x, y_hr))
        return out


def seed_all(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def safe_metrics(gt, pred):
    gt = np.asarray(gt, dtype=float)
    pred = np.asarray(pred, dtype=float)
    m = np.isfinite(gt) & np.isfinite(pred)
    if m.sum() == 0:
        return regression_metrics([0.0], [np.nan])
    return regression_metrics(gt[m], pred[m])


def evaluate_channel_baselines(ds: UBFCVideoPriorDataset) -> List[Tuple[str, float, float, float, float, float]]:
    gt = []
    preds = {name: [] for name in ds.channel_names}
    for i in range(len(ds.trace_ds)):
        item = ds.trace_ds[i]
        x = item["x"].numpy()
        gt.append(float(item["y_hr"]))
        for ch, name in enumerate(ds.channel_names):
            preds[name].append(estimate_hr_welch(x[ch], ds.fs).hr_bpm)
    rows = []
    for name, p in preds.items():
        m = safe_metrics(gt, p)
        rows.append((name, m.mae, m.rmse, m.me, m.sd, m.pearson))
    rows.sort(key=lambda r: r[1])
    return rows


def hr_distribution_ce(logits: torch.Tensor, hr_bpm: torch.Tensor, bins: torch.Tensor, sigma_bpm: float = 3.0) -> torch.Tensor:
    hr = hr_bpm.float().view(-1, 1)
    target = torch.exp(-0.5 * ((bins.view(1, -1) - hr) / float(sigma_bpm)) ** 2)
    target = target / (target.sum(dim=-1, keepdim=True) + 1e-8)
    logp = torch.log_softmax(logits, dim=-1)
    return -(target * logp).sum(dim=-1).mean()


def soft_selection_ce(selection_logits: torch.Tensor, target_probs: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    logits = selection_logits.masked_fill(~valid.bool(), -1e4)
    logp = torch.log_softmax(logits, dim=-1)
    denom = target_probs.sum(dim=-1)
    good = denom > 1e-6
    if good.sum() == 0:
        return logits.sum() * 0.0
    return -(target_probs[good] * logp[good]).sum(dim=-1).mean()



def apply_candidate_dropout(
    cand: torch.Tensor,
    valid: torch.Tensor,
    drop_prob: float = 0.0,
    noise_std: float = 0.0,
    protect_one: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Training-time corruption of candidate HRs.

    This is deliberately used only during training. It prevents the model from
    solving UBFC by always passing through CHROM_WIN and forces the video/trace
    encoders to learn a fallback/correction signal.
    """
    if drop_prob <= 0 and noise_std <= 0:
        return cand, valid

    cand2 = cand.clone()
    valid2 = valid.clone().bool()

    if noise_std > 0:
        noise = torch.randn_like(cand2) * float(noise_std)
        cand2 = torch.where(valid2, cand2 + noise, cand2)

    if drop_prob > 0:
        keep = torch.rand_like(cand2.float()) > float(drop_prob)
        keep = keep | (~valid2)
        valid2 = valid2 & keep

        if protect_one:
            # Ensure every row still has at least one valid candidate. If a row
            # lost all candidates, restore the first originally-valid candidate.
            empty = valid2.sum(dim=1) == 0
            if empty.any():
                orig = valid.bool()
                first = orig.float().argmax(dim=1)
                rows = torch.where(empty)[0]
                valid2[rows, first[rows]] = True

    return cand2, valid2

def quality_metrics(gt: np.ndarray, pred: np.ndarray, q: np.ndarray) -> List[Tuple[float, int, float, float]]:
    err = np.abs(pred - gt)
    out = []
    for cov in [1.0, 0.8, 0.6, 0.4, 0.2]:
        n = max(1, int(round(cov * len(err))))
        idx = np.argsort(-q)[:n]
        out.append((100.0 * cov, n, float(np.mean(err[idx])), float(np.median(err[idx]))))
    return out


@torch.no_grad()
def evaluate(model: VideoPriorHRNet, loader: DataLoader, device: torch.device) -> Dict:
    model.eval()
    gt, pred, video_pred, q, subjects = [], [], [], [], []
    sel_acc_n = sel_acc_d = 0
    sel_soft_sum = sel_soft_n = 0
    exp_err_sum = exp_err_n = 0
    gate_vals, resid_vals = [], []
    for batch in loader:
        video = batch["video"].to(device)
        x = batch["x"].to(device)
        y = batch["y_hr"].to(device)
        cand = batch["candidate_hrs"].to(device)
        valid = batch["candidate_valid"].to(device)
        out = model(video, x, candidate_hrs=cand, candidate_valid=valid, return_dict=True)
        gt.append(y.detach().cpu().numpy())
        pred.append(out["hr_bpm"].detach().cpu().numpy())
        if "video_hr_bpm" in out:
            video_pred.append(out["video_hr_bpm"].detach().cpu().numpy())
        q.append(out["quality"].detach().cpu().numpy())
        subjects.extend([str(s) for s in batch["subject_id"]])
        gate_vals.append(out["fusion_gate"].detach().cpu().numpy())
        resid_vals.append(out["residual_bpm"].detach().cpu().numpy())

        best_idx = batch["best_candidate_index"].to(device)
        valid_best = best_idx >= 0
        if valid_best.any():
            sel_pred = out["selection_probs"].argmax(dim=-1)
            sel_acc_n += int((sel_pred[valid_best] == best_idx[valid_best]).sum().item())
            sel_acc_d += int(valid_best.sum().item())
        st = batch["selection_target"].to(device)
        if st.sum(dim=1).gt(1e-6).any():
            good = st.sum(dim=1) > 1e-6
            logp = torch.log_softmax(out["selection_logits"], dim=-1)
            sel_soft_sum += float((-(st[good] * logp[good]).sum(dim=-1)).sum().item())
            sel_soft_n += int(good.sum().item())
        cerr = batch["candidate_errs"].to(device)
        probs = out["selection_probs"]
        finite_err = torch.isfinite(cerr) & (cerr < 900.0)
        if finite_err.any():
            err_safe = torch.where(finite_err, cerr, torch.zeros_like(cerr))
            exp_err = (probs * err_safe).sum(dim=-1)
            ok = finite_err.any(dim=1)
            exp_err_sum += float(exp_err[ok].sum().item())
            exp_err_n += int(ok.sum().item())
    gt_a = np.concatenate(gt) if gt else np.array([])
    pr_a = np.concatenate(pred) if pred else np.array([])
    vpr_a = np.concatenate(video_pred) if video_pred else np.array([])
    q_a = np.concatenate(q) if q else np.array([])
    gate_a = np.concatenate(gate_vals) if gate_vals else np.array([])
    resid_a = np.concatenate(resid_vals) if resid_vals else np.array([])
    m = safe_metrics(gt_a, pr_a)
    vm = safe_metrics(gt_a, vpr_a) if len(vpr_a) == len(gt_a) and len(gt_a) else None
    err = np.abs(pr_a - gt_a)
    qc = np.corrcoef(q_a, -err)[0, 1] if len(q_a) > 2 and np.std(q_a) > 1e-8 and np.std(err) > 1e-8 else np.nan
    return {
        "gt": gt_a,
        "pred": pr_a,
        "quality": q_a,
        "subjects": np.asarray(subjects, dtype=object),
        "metrics": m,
        "video_metrics": vm,
        "quality_error_corr": float(qc) if np.isfinite(qc) else float("nan"),
        "coverage": quality_metrics(gt_a, pr_a, q_a) if len(gt_a) else [],
        "selection_acc": (sel_acc_n / sel_acc_d) if sel_acc_d else float("nan"),
        "selection_soft_ce": (sel_soft_sum / sel_soft_n) if sel_soft_n else float("nan"),
        "selection_expected_err": (exp_err_sum / exp_err_n) if exp_err_n else float("nan"),
        "gate_mean": float(np.mean(gate_a)) if len(gate_a) else float("nan"),
        "residual_mae": float(np.mean(np.abs(resid_a))) if len(resid_a) else float("nan"),
    }


def print_eval(ev: Dict, prefix: str = "val") -> None:
    m = ev["metrics"]
    print(f"{prefix}_HR_MAE={m.mae:.3f} {prefix}_HR_RMSE={m.rmse:.3f} {prefix}_HR_r={m.pearson:.3f}")
    vm = ev.get("video_metrics")
    if vm is not None:
        print(f"{prefix}_VIDEO_ONLY_MAE={vm.mae:.3f} {prefix}_VIDEO_ONLY_RMSE={vm.rmse:.3f} {prefix}_VIDEO_ONLY_r={vm.pearson:.3f}")
    print(f"{prefix}_selection_acc={ev['selection_acc']:.3f} {prefix}_selection_soft_ce={ev['selection_soft_ce']:.3f} {prefix}_selection_expected_err={ev['selection_expected_err']:.3f}")
    print(f"{prefix}_quality_error_corr={ev['quality_error_corr']:.3f} quality_mean={np.mean(ev['quality']):.3f} gate_mean={ev['gate_mean']:.3f} residual_mae={ev['residual_mae']:.3f}")
    for cov, n, mae, med in ev["coverage"]:
        print(f"  top_quality_coverage={cov:5.1f}% N={n:4d} MAE={mae:7.3f} median={med:7.3f}")


def coverage_mae(ev: Dict, coverage: float) -> float:
    rows = ev.get("coverage", [])
    if not rows:
        return float("inf")
    row = min(rows, key=lambda r: abs(r[0] / 100.0 - float(coverage)))
    return float(row[2])


def save_checkpoint(path: Path, model, args, ds: UBFCVideoPriorDataset, tag: str, value: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "model_class": "VideoPriorHRNet",
        "checkpoint_tag": tag,
        "best_value": float(value),
        "channel_names": ds.channel_names,
        "candidate_names": ds.candidate_names,
        "priors": ds.prior_methods,
        "args": vars(args),
    }, path)


def main() -> None:
    ap = argparse.ArgumentParser(description="Train prior-guided video HR model on UBFC.")
    ap.add_argument("--ubfc-root", required=True)
    ap.add_argument("--video-cache-dir", default="cache_video_prior")
    ap.add_argument("--roi-cache-dir", default="cache_roi_oldpoints")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--fs", type=float, default=30.0)
    ap.add_argument("--clip-sec", type=float, default=10.0)
    ap.add_argument("--stride-sec", type=float, default=2.0)
    ap.add_argument("--clip-frames", type=int, default=96)
    ap.add_argument("--image-size", type=int, default=48)
    ap.add_argument("--roi", default="face")
    ap.add_argument("--priors", default=",".join(DEFAULT_PRIORS))
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-3)
    ap.add_argument("--dropout", type=float, default=0.20)
    ap.add_argument("--val-fraction", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--quality-tau-bpm", type=float, default=5.0)
    ap.add_argument("--selection-tau-bpm", type=float, default=3.0)
    ap.add_argument("--sigma-bpm", type=float, default=3.0)
    ap.add_argument("--dist-weight", type=float, default=0.25)
    ap.add_argument("--hr-reg-weight", type=float, default=1.0)
    ap.add_argument("--selection-weight", type=float, default=0.25)
    ap.add_argument("--soft-selection-weight", type=float, default=0.50)
    ap.add_argument("--quality-weight", type=float, default=0.10)
    ap.add_argument("--video-aux-dist-weight", type=float, default=0.15)
    ap.add_argument("--video-aux-reg-weight", type=float, default=0.25)
    ap.add_argument("--candidate-dropout-prob", type=float, default=0.25)
    ap.add_argument("--candidate-noise-std", type=float, default=1.5)
    ap.add_argument("--quality-save-coverage", type=float, default=0.60)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--out", default="checkpoints/video_prior_hrnet_ubfc.pt")
    args = ap.parse_args()

    seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    priors = [p.strip() for p in args.priors.split(",") if p.strip()]
    subjects = find_ubfc_subjects(args.ubfc_root)
    train_subj, val_subj = subject_split(subjects, val_fraction=args.val_fraction, seed=args.seed)
    print("Train subjects:", [s.subject_id for s in train_subj])
    print("Val subjects:", [s.subject_id for s in val_subj])

    train_ds = UBFCVideoPriorDataset(
        train_subj,
        fs=args.fs,
        clip_sec=args.clip_sec,
        stride_sec=args.stride_sec,
        clip_frames=args.clip_frames,
        image_size=args.image_size,
        video_cache_dir=args.video_cache_dir,
        roi_cache_dir=args.roi_cache_dir,
        roi=args.roi,
        prior_methods=priors,
        quality_tau_bpm=args.quality_tau_bpm,
        selection_tau_bpm=args.selection_tau_bpm,
    )
    val_ds = UBFCVideoPriorDataset(
        val_subj,
        fs=args.fs,
        clip_sec=args.clip_sec,
        stride_sec=args.stride_sec,
        clip_frames=args.clip_frames,
        image_size=args.image_size,
        video_cache_dir=args.video_cache_dir,
        roi_cache_dir=args.roi_cache_dir,
        roi=args.roi,
        prior_methods=priors,
        quality_tau_bpm=args.quality_tau_bpm,
        selection_tau_bpm=args.selection_tau_bpm,
    )
    print(f"Train samples: {len(train_ds)} | Val samples: {len(val_ds)}")
    print(f"Video tensor: T={args.clip_frames}, C=3, H=W={args.image_size}")
    print("Trace channels:", train_ds.channel_names)
    print("Candidate HR channels:", train_ds.candidate_names)

    print("\nValidation 1D trace/prior baselines on same subjects/windows:")
    rows = evaluate_channel_baselines(val_ds)
    for name, mae, rmse, me, sd, r in rows:
        print(f"{name:10s} MAE={mae:7.3f} RMSE={rmse:7.3f} ME={me:7.3f} SD={sd:7.3f} r={r:7.3f}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=(device.type == "cuda"))

    model = VideoPriorHRNet(
        trace_channels=len(train_ds.channel_names),
        num_candidates=len(train_ds.candidate_names),
        candidate_names=train_ds.candidate_names,
        frame_feature_dim=64,
        video_temporal_channels=64,
        trace_channels_hidden=48,
        num_video_blocks=3,
        num_trace_blocks=3,
        dropout=args.dropout,
    ).to(device)
    print(f"Model: VideoPriorHRNet | params={sum(p.numel() for p in model.parameters())/1e6:.3f}M")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, args.epochs))

    print("\nInitial evaluation:")
    init_ev = evaluate(model, val_loader, device)
    print_eval(init_ev)

    out_path = Path(args.out)
    quality_path = out_path.with_name(out_path.stem + f"_quality{int(round(100*args.quality_save_coverage))}" + out_path.suffix)
    best = init_ev["metrics"].mae
    best_q = coverage_mae(init_ev, args.quality_save_coverage)
    no_improve = 0
    best_epoch = 0
    save_checkpoint(out_path, model, args, train_ds, tag="epoch0_initial_overall", value=best)
    save_checkpoint(quality_path, model, args, train_ds, tag=f"epoch0_initial_quality_{int(100*args.quality_save_coverage)}", value=best_q)
    print(f"saved initial overall checkpoint: {out_path} MAE={best:.3f}")
    print(f"saved initial quality@{int(100*args.quality_save_coverage)}% checkpoint: {quality_path} MAE={best_q:.3f}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for batch in tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}"):
            video = batch["video"].to(device)
            x = batch["x"].to(device)
            y = batch["y_hr"].to(device)
            cand = batch["candidate_hrs"].to(device)
            valid = batch["candidate_valid"].to(device)
            qtar = batch["quality_target"].to(device)
            starget = batch["selection_target"].to(device)

            cand_train, valid_train = apply_candidate_dropout(
                cand,
                valid,
                drop_prob=args.candidate_dropout_prob,
                noise_std=args.candidate_noise_std,
            )

            out = model(video, x, candidate_hrs=cand_train, candidate_valid=valid_train, return_dict=True)
            dist = hr_distribution_ce(out["hr_logits"], y, model.hr_bins, sigma_bpm=args.sigma_bpm)
            reg = F.smooth_l1_loss(out["hr_bpm"], y)
            sel = soft_selection_ce(out["selection_logits"], starget, valid)
            qloss = F.binary_cross_entropy_with_logits(out["quality_logit"], qtar)
            video_dist = hr_distribution_ce(out["video_hr_logits"], y, model.hr_bins, sigma_bpm=args.sigma_bpm)
            video_reg = F.smooth_l1_loss(out["video_hr_bpm"], y)
            loss = (
                args.dist_weight * dist
                + args.hr_reg_weight * reg
                + (args.selection_weight + args.soft_selection_weight) * sel
                + args.quality_weight * qloss
                + args.video_aux_dist_weight * video_dist
                + args.video_aux_reg_weight * video_reg
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        scheduler.step()
        ev = evaluate(model, val_loader, device)
        m = ev["metrics"]
        print(f"epoch={epoch} train_loss={np.mean(losses):.4f} lr={scheduler.get_last_lr()[0]:.2e} val_HR_MAE={m.mae:.3f} val_HR_RMSE={m.rmse:.3f} val_HR_r={m.pearson:.3f}")
        print_eval(ev)
        q_mae = coverage_mae(ev, args.quality_save_coverage)
        if m.mae < best - 1e-6:
            best = m.mae
            best_epoch = epoch
            no_improve = 0
            save_checkpoint(out_path, model, args, train_ds, tag="best_overall", value=best)
            print(f"saved best overall: {out_path}")
        else:
            no_improve += 1
        if q_mae < best_q - 1e-6:
            best_q = q_mae
            save_checkpoint(quality_path, model, args, train_ds, tag=f"best_quality_{int(100*args.quality_save_coverage)}", value=best_q)
            print(f"saved best quality@{int(100*args.quality_save_coverage)}%: {quality_path} MAE={best_q:.3f}")
        if no_improve >= args.patience:
            print(f"Early stopping: no overall improvement for {args.patience} epochs. Best epoch={best_epoch}, best val_HR_MAE={best:.3f}; best quality@{int(100*args.quality_save_coverage)}% MAE={best_q:.3f}")
            break


if __name__ == "__main__":
    main()
