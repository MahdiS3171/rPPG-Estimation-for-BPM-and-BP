from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.signal import welch, find_peaks

from rppg_lab.classical import METHOD_FUNCS
from rppg_lab.datasets import load_ubfc_ground_truth
from rppg_lab.roi import extract_rgb_trace


EPS = 1e-12


@dataclass
class PeakCandidate:
    hr_bpm: float
    score: float
    rel_power: float
    raw_power: float
    method: str
    roi: str
    rank: int


@dataclass
class Cluster:
    center: float
    score: float
    candidates: List[PeakCandidate]

    @property
    def methods(self) -> set[str]:
        return {c.method for c in self.candidates}

    @property
    def rois(self) -> set[str]:
        return {c.roi for c in self.candidates}

    @property
    def sources(self) -> set[str]:
        return {f"{c.method}:{c.roi}" for c in self.candidates}


def parse_csv_arg(value: str) -> List[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def natural_subject_key(path: Path) -> Tuple[int, str]:
    name = path.name
    try:
        return int(name.replace("subject", "")), name
    except ValueError:
        return 10**9, name


def parabolic_refine_frequency(fb: np.ndarray, pb: np.ndarray, idx: int) -> float:
    if idx <= 0 or idx >= len(pb) - 1:
        return float(fb[idx])

    y0 = np.log(pb[idx - 1] + EPS)
    y1 = np.log(pb[idx] + EPS)
    y2 = np.log(pb[idx + 1] + EPS)

    denom = y0 - 2.0 * y1 + y2
    if abs(denom) < EPS:
        return float(fb[idx])

    delta = 0.5 * (y0 - y2) / denom
    delta = float(np.clip(delta, -1.0, 1.0))

    df = float(fb[1] - fb[0]) if len(fb) > 1 else 0.0
    return float(fb[idx] + delta * df)


def extract_peak_candidates(
    sig: np.ndarray,
    fs: float,
    method: str,
    roi: str,
    top_k: int = 5,
    fmin: float = 0.7,
    fmax: float = 3.5,
    window_sec: float = 30.0,
    zero_pad_factor: int = 4,
    method_weight: float = 1.0,
    roi_weight: float = 1.0,
) -> List[PeakCandidate]:
    x = np.asarray(sig, dtype=np.float64)
    x = np.nan_to_num(x - np.nanmean(x), nan=0.0)

    if x.size < int(fs * 3):
        return []

    nperseg = min(len(x), max(32, int(round(fs * window_sec))))
    nfft = int(2 ** np.ceil(np.log2(max(nperseg, zero_pad_factor * nperseg))))

    f, pxx = welch(x, fs=float(fs), nperseg=nperseg, nfft=nfft)
    band = (f >= fmin) & (f <= fmax)
    if not np.any(band):
        return []

    fb = f[band]
    pb = pxx[band]

    peaks, _ = find_peaks(pb)
    if len(peaks) == 0:
        peaks = np.array([int(np.argmax(pb))], dtype=int)

    order = peaks[np.argsort(pb[peaks])[::-1]]
    max_power = float(np.max(pb[order]) + EPS)

    candidates: List[PeakCandidate] = []
    for rank, idx in enumerate(order[:top_k]):
        refined_hz = parabolic_refine_frequency(fb, pb, int(idx))
        hr_bpm = 60.0 * refined_hz

        raw_power = float(pb[int(idx)])
        rel_power = raw_power / max_power

        # General signal-based score:
        # - top-ranked peaks are preferred
        # - strong relative PSD peaks are preferred
        # - method and ROI priors are mild weights, not UBFC-specific labels
        rank_score = 1.0 / (rank + 1.0)
        local_score = method_weight * roi_weight * (0.65 * rank_score + 0.35 * rel_power)

        candidates.append(
            PeakCandidate(
                hr_bpm=float(hr_bpm),
                score=float(local_score),
                rel_power=float(rel_power),
                raw_power=raw_power,
                method=method,
                roi=roi,
                rank=rank + 1,
            )
        )

    return candidates


def cluster_candidates(
    candidates: Sequence[PeakCandidate],
    cluster_width_bpm: float = 4.0,
) -> List[Cluster]:
    clusters: List[Cluster] = []

    # High-score candidates create clusters first.
    for cand in sorted(candidates, key=lambda c: c.score, reverse=True):
        assigned = False

        # Assign to the nearest compatible cluster.
        compatible = [
            (abs(cand.hr_bpm - cluster.center), cluster)
            for cluster in clusters
            if abs(cand.hr_bpm - cluster.center) <= cluster_width_bpm
        ]

        if compatible:
            _, cluster = min(compatible, key=lambda x: x[0])
            old_weight = sum(c.score for c in cluster.candidates) + EPS
            new_weight = old_weight + cand.score
            cluster.center = (cluster.center * old_weight + cand.hr_bpm * cand.score) / new_weight
            cluster.candidates.append(cand)
            assigned = True

        if not assigned:
            clusters.append(Cluster(center=cand.hr_bpm, score=0.0, candidates=[cand]))

    # Score clusters after assignment.
    for cluster in clusters:
        base = sum(c.score for c in cluster.candidates)
        n_methods = len(cluster.methods)
        n_rois = len(cluster.rois)
        n_sources = len(cluster.sources)

        # Agreement bonus: a peak supported by multiple methods/ROIs is more credible.
        agreement_bonus = 1.0 + 0.18 * (n_methods - 1) + 0.08 * (n_rois - 1)

        # Source-count bonus grows slowly so that one repeated family cannot dominate too much.
        source_bonus = np.sqrt(max(1, n_sources))

        cluster.score = float(base * agreement_bonus * source_bonus)

    return sorted(clusters, key=lambda c: c.score, reverse=True)


def choose_consensus_hr(
    candidates: Sequence[PeakCandidate],
    cluster_width_bpm: float = 4.0,
    min_hr: float = 42.0,
    max_hr: float = 180.0,
) -> Tuple[float, Cluster, List[Cluster]]:
    valid = [c for c in candidates if min_hr <= c.hr_bpm <= max_hr]
    if not valid:
        raise ValueError("No valid HR candidates found.")

    clusters = cluster_candidates(valid, cluster_width_bpm=cluster_width_bpm)
    if not clusters:
        raise ValueError("No clusters found.")

    best = clusters[0]
    return float(best.center), best, clusters


def summarize_errors(gt: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    err = pred - gt
    abs_err = np.abs(err)

    if len(gt) > 1 and np.std(gt) > EPS and np.std(pred) > EPS:
        r = float(np.corrcoef(gt, pred)[0, 1])
    else:
        r = float("nan")

    return {
        "N": int(len(gt)),
        "MAE": float(np.mean(abs_err)),
        "RMSE": float(np.sqrt(np.mean(err**2))),
        "ME": float(np.mean(err)),
        "SD": float(np.std(err, ddof=1)) if len(err) > 1 else float("nan"),
        "r": r,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ubfc-root", required=True)
    parser.add_argument("--fs", type=float, default=30.0)
    parser.add_argument("--cache-dir", default="cache_roi")
    parser.add_argument("--out", default="outputs/consensus_ubfc.csv")

    parser.add_argument("--methods", default="CHROM,PBV,GREEN")
    parser.add_argument("--rois", default="face,forehead,left_cheek,right_cheek")

    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--cluster-width", type=float, default=4.0)
    parser.add_argument("--fmin", type=float, default=0.7)
    parser.add_argument("--fmax", type=float, default=3.5)
    parser.add_argument("--window-sec", type=float, default=30.0)

    parser.add_argument(
        "--debug-subjects",
        default="",
        help="Comma-separated subject ids, e.g. subject8,subject15,subject16",
    )

    args = parser.parse_args()

    methods = parse_csv_arg(args.methods)
    rois = parse_csv_arg(args.rois)
    debug_subjects = set(parse_csv_arg(args.debug_subjects))

    # Mild, literature/robustness-inspired priors.
    # Keep these conservative; do not tune them for UBFC outliers.
    method_weights = {
        "CHROM": 1.00,
        "CHROM_WIN": 1.00,
        "PBV": 0.90,
        "GREEN": 0.75,
        "POS": 0.55,
        "POS_WIN": 0.55,
        "OMIT": 0.55,
        "LGI": 0.55,
    }

    roi_weights = {
        "face": 1.00,
        "forehead": 0.95,
        "left_cheek": 0.95,
        "right_cheek": 0.95,
    }

    unknown_methods = [m for m in methods if m not in METHOD_FUNCS]
    if unknown_methods:
        raise ValueError(f"Unknown methods: {unknown_methods}. Available: {sorted(METHOD_FUNCS)}")

    root = Path(args.ubfc_root)
    subject_dirs = sorted(root.glob("subject*"), key=natural_subject_key)

    rows = []

    for subject_dir in subject_dirs:
        video_path = subject_dir / "vid.avi"
        gt_path = subject_dir / "ground_truth.txt"
        if not video_path.exists() or not gt_path.exists():
            continue

        gt_t, gt_hr, ppg = load_ubfc_ground_truth(gt_path)
        gt_value = float(np.nanmean(gt_hr))

        candidates: List[PeakCandidate] = []
        n_frames = None

        rgb_cache: Dict[str, Tuple[np.ndarray, float]] = {}

        for roi in rois:
            rgb, fs, t, q = extract_rgb_trace(
                str(video_path),
                roi=roi,
                cache_dir=args.cache_dir,
                fs_target=args.fs,
            )
            rgb_cache[roi] = (rgb, fs)
            n_frames = len(rgb)

        for method in methods:
            for roi in rois:
                rgb, fs = rgb_cache[roi]
                sig = METHOD_FUNCS[method](rgb, fs)

                candidates.extend(
                    extract_peak_candidates(
                        sig=sig,
                        fs=fs,
                        method=method,
                        roi=roi,
                        top_k=args.top_k,
                        fmin=args.fmin,
                        fmax=args.fmax,
                        window_sec=args.window_sec,
                        method_weight=method_weights.get(method, 0.50),
                        roi_weight=roi_weights.get(roi, 0.90),
                    )
                )

        pred_hr, best_cluster, clusters = choose_consensus_hr(
            candidates,
            cluster_width_bpm=args.cluster_width,
        )

        abs_err = abs(pred_hr - gt_value)

        best_sources = sorted(best_cluster.sources)
        best_methods = sorted(best_cluster.methods)
        best_rois = sorted(best_cluster.rois)

        rows.append(
            {
                "subject_id": subject_dir.name,
                "gt_hr": gt_value,
                "pred_hr": pred_hr,
                "abs_err": abs_err,
                "n_frames": n_frames,
                "n_candidates": len(candidates),
                "n_clusters": len(clusters),
                "best_cluster_score": best_cluster.score,
                "best_cluster_sources": ";".join(best_sources),
                "best_cluster_methods": ";".join(best_methods),
                "best_cluster_rois": ";".join(best_rois),
            }
        )

        print(
            f"{subject_dir.name}: gt={gt_value:.2f} pred={pred_hr:.2f} "
            f"err={abs_err:.2f} sources={','.join(best_sources[:5])}"
        )

        if subject_dir.name in debug_subjects:
            print("  Top clusters:")
            for i, cluster in enumerate(clusters[:8], start=1):
                cand_preview = sorted(cluster.candidates, key=lambda c: c.score, reverse=True)[:6]
                cand_str = " | ".join(
                    f"{c.method}:{c.roi}:{c.hr_bpm:.1f}(r{c.rank})"
                    for c in cand_preview
                )
                print(
                    f"  {i:02d}) center={cluster.center:.2f} score={cluster.score:.3f} "
                    f"methods={sorted(cluster.methods)} rois={sorted(cluster.rois)} :: {cand_str}"
                )

    df = pd.DataFrame(rows)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    summary = summarize_errors(df["gt_hr"].to_numpy(), df["pred_hr"].to_numpy())

    print("\nSummary")
    print(
        f"N={summary['N']} "
        f"MAE={summary['MAE']:.3f} "
        f"RMSE={summary['RMSE']:.3f} "
        f"ME={summary['ME']:.3f} "
        f"SD={summary['SD']:.3f} "
        f"r={summary['r']:.3f}"
    )
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()