"""Non-overwriting research artifacts with input/config/software provenance."""
from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
import csv
import hashlib
import importlib.metadata
import json
import subprocess
import numpy as np
from .types import DualROIResult
from .bp import FEATURE_KEYS


def json_safe(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k,v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return value


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(json_safe(value), indent=2, allow_nan=False), encoding="utf-8")


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


def provenance(input_path: str | Path | None = None) -> dict:
    root = Path(__file__).resolve().parents[1]
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True, check=True).stdout.strip())
    except (OSError, subprocess.CalledProcessError):
        commit,dirty = None,None
    versions = {}
    for package in ("numpy", "scipy", "scikit-learn", "opencv-python", "mediapipe", "rppg-lab"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return {"created_at_utc": datetime.now(timezone.utc).isoformat(), "git_commit": commit,
            "git_dirty": dirty, "versions": versions, "input_path": str(Path(input_path).resolve()) if input_path else None,
            "input_sha256": file_sha256(input_path) if input_path else None,
            "source_sha256": {str(p.relative_to(root)):file_sha256(p) for folder in ("rppg_lab","scripts") for p in sorted((root/folder).glob("*.py"))}}


def reserve_run(output: str | Path) -> Path:
    """Refuse existing directories, raw videos, or previous runs."""
    path = Path(output)
    path.mkdir(parents=True, exist_ok=False)
    return path


def save_result(result: DualROIResult, output: Path, run_provenance: dict) -> None:
    from .config import PipelineConfig
    from .processing import resample_trace, extract_signal
    config = PipelineConfig(**result.config)
    arrays = {"timestamps_original": result.shared.original_timestamps,
              "timestamps_uniform": result.shared.timestamps,
              "backend_timestamps": np.asarray(result.video.backend_timestamps)}
    uniform = {}
    for site, region in (("face",result.face),("hand",result.hand)):
        arrays[site+"_rgb"] = region.rgb.values
        arrays[site+"_valid"] = region.rgb.validity_mask
        arrays[site+"_rppg"] = region.rppg.values
        uniform[site], imputed = resample_trace(region.rgb,result.shared.timestamps,config.max_gap_sec)
        arrays[site+"_rgb_uniform"] = uniform[site]
        arrays[site+"_interpolated"] = imputed
        arrays[site+"_bbox"] = np.asarray([obs.bbox if obs.bbox else (np.nan,)*4 for obs in region.observations])
        # NaN-filled landmarks preserve per-frame observation identity.
        count = max((len(obs.landmarks) for obs in region.observations if obs.landmarks is not None), default=0)
        landmarks = np.full((len(region.observations),count,2),np.nan)
        for i,obs in enumerate(region.observations):
            if obs.landmarks is not None:
                landmarks[i,:len(obs.landmarks)] = obs.landmarks
        arrays[site+"_landmarks"] = landmarks
        for name,values in region.rgb.quality.items():
            arrays[site+"_"+name] = values
    arrays["face_roi_names"] = np.asarray(list(result.face_rois), dtype=str)
    for name, region in result.face_rois.items():
        prefix = "face_roi_" + name + "_"
        arrays[prefix + "rgb"] = region.rgb.values
        arrays[prefix + "valid"] = region.rgb.validity_mask
        arrays[prefix + "rppg"] = region.rppg.values
        arrays[prefix + "rgb_uniform"], arrays[prefix + "interpolated"] = resample_trace(
            region.rgb, result.shared.timestamps, config.max_gap_sec)
        arrays[prefix + "reason"] = np.asarray([obs.reason or "" for obs in region.observations], dtype=str)
        for metric, values in region.rgb.quality.items():
            arrays[prefix + metric] = values
    joint = np.isfinite(uniform["face"]).all(axis=1)&np.isfinite(uniform["hand"]).all(axis=1)
    paired = {site: extract_signal(uniform[site],result.shared.timestamps,site,config,allowed=joint) for site in uniform}
    arrays.update({site+"_rppg_paired":signal.values for site,signal in paired.items()})
    np.savez_compressed(output/"signals.npz", **arrays)
    write_json(output/"config.json", result.config)
    write_json(output/"summary.json", {"hr": result.hr, "delay": result.delay,
        "quality": {"face": result.face.quality,"hand": result.hand.quality},
        "preprocessing": {"face": result.face.rppg.preprocessing,"hand": result.hand.rppg.preprocessing},
        "face_rois": {name: {"quality": region.quality, "preprocessing": region.rppg.preprocessing}
                      for name, region in result.face_rois.items()},
        "paired_preprocessing": {site:signal.preprocessing for site,signal in paired.items()},
        "video": asdict(result.video), "exclusions": result.exclusions,
        "observations": {site: [{"reason":obs.reason,"identity":obs.identity,"valid":obs.valid,
            "confidence_kind":obs.confidence_kind} for obs in region.observations] for site,region in (("face",result.face),("hand",result.hand))}})
    write_json(output/"provenance.json", run_provenance)
    write_json(output/"feature_windows.json", result.bp_features)
    columns = ["window_index","start_sec","end_sec","status","reasons",*FEATURE_KEYS]
    with (output/"features.csv").open("w",encoding="utf-8",newline="") as f:
        writer = csv.DictWriter(f,fieldnames=columns)
        writer.writeheader()
        for row in result.bp_features:
            row = json_safe(row)
            writer.writerow({**row,"reasons": "|".join(row["reasons"])})


def overlay_writer(output: Path):
    import cv2
    output.mkdir(parents=True,exist_ok=True)
    def callback(index,timestamp,frame,masks,observations):
        rgb = frame.copy()
        for site,color in (("face",np.array([0,255,80])),("hand",np.array([255,180,0]))):
            mask = masks[site]
            rgb[mask] = (0.6*rgb[mask]+0.4*color).astype(np.uint8)
            obs = observations[site]
            if obs.bbox:
                x1,y1,x2,y2 = obs.bbox
                cv2.rectangle(rgb,(x1,y1),(x2,y2),color.tolist(),1)
        cv2.putText(rgb,f"frame={index} t={timestamp:.4f}s",(8,20),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)
        if not cv2.imwrite(str(output/f"frame_{index:07d}.png"),cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)):
            raise OSError("Could not save ROI overlay")
    return callback


def plot_result(result: DualROIResult, output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axes = plt.subplots(4,1,figsize=(12,11),constrained_layout=True)
    for site,region in (("face",result.face),("hand",result.hand)):
        for i,channel in enumerate("RGB"):
            axes[0].plot(region.rgb.timestamps,region.rgb.values[:,i],label=f"{site} {channel}",alpha=0.7)
        axes[1].plot(region.rppg.timestamps,region.rppg.values,label=site)
        axes[2].plot(region.rgb.timestamps,region.rgb.validity_mask.astype(int),label=site,alpha=0.7)
    rows = result.bp_features
    accepted = [r for r in rows if r["status"] == "accepted"]
    axes[3].plot([r["start_sec"] for r in accepted],[r["delay_xcorr_sec"] for r in accepted],"o",label="accepted optical delay")
    for ax,title in zip(axes,("Original RGB traces","Shared-clock rPPG (gaps retained)","Original ROI validity","Positive delay: hand lags face")):
        ax.set_title(title)
        ax.set_xlabel("Video time (s)")
        ax.legend(loc="upper right")
    axes[3].set_ylabel("Delay (s)")
    fig.savefig(output/"signals.png",dpi=140)
    plt.close(fig)
    if accepted:
        from scipy.signal import correlate, correlation_lags
        from .processing import resample_trace, extract_signal
        from .config import PipelineConfig
        config = PipelineConfig(**result.config)
        grid = result.shared.timestamps
        rgb = {site: resample_trace(region.rgb,grid,config.max_gap_sec)[0] for site,region in (("face",result.face),("hand",result.hand))}
        joint = np.isfinite(rgb["face"]).all(axis=1)&np.isfinite(rgb["hand"]).all(axis=1)
        paired = {site: extract_signal(rgb[site],grid,site,config,allowed=joint) for site in rgb}
        row = accepted[0]
        keep = (grid >= row["start_sec"]-1e-8)&(grid < row["end_sec"]-1e-8)
        x,y = paired["face"].values[keep],paired["hand"].values[keep]
        t = grid[keep]
        x,y = (x-x.mean())/x.std(),(y-y.mean())/y.std()
        fig,axes = plt.subplots(2,1,figsize=(10,6),constrained_layout=True)
        axes[0].plot(t,x,label="face")
        axes[0].plot(t,y,label="hand original")
        axes[0].plot(t-row["delay_xcorr_sec"],y,label="hand shifted for display only",alpha=0.6)
        axes[0].set_xlabel("Video time (s)")
        axes[0].legend()
        c = correlate(y,x,method="fft")/np.sqrt(np.sum(x*x)*np.sum(y*y))
        lags = correlation_lags(len(y),len(x))/config.sample_rate
        keep = abs(lags) <= config.max_delay_sec
        axes[1].plot(lags[keep],c[keep])
        axes[1].axvline(row["delay_xcorr_sec"],color="red")
        axes[1].set_xlabel("Hand minus face lag (s)")
        axes[1].set_ylabel("Normalized cross-correlation")
        fig.savefig(output/"timing_debug.png",dpi=140)
        plt.close(fig)
