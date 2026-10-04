#!/usr/bin/env python
"""Create a machine-readable audit of stored checkpoints and result CSV files.

The script does not rerun training. It summarizes the artifacts already present
in the project so report values can be checked against a single manifest.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


def to_jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def regression_summary(df: pd.DataFrame) -> dict[str, float | int]:
    if not {"gt_hr", "pred_hr"}.issubset(df.columns):
        return {}
    y = pd.to_numeric(df["gt_hr"], errors="coerce").to_numpy(float)
    p = pd.to_numeric(df["pred_hr"], errors="coerce").to_numpy(float)
    keep = np.isfinite(y) & np.isfinite(p)
    y, p = y[keep], p[keep]
    if y.size == 0:
        return {"n": 0}
    err = p - y
    corr = float(np.corrcoef(y, p)[0, 1]) if y.size > 1 and np.std(y) > 0 and np.std(p) > 0 else float("nan")
    return {
        "n": int(y.size),
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "me": float(np.mean(err)),
        "sd": float(np.std(err, ddof=1)) if y.size > 1 else 0.0,
        "pearson": corr,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", default=str(Path(__file__).resolve().parents[1]))
    ap.add_argument("--out", default="outputs/results_audit.json")
    args = ap.parse_args()

    root = Path(args.project_root).resolve()
    manifest: dict[str, Any] = {
        "note": (
            "Artifact audit only: values are read from existing checkpoints and CSV files; "
            "training and dataset extraction are not rerun."
        ),
        "checkpoints": {},
        "csv_results": {},
    }

    for path in sorted((root / "checkpoints").glob("*.pt")):
        try:
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
            if not isinstance(ckpt, dict):
                manifest["checkpoints"][path.name] = {"type": type(ckpt).__name__}
                continue
            keep_keys = [
                "model_class", "model_version", "checkpoint_tag", "best_val_hr_mae",
                "best_val_mean_mae", "best_value", "best_mae", "epoch", "quality_coverage",
                "quality_mae", "in_channels", "channel_names", "candidate_names", "priors",
                "train_subjects", "val_subjects", "test_subjects", "args",
            ]
            manifest["checkpoints"][path.name] = {
                k: to_jsonable(ckpt[k]) for k in keep_keys if k in ckpt
            }
        except Exception as exc:  # audit should continue and record the failure
            manifest["checkpoints"][path.name] = {"load_error": str(exc)}

    for path in sorted((root / "outputs").glob("*.csv")):
        try:
            df = pd.read_csv(path)
            manifest["csv_results"][path.name] = {
                "columns": list(df.columns),
                "summary": regression_summary(df),
            }
        except Exception as exc:
            manifest["csv_results"][path.name] = {"load_error": str(exc)}

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = root / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(to_jsonable(manifest), ensure_ascii=False, indent=2), encoding="utf-8")
    print("Wrote", out_path)


if __name__ == "__main__":
    main()
