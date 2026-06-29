#!/usr/bin/env python
"""Train a feature-level BP baseline on future collected sessions.

This is not meant for face-only BP claims. Use only after you have features.json
and labels.json with cuff references and subject-wise splits.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

sys.path.append(str(Path(__file__).resolve().parents[1]))

from rppg_lab.datasets import SessionBPDataset
from rppg_lab.models import BPFeatureMLP
from rppg_lab.metrics import aami_summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", default="checkpoints/bp_feature_mlp.pt")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    ds = SessionBPDataset(args.data_dir)
    n_val = max(1, int(0.2 * len(ds)))
    n_train = len(ds) - n_val
    train_ds, val_ds = random_split(ds, [n_train, n_val], generator=torch.Generator().manual_seed(args.seed))
    tr = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    va = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BPFeatureMLP(in_features=len(ds.feature_keys)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = torch.nn.MSELoss()
    best = np.inf
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for ep in range(1, args.epochs + 1):
        model.train()
        for x, y in tr:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = loss_fn(pred, y)
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        Y, P = [], []
        with torch.no_grad():
            for x, y in va:
                pred = model(x.to(device)).cpu().numpy()
                Y.append(y.numpy()); P.append(pred)
        Y = np.vstack(Y); P = np.vstack(P)
        mae = float(np.mean(np.abs(P - Y)))
        if ep % 10 == 0 or ep == 1:
            print(f"epoch={ep} val_mae_avg={mae:.2f} SBP={aami_summary(Y[:,0],P[:,0])} DBP={aami_summary(Y[:,1],P[:,1])}")
        if mae < best:
            best = mae
            torch.save({"model_state": model.state_dict(), "feature_keys": ds.feature_keys, "best_val_mae_avg": best}, out_path)
    print("saved:", out_path)


if __name__ == "__main__":
    main()
