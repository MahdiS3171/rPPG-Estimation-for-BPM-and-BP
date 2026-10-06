"""Interpretable train-only BP baselines with a common fit/predict/save API."""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import pickle
import numpy as np
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from .metrics import regression_metrics, bland_altman
from .bp import FEATURE_KEYS
from .splits import assert_subject_disjoint, validate_sample_partitions
from .study import BPReference, ReferenceAssociation


class BPModel:
    """Columns in every target/prediction are [SBP, DBP], in mmHg."""
    def __init__(self, kind: str = "ridge", seed: int = 42):
        estimators = {"mean": DummyRegressor(strategy="mean"), "linear": LinearRegression(),
                      "ridge": Ridge(alpha=10.0),
                      "random_forest": RandomForestRegressor(n_estimators=200,min_samples_leaf=3,random_state=seed,n_jobs=-1)}
        if kind not in estimators:
            raise ValueError("Unknown BP baseline")
        self.kind, self.seed = kind, seed
        self.pipeline = Pipeline([("imputer",SimpleImputer(strategy="median",keep_empty_features=True)),
                                  ("scaler",StandardScaler()),("model",estimators[kind])])
        self.feature_keys: list[str] = []
        self.train_subjects: list[str] = []

    def fit(self, x_train: np.ndarray, y_train: np.ndarray, *, feature_keys: list[str], subject_ids: list[str]) -> "BPModel":
        x,y = np.asarray(x_train,dtype=float),np.asarray(y_train,dtype=float)
        if x.ndim != 2 or y.shape != (len(x),2) or x.shape[1] != len(feature_keys) or len(x) != len(subject_ids) or not len(x):
            raise ValueError("Invalid training shapes/schema/subject IDs")
        if not np.isfinite(y).all() or np.isinf(x).any() or not set(feature_keys) <= set(FEATURE_KEYS):
            raise ValueError("Invalid targets or non-signal/label feature columns")
        self.feature_keys, self.train_subjects = list(feature_keys),sorted(set(subject_ids))
        self.pipeline.fit(x,y)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x,dtype=float)
        if x.ndim != 2 or x.shape[1] != len(self.feature_keys) or np.isinf(x).any():
            raise ValueError("Prediction features do not match training schema")
        return self.pipeline.predict(x)

    def save(self, path: str | Path) -> None:
        # Exclusive create: never overwrite a selected model accidentally.
        with Path(path).open("xb") as f:
            pickle.dump(self,f)

    @classmethod
    def load(cls, path: str | Path) -> "BPModel":
        """Load only trusted local artifacts (pickle is executable)."""
        with Path(path).open("rb") as f:
            model = pickle.load(f)
        if not isinstance(model,cls):
            raise TypeError("Artifact is not a BPModel")
        return model


def bp_metrics(y: np.ndarray, predictions: np.ndarray, subjects: list[str], strata: dict | None = None) -> dict:
    """Separate outputs; Bland–Altman points and descriptive subgroup errors."""
    y,p = np.asarray(y,dtype=float),np.asarray(predictions,dtype=float)
    if y.shape != p.shape or y.ndim != 2 or y.shape[1] != 2 or len(subjects) != len(y):
        raise ValueError("BP metrics require matching (N,2) arrays and N subjects")
    out = {}
    for k,name in enumerate(("sbp","dbp")):
        out[name] = {**asdict(regression_metrics(y[:,k],p[:,k])), "bland_altman": bland_altman(y[:,k],p[:,k]),
                     "bland_altman_mean_mmHg": ((y[:,k]+p[:,k])/2).tolist(),
                     "bland_altman_difference_mmHg": (p[:,k]-y[:,k]).tolist(),
                     "error_vs_reference": {"reference_mmHg":y[:,k].tolist(),"error_mmHg":(p[:,k]-y[:,k]).tolist()},
                     "per_subject": {}}
        for sid in sorted(set(subjects)):
            idx = np.asarray(subjects)==sid
            out[name]["per_subject"][sid] = asdict(regression_metrics(y[idx,k],p[idx,k]))
        out[name]["strata"] = {}
        for label,values in (strata or {}).items():
            if len(values) != len(y):
                raise ValueError("Stratum labels must match samples")
            values = np.asarray(values,dtype=str)
            out[name]["strata"][label] = {group: asdict(regression_metrics(y[values==group,k],p[values==group,k])) for group in sorted(set(values))}
    out["note"] = "Pilot agreement only; overlapping windows are not independent observations; no standards compliance claim"
    return out


def train_baselines(rows: list[dict], partitions: dict[str,str], seed: int = 42,
                    kinds=("mean","linear","ridge","random_forest")) -> tuple[BPModel,dict]:
    """Fit on training subjects, select on validation, evaluate locked test once.

    Fixed allowed schema prevents feature discovery on test data. Columns that
    are entirely missing in training are dropped using training only.
    """
    if len(set(kinds)) != len(kinds) or not kinds:
        raise ValueError("Choose at least one unique baseline")
    assert_subject_disjoint(*[[sid for sid,part in partitions.items() if part == name] for name in ("train","validation","test")])
    validate_sample_partitions(rows,partitions)
    eligible = []
    excluded = []
    for i,row in enumerate(rows):
        try:
            y = np.asarray([float(row["sbp_mmHg"]),float(row["dbp_mmHg"])])
        except (ValueError,KeyError,TypeError):
            y = np.full(2,np.nan)
        reference_fields = ("reference_start_time","reference_end_time","reference_device","association_notes")
        reference_valid = all(row.get(k) for k in reference_fields)
        if reference_valid:
            try:
                BPReference(y[0],y[1],int(row.get("reference_index",0)),row["reference_start_time"],
                            row["reference_end_time"],row.get("reference_relation",""),row["reference_device"])
                ReferenceAssociation(int(row.get("reference_index",0)),float(row["start_sec"]),float(row["end_sec"]),
                                     row.get("association_method",""),row["association_notes"])
            except (ValueError,TypeError):
                reference_valid = False
        if row.get("status") != "accepted" or not np.isfinite(y).all() or not 0 < y[1] < y[0] or not reference_valid:
            excluded.append({"row_index":i,"subject_id":row["subject_id"],"reason":row.get("reasons") or "invalid_bp_reference_or_quality"})
            continue
        eligible.append(row)
    indices = {name:[i for i,row in enumerate(eligible) if partitions[row["subject_id"]] == name] for name in ("train","validation","test")}
    if any(not indices[name] for name in indices):
        raise ValueError("Each subject partition needs at least one accepted cuff-associated sample")
    def number(row,key):
        value = row.get(key)
        return float(value) if value not in (None,"") else np.nan
    raw = np.asarray([[number(row,key) for key in FEATURE_KEYS] for row in eligible])
    if np.isinf(raw).any():
        raise ValueError("Infinite feature values")
    train = indices["train"]
    keep = np.isfinite(raw[train]).any(axis=0)
    keys = [key for key,allowed in zip(FEATURE_KEYS,keep) if allowed]
    if not keys:
        raise ValueError("No observed training features")
    x = raw[:,keep]
    y = np.asarray([[float(r["sbp_mmHg"]),float(r["dbp_mmHg"])] for r in eligible])
    subjects = [r["subject_id"] for r in eligible]
    validation = indices["validation"]
    models, reports, scores = {}, {}, {}
    for kind in kinds:
        model = BPModel(kind,seed).fit(x[train],y[train],feature_keys=keys,subject_ids=[subjects[i] for i in train])
        pred = model.predict(x[validation])
        reports[kind] = bp_metrics(y[validation],pred,[subjects[i] for i in validation])
        scores[kind] = float(np.mean(np.abs(y[validation]-pred)))
        models[kind] = model
    best = min(scores,key=scores.get)
    test = indices["test"]
    selected = models[best]
    pred = selected.predict(x[test])
    report = {"partitions":partitions,"feature_keys":keys,"dropped_train_empty_features":[k for k,v in zip(FEATURE_KEYS,keep) if not v],
              "selected_model":best,"validation":reports,"validation_mean_mae_mmHg":scores,
              "locked_test":bp_metrics(y[test],pred,[subjects[i] for i in test]),
              "excluded_samples":excluded,"partition_sample_counts":{k:len(v) for k,v in indices.items()},
              "test_predictions":[{"subject_id":subjects[i],"recording_id":eligible[i]["recording_id"],"start_sec":eligible[i]["start_sec"],
                                   "sbp_true_mmHg":y[i,0],"dbp_true_mmHg":y[i,1],"sbp_pred_mmHg":p[0],"dbp_pred_mmHg":p[1]} for i,p in zip(test,pred)]}
    if "mean" in models:
        report["locked_test_mean_baseline"] = bp_metrics(y[test],models["mean"].predict(x[test]),[subjects[i] for i in test])
    report["locked_test_error_vs_quality"] = {
        "face_snr_db":[number(eligible[i],"face_snr_db") for i in test],
        "hand_snr_db":[number(eligible[i],"hand_snr_db") for i in test],
        "sbp_error_mmHg":(pred[:,0]-y[test,0]).tolist(), "dbp_error_mmHg":(pred[:,1]-y[test,1]).tolist()}
    return selected,report
