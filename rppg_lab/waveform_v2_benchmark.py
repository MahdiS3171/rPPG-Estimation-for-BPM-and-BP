"""Waveform v2 *interface* for matched baseline evaluation, not a new evaluator.

Preserves failures and full attempted denominator. No inference, signal filtering,
or retrospective choice of the strongest comparator happens in this module.
"""
from __future__ import annotations

from collections import defaultdict

from .waveform_v2_registry import RESERVED_OWN_SUBJECTS
from typing import Any, Mapping, Sequence
import math

PROTOCOL_VERSION = "waveform_v2_matched_benchmark_v0.1"
MANDATORY_METHODS = ("GREEN", "CHROM", "POS_WIN", "FrozenWaveformV1", "CandidateV2")


def benchmark_protocol() -> dict[str, Any]:
    return dict(
        protocol_version=PROTOCOL_VERSION, status="DRAFT_NOT_FROZEN",
        mandatory_methods=list(MANDATORY_METHODS),
        optional_methods=["POS", "CHROM_WIN", "PBV", "OMIT"],
        primary_metric="subject_balanced_HR_MAE_bpm",
        matching_keys=["dataset_id", "participant_id", "recording_id", "start_ns", "end_ns"],
        required_shared_contract=["source_sha256", "reference_sha256", "input_contract_id",
                                  "reference_hr_bpm"],
        scores_allowed_only_when="Reference source, permission and synchrony accepted",
        provenance=["source video timebase", "ROI/mask", "filter", "model rate", "reference clock"],
        coverage_policy="Every method is compared on the same intended interval denominator."
                        " Failures and missing predictions remain explicit.",
        future_timing_policy="Absolute timing metrics require independently verified alignment."
                             " Never use reference-selected shifts as evidence of native pulse arrival.",
        excluded_from_development=["S007", "S008", "S009", "S010", "S011"],
        historically_exposed=["UBFC_v1_locked_test", "rPPG-10_external", "pilot_S001-S006"],
    )


def _identity(row: Mapping[str, Any]) -> tuple:
    return tuple(row[k] for k in
                 ("dataset_id", "participant_id", "recording_id", "start_ns", "end_ns"))


def matched_method_index(rows: Sequence[Mapping[str, Any]],
                         mandatory_methods: Sequence[str] = MANDATORY_METHODS) -> list[dict[str, Any]]:
    """Build paired method groups, retaining missing and invalid rows.

    Requires a registry of intended intervals: one 'intent' row per interval and
    zero/one prediction rows per method. Shared provenance is compared exactly.
    Output does not compute physiological performance or endorse any dataset.
    """
    if not mandatory_methods or len(set(mandatory_methods)) != len(mandatory_methods):
        raise ValueError("Mandatory methods must be unique and nonempty")
    by_key: dict[tuple, dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in rows:
        key = _identity(row)
        if not isinstance(key[0], str) or not key[0].strip():
            raise ValueError("Missing dataset ID")
        participant = str(key[1])
        if participant.split(":")[-1] in RESERVED_OWN_SUBJECTS:
            raise ValueError("Restricted subject must not enter benchmark")
        start, end = key[-2:]
        if any(isinstance(v, bool) or not isinstance(v, int) for v in (start, end)) or end <= start:
            raise ValueError("Intervals require positive integer nanosecond bounds")
        method = row["method"]
        if method not in (*mandatory_methods, "__intent__"):
            raise ValueError(f"Unrecognized method for mandatory matched benchmark: {method}")
        if method in by_key[key]:
            raise ValueError(f"Duplicate benchmark record: {key} {method}")
        by_key[key][method] = row
    output = []
    for key, entries in sorted(by_key.items()):
        intent = entries.get("__intent__")
        if intent is None:
            raise ValueError(f"Prediction without an independently registered intended interval: {key}")
        shared = ["source_sha256", "reference_sha256", "input_contract_id", "reference_hr_bpm"]
        if not all(k in intent for k in shared):
            raise ValueError("Intended interval missing shared provenance")
        for method, row in entries.items():
            if method == "__intent__":
                continue
            for k in shared:
                if k not in row or row[k] != intent[k]:
                    raise ValueError(f"Mismatched or missing shared {k} for {key}: {method}")
        methods = {}
        for method in mandatory_methods:
            pred = entries.get(method)
            if pred is None:
                methods[method] = {"status": "missing_prediction", "hr_bpm": None}
            else:
                status = pred.get("status")
                if status not in ("valid", "invalid"):
                    raise ValueError("Each prediction must explicitly declare valid/invalid")
                bpm = pred.get("hr_bpm")
                if status == "valid" and (not isinstance(bpm, (int, float)) or
                                          isinstance(bpm, bool) or not math.isfinite(bpm) or bpm <= 0):
                    raise ValueError("A valid prediction requires a finite positive HR")
                if status == "invalid" and not pred.get("failure_reason"):
                    raise ValueError("Invalid prediction must provide failure_reason")
                methods[method] = {"status": status, "hr_bpm": bpm if status == "valid" else None,
                                   "failure_reason": pred.get("failure_reason")}
        output.append({"window": key, "reference_hr_bpm": intent["reference_hr_bpm"],
                       "methods": methods, "paired_all_valid": all(
                           methods[method]["status"] == "valid" for method in mandatory_methods)})
    return output


def coverage_only(index: Sequence[Mapping[str, Any]],
                  mandatory_methods: Sequence[str] = MANDATORY_METHODS) -> dict[str, Any]:
    """Report explicit attempted-denominator coverage without computing performance."""
    denominator = len(index)
    counts = {m: 0 for m in mandatory_methods}
    failure_reasons: dict[str, dict[str, int]] = {m: {} for m in mandatory_methods}
    for matched in index:
        for m in mandatory_methods:
            pred = matched["methods"][m]
            if pred["status"] == "valid":
                counts[m] += 1
            else:
                reason = pred.get("failure_reason") or pred["status"]
                failure_reasons[m][reason] = failure_reasons[m].get(reason, 0) + 1
    return dict(intended_windows=denominator, matched_all_valid=sum(
        bool(r["paired_all_valid"]) for r in index),
        valid_by_method=counts, failure_reasons=failure_reasons)
