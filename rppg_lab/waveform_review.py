"""Artifact-only candidate review. This module never loads a dataset or runs inference."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import json
import math

import numpy as np

from .artifacts import file_sha256, json_safe, write_json
from .waveform_training import (SELECTION_TIE_TOLERANCE, WaveformCheckpointSelector,
                                checkpoint_dataset_config, split_manifest_hash)

REVIEW_VERSION = "waveform_v1_part4_review_v1"
MAIN_METRICS = ("wave_corr_aligned", "d1_corr_same_lag", "morphology_score", "aligned_nrmse",
                "spectral_distance", "spectral_similarity", "hr_mae")
BOOTSTRAP_METRICS = ("wave_corr_aligned", "d1_corr_same_lag", "morphology_score", "hr_mae")
FUTURE_TEST_POLICY = ("The UBFC locked test is consumed. Future architecture or loss changes require "
                      "an external dataset, a new held-out dataset, or a newly designed validation protocol; "
                      "do not repeatedly optimize against this test.")


def metric_summary(metrics: dict) -> dict:
    result = {k: metrics.get(k) for k in MAIN_METRICS if k != "morphology_score"}
    a, b = result["wave_corr_aligned"], result["d1_corr_same_lag"]
    result["morphology_score"] = .5 * (a + b) if finite(a) and finite(b) else None
    return result


def finite(value) -> bool:
    return isinstance(value, (int, float, np.number)) and math.isfinite(value)


def subject_bootstrap(per_subject: dict, resamples: int = 2000, seed: int = 42) -> dict:
    """Equal participant weighting; each draw resamples entire subject summaries."""
    if type(resamples) is not int or resamples < 1 or not per_subject:
        raise ValueError("Require subjects and positive integer bootstrap resamples")
    ids = sorted(per_subject)
    summaries = [metric_summary(per_subject[sid]) for sid in ids]
    indices = np.random.default_rng(seed).integers(0, len(ids), size=(resamples, len(ids)))
    intervals = {}
    for key in BOOTSTRAP_METRICS:
        values = np.array([s[key] if finite(s[key]) else np.nan for s in summaries])
        if not np.isfinite(values).all():
            intervals[key] = None  # Never silently drop a failed participant.
            continue
        draws = values[indices].mean(axis=1)
        lo, hi = np.percentile(draws, [2.5, 97.5])
        intervals[key] = dict(estimate=float(values.mean()), lower_95=float(lo), upper_95=float(hi))
    return dict(seed=seed, resamples=resamples, method="participant-level percentile bootstrap (PCG64)",
                unit="subject", subject_ids=ids, intervals=intervals,
                interpretation="uncertainty reporting; not a significance claim")


def subject_distribution(per_subject: dict) -> dict:
    result = {}
    for key in MAIN_METRICS:
        values = [metric_summary(s)[key] for s in per_subject.values()]
        values = np.array([v for v in values if finite(v)])
        if not len(values):
            result[key] = None
        else:
            q1, median, q3 = np.percentile(values, [25, 50, 75])
            result[key] = dict(median=float(median), q1=float(q1), q3=float(q3),
                               iqr=float(q3 - q1), minimum=float(values.min()), maximum=float(values.max()),
                               finite_subjects=len(values), total_subjects=len(per_subject))
    return result


def compare_metrics(epoch0: dict, selected: dict) -> dict:
    result = {}
    for label, key in (("subject_balanced", "subject_balanced_metrics"), ("window_weighted", "window_metrics")):
        before, after = metric_summary(epoch0[key]), metric_summary(selected[key])
        result[label] = dict(epoch0=before, selected=after,
            selected_minus_epoch0={k: after[k] - before[k] if finite(after[k]) and finite(before[k])
                                  else None for k in MAIN_METRICS})
    return result


def window_identity(rows: list[dict]) -> list[tuple]:
    result = [(r["subject_id"], r["start_sec"], r["target_sha256"]) for r in rows]
    if len(result) != len({(sid, start) for sid, start, _ in result}):
        raise ValueError("Duplicate evaluation windows")
    return result


def validate_evaluation_identity(evaluation: dict, checkpoint: dict, manifest: dict, candidate_path,
                                 manifest_path, split: str) -> None:
    checkpoint_dataset_config(checkpoint, manifest)
    if evaluation["split"] != split:
        raise ValueError(f"Require an explicit {split} evaluation artifact")
    if evaluation["checkpoint_sha256"] != file_sha256(candidate_path) or evaluation["epoch"] != checkpoint["epoch"]:
        raise ValueError("Evaluation candidate hash/epoch mismatch")
    if (evaluation["split_manifest_sha256"] != split_manifest_hash(manifest)
            or evaluation["split_manifest"] != manifest
            or evaluation.get("split_manifest_file_sha256") != file_sha256(manifest_path)):
        raise ValueError("Evaluation split manifest identity mismatch")
    ids = manifest["validation_ids" if split == "val" else "test_ids"]
    if evaluation["subject_ids"] != ids:
        raise ValueError("Evaluation subject IDs do not match the requested partition")
    if not set(evaluation["metrics"]["per_subject"]) <= set(ids):
        raise ValueError("Evaluation contains subjects outside the requested partition")


def validation_gate(epoch0: dict, selected: dict, checkpoint: dict, expected_ids: list[str],
                    baselines_present: bool) -> dict:
    before = metric_summary(epoch0["subject_balanced_metrics"])
    after = metric_summary(selected["subject_balanced_metrics"])
    tol = SELECTION_TIE_TOLERANCE
    checks = {}
    checks["checkpoint_hr_eligible"] = checkpoint["validation_metrics"].get("hr_eligible") is True
    checks["finite_main_metrics"] = all(finite(v) for v in after.values()) and all(
        finite(v) for v in metric_summary(selected["window_metrics"]).values())
    checks["morphology_not_worse_than_epoch0"] = (finite(after["morphology_score"])
        and finite(before["morphology_score"]) and after["morphology_score"] >= before["morphology_score"] - tol)
    a, b = "wave_corr_aligned", "d1_corr_same_lag"
    checks["one_correlation_improves_without_large_deterioration"] = (all(
        finite(s[k]) for s in (before, after) for k in (a, b)) and
        ((after[a] > before[a] + tol and after[b] >= before[b] - .05 - tol)
         or (after[b] > before[b] + tol and after[a] >= before[a] - .05 - tol)))
    checks["subject_balanced_hr_within_epoch0_plus_1_bpm"] = (finite(after["hr_mae"])
        and finite(before["hr_mae"]) and after["hr_mae"] <= before["hr_mae"] + 1.0)
    checks["all_expected_subjects_have_valid_windows"] = all(
        selected.get("per_subject", {}).get(sid, {}).get("valid_windows", 0) > 0 for sid in expected_ids)
    checks["finite_main_metrics_for_every_subject_and_window"] = all(
        all(finite(v) for v in metric_summary(s).values()) and all(
            s["finite_window_counts"].get(k, 0) == s["windows"] for k in MAIN_METRICS if k != "morphology_score")
        for s in selected["per_subject"].values())
    residual = [selected.get("residual_contribution_overall_mean_abs"), selected.get("residual_contribution_max_abs"),
                *selected.get("residual_contribution_mean_abs", {}).values(),
                *selected.get("residual_contribution_max_abs_per_roi", {}).values()]
    checks["finite_residual_diagnostics"] = (selected.get("residual_diagnostics_all_finite") is True
                                             and all(finite(v) for v in residual))
    checks["classical_baselines_compared"] = baselines_present
    failed = [name for name, passed in checks.items() if not passed]
    return dict(validation_decision="DO_NOT_TEST_OR_FREEZE" if failed else "ACCEPT_FOR_LOCKED_TEST",
                checks=checks, reasons=failed if failed else ["All predeclared validation conditions passed."],
                numerical_tolerance=tol, large_deterioration_absolute_correlation=.05,
                thresholds_are="engineering guards, not physiological or clinical thresholds")


def build_validation_review(log: dict, checkpoint: dict, evaluation: dict, manifest: dict,
                            candidate_path, evaluation_path, log_path, manifest_path,
                            resamples=2000, seed=42) -> dict:
    validate_evaluation_identity(evaluation, checkpoint, manifest, candidate_path, manifest_path, "val")
    epochs = {e["epoch"]: e for e in log["epochs"]}
    if len(epochs) != len(log["epochs"]) or 0 not in epochs or checkpoint["epoch"] not in epochs:
        raise ValueError("Training log is missing unique epoch-0/selected records")
    if sorted(epochs) != list(range(log["metadata"]["args"]["epochs"] + 1)):
        raise ValueError("Training did not complete the declared epochs")
    if log["metadata"]["split_manifest"] != manifest:
        raise ValueError("Training log split mismatch")
    selector = WaveformCheckpointSelector(checkpoint["hr_regression_tolerance_bpm"])
    for epoch in sorted(epochs):
        # JSON encodes undefined numbers as null; selection's in-memory API
        # expects NaN. Diagnostic epochs with missing HR stay ineligible.
        stored = epochs[epoch]["subject_balanced_metrics"]
        selector.consider(epoch, dict(subject_balanced_metrics={k: v if finite(v) else float("nan")
            for k, v in stored.items() if k in ("wave_corr_aligned", "d1_corr_same_lag", "hr_mae")}))
    if selector.best_candidate["epoch"] != checkpoint["epoch"]:
        raise ValueError("Checkpoint is not the HR-gated candidate selected by the completed log")
    epoch0, metrics = epochs[0], evaluation["metrics"]
    for label in ("subject_balanced_metrics", "window_metrics"):
        for key in MAIN_METRICS:
            logged = metric_summary(epochs[checkpoint["epoch"]][label])[key]
            actual = metric_summary(metrics[label])[key]
            if finite(logged) != finite(actual) or (finite(logged) and abs(logged - actual) > 1e-6):
                raise ValueError(f"Standalone validation disagrees with selected training epoch: {label}/{key}")
            for stored, reference in ((checkpoint["validation_metrics"], epochs[checkpoint["epoch"]]),
                                      (checkpoint["epoch0_reference_metrics"], epoch0)):
                saved = metric_summary(stored[label])[key]
                log_value = metric_summary(reference[label])[key]
                if finite(saved) != finite(log_value) or (finite(saved) and abs(saved - log_value) > 1e-8):
                    raise ValueError(f"Checkpoint/log reference disagreement: {label}/{key}")
    if epoch0["valid_windows_per_subject"] != metrics["valid_windows_per_subject"]:
        raise ValueError("Epoch-0 and candidate do not use the same subject/window counts")
    identity = window_identity(metrics["window_results"])
    baseline_table = []
    baselines = metrics.get("classical_baselines", {})
    expected_names = {f"{r}/{p}" for r in checkpoint["roi_names"] for p in checkpoint["prior_names"]}
    expected_names |= {f"{p} multi-ROI average" for p in checkpoint["prior_names"]}
    if baselines and set(baselines) != expected_names:
        raise ValueError("Incomplete classical baseline set")
    for name, baseline in baselines.items():
        if window_identity(baseline["window_results"]) != identity:
            raise ValueError(f"Baseline windows/targets differ: {name}")
        baseline_table.append(dict(name=name, **metric_summary(baseline["subject_balanced_metrics"]),
            valid_windows_per_subject=baseline["valid_windows_per_subject"],
            finite_subject_counts=baseline["subject_balanced_metrics"]["finite_subject_counts"],
            same_registered_windows_and_targets=True))
    classical = [b for b in baseline_table if all(finite(b[k]) for k in BOOTSTRAP_METRICS)
                 and all(b["valid_windows_per_subject"].get(sid, 0) > 0 for sid in manifest["validation_ids"])]
    best = max(classical, key=lambda b: b["morphology_score"]) if classical else None
    baseline_table.append(dict(name="epoch-0 conservative model", **metric_summary(epoch0["subject_balanced_metrics"]),
                               valid_windows_per_subject=epoch0["valid_windows_per_subject"]))
    comparison = compare_metrics(epoch0, metrics)
    gate = validation_gate(epoch0, metrics, checkpoint, manifest["validation_ids"], bool(baselines))
    improves = comparison["subject_balanced"]["selected_minus_epoch0"]["morphology_score"]
    review = dict(review_version=REVIEW_VERSION, stage="validation_before_locked_test",
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        git_commit_used=checkpoint["provenance"]["git_commit"],
        checkpoint=str(Path(candidate_path).resolve()), checkpoint_sha256=file_sha256(candidate_path),
        selected_epoch=checkpoint["epoch"], training_log_path=str(Path(log_path).resolve()),
        training_log_sha256=file_sha256(log_path),
        validation_evaluation_path=str(Path(evaluation_path).resolve()),
        validation_evaluation_sha256=file_sha256(evaluation_path),
        split_manifest=manifest, split_manifest_sha256=split_manifest_hash(manifest),
        split_manifest_file_sha256=file_sha256(manifest_path),
        counts={k: len(manifest[k]) for k in ("all_subject_ids", "train_ids", "validation_ids", "test_ids")},
        training_completed_epochs=max(epochs), epoch0_comparison=comparison,
        learned_stage_added_value=bool(finite(improves) and improves > SELECTION_TIE_TOLERANCE),
        learned_stage_assessment=("Training improved subject-balanced morphology relative to conservative initialization."
            if finite(improves) and improves > SELECTION_TIE_TOLERANCE else
            "Training did not improve subject-balanced morphology over conservative initialization."),
        classical_baseline_table=baseline_table, best_classical_baseline_by_morphology=best,
        candidate_minus_best_classical={k: metric_summary(metrics["subject_balanced_metrics"])[k] - best[k]
            if finite(metric_summary(metrics["subject_balanced_metrics"])[k]) and finite(best[k]) else None
            for k in MAIN_METRICS} if best else None,
        attention={k: metrics[k] for k in ("roi_attention_mean", "roi_attention_median",
            "roi_attention_per_subject_mean", "roi_valid_fraction", "roi_highest_attention_percent",
            "highly_concentrated_attention_percent", "attention_concentration_rule", "prior_weights_mean")},
        attention_interpretation="descriptive model allocation; neither causal nor physiological explanation",
        residual={k: v for k, v in metrics.items() if k.startswith("residual_")},
        per_subject={sid: dict(**metric_summary(metrics["per_subject"].get(sid, {})),
            valid_windows=metrics["per_subject"].get(sid, {}).get("valid_windows", 0),
            windows=metrics["per_subject"].get(sid, {}).get("windows", 0)) for sid in manifest["validation_ids"]},
        subject_distribution=subject_distribution(metrics["per_subject"]),
        validation_bootstrap=subject_bootstrap({sid: metrics["per_subject"].get(sid, {})
            for sid in manifest["validation_ids"]}, resamples, seed), **gate,
        locked_test_consumed=False, freeze_decision="NOT_FROZEN_PENDING_LOCKED_TEST"
            if gate["validation_decision"] == "ACCEPT_FOR_LOCKED_TEST" else "NOT_FROZEN_VALIDATION_REJECTED",
        waveform_v1_status="candidate_not_frozen",
        limitations=["Facial video versus fingertip contact PPG does not establish exact anatomical equivalence.",
            "No validated notch, reflection index, APG, hand generalization, BP morphology or clinical BP/PTT claims.",
            "Morphology extraction only; face-hand timing remains the matched GREEN path."])
    return json_safe(review)


def require_accepted_validation(review: dict, candidate_path, manifest_path) -> None:
    """Fail closed before any locked-test dataset can be constructed."""
    if (review.get("review_version") != REVIEW_VERSION
            or review.get("validation_decision") != "ACCEPT_FOR_LOCKED_TEST"
            or not review.get("checks") or not all(review["checks"].values())):
        raise ValueError("Validation review did not accept this candidate; locked test must remain unopened")
    if (review["checkpoint_sha256"] != file_sha256(candidate_path)
            or review["split_manifest_file_sha256"] != file_sha256(manifest_path)):
        raise ValueError("Accepted review does not describe the current candidate/split")
    if file_sha256(review["validation_evaluation_path"]) != review["validation_evaluation_sha256"]:
        raise ValueError("Accepted validation artifact changed")


def build_locked_test_review(validation_review: dict, checkpoint: dict, test: dict,
                            candidate_path, manifest_path, test_path, generalization_review: dict,
                            resamples=2000, seed=42) -> dict:
    """Assess the already-consumed fixed candidate; never select another epoch.

    Generalization is an explicit reasoned engineering assessment, not a newly
    invented clinical cutoff. Its three judgments must each include evidence.
    """
    require_accepted_validation(validation_review, candidate_path, manifest_path)
    manifest = validation_review["split_manifest"]
    validate_evaluation_identity(test, checkpoint, manifest, candidate_path, manifest_path, "test")
    receipt = test.get("locked_test_receipt")
    if (test.get("locked_test_consumed") is not True or not receipt
            or receipt.get("locked_test_consumed") is not True
            or receipt["checkpoint_sha256"] != file_sha256(candidate_path)
            or receipt["checkpoint_epoch"] != checkpoint["epoch"]
            or receipt["split_manifest_file_sha256"] != file_sha256(manifest_path)
            or receipt["validation_evaluation_sha256"] != validation_review["validation_evaluation_sha256"]):
        raise ValueError("Missing or incompatible locked-test consumption receipt")
    criteria = ("hr_usable_without_catastrophic_regression", "waveform_correlations_do_not_collapse",
                "no_broad_systematic_subject_failure")
    if set(generalization_review) != set(criteria) or any(
        type(generalization_review[k].get("passed")) is not bool
        or not str(generalization_review[k].get("reason", "")).strip() for k in criteria):
        raise ValueError("Generalization review requires three explicit judgments with evidence")
    metrics = test["metrics"]
    summary = metric_summary(metrics["subject_balanced_metrics"])
    finite_metrics = all(finite(v) for v in summary.values()) and all(
        finite(v) for v in metric_summary(metrics["window_metrics"]).values())
    coverage = all(metrics.get("per_subject", {}).get(sid, {}).get("valid_windows", 0) > 0
                   for sid in manifest["test_ids"])
    per_subject_finite = all(all(finite(v) for v in metric_summary(s).values())
        and all(s["finite_window_counts"].get(k, 0) == s["windows"]
                for k in MAIN_METRICS if k != "morphology_score") for s in metrics["per_subject"].values())
    checks = dict(finite_main_metrics=finite_metrics, all_test_subjects_evaluable=coverage,
                  finite_subject_and_window_metrics=per_subject_finite,
                  finite_residuals=metrics.get("residual_diagnostics_all_finite") is True,
                  **{k: generalization_review[k]["passed"] for k in criteria})
    final = dict(validation_review)
    final.update(stage="validation_and_consumed_locked_test", locked_test_consumed=True,
        locked_test=dict(checkpoint_sha256=test["checkpoint_sha256"], checkpoint_epoch=test["epoch"],
            evaluation_path=str(Path(test_path).resolve()), evaluation_sha256=file_sha256(test_path), receipt=receipt,
            subject_balanced_metrics=summary, window_weighted_metrics=metric_summary(metrics["window_metrics"]),
            test_minus_validation={k: summary[k] - validation_review["epoch0_comparison"]["subject_balanced"]["selected"][k]
                if finite(summary[k]) else None for k in MAIN_METRICS},
            per_subject={sid: dict(**metric_summary(metrics["per_subject"].get(sid, {})),
                valid_windows=metrics["per_subject"].get(sid, {}).get("valid_windows", 0)) for sid in manifest["test_ids"]},
            subject_distribution=subject_distribution(metrics["per_subject"]),
            bootstrap=subject_bootstrap({sid: metrics["per_subject"].get(sid, {}) for sid in manifest["test_ids"]}, resamples, seed),
            generalization_review=generalization_review, checks=checks, future_test_policy=FUTURE_TEST_POLICY),
        freeze_decision="FREEZE_WAVEFORM_V1" if all(checks.values()) else "REJECTED_AFTER_LOCKED_TEST",
        freeze_reasons=[k for k, passed in checks.items() if not passed] or ["Validation gate and generalization sanity review passed."])
    return json_safe(final)


def reserve_locked_test(candidate_path, manifest_path, review_path, output_path) -> dict:
    """Irreversibly record consumption BEFORE touching the held-out data.

    Refuse repeats even if a different output path is requested. A failed test
    attempt remains consumed; recovering it needs an explicit separate decision.
    """
    review = json.loads(Path(review_path).read_text(encoding="utf-8"))
    require_accepted_validation(review, candidate_path, manifest_path)
    if review.get("locked_test_consumed") is not False:
        raise ValueError("Review already records locked-test consumption")
    output = Path(output_path)
    if output.exists():
        raise FileExistsError("Refusing to overwrite a locked-test evaluation")
    receipt = dict(locked_test_consumed=True, consumed_at_utc=datetime.now(timezone.utc).isoformat(),
        checkpoint_sha256=file_sha256(candidate_path), checkpoint_epoch=review["selected_epoch"],
        split_manifest_sha256=review["split_manifest_sha256"],
        split_manifest_file_sha256=file_sha256(manifest_path),
        validation_review_sha256=file_sha256(review_path),
        validation_evaluation_sha256=review["validation_evaluation_sha256"],
        output_path=str(output.resolve()), policy=FUTURE_TEST_POLICY)
    marker = Path(candidate_path).parent / "locked_test_consumption.json"
    with marker.open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False)
    return receipt


def write_review(review: dict, json_path, markdown_path) -> None:
    write_json(Path(json_path), review)
    def number(v):
        return f"{v:.6f}" if finite(v) else "unavailable"
    lines = ["# Waveform v1 candidate review", "", f"Decision: **{review['validation_decision']}**", "",
             f"Freeze decision: **{review['freeze_decision']}**; status: `{review['waveform_v1_status']}`.", "",
             f"Git commit used: `{review['git_commit_used']}`; selected epoch: {review['selected_epoch']}.", "",
             f"Locked test consumed: **{review['locked_test_consumed']}**.", "", review["learned_stage_assessment"], "",
             "## Participant split", ""]
    for key in ("all_subject_ids", "train_ids", "validation_ids", "test_ids"):
        ids = review["split_manifest"][key]
        lines += [f"- {key} ({len(ids)}): {', '.join(ids)}"]
    lines += ["", "## Epoch-0 comparison", ""]
    for label, values in review["epoch0_comparison"].items():
        lines += [f"### {label}", "", "| Metric | Epoch 0 | Selected | Selected minus epoch 0 |",
                  "|---|---:|---:|---:|"]
        for key in MAIN_METRICS:
            lines.append(f"| {key} | {number(values['epoch0'][key])} | {number(values['selected'][key])} | "
                         f"{number(values['selected_minus_epoch0'][key])} |")
        lines.append("")
    lines += ["## Classical and conservative baselines", "", "Same registered windows, targets and alignment rules; "
              "unavailable ROI/prior windows remain unavailable and coverage is recorded in JSON.", "",
              "| Baseline | Aligned corr | Same-lag d1 corr | Morphology | HR MAE (bpm) |", "|---|---:|---:|---:|---:|"]
    for row in review["classical_baseline_table"]:
        lines.append(f"| {row['name']} | " + " | ".join(number(row[k]) for k in BOOTSTRAP_METRICS) + " |")
    lines += ["", "## Validation gate", ""]
    lines += [f"- {key}: {'PASS' if passed else 'FAIL'}" for key, passed in review["checks"].items()]
    lines += ["", "Reasons: " + "; ".join(review["reasons"]), "", "The 0.05 correlation and +1 bpm guards are engineering "
              "rules for this experiment, not physiological or clinical thresholds.", "", "## Per-subject validation", "",
              "| Subject | Valid windows | Aligned corr | d1 corr | Morphology | HR MAE | nRMSE |", "|---|---:|---:|---:|---:|---:|---:|"]
    for sid, s in review["per_subject"].items():
        lines.append(f"| {sid} | {s['valid_windows']} | " + " | ".join(number(s[k]) for k in
            ("wave_corr_aligned", "d1_corr_same_lag", "morphology_score", "hr_mae", "aligned_nrmse")) + " |")
    for title, key in (("Subject distributions", "subject_distribution"), ("Validation subject bootstrap", "validation_bootstrap"),
                       ("Attention and prior weights (descriptive only)", "attention"), ("Scaled residual diagnostics", "residual")):
        lines += ["", f"## {title}", "", "```json", json.dumps(review[key], indent=2), "```"]
    if review.get("locked_test"):
        lines += ["", "## Locked-test review", "", FUTURE_TEST_POLICY, "", "```json",
                  json.dumps(review["locked_test"], indent=2), "```"]
    if review.get("frozen_checkpoint"):
        lines += ["", "## Frozen identity", "", f"Checkpoint: `{review['frozen_checkpoint']}`", "",
                  f"Frozen SHA256: `{review['frozen_checkpoint_sha256']}`", "",
                  f"Source candidate SHA256: `{review['checkpoint_sha256']}`", "",
                  f"Manifest: `{review['freeze_manifest_path']}`"]
    lines += ["", "## Limitations", ""] + [f"- {s}" for s in review["limitations"]]
    if not review["locked_test_consumed"]:
        lines += ["", "Locked test was NOT evaluated. Waveform v1 is NOT frozen."]
    Path(markdown_path).write_text("\n".join(lines) + "\n", encoding="utf-8")
