"""Provenance-checked finalization of an already-reviewed locked-test candidate."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import json
import torch

from .artifacts import file_sha256, write_json
from .waveform_review import REVIEW_VERSION, require_accepted_validation, validate_evaluation_identity


def freeze_waveform_candidate(candidate_path, manifest_path, review: dict, output_dir) -> dict:
    """Repack metadata only; never change weights or retain an optimizer state."""
    require_accepted_validation(review, candidate_path, manifest_path)
    if (review.get("freeze_decision") != "FREEZE_WAVEFORM_V1"
            or review.get("locked_test_consumed") is not True or not review.get("locked_test")):
        raise ValueError("Freeze requires an accepted consumed locked-test artifact and generalization review")
    locked = review["locked_test"]
    if not locked.get("checks") or not all(locked["checks"].values()):
        raise ValueError("Locked-test sanity checks failed")
    if file_sha256(locked["evaluation_path"]) != locked["evaluation_sha256"]:
        raise ValueError("Locked-test evaluation changed")
    checkpoint = torch.load(candidate_path, map_location="cpu", weights_only=True)
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    test = json.loads(Path(locked["evaluation_path"]).read_text(encoding="utf-8"))
    validate_evaluation_identity(test, checkpoint, manifest, candidate_path, manifest_path, "test")
    receipt = test.get("locked_test_receipt", {})
    marker_path = Path(candidate_path).parent / "locked_test_consumption.json"
    if (test.get("locked_test_consumed") is not True or receipt.get("locked_test_consumed") is not True
            or not marker_path.is_file() or json.loads(marker_path.read_text(encoding="utf-8")) != receipt
            or receipt.get("checkpoint_sha256") != file_sha256(candidate_path)
            or receipt.get("split_manifest_file_sha256") != file_sha256(manifest_path)
            or receipt.get("checkpoint_epoch") != checkpoint["epoch"]):
        raise ValueError("Locked-test consumption record is missing or inconsistent")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    frozen_path, freeze_manifest_path = output / "waveform_v1_frozen.pt", output / "waveform_v1_freeze_manifest.json"
    if frozen_path.exists() or freeze_manifest_path.exists():
        raise FileExistsError("Refusing to replace an existing frozen identity")
    date = datetime.now(timezone.utc).isoformat()
    frozen = {k: v for k, v in checkpoint.items() if k != "optimizer_state"}
    frozen.update(waveform_v1_status="frozen_v1", frozen_from_checkpoint=str(Path(candidate_path).resolve()),
        frozen_from_checkpoint_sha256=file_sha256(candidate_path), selected_epoch=checkpoint["epoch"],
        validation_evaluation_sha256=review["validation_evaluation_sha256"],
        locked_test_evaluation_sha256=locked["evaluation_sha256"],
        split_manifest_file_sha256=file_sha256(manifest_path), freeze_date=date,
        freeze_decision="FREEZE_WAVEFORM_V1", freeze_review_version=REVIEW_VERSION)
    torch.save(frozen, frozen_path)
    reloaded = torch.load(frozen_path, map_location="cpu", weights_only=True)
    for name, tensor in checkpoint["model_state"].items():
        other = reloaded["model_state"][name]
        # Compare raw bytes as well as dtype/shape, including signed zero bits.
        if (tensor.dtype != other.dtype or tensor.shape != other.shape
                or not torch.equal(tensor.contiguous().reshape(-1).view(torch.uint8),
                                   other.contiguous().reshape(-1).view(torch.uint8))):
            raise RuntimeError(f"Frozen model state changed: {name}")
    result = dict(model_sha256=file_sha256(frozen_path), model_path=str(frozen_path.resolve()),
        source_candidate_sha256=file_sha256(candidate_path), git_commit=checkpoint["provenance"]["git_commit"],
        model_class=checkpoint["model_class"], model_config=checkpoint["model_config"],
        roi_ordering=checkpoint["roi_names"], prior_ordering=checkpoint["prior_names"],
        channel_ordering=checkpoint["channel_ordering"], fs=checkpoint["fs"], win_sec=checkpoint["win_sec"],
        window_length_samples=checkpoint["window_length_samples"], stride_sec=checkpoint["stride_sec"],
        extraction_config=checkpoint["extraction_config"], extraction_cache_version=checkpoint["extraction_cache_version"],
        min_valid_fraction=checkpoint["min_valid_fraction"], loss_config=checkpoint["loss_config"],
        objective_name=checkpoint["objective_name"], selected_epoch=checkpoint["epoch"],
        train_ids=manifest["train_ids"], validation_ids=manifest["validation_ids"], test_ids=manifest["test_ids"],
        validation_metrics=review["epoch0_comparison"]["subject_balanced"]["selected"],
        test_metrics=locked["subject_balanced_metrics"], validation_bootstrap=review["validation_bootstrap"],
        test_bootstrap=locked["bootstrap"], split_manifest_sha256=review["split_manifest_sha256"],
        split_manifest_file_sha256=file_sha256(manifest_path),
        validation_evaluation_sha256=review["validation_evaluation_sha256"],
        locked_test_evaluation_sha256=locked["evaluation_sha256"], locked_test_receipt=receipt,
        date=date, status="frozen_v1", freeze_decision="FREEZE_WAVEFORM_V1", freeze_review_version=REVIEW_VERSION,
        model_state_bit_identical=True, training_source_provenance=checkpoint["provenance"])
    write_json(freeze_manifest_path, result)
    return result
