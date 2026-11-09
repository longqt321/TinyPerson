"""Shared per-image validation prediction collection.

`WandbMediaCallback` and `SizeStratifiedValCallback` both need the per-image
validation predictions in source-pixel xyxy format (the format the TinyPerson
benchmark consumes). They wrap `validator.update_metrics` differently but the
row-building logic must be identical, so it lives here — one implementation,
two consumers.
"""


def build_prediction_row(validator, index: int, pred, batch) -> dict:
    """Build one per-image prediction row via the validator's own pipeline.

    Uses `validator._prepare_batch` / `_prepare_pred` / `scale_preds`, so the
    boxes are in source-pixel xyxy coordinates, exactly matching the input
    format of `evaluate_tinyperson`.
    """
    prepared = validator._prepare_batch(index, batch)
    row = {"path": str(prepared["im_file"]), "boxes": [], "scores": [], "classes": []}
    if pred["conf"].numel():
        predn = validator._prepare_pred(pred)
        scaled = validator.scale_preds(predn, prepared)
        row["boxes"] = scaled["bboxes"].detach().cpu().numpy().tolist()
        row["scores"] = predn["conf"].detach().cpu().numpy().tolist()
        row["classes"] = predn["cls"].detach().cpu().numpy().tolist()
    return row
