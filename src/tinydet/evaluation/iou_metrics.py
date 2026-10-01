"""AP values at the ten IoU thresholds already computed by Ultralytics."""


def ap_by_iou(box) -> dict[str, float]:
    all_ap = box.all_ap
    if len(all_ap) == 0:
        return {f"AP{iou / 100:.2f}": 0.0 for iou in range(50, 100, 5)}
    if all_ap.shape[1] != 10:
        raise ValueError(f"Expected 10 IoU thresholds, got {all_ap.shape[1]}")
    return {
        f"AP{iou / 100:.2f}": float(all_ap[:, index].mean())
        for index, iou in enumerate(range(50, 100, 5))
    }
