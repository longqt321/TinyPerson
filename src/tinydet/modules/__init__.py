"""Tiny object detection modules: QAC and QAC v2."""

from tinydet.modules.qac import QACDetect, QualityAwareDetectionLoss, QualityAwareE2ELoss
from tinydet.modules.qac_v2 import (
    QACDetectV2,
    QualityAwareDetectionLossV2,
    QualityAwareE2ELossV2,
)

__all__ = [
    "QACDetect",
    "QACDetectV2",
    "QualityAwareDetectionLoss",
    "QualityAwareDetectionLossV2",
    "QualityAwareE2ELoss",
    "QualityAwareE2ELossV2",
]
