"""Training-only tiny object detection modules and Ultralytics adapters."""

from tinydet.modules.btrm import BTRM
from tinydet.modules.composite import (
    AssignerMLP,
    CandidateRefiner,
    CompositeCriterion,
    CompositeDetect,
    ContextEvidenceGate,
    DensityRouter,
    DuplicateScorer,
    QualityHead,
    ScaleAligner,
)
from tinydet.modules.msr import (
    MSR,
    EvidenceGuidedModulation,
    MultiScaleEvidenceFusion,
    ResidualRefinement,
    SingleScaleEvidence,
)
from tinydet.modules.qac import QACDetect, QualityAwareDetectionLoss, QualityAwareE2ELoss
from tinydet.modules.qac_v2 import (
    QACDetectV2,
    QualityAwareDetectionLossV2,
    QualityAwareE2ELossV2,
)

__all__ = [
    "BTRM",
    "MSR",
    "AssignerMLP",
    "CandidateRefiner",
    "CompositeCriterion",
    "CompositeDetect",
    "ContextEvidenceGate",
    "DensityRouter",
    "DuplicateScorer",
    "EvidenceGuidedModulation",
    "MultiScaleEvidenceFusion",
    "QACDetect",
    "QACDetectV2",
    "QualityAwareDetectionLoss",
    "QualityAwareDetectionLossV2",
    "QualityAwareE2ELoss",
    "QualityAwareE2ELossV2",
    "QualityHead",
    "ResidualRefinement",
    "ScaleAligner",
    "SingleScaleEvidence",
]
