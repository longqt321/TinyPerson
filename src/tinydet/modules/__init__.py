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
    "QualityHead",
    "ResidualRefinement",
    "ScaleAligner",
    "SingleScaleEvidence",
]
