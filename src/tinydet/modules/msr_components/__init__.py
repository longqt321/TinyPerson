"""Independent feature-processing stages used by :class:`MSR`."""

from tinydet.modules.msr_components.fusion import MultiScaleEvidenceFusion
from tinydet.modules.msr_components.modulation import EvidenceGuidedModulation
from tinydet.modules.msr_components.refinement import ResidualRefinement
from tinydet.modules.msr_components.single_scale import SingleScaleEvidence

__all__ = [
    "EvidenceGuidedModulation",
    "MultiScaleEvidenceFusion",
    "ResidualRefinement",
    "SingleScaleEvidence",
]
