"""Compatibility shim for checkpoints written before the DCAF -> GauSS rename.

Those checkpoints pickle their classes under ``tinydet.modules.dcaf``.  This
module re-exports the current implementations so such checkpoints can still be
loaded for evaluation or resume.  No logic lives here; see
``tinydet.modules.gauss``.
"""

from tinydet.modules.gauss import (
    BandBranch,
    DilatedContextBranch,
    GauSS,
    LearnableGaussianScaleSpace,
    RefinementBranch,
)

# Legacy class name used by pre-rename checkpoints.
DCAF = GauSS

__all__ = [
    "DCAF",
    "BandBranch",
    "DilatedContextBranch",
    "GauSS",
    "LearnableGaussianScaleSpace",
    "RefinementBranch",
]
