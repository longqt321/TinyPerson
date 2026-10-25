"""Evidence-guided feature modulation."""

from torch import Tensor, nn


class EvidenceGuidedModulation(nn.Module):
    """Gate original features from fused evidence."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        if type(channels) is not int or channels < 1:
            raise ValueError("channels must be a positive integer")
        self.channels = channels
        self.gate = nn.Sequential(nn.Conv2d(channels, channels, 1), nn.Sigmoid())

    def forward(self, x: Tensor, evidence: Tensor) -> Tensor:
        if x.ndim != 4 or x.shape[1] != self.channels or evidence.shape != x.shape:
            raise ValueError("x and evidence must have the same BCHW shape and configured channels")
        return x * self.gate(evidence)
