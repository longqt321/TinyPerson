"""P2Refine: stride-4 identity-initialised refinement for YOLO26n-P2.

Replaces DCAF v2.1 as the P2 producer. Identity at step 0 (ReZero-style):
``out`` conv is zero-initialised with no BN after it, so ``y == x0`` bit-exact
at init, eliminating the cls-head desync measured for DCAF at step 0.

Provenance: authored against a numpy reference (verify_numpy.py); the PyTorch
implementation self-test passed in-repo on CPU (params=6000, train/eval both
rel=0.000000 exact=True, |grad W_out|=0.437, |grad dw1|=0 at step 0).
"""

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import nn


class P2Refine(nn.Module):
    """inputs: [C2 (B,c_c2,H,W), P3 (B,c_p3,H/2,W/2)]  ->  y (B,c,H,W)

    x0 = SiLU(BN(PW(C2) + Up(PW(P3))))                      # stem (stride 4)
    h  = SiLU(BN(DW3x3(x0)))                                # local
    h  = BN(PW_{c->mid}(h))                                 # channel mix
    h  = SiLU(BN(DW5x5(h)))                                 # wider context
    y  = x0 + W_out h                                       # W_out == 0 at init, NO activation

    Supports both Ultralytics YAML parser convention ``P2Refine(c1=[64, 64], c2=32)``
    and explicit channel arguments ``P2Refine(c_c2=64, c_p3=64, c=32)``.
    """

    def __init__(
        self,
        c1: Sequence[int] | int = (64, 64),
        c2: int = 32,
        mid: int = 16,
        bn_eps: float = 1e-3,
        bn_momentum: float = 0.03,
        c_p3: int | None = None,
        c: int | None = None,
    ):
        super().__init__()
        if isinstance(c1, Sequence):
            if len(c1) != 2:
                raise ValueError(f"P2Refine expects 2 input channels [C2, P3], got {list(c1)}")
            c_c2, c_p3_ = int(c1[0]), int(c1[1])
            c_out = int(c2)
        else:
            if c_p3 is None or c is None:
                raise ValueError("When c1 is an int, provide c_p3 and c as well.")
            c_c2, c_p3_, c_out = int(c1), int(c_p3), int(c)
        bn = lambda n: nn.BatchNorm2d(n, eps=bn_eps, momentum=bn_momentum)
        # stem
        self.proj_c2 = nn.Conv2d(c_c2, c_out, 1, bias=False)
        self.proj_p3 = nn.Conv2d(c_p3_, c_out, 1, bias=False)
        self.bn_stem = bn(c_out)
        # residual branch
        self.dw1 = nn.Conv2d(c_out, c_out, 3, padding=1, groups=c_out, bias=False)
        self.bn1 = bn(c_out)
        self.pw = nn.Conv2d(c_out, mid, 1, bias=False)
        self.bn2 = bn(mid)
        self.dw2 = nn.Conv2d(mid, mid, 5, padding=2, groups=mid, bias=False)
        self.bn3 = bn(mid)
        self.out = nn.Conv2d(mid, c_out, 1, bias=False)  # exit projection, zero-init, NO BN after it
        self.act = nn.SiLU()
        self._reset_parameters()

    def _reset_parameters(self):
        for m in (self.proj_c2, self.proj_p3, self.dw1, self.pw, self.dw2):
            # fan_in is the correct fan for depthwise (fan_in = k*k) and pointwise convs
            nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
        for m in (self.bn_stem, self.bn1, self.bn2, self.bn3):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        nn.init.zeros_(self.out.weight)  # exact identity at step 0

    def stem(self, c2, p3):
        p3 = F.interpolate(self.proj_p3(p3), size=c2.shape[-2:], mode="nearest")
        return self.act(self.bn_stem(self.proj_c2(c2) + p3))

    def residual(self, x0):
        h = self.act(self.bn1(self.dw1(x0)))
        h = self.bn2(self.pw(h))
        h = self.act(self.bn3(self.dw2(h)))
        return self.out(h)

    def forward(self, xs):
        c2, p3 = xs
        x0 = self.stem(c2, p3)
        return x0 + self.residual(x0)  # no activation on the sum

    @torch.no_grad()
    def get_diagnostics(self, xs):
        """Residual-to-stem ratio and direction stats for one batch of inputs."""
        c2, p3 = xs
        x0 = self.stem(c2, p3)
        d = self.residual(x0)
        n0, nd = x0.flatten(1).norm(dim=1), d.flatten(1).norm(dim=1)
        cos = F.cosine_similarity(x0.flatten(1), d.flatten(1), dim=1)
        return {
            "rho_mean": (nd / n0).mean().item(),
            "rho_max": (nd / n0).max().item(),
            "cos_x0_delta": cos.mean().item(),
            "w_out_norm": self.out.weight.norm().item(),
        }
