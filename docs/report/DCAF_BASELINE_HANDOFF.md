# Hồ sơ bàn giao nghiên cứu: DCAF v2.1 và YOLO26n-P2

Tài liệu này tự chứa mô tả, mã triển khai DCAF, cấu hình mô hình/huấn luyện và kết quả thực nghiệm cần cho người viết báo cáo. Mọi con số benchmark dưới đây là kết quả mô tả từ một lần chạy mỗi mô hình, seed 42; không có kiểm định ý nghĩa thống kê. Không coi đây là bằng chứng tuân thủ benchmark chính thức.

## 1. Câu hỏi và phạm vi

- **Bài toán:** phát hiện người rất nhỏ trên TinyPerson.
- **Đối chứng:** YOLO26n-P2 với detection head trên P2/P3/P4/P5.
- **Phương pháp:** thay P2 neck feature bằng feature được DCAF (*Dilated-Context Asymmetric Feature Refinement*) tinh chỉnh từ C2, P3 và P4.
- **Câu hỏi:** trong paired setup đã ghi nhận, DCAF v2.1 thay đổi AP/MR như thế nào ở inference tiled và full-frame?
- **Giới hạn:** một seed; test checkpoint là `last.pt`; evaluator standalone; chưa xác minh tương đương evaluator tham chiếu. Không có ablation tách riêng từng thành phần.

## 2. Thiết kế mô hình

### 2.1. Luồng tính toán

Đầu vào DCAF là `[C2, P3, P4]`, channels `[64, 64, 128]`; đầu ra là P2 feature 32 channels. Ở đầu vào 1280×1280, tensor C2/P2 có kích thước 320×320 theo stride 4. Module gồm ba khối:

1. **Multi-scale band encoder:** chiếu C2/P3/P4 về 32 channels; hợp nhất C2 với P3 nội suy lên P2. Hai Gaussian scale-space separable, support 5×5 và 9×9, tách `high = x0 − L1`, `mid = L1 − L2`, `low = L2`. Mỗi band được PixelUnshuffle(2) rồi depthwise-separable refinement.
2. **Dilated context controller:** cộng P3 với P4 nội suy; ba depthwise convolution 3×3 với dilation 1/2/3, softmax gate theo nhánh; cộng context với ba band-refined features và sinh spatial routing logits.
3. **Asymmetric refinement/reconstruction:** softmax trộn ba band; detail branch 3×3 có neo high-band H-BRA; structural branch 5×5 có BSG saliency attenuation trên low-band; mixing gate chọn detail/structure; projection + PixelShuffle(2); residual gain bị chặn rồi cộng với x0.

Hệ số Gaussian được chặn: sigma1 ∈ [0.4, 0.9], sigma2 ∈ [1.1, 1.8]. Khởi tạo raw sigma bằng 0, do đó sigma ban đầu lần lượt là 0.65 và 1.45. Residual gain `alpha = 0.5 sigmoid(alpha_raw)` khởi tạo 0.25. H-BRA gain `gamma_h = 0.5 sigmoid(gamma_h_raw)` khởi tạo xấp xỉ 0.20. BSG là `sigmoid(Conv1x1(c_ctx))`, attenuation `r_l * (1 - 0.5*mask)`.

**Implementation contract:** 34,666 trainable parameters, 24 Conv2d modules, 130 control scalars; tại batch 1, C2=64/P3=64/P4=128, P2 320×320: 1,016,750,080 MAC = 1.01675008 GMAC = 2.03350016 GFLOPs theo quy ước 2 FLOPs/MAC. Đây là chi phí module phân tích, không phải FLOPs end-to-end của detector.

### 2.2. Mã đầy đủ module DCAF

Mã dưới đây là phần triển khai PyTorch đầy đủ, gồm module, nhóm optimizer cho tham số điều khiển và kiểm tra phân tích MAC. Các import phụ thuộc PyTorch.

```python
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

CONTROL_PARAM_SUFFIXES: tuple[str, ...] = (
    "sigma1x_raw", "sigma1y_raw", "sigma2x_raw", "sigma2y_raw", "alpha_raw", "gamma_h_raw",
)


class LearnableGaussianScaleSpace(nn.Module):
    """Separable learnable Gaussian scale-space filter with single reflect pad."""

    coord1: Tensor
    coord2: Tensor

    def __init__(self, channels: int = 32) -> None:
        super().__init__()
        self.channels = int(channels)
        self.r1, self.r2 = 2, 4
        self.sigma1x_raw = nn.Parameter(torch.zeros(self.channels, dtype=torch.float32))
        self.sigma1y_raw = nn.Parameter(torch.zeros(self.channels, dtype=torch.float32))
        self.sigma2x_raw = nn.Parameter(torch.zeros(self.channels, dtype=torch.float32))
        self.sigma2y_raw = nn.Parameter(torch.zeros(self.channels, dtype=torch.float32))
        self.register_buffer("coord1", torch.arange(-self.r1, self.r1 + 1, dtype=torch.float32), persistent=False)
        self.register_buffer("coord2", torch.arange(-self.r2, self.r2 + 1, dtype=torch.float32), persistent=False)

    def compute_sigmas(self) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        s1x = 0.4 + 0.5 * torch.sigmoid(self.sigma1x_raw.float())
        s1y = 0.4 + 0.5 * torch.sigmoid(self.sigma1y_raw.float())
        s2x = 1.1 + 0.7 * torch.sigmoid(self.sigma2x_raw.float())
        s2y = 1.1 + 0.7 * torch.sigmoid(self.sigma2y_raw.float())
        return s1x, s1y, s2x, s2y

    @staticmethod
    def _kernel_1d(sigma: Tensor, coords: Tensor) -> Tensor:
        logits = -coords.unsqueeze(0).square() / (2.0 * sigma.unsqueeze(1).square())
        return torch.softmax(logits, dim=-1)

    def _apply_separable(self, x: Tensor, sigma_x: Tensor, sigma_y: Tensor,
                         coords: Tensor, radius: int) -> Tensor:
        c = self.channels
        gx = self._kernel_1d(sigma_x, coords).to(dtype=x.dtype).view(c, 1, 1, -1)
        gy = self._kernel_1d(sigma_y, coords).to(dtype=x.dtype).view(c, 1, -1, 1)
        x_pad = F.pad(x, (radius, radius, radius, radius), mode="reflect")
        x_h = F.conv2d(x_pad, gx, bias=None, stride=1, padding=0, groups=c)
        return F.conv2d(x_h, gy, bias=None, stride=1, padding=0, groups=c)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, tuple[Tensor, Tensor, Tensor, Tensor]]:
        s1x, s1y, s2x, s2y = self.compute_sigmas()
        l1 = self._apply_separable(x, s1x, s1y, self.coord1, self.r1)
        l2 = self._apply_separable(x, s2x, s2y, self.coord2, self.r2)
        return l1, l2, (s1x, s1y, s2x, s2y)


class BandBranch(nn.Module):
    """PixelUnshuffle and residual depthwise-separable band encoder."""

    def __init__(self, hidden_dim: int = 32) -> None:
        super().__init__()
        d = int(hidden_dim)
        self.pu = nn.PixelUnshuffle(downscale_factor=2)
        self.proj = nn.Conv2d(4*d, d, 1, bias=False)
        self.bn_proj = nn.BatchNorm2d(d)
        self.dw = nn.Conv2d(d, d, 3, padding=1, groups=d, bias=False)
        self.bn_dw = nn.BatchNorm2d(d)
        self.pw = nn.Conv2d(d, d, 1, bias=False)
        self.bn_pw = nn.BatchNorm2d(d)
        self.act = nn.SiLU(inplace=False)

    def forward(self, b_band: Tensor) -> Tensor:
        z = self.act(self.bn_proj(self.proj(self.pu(b_band))))
        d_feat = self.act(self.bn_dw(self.dw(z)))
        p_feat = self.bn_pw(self.pw(d_feat))
        return self.act(z + p_feat)


class DilatedContextBranch(nn.Module):
    def __init__(self, hidden_dim: int = 32, dilation: int = 1) -> None:
        super().__init__()
        d, dil = int(hidden_dim), int(dilation)
        self.dw = nn.Conv2d(d, d, 3, padding=dil, dilation=dil, groups=d, bias=False)
        self.bn = nn.BatchNorm2d(d)
        self.act = nn.SiLU(inplace=False)

    def forward(self, c0: Tensor) -> Tensor:
        return self.act(self.bn(self.dw(c0)))


class RefinementBranch(nn.Module):
    """Residual depthwise-separable refinement, k=3 detail or k=5 structure."""

    def __init__(self, hidden_dim: int = 32, kernel_size: int = 3) -> None:
        super().__init__()
        d, k = int(hidden_dim), int(kernel_size)
        self.dw = nn.Conv2d(d, d, k, padding=k//2, groups=d, bias=False)
        self.bn_dw = nn.BatchNorm2d(d)
        self.pw = nn.Conv2d(d, d, 1, bias=False)
        self.bn_pw = nn.BatchNorm2d(d)
        self.act = nn.SiLU(inplace=False)

    def forward(self, x: Tensor) -> Tensor:
        h = self.act(self.bn_dw(self.dw(x)))
        h = self.bn_pw(self.pw(h))
        return self.act(x + h)


class DCAF(nn.Module):
    """DCAF v2.1; accepts [C2,P3,P4] channel list or explicit channel args."""

    def __init__(self, c1: Sequence[int] | int = (64, 64, 128), c2: int = 32,
                 p4_ch: int | None = None, c_out: int | None = None) -> None:
        super().__init__()
        if isinstance(c1, Sequence):
            if len(c1) != 3:
                raise ValueError(f"DCAF expects 3 input channels [C2, P3, P4], got {list(c1)}")
            c2_in, p3_in, p4_in = int(c1[0]), int(c1[1]), int(c1[2])
            d = int(c2)
        else:
            if p4_ch is None or c_out is None:
                raise ValueError("When c1 is an int, provide (c2_ch, p3_ch, p4_ch, c_out) or pass c1=[c2, p3, p4].")
            c2_in, p3_in, p4_in, d = int(c1), int(c2), int(p4_ch), int(c_out)

        self.c2_channels, self.p3_channels, self.p4_channels = c2_in, p3_in, p4_in
        self.hidden_dim = self.out_channels = d
        self.proj_c2 = nn.Conv2d(c2_in, d, 1, bias=False)
        self.proj_p3 = nn.Conv2d(p3_in, d, 1, bias=False)
        self.proj_p4 = nn.Conv2d(p4_in, d, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(d)
        self.gaussian = LearnableGaussianScaleSpace(d)
        self.band_h, self.band_m, self.band_l = (BandBranch(d) for _ in range(3))
        self.bn3 = nn.BatchNorm2d(d)
        self.ctx_d1, self.ctx_d2, self.ctx_d3 = (DilatedContextBranch(d, k) for k in (1, 2, 3))
        self.ctx_gate = nn.Conv2d(d, 3, 1, bias=True)
        self.saliency_gate = nn.Conv2d(d, 1, 1, bias=True)
        self.route_dw = nn.Conv2d(d, d, 3, padding=1, groups=d, bias=False)
        self.route_bn = nn.BatchNorm2d(d)
        self.route_out = nn.Conv2d(d, 4, 1, bias=True)
        self.detail_branch = RefinementBranch(d, 3)
        self.struct_branch = RefinementBranch(d, 5)
        self.recon_proj = nn.Conv2d(d, 4*d, 1, bias=False)
        self.recon_bn = nn.BatchNorm2d(4*d)
        self.ps = nn.PixelShuffle(2)
        self.alpha_raw = nn.Parameter(torch.zeros(1, dtype=torch.float32))
        self.gamma_h_raw = nn.Parameter(torch.tensor([-0.4055], dtype=torch.float32))
        self.act = nn.SiLU(inplace=False)
        self.last_diag: dict[str, float] = {}
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        with torch.no_grad():
            self.route_out.weight[3].mul_(0.01)
            if self.route_out.bias is not None:
                self.route_out.bias.zero_()
            self.gaussian.sigma1x_raw.zero_(); self.gaussian.sigma1y_raw.zero_()
            self.gaussian.sigma2x_raw.zero_(); self.gaussian.sigma2y_raw.zero_()
            self.alpha_raw.zero_(); self.gamma_h_raw.fill_(-0.4055)
            self.saliency_gate.weight.mul_(0.01)
            if self.saliency_gate.bias is not None:
                self.saliency_gate.bias.zero_()

    def compute_alpha(self) -> Tensor:
        return 0.5 * torch.sigmoid(self.alpha_raw.float())

    def compute_gamma_h(self) -> Tensor:
        return 0.5 * torch.sigmoid(self.gamma_h_raw.float())

    def forward(self, x: Sequence[Tensor]) -> Tensor:
        if not isinstance(x, Sequence) or len(x) != 3:
            raise ValueError("DCAF.forward expects a list/tuple of 3 tensors: [C2, P3, P4].")
        c2, p3, p4 = x[0], x[1], x[2]
        s2, s3, s4 = self.proj_c2(c2), self.proj_p3(p3), self.proj_p4(p4)
        s3_up = F.interpolate(s3, size=s2.shape[-2:], mode="nearest")
        x0 = self.act(self.bn2(s2 + s3_up))
        l1, l2, sigmas = self.gaussian(x0)
        b_h, b_m, b_l = x0-l1, l1-l2, l2
        r_h, r_m, r_l = self.band_h(b_h), self.band_m(b_m), self.band_l(b_l)
        p4_up = F.interpolate(s4, size=s3.shape[-2:], mode="nearest")
        c0 = self.act(self.bn3(s3 + p4_up))
        t1, t2, t3 = self.ctx_d1(c0), self.ctx_d2(c0), self.ctx_d3(c0)
        pi = torch.softmax(self.ctx_gate(c0), dim=1)
        pi1, pi2, pi3 = pi[:, 0:1], pi[:, 1:2], pi[:, 2:3]
        c_ctx = self.act(c0 + pi1*t1 + pi2*t2 + pi3*t3)
        u = self.act(r_h + r_m + r_l + c_ctx)
        v = self.act(self.route_bn(self.route_dw(u)))
        logits = self.route_out(v)
        a_bands = torch.softmax(logits[:, :3], dim=1)
        a_h, a_m, a_l = a_bands[:, 0:1], a_bands[:, 1:2], a_bands[:, 2:3]
        m_gate = torch.sigmoid(logits[:, 3:4])
        f_band = a_h*r_h + a_m*r_m + a_l*r_l
        gamma_h = self.compute_gamma_h().to(dtype=r_h.dtype)
        x_d = f_band + c_ctx + gamma_h*r_h
        f_d = self.detail_branch(x_d)
        saliency_mask = torch.sigmoid(self.saliency_gate(c_ctx))
        r_l_attenuated = r_l*(1.0 - 0.5*saliency_mask)
        f_s = self.struct_branch(r_l_attenuated + c_ctx)
        f_route = m_gate*f_d + (1.0-m_gate)*f_s
        z = self.ps(self.recon_bn(self.recon_proj(f_route)))
        alpha = self.compute_alpha().to(dtype=z.dtype)
        alpha_z = alpha*z
        y_p2 = self.act(x0 + alpha_z)
        with torch.no_grad():
            s1x, s1y, s2x, s2y = sigmas
            sigma1, sigma2 = torch.cat((s1x,s1y)), torch.cat((s2x,s2y))
            m_f = m_gate.detach().float()
            rms_res = alpha_z.detach().float().square().mean().sqrt()
            rms_in = x0.detach().float().square().mean().sqrt().clamp_min(1e-12)
            self.last_diag = {
                "routing/mixing_mean": float(m_f.mean().item()),
                "routing/mixing_std": float(m_f.std(unbiased=False).item()),
                "routing/high_band_mean": float(a_h.detach().float().mean().item()),
                "routing/mid_band_mean": float(a_m.detach().float().mean().item()),
                "routing/low_band_mean": float(a_l.detach().float().mean().item()),
                "context/dilation_1_mean": float(pi1.detach().float().mean().item()),
                "context/dilation_2_mean": float(pi2.detach().float().mean().item()),
                "context/dilation_3_mean": float(pi3.detach().float().mean().item()),
                "refinement/residual_ratio": float((rms_res/rms_in).item()),
                "gaussian/sigma1_mean": float(sigma1.mean().item()),
                "gaussian/sigma2_mean": float(sigma2.mean().item()),
                "gaussian/sigma1_std": float(sigma1.std(unbiased=False).item()),
                "gaussian/sigma2_std": float(sigma2.std(unbiased=False).item()),
            }
        return y_p2

    def get_diagnostics(self, optimizer: torch.optim.Optimizer | None = None) -> dict[str, float]:
        out = dict(self.last_diag)
        ctrl_lr = ctrl_wd = float("nan")
        if optimizer is not None:
            for pg in optimizer.param_groups:
                if pg.get("param_group") == "control":
                    ctrl_lr, ctrl_wd = float(pg.get("lr", float("nan"))), float(pg.get("weight_decay", float("nan")))
                    break
        out["optimizer/control_lr"], out["optimizer/control_weight_decay"] = ctrl_lr, ctrl_wd
        return out


def is_dcaf_control_param_name(name: str) -> bool:
    return name.endswith(CONTROL_PARAM_SUFFIXES)


def collect_dcaf_control_params(model: nn.Module) -> list[nn.Parameter]:
    params: list[nn.Parameter] = []
    seen: set[int] = set()
    for name, param in model.named_parameters():
        if param.requires_grad and is_dcaf_control_param_name(name):
            pid = id(param)
            if pid not in seen:
                seen.add(pid)
                params.append(param)
    return params


def configure_dcaf_control_param_group(optimizer: torch.optim.Optimizer, model: nn.Module,
                                      base_lr: float | None = None) -> dict[str, Any] | None:
    """Idempotently place DCAF control scalars in zero-weight-decay optimizer group."""
    control_params = collect_dcaf_control_params(model)
    if not control_params:
        return None
    ctrl_ids = {id(p) for p in control_params}
    inferred_lr: float | None = base_lr
    existing_ctrl_group: dict[str, Any] | None = None
    for pg in optimizer.param_groups:
        if pg.get("param_group") == "control":
            existing_ctrl_group = pg
            continue
        if inferred_lr is None and pg.get("param_group") in ("weight", None):
            inferred_lr = float(pg.get("lr", 1e-3))
        pg["params"] = [p for p in pg["params"] if id(p) not in ctrl_ids]
    if inferred_lr is None:
        inferred_lr = float(optimizer.param_groups[0].get("lr", 1e-3))
    if existing_ctrl_group is not None:
        existing_ctrl_group["params"] = control_params
        existing_ctrl_group["weight_decay"] = 0.0
        existing_ctrl_group.setdefault("lr", inferred_lr)
        existing_ctrl_group.setdefault("initial_lr", inferred_lr)
        return existing_ctrl_group
    ctrl_group: dict[str, Any] = {"params": control_params, "lr": inferred_lr,
                                  "initial_lr": inferred_lr, "weight_decay": 0.0,
                                  "param_group": "control"}
    for k, v in optimizer.param_groups[0].items():
        if k not in ctrl_group and k != "params":
            ctrl_group[k] = v
    optimizer.param_groups.append(ctrl_group)
    return ctrl_group


def compute_dcaf_analytical_macs(c2_ch: int = 64, p3_ch: int = 64, p4_ch: int = 128,
                                 d: int = 32, h_p2: int = 320,
                                 w_p2: int = 320) -> dict[str, float]:
    h_p3, w_p3 = h_p2//2, w_p2//2
    h_p4, w_p4 = h_p2//4, w_p2//4
    n_p2 = d*h_p2*w_p2
    m_c2, m_p3, m_p4 = h_p2*w_p2*d*c2_ch, h_p3*w_p3*d*p3_ch, h_p4*w_p4*d*p4_ch
    m_gauss_1 = d*(h_p2+4)*w_p2*5 + n_p2*5
    m_gauss_2 = d*(h_p2+8)*w_p2*9 + n_p2*9
    m_gauss = m_gauss_1 + m_gauss_2
    m_band_proj = 3*(h_p3*w_p3*d*(4*d))
    m_band_ref = 3*(h_p3*w_p3*(d*9+d*d))
    m_dilated = 3*(h_p3*w_p3*d*9)
    m_ctx_gate = h_p3*w_p3*3*d
    m_routing = h_p3*w_p3*(d*9+4*d)
    m_detail = h_p3*w_p3*(d*9+d*d)
    m_struct = h_p3*w_p3*(d*25+d*d)
    m_recon = h_p3*w_p3*(4*d)*d
    total = sum((m_c2,m_p3,m_p4,m_gauss,m_band_proj,m_band_ref,m_dilated,
                 m_ctx_gate,m_routing,m_detail,m_struct,m_recon))
    return {"c2_proj":float(m_c2), "p3_proj":float(m_p3), "p4_proj":float(m_p4),
            "gaussian_separable_single_pad":float(m_gauss), "band_projections":float(m_band_proj),
            "band_refinements":float(m_band_ref), "dilated_context":float(m_dilated),
            "context_gate":float(m_ctx_gate), "routing_controller":float(m_routing),
            "detail_refinement":float(m_detail), "structural_refinement":float(m_struct),
            "reconstruction_proj":float(m_recon), "total_macs":float(total),
            "total_gmacs":float(total)/1e9, "total_gflops":2.0*float(total)/1e9}
```

### 2.3. Cấu hình graph detector

Cả hai dùng YOLO26 nano, `nc=2`, `end2end=true`, `reg_max=1`, `tal_beta=6.0`, cùng backbone/neck P2→P5. Khác biệt graph chính: baseline tạo P2 qua neck C3k2 thông thường; DCAF lấy `[C2, P3, P4]` tại layer 20 để sinh P2 32 channels. Detection head nhận `[P2, P3, P4, P5]` ở cả hai.

**Cấu hình chung:**

```yaml
nc: 2
end2end: true
reg_max: 1
tal_beta: 6.0
scale: n
scales:
  n: [0.50, 0.25, 1024]
  s: [0.50, 0.50, 1024]
  m: [0.50, 1.00, 512]
  l: [1.00, 1.00, 512]
  x: [1.00, 1.50, 1024]
```

**Đoạn graph khác nhau ở neck P2 và detection head:**

```yaml
# YOLO26n-P2 baseline: tiếp tục neck từ P2 C3k2
head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 6], 1, Concat, [1]]
  - [-1, 2, C3k2, [512, True]]        # P4
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 4], 1, Concat, [1]]
  - [-1, 2, C3k2, [256, True]]        # P3
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 2], 1, Concat, [1]]
  - [-1, 2, C3k2, [128, True]]        # P2
  - [-1, 1, Conv, [128, 3, 2]]
  - [[-1, 16], 1, Concat, [1]]
  - [-1, 2, C3k2, [256, True]]        # P3 neck
  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 13], 1, Concat, [1]]
  - [-1, 2, C3k2, [512, True]]        # P4 neck
  - [-1, 1, Conv, [512, 3, 2]]
  - [[-1, 10], 1, Concat, [1]]
  - [-1, 1, C3k2, [1024, True, 0.5, True]] # P5 neck
  - [[19, 22, 25, 28], 1, Detect, [nc]]

# DCAF variant: same upstream backbone/neck; replace P2 output with DCAF
head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 6], 1, Concat, [1]]
  - [-1, 2, C3k2, [512, True]]        # P4
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 4], 1, Concat, [1]]
  - [-1, 2, C3k2, [256, True]]        # P3
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 2], 1, Concat, [1]]
  - [-1, 2, C3k2, [128, True]]        # P2 intermediate
  - [[2, 16, 13], 1, DCAF, [32]]      # [C2=64,P3=64,P4=128] -> P2 32
  - [-1, 1, Conv, [128, 3, 2]]
  - [[-1, 16], 1, Concat, [1]]
  - [-1, 2, C3k2, [256, True]]        # P3 neck
  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 13], 1, Concat, [1]]
  - [-1, 2, C3k2, [512, True]]        # P4 neck
  - [-1, 1, Conv, [512, 3, 2]]
  - [[-1, 10], 1, Concat, [1]]
  - [-1, 1, C3k2, [1024, True, 0.5, True]] # P5 neck
  - [[20, 23, 26, 29], 1, Detect, [nc]]
```

## 3. Cấu hình huấn luyện paired

Hai lần chạy khai báo cùng seed, train settings, augmentations và hardware; khác tên run và graph model.

```yaml
seed: 42
train:
  epochs: 150
  patience: 0
  val: false
  imgsz: 1280
  batch: 8
  workers: 8
  optimizer: AdamW
  lr0: 0.002
  lrf: 0.1
  cos_lr: true
  warmup_epochs: 5.0
  amp: true
  deterministic: false
  degrees: 0.0
  translate: 0.05
  scale: 0.15
  shear: 0.0
  perspective: 0.0
  flipud: 0.0
  fliplr: 0.5
  hsv_h: 0.005
  hsv_s: 0.1
  hsv_v: 0.15
  bgr: 0.0
  mosaic: 0.0
  close_mosaic: 10
  mixup: 0.0
  cutmix: 0.0
  copy_paste: 0.0
  copy_paste_mode: flip
  multi_scale: 0.0
  compile: false
hardware:
  gpu: A10
```

| Run | Model graph | Run name |
|---|---|---|
| Baseline | YOLO26n-P2 | `baseline_tinyperson_wacv` |
| DCAF v2.1 | YOLO26n-P2 + DCAF | `dcaf_v2_1_tinyperson_wacv` |

Dataset YAML runtime path: `/mnt/data/tinyperson-yolo/data.yaml`. Evaluation JSONs trỏ tới `training/weights/last.pt`, không phải `best.pt`. `deterministic=false`; seed không bảo đảm tái lập bitwise. Raw run metadata/config hash chưa được đối chiếu trong hồ sơ này.

## 4. Dataset và protocol

- Test artifact: 786 ảnh, 13,687 positive ground truths; JSON khai báo dense-image exclusions = 0.
- Size bins: Tiny 2–20 px; Tiny1 2–8 px; Tiny2 8–12 px; Tiny3 12–20 px; Small 20–32 px; Reasonable >32 px.
- Tiled: crop 640×512, NMS merge IoU 0.50; tối đa 200 detections/ảnh; IoU thresholds 0.25/0.50/0.75.
- Full-frame: inference ảnh gốc ở imgsz 1280.
- Evaluator là standalone implementation; tương đương reference/official protocol chưa xác minh. MR được chép đúng theo artifact standalone, không diễn giải như official MR.
- Báo cáo riêng validation, single-frame run khác, PR/F1 diagnostics và profiling; không trộn chúng vào paired test chính.

## 5. Kết quả chính: tiled 150 epoch

AP/MR theo phần trăm; Δ tính DCAF trừ baseline, đơn vị điểm phần trăm (pp). Cùng protocol khai báo; mỗi phương pháp một lần chạy.

### AP

| Size bin | GT | AP@0.25 Base | DCAF | Δ pp | AP@0.50 Base | DCAF | Δ pp | AP@0.75 Base | DCAF | Δ pp |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| All | 13,687 | 21.2816 | 20.6751 | −0.6067 | 11.9538 | 12.9360 | +0.9822 | 1.2834 | 1.3973 | +0.1139 |
| Tiny | 8,878 | 15.9616 | 15.4235 | −0.5382 | 9.0711 | 9.7583 | +0.6872 | 0.7652 | 1.1967 | +0.4315 |
| Tiny1 | 2,333 | 9.6772 | 8.7564 | −0.9204 | 4.7164 | 5.1776 | +0.4612 | 0.4143 | 1.1080 | +0.6937 |
| Tiny2 | 2,778 | 13.1955 | 14.4023 | +1.2068 | 6.9036 | 9.4962 | +2.5926 | 1.1692 | 0.9531 | −0.2162 |
| Tiny3 | 3,767 | 22.0455 | 20.7277 | −1.3177 | 13.5084 | 13.3721 | −0.1363 | 1.2073 | 1.5648 | +0.3575 |
| Small | 2,485 | 28.6015 | 28.9236 | +0.3229 | 17.4276 | 19.2287 | +1.8011 | 1.7905 | 2.3549 | +0.5644 |
| Reasonable | 2,324 | 36.0402 | 34.0618 | −1.9785 | 20.4852 | 21.2033 | +0.7181 | 2.3582 | 2.0370 | −0.3213 |

### Miss Rate

| Size bin | MR@0.25 Base | DCAF | Δ pp | MR@0.50 Base | DCAF | Δ pp | MR@0.75 Base | DCAF | Δ pp |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| All | 91.5559 | 92.9544 | +1.3985 | 96.9346 | 96.7293 | −0.2054 | 99.5324 | 99.5486 | +0.0162 |
| Tiny | 91.5559 | 92.9539 | +1.3980 | 95.8987 | 95.8712 | −0.0275 | 99.4168 | 99.2891 | −0.1277 |
| Tiny1 | 92.4751 | 93.7229 | +1.2477 | 95.8327 | 96.0089 | +0.1762 | 99.4189 | 99.2332 | −0.1858 |
| Tiny2 | 91.1647 | 90.4368 | −0.7279 | 95.0964 | 93.5245 | −1.5719 | 99.2481 | 98.8521 | −0.3960 |
| Tiny3 | 84.9925 | 87.8713 | +2.8788 | 92.1098 | 92.5699 | +0.4601 | 98.6844 | 98.5488 | −0.1356 |
| Small | 79.8301 | 80.5231 | +0.6930 | 89.9844 | 88.7279 | −1.2565 | 98.0371 | 97.6481 | −0.3890 |
| Reasonable | 78.3324 | 80.8615 | +2.5292 | 89.2427 | 89.0945 | −0.1482 | 97.6573 | 98.0923 | +0.4350 |

### AP trung bình ba IoU

| Size bin | Baseline | DCAF v2.1 | Δ pp |
|---|---:|---:|---:|
| All | 11.5064 | 11.6695 | +0.1631 |
| Tiny | 8.5993 | 8.7928 | +0.1935 |
| Tiny1 | 4.9360 | 5.0140 | +0.0780 |
| Tiny2 | 7.0895 | 8.2839 | +1.1944 |
| Tiny3 | 12.2537 | 11.8882 | −0.3655 |
| Small | 15.9399 | 16.8357 | +0.8958 |
| Reasonable | 19.6279 | 19.1007 | −0.5272 |

**Diễn giải:** AP@0.50 cao hơn ở 5/7 hàng (All và sáu size bins), mức tăng lớn nhất ở Tiny2 (+2.5926 pp). AP@0.25 giảm ở All/Tiny/Tiny1/Tiny3/Reasonable; AP@0.75 và MR biến thiên theo bin. Chỉ số AP trung bình ba IoU tăng nhẹ ở All (+0.1631 pp), Tiny2 tăng nhiều nhất (+1.1944 pp). Đây là khác biệt mô tả, không có CI hay kiểm định.

## 6. Kết quả full-frame 150 epoch

Ảnh 1280, 786 ảnh, 13,687 GT; cùng provenance caveat về checkpoint/evaluator.

| Size bin | GT | AP@0.25 Base | DCAF | AP@0.50 Base | DCAF | AP@0.75 Base | DCAF | MR@0.50 Base | DCAF |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| All | 13,687 | 20.6517 | 20.1976 | 12.4307 | 12.5274 | 1.9644 | 1.3064 | 97.1563 | 97.0369 |
| Tiny | 8,878 | 16.7814 | 15.7878 | 9.9717 | 9.7606 | 0.9790 | 1.4696 | 95.7410 | 96.0665 |
| Tiny1 | 2,333 | 10.7923 | 9.4267 | 5.6587 | 5.7922 | 0.6372 | 0.6195 | 95.1803 | 95.2374 |
| Tiny2 | 2,778 | 16.2780 | 16.2430 | 9.8958 | 9.6007 | 1.2882 | 0.7382 | 93.0766 | 93.4605 |
| Tiny3 | 3,767 | 20.9112 | 19.7377 | 12.8254 | 12.5304 | 1.1971 | 1.7670 | 92.8885 | 93.3398 |
| Small | 2,485 | 25.6971 | 24.8648 | 15.8047 | 16.3505 | 2.0742 | 1.6476 | 91.2229 | 90.8071 |
| Reasonable | 2,324 | 34.0570 | 34.2579 | 20.4853 | 21.7051 | 2.3582 | 2.7506 | 89.2427 | 89.7160 |

All AP@0.50 +0.0968 pp; All AP@0.75 −0.6579 pp. Tiny AP@0.50 −0.2111 pp; results therefore do not support a broad full-frame improvement claim.

## 7. Kết quả bổ sung, không gộp với benchmark chính

- **Single-frame run khác (WACV-named):** all AP@0.50 baseline 8.74753%, DCAF 9.09815%; imgsz1280, 786 images. Run/checkpoint khác với paired 150e full-frame, không thay thế được kết quả ở mục 6.
- **Validation derivative:** artifact ghi AP@0.50 26.69418% (all), 27.79618% (tiny), 84 ảnh/3,608 GT, checkpoint `best.pt`. Cycle validation khác ghi AP 8.00971%, AP50 26.17159%, AP75 2.20290%, precision 40.44691%, recall 31.51330%. Không so trực tiếp với standalone test.
- **PR/F1 operating-point diagnostics:** all max F1 baseline 45.36, DCAF 44.08; precision tại max F1 51.55/50.80; recall 40.49/38.93; max recall 71.62/70.94. Max F1 theo nhóm: Tiny 42.68/40.11; Tiny1 34.19/29.12; Tiny2 42.39/40.16; Tiny3 47.96/47.35; Small 52.62/53.29; Reasonable 49.22/49.19 (baseline/DCAF). Đây không phải AP.
- **Density recall operating point:** sparse 74.93/76.85%; medium 77.71/76.87%; crowded 66.94/65.35% (baseline/DCAF). Không diễn giải là AP.
- **Profiling NVIDIA A10G, methodology chưa rõ:** params 2,516,240/2,550,906; FLOPs 31.97/33.99 GFLOPs ở 1280 và 7.62/8.13 GFLOPs ở 640; full-frame latency 11.85/16.58 ms, 84.4/60.3 FPS; tiled latency 11.28/15.19 ms, 88.6/65.8 FPS; peak VRAM full 244.65/367.16 MB, tiled 91.51/122.10 MB (baseline/DCAF). Timing narrative khác artifact raw; không dùng để khẳng định tốc độ triển khai nếu chưa kiểm tra phương pháp đo.
- Cycle 2 được khởi tạo từ best checkpoint Cycle 1; đây không phải hai seed/run độc lập. Validation và geometry diagnostics của các cycle không phải paired 150e test.

## 8. Kết luận được phép và giới hạn

Trong artifact tiled đơn seed, DCAF v2.1 có AP@0.50 cao hơn baseline 0.9822 pp toàn bộ và 2.5926 pp ở Tiny2; AP trung bình ba IoU toàn bộ chỉ tăng 0.1631 pp. Full-frame AP@0.50 toàn bộ tăng 0.0968 pp, trong khi AP@0.75 giảm 0.6579 pp. Đây là kết quả phụ thuộc protocol/metric, không chứng minh cải thiện phổ quát, ý nghĩa thống kê, hay ưu thế so với phương pháp ngoài.

Giới hạn cần ghi rõ trong bất kỳ bài viết nào:

1. Một seed mỗi phương pháp; không có mean/std, CI hoặc paired significance test.
2. Test artifacts đánh giá `last.pt`; không đổi nhãn thành `best.pt`.
3. Evaluator standalone; 786 ảnh, dense exclusions khai báo 0; equivalence benchmark chính thức chưa xác minh.
4. Chưa có ablation tách H-BRA, BSG, Gaussian scale-space hoặc residual gain.
5. Raw run metadata/config hash chưa xác minh rằng các JSON test khớp chính xác YAML paired.
6. Profiling methodology còn thiếu; không ngoại suy latency/VRAM sang edge hardware.
7. Chưa có tài liệu related work/citation ngoài được kiểm chứng trong hồ sơ này; không dùng tuyên bố SOTA.

## 9. Ghi chú giải thích kết quả

Kết quả 150e tiled và full-frame được giữ thành hai protocol riêng. Validation, single-frame WACV-named run, F1/density operating points, profiling và Cycle 1/2 không phải phép đo thay thế cho hai bảng test paired. Khi viết báo cáo, trình bày cả kết quả tăng lẫn giảm và coi cơ chế kiến trúc là giả thuyết thiết kế cho tới khi có ablation nhân quả.
