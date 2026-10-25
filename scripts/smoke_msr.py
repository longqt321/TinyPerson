"""Run one YOLO26 criterion/optimizer step through the P2 MSR model."""
import argparse
from pathlib import Path

import torch
from ultralytics.utils import DEFAULT_CFG

from tinydet.models.registry import build_model
from tinydet.modules.msr import MSR


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="configs/model/yolo26n_tiny_A2_p2_msr.yaml")
    parser.add_argument("--checkpoint", type=Path, default=Path("/tmp/msr-smoke.pt"))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wrapper = build_model({"config": args.model})
    model = wrapper.model.to(device).train()
    model.args = DEFAULT_CFG
    batch = {
        "img": torch.rand(2, 3, 64, 64, device=device),
        "batch_idx": torch.tensor([0.0, 1.0], device=device),
        "cls": torch.zeros(2, 1, device=device),
        "bboxes": torch.tensor([[0.5, 0.5, 0.15, 0.15], [0.4, 0.6, 0.1, 0.12]], device=device),
    }
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    amp = device.type == "cuda"
    before = model.model[29].refinement.gamma.detach().clone()
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast("cuda", enabled=amp):
        loss, _ = model.loss(batch)
    loss.sum().backward()
    if not torch.isfinite(loss).all() or any(
        p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()
    ):
        raise RuntimeError("Non-finite loss or gradient")
    optimizer.step()
    if torch.equal(before, model.model[29].refinement.gamma.detach()):
        raise RuntimeError("MSR residual scale did not update")

    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.checkpoint)
    restored = build_model({"config": args.model})
    restored.model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True))
    restored.model.eval()
    result = restored.predict(torch.zeros(1, 3, 64, 64), device="cpu", verbose=False)
    if not isinstance(restored.model.model[29], MSR) or len(result) != 1:
        raise RuntimeError("Checkpoint reload or prediction smoke failed")
    print(f"PASS: {device} criterion/backward/optimizer/checkpoint/predict; loss={loss.tolist()}")
    if not torch.cuda.is_available():
        print("CUDA AMP check skipped: CUDA is unavailable")


if __name__ == "__main__":
    main()
