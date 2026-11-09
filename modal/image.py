from pathlib import Path

import modal

# CUDA 13.2 wheels: Modal's host driver (610.57.04, CUDA driver API 13.3)
# supports cu132, and cu132 carries the newest torch/torchvision pairing
# (torch 2.15.0 + torchvision 0.29.2) — cu128 tops out at torch 2.11.0, which
# skews three minors behind the local CPU environment (torch 2.14.0).
# A10 = Ampere (sm_86); CUDA 13 only dropped Maxwell/Pascal/Volta.
base_image = (
    modal.Image.debian_slim(python_version="3.14").apt_install("libgl1", "libglib2.0-0")
    .uv_pip_install("torch>=2.15,<3", "torchvision>=0.29,<1", index_url="https://download.pytorch.org/whl/cu132")
    .uv_pip_install(
        "ultralytics==8.4.208", "numpy>=2.5,<3", "pandas>=3.1,<4", "PyYAML>=6,<7",
        "Pillow>=13,<14", "opencv-python-headless>=4.10,<5", "matplotlib>=3.11,<4",
        "wandb>=0.31,<1", "scikit-learn>=1.9,<2", "polars>=2.1,<3",
    )
)

local_files = ("tinydet", "image", "paths", "app")
config_dir = str(Path(__file__).resolve().parents[1] / "configs")
scripts_dir = str(Path(__file__).resolve().parents[1] / "scripts")

image = base_image.add_local_python_source(*local_files, copy=False).add_local_dir(
    config_dir, "/root/configs", copy=False
).add_local_dir(scripts_dir, "/root/scripts", copy=False)
