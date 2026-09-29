from pathlib import Path

import modal

image = (
    modal.Image.debian_slim(python_version="3.12").apt_install("libgl1", "libglib2.0-0")
    .uv_pip_install("torch>=2.7,<3", "torchvision>=0.22,<1", index_url="https://download.pytorch.org/whl/cu128")
    .uv_pip_install("ultralytics>=8.4,<9", "numpy>=1.26,<3", "pandas>=2.2,<3", "PyYAML>=6,<7", "Pillow>=10,<13", "opencv-python-headless>=4.10,<5", "matplotlib>=3.9,<4")
    .add_local_python_source("tinydet", "image", "paths", copy=False)
    .add_local_dir(str(Path(__file__).resolve().parents[1] / "configs"), "/root/configs", copy=False)
)
