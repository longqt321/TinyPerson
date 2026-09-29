from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def save_frequency(feature_map: np.ndarray, output: Path) -> None:
    spectrum = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(feature_map))))
    output.mkdir(parents=True, exist_ok=True)
    np.save(output / "spectrum.npy", spectrum)
    plt.imsave(output / "spectrum.png", spectrum, cmap="magma")
