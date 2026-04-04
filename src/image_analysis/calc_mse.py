from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.typing as npt
from PIL import Image

from image_analysis.fft_trans import DoGSubbandFilter, SIGMA_SCALES


def load_img_gray_float(path: Path) -> npt.NDArray[np.float64]:
    """Load image as grayscale float64 array."""
    img = Image.open(path).convert("L")
    return np.asarray(img).astype(np.float64)


def calc_dog_subbands(
    img: npt.NDArray[np.float64], sigma_list: list[float] | None = None
) -> list[npt.NDArray[np.float64]]:
    """Apply DoG subband filter and return dog_subbands."""
    f = DoGSubbandFilter(img, sigma_list)
    f.process()
    return f.dog_subbands


def calculate_mse_dog_subbands(img_path1: Path, img_path2: Path, sigma_list: list[float] | None = None) -> list[float]:
    """Compute MSE for each DoG subband between two images.

    Args:
        img_path1: Path to the first image.
        img_path2: Path to the second image.
        sigma_list: Sigma values passed to DoGSubbandFilter. Defaults to SIGMA_SCALES.

    Returns:
        List of MSE values, one per DoG subband.

    Raises:
        ValueError: If the number of subbands differ between the two images.
    """
    img1 = load_img_gray_float(img_path1)
    img2 = load_img_gray_float(img_path2)

    subbands1 = calc_dog_subbands(img1, sigma_list)
    subbands2 = calc_dog_subbands(img2, sigma_list)

    if len(subbands1) != len(subbands2):
        msg = f"Number of subbands differ: {len(subbands1)} vs {len(subbands2)}"
        raise ValueError(msg)

    mse_list: list[float] = []
    for sb1, sb2 in zip(subbands1, subbands2, strict=False):
        if sb1.shape != sb2.shape:
            msg = f"Subband shapes do not match: {sb1.shape} vs {sb2.shape}"
            raise ValueError(msg)
        mse = float(np.mean((sb1 - sb2) ** 2))
        mse_list.append(mse)

    return mse_list


def main() -> None:
    sigma_list = SIGMA_SCALES
    img_path1 = Path("tests/test_data/5dmark3_iso3200_1_mean.png")
    img_path2 = Path("tests/test_data/5dmark3_iso3200_1_real.png")

    mse_values = calculate_mse_dog_subbands(img_path1, img_path2, sigma_list)
    for i, mse in enumerate(mse_values):
        if i == 0:
            print(f"MSE subband {i} (sigma={sigma_list[0]:.1f}-original): {mse:.6f}")
        elif i < len(sigma_list):
            print(f"MSE subband {i} (sigma={sigma_list[i]:.1f}-{sigma_list[i - 1]:.1f}): {mse:.6f}")
        else:
            print(f"MSE subband {i} (sigma={sigma_list[-1]:.1f}): {mse:.6f}")


if __name__ == "__main__":
    main()
