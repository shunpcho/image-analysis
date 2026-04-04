from pathlib import Path

import numpy as np
import torch
from PIL import Image

from image_analysis.subband_loss.filter import MultiScaleDogFilter
from image_analysis.subband_loss.IER import SFLLoss


def _format_subband_label(index: int, sigma_list: list[float]) -> str:
    if index == 0:
        return f"sigma={sigma_list[0]:.1f}-original"
    if index < len(sigma_list):
        return f"sigma={sigma_list[index]:.1f}-{sigma_list[index - 1]:.1f}"
    return f"sigma={sigma_list[-1]:.1f}"


def _load_grayscale_tensor(path: Path) -> torch.Tensor:
    img = Image.open(path).convert("L")
    arr = np.asarray(img, dtype=np.uint8)
    return torch.tensor(arr, dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 255.0
    # return torch.tensor(arr, dtype=torch.float32).unsqueeze(0).unsqueeze(0)


def test_print_subband_mse_like_calc_mse() -> None:
    sigma_list = [0.6 * (2**i) for i in range(6)]  # Example scales: [0.6, 1.2, 2.4, 4.8, 9.6, 19.2]

    data_dir = Path(__file__).parent / "test_data"
    predicted = _load_grayscale_tensor(data_dir / "5dmark3_iso3200_1_mean.png")
    target = _load_grayscale_tensor(data_dir / "5dmark3_iso3200_1_real.png")

    assert predicted.shape == target.shape

    # Use the loss implementation from IER.py to compute per-subband MSE.
    loss_fn = SFLLoss(scales=sigma_list)
    mse_per_subband = loss_fn.e_sfl(predicted, target).squeeze(0)

    # Also use MultiScaleDogFilter directly to verify e_sfl numerically.
    dog_filter = MultiScaleDogFilter(scales=sigma_list)
    predicted_subbands = dog_filter(predicted)
    target_subbands = dog_filter(target)
    manual_mse = ((predicted_subbands - target_subbands) ** 2).mean(dim=(2, 3, 4)).squeeze(0)

    assert mse_per_subband.shape[0] == len(sigma_list) + 1
    assert torch.all(mse_per_subband >= 0)
    assert torch.allclose(mse_per_subband, manual_mse, atol=1e-6)

    for i in range(mse_per_subband.shape[0]):
        mse = float(mse_per_subband[i].item())
        print(f"MSE subband {i} ({_format_subband_label(i, sigma_list)}): {mse}")
