import torch
from torch import nn

from image_analysis.subband_loss.filter import MultiScaleDogFilter

# In paper, input is noisy image, output is denoised image, target is clean image
# In pytorch, input is predicted image, target is ground truth image
# input, target: (B, C, H, W)


SIGMA_SCALES = [0.6 * (2**i) for i in range(5)]


class SFLLoss(nn.Module):
    def __init__(self, scales: list[float] | None = None) -> None:
        super().__init__()
        # Output MSE per subband
        self.L2loss_fn = nn.MSELoss(reduction="none")
        self.dog_bunk = MultiScaleDogFilter(scales=scales)
        self.scales = scales or SIGMA_SCALES

    def e_sfl(self, predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute the E_SFL between the denoised and noisy images."""
        predicted_subbands = self.dog_bunk(predicted)
        target_subbands = self.dog_bunk(target)
        return self.L2loss_fn(predicted_subbands, target_subbands).mean(dim=(2, 3, 4))

    def sfl_loss(self, predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute the final SFL loss from the E_SFL."""
        l_sfl = self.w_sfl(predicted, target) * self.e_sfl(predicted, target)
        return l_sfl.mean()

    def w_sfl(self, predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute the weight for the SFL loss based on the number of subbands."""
        return self.ier(predicted, target).mean(dim=0)

    def ier(self, predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        e_l2 = self.L2loss_fn(predicted, target).mean(dim=(1, 2, 3))
        return e_l2.unsqueeze(1) / self.e_sfl(predicted, target)

    def forward(self, predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.sfl_loss(predicted, target)
