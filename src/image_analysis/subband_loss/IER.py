import torch
from torch import nn

from image_analysis.subband_loss.filter import MultiScaleDogFilter

# In paper, input is noisy image, output is denoised image, target is clean image
# In pytorch, input is predicted image, target is ground truth image
# input, target: (B, C, H, W)


SIGMA_SCALES = [0.6 * (2**i) for i in range(5)]


class SFLLoss(nn.Module):
    def __init__(self, predicted: torch.Tensor, target: torch.Tensor, scales: list[float] | None = None) -> None:
        super().__init__()
        self.predicted = predicted
        self.target = target
        # Output MSE per subband
        self.L2loss_fn = nn.MSELoss(reduction="none")
        self.dog_bunk = MultiScaleDogFilter(scales=scales)
        self.batch_num = self.predicted.size(0)
        self.scales = scales or SIGMA_SCALES

    def e_sfl(self) -> torch.Tensor:
        """Compute the E_SFL between the denoised and noisy images."""
        predicted_subbands = self.dog_bunk(self.predicted)
        target_subbands = self.dog_bunk(self.target)
        return self.L2loss_fn(predicted_subbands, target_subbands).mean(dim=(2, 3, 4))

    def sfl_loss(self) -> torch.Tensor:
        """Compute the final SFL loss from the E_SFL."""
        l_sfl = self.w_sfl() * self.e_sfl()
        return l_sfl.mean()

    def w_sfl(self) -> torch.Tensor:
        """Compute the weight for the SFL loss based on the number of subbands."""
        return self.ier().mean(dim=0)

    def ier(self) -> torch.Tensor:
        e_l2 = self.L2loss_fn(self.predicted, self.target).mean(dim=(1, 2, 3))
        return e_l2.unsqueeze(1) / self.e_sfl()
