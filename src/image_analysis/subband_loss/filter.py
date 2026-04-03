import math

import torch
import torch.nn.functional as f
from torch import nn

SIGMA_SCALES = [0.6 * (2**i) for i in range(6)]
# Example scales: [0.6, 1.2, 2.4, 4.8, 9.6, 19.2]


class MultiScaleDogFilter(nn.Module):
    def __init__(self, scales: list[float] | None = None, kernel_size: int | None = None) -> None:
        """Maltui-scale Difference of Gaussian (DoG) filter implemented as a PyTorch module.

        Args:
            scales: List of sigma values for Gaussian filters.
                    DoG subbands are computed as differences between consecutive Gaussians.
                    Default is [0.6, 1.2, 2.4, 4.8, 9.6, 19.2].
            kernel_size: If None, kernel size is computed per sigma as 2*ceil(3*sigma)+1 (minimum 7).
                        If int, it is used as a fixed size (must be odd). Deprecated in favor of sigma-dependent sizing.

        if scales=[1.0, 2.0, 4.0, 8.0], the subbands are:
            - Filter1: Original image - G(1.0)
            - Filter2: G(1.0) - G(2.0)
            - Filter3: G(2.0) - G(4.0)
            - Filter4: G(4.0) - G(8.0)
            - Filter5: G(8.0)

        Raises:
            ValueError: If any sigma value is non-positive.
            ValueError: If less than 2 sigma values are provided.
        """
        super().__init__()
        self.scales = scales or SIGMA_SCALES
        if any(s <= 0 for s in self.scales):
            msg = "Sigma values must be positive."
            raise ValueError(msg)

        self.gaussian_kernels: list[torch.Tensor] = []
        self.kernel_sizes: list[int] = []

        for sigma in self.scales:
            # Compute kernel size based on sigma if not provided
            if kernel_size is None:
                ks = max(7, 2 * math.ceil(3.0 * sigma) + 1)
            else:
                ks = kernel_size
                if ks % 2 == 0:
                    msg = "Kernel size must be odd."
                    raise ValueError(msg)
            self.kernel_sizes.append(ks)
            gaussian_kernel = self.create_gaussian_kernel(sigma, ks)
            self.gaussian_kernels.append(gaussian_kernel)

    @staticmethod
    def create_gaussian_kernel(sigma: float, kernel_size: int) -> torch.Tensor:
        """Create a 2D Gaussian kernel.

        Args:
            sigma: Standard deviation of the Gaussian
            kernel_size: Size of the kernel (must be odd)

        Returns:
            2D Gaussian kernel tensor of shape (1, kernel_size, kernel_size)
        """
        ax = torch.arange(-kernel_size // 2 + 1.0, kernel_size // 2 + 1.0)
        xx, yy = torch.meshgrid(ax, ax, indexing="ij")

        gaussian = torch.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
        gaussian /= gaussian.sum()
        return gaussian.unsqueeze(0)

    @staticmethod
    def _apply_gaussian_filter(img: torch.Tensor, gaussian_kernel: torch.Tensor, kernel_size: int) -> torch.Tensor:
        """Apply Gaussian filter to the input image with reflect padding.

        Args:
            img: Input image tensor of shape (B, C, H, W)
            gaussian_kernel: 2D Gaussian kernel tensor of shape (1, kernel_size, kernel_size)
            kernel_size: Size of the kernel for padding calculation

        Returns:
            Filtered image tensor of shape (B, C, H, W)
        """
        num_channels = img.size(1)
        # Apply reflect padding to avoid boundary artifacts
        pad_size = kernel_size // 2
        img_padded = f.pad(img, (pad_size, pad_size, pad_size, pad_size), mode="reflect")

        kernel = gaussian_kernel.to(device=img.device, dtype=img.dtype).unsqueeze(1)
        kernel_expanded = kernel.expand(num_channels, 1, kernel_size, kernel_size)
        # Use padding=0 since we already applied reflect padding
        filtered = f.conv2d(img_padded, kernel_expanded, padding=0, groups=num_channels)
        return filtered

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """Apply DoG filter to the input image and return subbands.

        Args:
            img: Input image tensor of shape (B, C, H, W)

        Returns:
            DoG subbands tensor of shape (B, len(scales) + 1, C, H, W)
        """
        gaussians = [img]

        for gaussian_kernel, kernel_size in zip(self.gaussian_kernels, self.kernel_sizes, strict=False):
            filtered = self._apply_gaussian_filter(img, gaussian_kernel, kernel_size)
            gaussians.append(filtered)

        subbands = [gaussians[i + 1] - gaussians[i] for i in range(len(gaussians) - 1)]
        # Add the lowest frequency residual as the last subband
        subbands.append(gaussians[-1])

        return torch.stack(subbands, dim=1)
