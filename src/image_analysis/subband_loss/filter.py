import torch
from torch import nn

SIGMA_SCALES = [0.6 * (2**i) for i in range(5)]


class MultiScaleDogFilter(nn.Module):
    def __init__(self, scales: list[float] | None = None, kernel_size: int = 5) -> None:
        """Maltui-scale Difference of Gaussian (DoG) filter implemented as a PyTorch module.

        Args:
            scales: List of sigma values for Gaussian filters.
                    DoG subbands are computed as differences between consecutive Gaussians.
                    Default is [0.6, 1.2, 2.4, 4.8, 9.6].
            kernel_size: Size of the Gaussian kernel (must be odd).

        if scales=[1.0, 2.0, 4.0, 8.0], the subbands are:
            - Filter1: Original image - G(1.0)
            - Filter2: G(1.0) - G(2.0)
            - Filter3: G(2.0) - G(4.0)
            - Filter4: G(4.0) - G(8.0)
            - Filter5: G(8.0)

        Raises:
            ValueError: If kernel_size is not odd or if any sigma value is non-positive.
            ValueError: If less than 2 sigma values are provided.
        """
        super().__init__()
        if kernel_size % 2 == 0:
            msg = "Kernel size must be odd."
            raise ValueError(msg)
        self.kernel_size = kernel_size
        self.scales = scales or SIGMA_SCALES
        if any(s <= 0 for s in self.scales):
            msg = "Sigma values must be positive."
            raise ValueError(msg)

        self.gaussian_kernels: list[torch.Tensor] = []
        self.paddings: list[int] = []

        for sigma in self.scales:
            gaussian_kernel = self.create_gaussian_kernel(sigma)
            self.gaussian_kernels.append(gaussian_kernel)
            self.paddings.append(self.kernel_size // 2)

    def create_gaussian_kernel(self, sigma: float) -> torch.Tensor:
        """Create a 2D Gaussian kernel.

        Args:
            sigma: Standard deviation of the Gaussian

        Returns:
            2D Gaussian kernel tensor of shape (1, kernel_size, kernel_size)
        """
        ax = torch.arange(-self.kernel_size // 2 + 1.0, self.kernel_size // 2 + 1.0)
        xx, yy = torch.meshgrid(ax, ax, indexing="ij")

        gaussian = torch.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
        gaussian /= gaussian.sum()
        return gaussian.unsqueeze(0)

    def _apply_gaussian_filter(self, img: torch.Tensor, gaussian_kernel: torch.Tensor, padding: int) -> torch.Tensor:
        """Apply Gaussian filter to the input image.

        Args:
            img: Input image tensor of shape (B, C, H, W)
            gaussian_kernel: 2D Gaussian kernel tensor of shape (1, kernel_size, kernel_size)
            padding: Padding size for convolution

        Returns:
            Filtered image tensor of shape (B, C, H, W)
        """
        num_channels = img.size(1)
        kernel = gaussian_kernel.to(device=img.device, dtype=img.dtype).unsqueeze(1)
        kernel_expanded = kernel.expand(num_channels, 1, self.kernel_size, self.kernel_size)
        filtered = nn.functional.conv2d(img, kernel_expanded, padding=padding, groups=num_channels)
        return filtered

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """Apply DoG filter to the input image and return subbands.

        Args:
            img: Input image tensor of shape (B, C, H, W)

        Returns:
            DoG subbands tensor of shape (B, len(scales) + 1, C, H, W)
        """
        gaussians = [img]

        for gaussian_kernel, padding in zip(self.gaussian_kernels, self.paddings, strict=False):
            filtered = self._apply_gaussian_filter(img, gaussian_kernel, padding)
            gaussians.append(filtered)

        subbands = [gaussians[i + 1] - gaussians[i] for i in range(len(gaussians) - 1)]
        # Add the lowest frequency residual as the last subband
        subbands.append(gaussians[-1])

        return torch.stack(subbands, dim=1)
