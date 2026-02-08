from pathlib import Path

import numpy as np
import numpy.typing as npt
from PIL import Image
from scipy import ndimage


class DoGSubbandFilter:
    """Class for creating subband images using Difference of Gaussian (DoG) filters."""

    def __init__(
        self,
        image_path: Path | str,
        sigma_list: list[float] | None = None,
        output_dir: Path | str = "output",
    ) -> None:
        """Initialize the DoG subband filter.

        Args:
            image_path: Path to the input image
            sigma_list: List of sigma values for Gaussian filters.
                       DoG subbands are computed as differences between consecutive Gaussians.
                       Default is [1.0, 2.0, 4.0, 8.0] which creates 3 DoG subbands plus
                       the lowest frequency residual.
            output_dir: Directory to save output images
        """
        self.image_path = Path(image_path)
        self.sigma_list = sigma_list or [1.0, 2.0, 4.0, 8.0]
        self.img: npt.NDArray[np.float64] | None = None
        self.gaussian_pyramid: list[npt.NDArray[np.float64]] = []
        self.dog_subbands: list[npt.NDArray[np.float64]] = []
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_image(self) -> None:
        """Load image and convert to grayscale float."""
        img_uint8 = np.array(Image.open(self.image_path).convert("L"))
        self.img = img_uint8.astype(np.float64)

    def create_gaussian_pyramid(self) -> None:
        """Create Gaussian pyramid by applying different sigma values.

        Raises:
            ValueError: If image is not loaded before calling this method.
        """
        if self.img is None:
            msg = "Image must be loaded before creating Gaussian pyramid"
            raise ValueError(msg)

        self.gaussian_pyramid = []
        for sigma in self.sigma_list:
            gaussian_img: npt.NDArray[np.float64] = ndimage.gaussian_filter(self.img, sigma=sigma)  # type: ignore[assignment]
            self.gaussian_pyramid.append(gaussian_img)

    def compute_dog_subbands(self) -> None:
        """Compute DoG (Difference of Gaussian) subbands.

        Raises:
            ValueError: If less than 2 Gaussian images are available.
        """
        min_gaussian_images = 2
        if len(self.gaussian_pyramid) < min_gaussian_images:
            msg = "At least 2 Gaussian images are needed to compute DoG"
            raise ValueError(msg)

        self.dog_subbands = []
        for i in range(len(self.gaussian_pyramid) - 1):
            dog = self.gaussian_pyramid[i] - self.gaussian_pyramid[i + 1]
            self.dog_subbands.append(dog)

    @staticmethod
    def normalize_image(img_array: npt.NDArray[np.float64]) -> npt.NDArray[np.uint8]:
        """Normalize image array to 0-255 range.

        Args:
            img_array: Input image array

        Returns:
            Normalized uint8 array
        """
        # Handle negative values (common in DoG)
        img_normalized = img_array - img_array.min()
        if img_normalized.max() > 0:
            img_normalized = img_normalized / img_normalized.max() * 255
        return img_normalized.astype(np.uint8)

    @staticmethod
    def compute_fft_magnitude(img_array: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Compute FFT magnitude spectrum of an image.

        Args:
            img_array: Input image array

        Returns:
            FFT magnitude spectrum (shifted to center)
        """
        fft = np.fft.fft2(img_array)
        fft_shift = np.fft.fftshift(fft)
        magnitude = np.abs(fft_shift)
        return magnitude

    @staticmethod
    def normalize_spectrum(magnitude: npt.NDArray[np.float64]) -> npt.NDArray[np.uint8]:
        """Normalize magnitude spectrum with log scaling.

        Args:
            magnitude: FFT magnitude array

        Returns:
            Normalized uint8 array
        """
        magnitude_normalized = np.log(magnitude + 1)
        if magnitude_normalized.max() > 0:
            magnitude_normalized = magnitude_normalized / magnitude_normalized.max() * 255
        return magnitude_normalized.astype(np.uint8)

    def save_gaussian_pyramid(self) -> None:
        """Save all Gaussian pyramid images."""
        for idx, gaussian_img in enumerate(self.gaussian_pyramid):
            img_normalized = self.normalize_image(gaussian_img)
            Image.fromarray(img_normalized).save(self.output_dir / f"gaussian_sigma_{self.sigma_list[idx]:.1f}.png")

    def save_dog_subbands(self) -> None:
        """Save all DoG subband images and their FFT magnitude spectrums."""
        for idx, dog_img in enumerate(self.dog_subbands):
            # Save normalized DoG (with sign information)
            img_normalized = self.normalize_image(dog_img)
            Image.fromarray(img_normalized).save(
                self.output_dir
                / f"dog_subband_{idx}_sigma_{self.sigma_list[idx]:.1f}-{self.sigma_list[idx + 1]:.1f}.png"
            )

            # Save FFT magnitude spectrum
            fft_magnitude = self.compute_fft_magnitude(dog_img)
            magnitude_normalized = self.normalize_spectrum(fft_magnitude)
            Image.fromarray(magnitude_normalized).save(
                self.output_dir
                / f"dog_fft_magnitude_{idx}_sigma_{self.sigma_list[idx]:.1f}-{self.sigma_list[idx + 1]:.1f}.png"
            )

    def save_residual(self) -> None:
        """Save the residual (lowest frequency component)."""
        if len(self.gaussian_pyramid) > 0:
            residual = self.gaussian_pyramid[-1]
            img_normalized = self.normalize_image(residual)
            Image.fromarray(img_normalized).save(self.output_dir / f"residual_sigma_{self.sigma_list[-1]:.1f}.png")

    def process(self) -> None:
        """Execute the complete DoG filtering process."""
        # Load and process image
        self.load_image()
        self.create_gaussian_pyramid()
        self.compute_dog_subbands()

        # Save results
        self.save_gaussian_pyramid()
        self.save_dog_subbands()
        self.save_residual()

        print("Saved images")
        print(f"Sigma values: {self.sigma_list}")


if __name__ == "__main__":
    image_path = "path/to/your/image.jpg"

    # Example 1: Default 4 sigma values (creates 3 DoG subbands)
    # dog_filter = DoGSubbandFilter(image_path)

    # Example 2: Custom sigma values (creates 4 DoG subbands)
    # dog_filter = DoGSubbandFilter(image_path, sigma_list=[0.5, 1.0, 2.0, 4.0, 8.0])

    # Example 3: 5 sigma values
    dog_filter = DoGSubbandFilter(image_path, sigma_list=[1.0, 2.0, 4.0, 8.0, 16.0])

    dog_filter.process()
