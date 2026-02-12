from pathlib import Path

import numpy as np
import numpy.typing as npt
from PIL import Image
from scipy import ndimage


class DoGSubbandFilter:
    """Class for creating subband images using Difference of Gaussian (DoG) filters."""

    def __init__(
        self,
        img: npt.NDArray[np.float64],
        sigma_list: list[float] | None = None,
    ) -> None:
        """Initialize the DoG subband filter.

        Args:
            img: Input grayscale image as a 2D numpy array.
            sigma_list: List of sigma values for Gaussian filters.
                       DoG subbands are computed as differences between consecutive Gaussians.
                       Default is [1.0, 2.0, 4.0, 8.0] which creates 3 DoG subbands plus
                       the lowest frequency residual.
        """
        self.img = img
        self.sigma_list = sigma_list or [1.0, 2.0, 4.0, 8.0]
        self.gaussian_pyramid: list[npt.NDArray[np.float64]] = []
        self.dog_subbands: list[npt.NDArray[np.float64]] = []

    def create_gaussian_pyramid(self) -> None:
        """Create Gaussian pyramid by applying different sigma values."""
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

    def process(self) -> None:
        """Execute the complete DoG filtering process."""
        self.create_gaussian_pyramid()
        self.compute_dog_subbands()


class DoGImageSaver:
    """Class for saving DoG filter results to disk."""

    def __init__(self, output_dir: Path | str = "results") -> None:
        """Initialize the image saver.

        Args:
            output_dir: Directory to save output images
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

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

    def save_gaussian_pyramid(
        self,
        gaussian_pyramid: list[npt.NDArray[np.float64]],
        sigma_list: list[float],
    ) -> None:
        """Save all Gaussian pyramid images.

        Args:
            gaussian_pyramid: List of Gaussian filtered images
            sigma_list: List of sigma values used for Gaussian filters
        """
        for idx, gaussian_img in enumerate(gaussian_pyramid):
            img_normalized = self.normalize_image(gaussian_img)
            Image.fromarray(img_normalized).save(self.output_dir / f"gaussian_sigma_{sigma_list[idx]:.1f}.png")

    def save_dog_subbands(
        self,
        dog_subbands: list[npt.NDArray[np.float64]],
        sigma_list: list[float],
    ) -> None:
        """Save all DoG subband images and their FFT magnitude spectrums.

        Args:
            dog_subbands: List of DoG subband images
            sigma_list: List of sigma values used for Gaussian filters
        """
        for idx, dog_img in enumerate(dog_subbands):
            # Save normalized DoG (with sign information)
            img_normalized = self.normalize_image(dog_img)
            Image.fromarray(img_normalized).save(
                self.output_dir / f"dog_subband_{idx}_sigma_{sigma_list[idx]:.1f}-{sigma_list[idx + 1]:.1f}.png"
            )

            # Save FFT magnitude spectrum
            fft_magnitude = self.compute_fft_magnitude(dog_img)
            magnitude_normalized = self.normalize_spectrum(fft_magnitude)
            Image.fromarray(magnitude_normalized).save(
                self.output_dir / f"dog_fft_magnitude_{idx}_sigma_{sigma_list[idx]:.1f}-{sigma_list[idx + 1]:.1f}.png"
            )

    def save_residual(
        self,
        residual: npt.NDArray[np.float64],
        sigma: float,
    ) -> None:
        """Save the residual (lowest frequency component).

        Args:
            residual: Residual image array
            sigma: Sigma value of the residual
        """
        img_normalized = self.normalize_image(residual)
        Image.fromarray(img_normalized).save(self.output_dir / f"residual_sigma_{sigma:.1f}.png")


class DoGPipeline:
    """Pipeline for loading, processing, and saving DoG filter results."""

    def __init__(
        self,
        image_path: Path | str,
        sigma_list: list[float] | None = None,
        output_dir: Path | str = "results",
    ) -> None:
        """Initialize the DoG pipeline.

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
        self.output_dir = Path(output_dir)
        self.dog_filter: DoGSubbandFilter | None = None

    def load_image(self) -> npt.NDArray[np.float64]:
        """Load image from the specified path and convert to grayscale float.

        Returns:
            Grayscale image as float64 array
        """
        img_uint8 = np.array(Image.open(self.image_path).convert("L"))
        return img_uint8.astype(np.float64)

    def process(self) -> None:
        """Execute the complete DoG filtering and saving process."""
        # Load image
        img = self.load_image()

        # Process with DoG filter
        self.dog_filter = DoGSubbandFilter(img, self.sigma_list)
        self.dog_filter.process()

        # Save results
        saver = DoGImageSaver(self.output_dir)
        saver.save_gaussian_pyramid(self.dog_filter.gaussian_pyramid, self.sigma_list)
        saver.save_dog_subbands(self.dog_filter.dog_subbands, self.sigma_list)
        if len(self.dog_filter.gaussian_pyramid) > 0:
            saver.save_residual(self.dog_filter.gaussian_pyramid[-1], self.sigma_list[-1])

        print("Saved images")
        print(f"Sigma values: {self.sigma_list}")


if __name__ == "__main__":
    image_path = "path/to/your/image.jpg"

    # Example 1: Default 4 sigma values (creates 3 DoG subbands)
    # pipeline = DoGPipeline(image_path)

    # Example 2: Custom sigma values (creates 4 DoG subbands)
    # pipeline = DoGPipeline(image_path, sigma_list=[0.5, 1.0, 2.0, 4.0, 8.0])

    # Example 3: 5 sigma values
    pipeline = DoGPipeline(image_path, sigma_list=[1.0, 2.0, 4.0, 8.0, 16.0])

    pipeline.process()
