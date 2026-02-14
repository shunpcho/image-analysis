# image-analysis

Implementation of subband-based loss design inspired by the paper:

**Naoyuki Ichimura, _Inverse Error Ratio: Subband Weighting in Training DNNs_**

This repository provides:

- Multi-scale DoG (Difference of Gaussian) subband decomposition
- Spatial Frequency Loss (SFL)
- Inverse Error Ratio (IER)-based subband weighting

---

## 1. Method overview (paper-aligned)

The main idea is to compare prediction and target not only in pixel space but also in frequency-oriented subbands.

1. Decompose images into DoG subbands.
2. Compute per-subband error $E_{\mathrm{SFL},k}$.
3. Compute IER-based weight per subband.
4. Aggregate weighted subband errors into final SFL loss.

In this implementation:

- `predicted` = model output
- `target` = ground-truth clean image
- tensor shape = `(B, C, H, W)`

---

## 2. Mathematical form used here

For each subband $k$:

$$
E_{\mathrm{SFL},k}
=
\operatorname{mean}_{c,h,w}
\left(\left(S_k(\hat{x}) - S_k(x)\right)^2\right)
$$

where $S_k(\cdot)$ is the $k$-th DoG subband transform.

Image-space MSE per sample:

$$
E_{\mathrm{L2}} = \operatorname{mean}_{c,h,w}\left((\hat{x}-x)^2\right)
$$

IER per sample and subband:

$$
\mathrm{IER}_k = \frac{E_{\mathrm{L2}}}{E_{\mathrm{SFL},k}}
$$

Subband weight (batch-mean IER):

$$
w_k = \operatorname{mean}_{b}(\mathrm{IER}_{b,k})
$$

Final loss:

$$
L_{\mathrm{SFL}} = \operatorname{mean}_{b,k}\left(w_k \cdot E_{\mathrm{SFL},b,k}\right)
$$

---

## 3. Project structure

- `src/image_analysis/subband_loss/filter.py`
  - `MultiScaleDogFilter`: PyTorch DoG filter bank for `(B, C, H, W)` input
- `src/image_analysis/subband_loss/IER.py`
  - `SFLLoss`: `e_sfl`, `ier`, `w_sfl`, and final `forward` loss
- `src/image_analysis/fft_trans.py`
  - NumPy/SciPy pipeline for generating/saving DoG and FFT visualization images

---

## 4. Usage

### 4.1 SFL loss in training (PyTorch)

```python
import torch
from image_analysis.subband_loss.IER import SFLLoss

predicted = torch.randn(2, 3, 64, 64)
target = torch.randn(2, 3, 64, 64)

criterion = SFLLoss(scales=[0.6, 1.2, 2.4, 4.8, 9.6])
loss = criterion(predicted, target)
loss.backward()
```

### 4.2 Inspect intermediate terms

```python
e_sfl = criterion.e_sfl(predicted, target)   # (B, K)
ier = criterion.ier(predicted, target)       # (B, K)
w_sfl = criterion.w_sfl(predicted, target)   # (K,)
```

### 4.3 DoG subband image generation

```python
from image_analysis.fft_trans import DoGPipeline

pipeline = DoGPipeline(
		image_path="/path/to/input.png",
		sigma_list=[1.0, 2.0, 4.0, 8.0],
		output_dir="results",
)
pipeline.process()
```

This saves Gaussian pyramid images, DoG subbands, FFT magnitude maps, and residual low-frequency image.

---

## 5. Notes

- This repository focuses on the loss/filter implementation side of the paper method.
- `fft_trans.py` contains a script-style example with a local absolute image path; replace it with your own path before running.
- For full reproducibility against the exact paper setup, training protocol/data settings should be configured in your downstream training project.
