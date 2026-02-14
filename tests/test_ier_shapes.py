import torch

from image_analysis.subband_loss.filter import MultiScaleDogFilter
from image_analysis.subband_loss.IER import SFLLoss


def test_multiscale_dog_filter_forward_shape() -> None:
    batch_size, channels, height, width = 2, 3, 16, 16
    scales = [1.0, 2.0, 4.0]

    img = torch.randn(batch_size, channels, height, width)
    dog_filter = MultiScaleDogFilter(scales=scales)

    subbands = dog_filter(img)

    assert subbands.shape == (batch_size, len(scales) + 1, channels, height, width)


def test_sfl_loss_function_output_shapes() -> None:
    batch_size, channels, height, width = 2, 3, 16, 16
    scales = [1.0, 2.0, 4.0]
    num_subbands = len(scales) + 1

    predicted = torch.randn(batch_size, channels, height, width)
    target = torch.randn(batch_size, channels, height, width)
    loss_fn = SFLLoss(scales=scales)

    e_sfl = loss_fn.e_sfl(predicted, target)
    ier = loss_fn.ier(predicted, target)
    w_sfl = loss_fn.w_sfl(predicted, target)
    sfl_loss = loss_fn.sfl_loss(predicted, target)

    assert e_sfl.shape == (batch_size, num_subbands)
    assert ier.shape == (batch_size, num_subbands)
    assert w_sfl.shape == (num_subbands,)
    assert sfl_loss.shape == torch.Size([])
