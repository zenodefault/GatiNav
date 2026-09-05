import torch

from python.ml.cnn import NoiseNet


def test_noisenet_output_shape_and_positive_variances():
    model = NoiseNet()
    output = model(torch.zeros(4, 6, 100))
    assert output.shape == (4, 3)
    assert torch.all(output > 0)


def test_noisenet_parameter_count():
    model = NoiseNet()
    assert sum(parameter.numel() for parameter in model.parameters()) == 125187


def test_noisenet_forward_is_deterministic_with_fixed_seed():
    torch.manual_seed(7)
    first = NoiseNet()(torch.randn(2, 6, 100))
    torch.manual_seed(7)
    second = NoiseNet()(torch.randn(2, 6, 100))
    assert torch.equal(first, second)
