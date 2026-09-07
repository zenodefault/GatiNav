import torch

from python.ml.cnn import NoiseNet


def test_noisenet_output_shape_and_positive_variances():
    model = NoiseNet()
    output = model(torch.zeros(4, 6, 100))
    assert output.shape == (4, 4)
    # variance head: strictly positive
    assert torch.all(output[:, :3] > 0)
    # speed head: non-negative log(1+v) encoding
    assert torch.all(output[:, 3] >= 0)


def test_noisenet_parameter_count():
    model = NoiseNet()
    # 1600x64 + 64x4 projection/heads: +260 params over the 3-output net
    assert sum(parameter.numel() for parameter in model.parameters()) == 125252


def test_speed_head_recovers_speed_from_log_encoding():
    model = NoiseNet()
    out = model(torch.zeros(2, 6, 100))
    speed = torch.expm1(out[:, 3])
    assert speed.shape == (2,)
    assert torch.all(speed >= 0)


def test_noisenet_forward_is_deterministic_with_fixed_seed():
    torch.manual_seed(7)
    first = NoiseNet()(torch.randn(2, 6, 100))
    torch.manual_seed(7)
    second = NoiseNet()(torch.randn(2, 6, 100))
    assert torch.equal(first, second)
