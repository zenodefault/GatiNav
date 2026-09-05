import pytest

torch = pytest.importorskip("torch")

from python.ml.cnn import NoiseNet
from python.ml.export_onnx import export_onnx


def test_export_onnx(tmp_path):
    checkpoint = tmp_path / "model.pt"
    output = tmp_path / "model.onnx"
    torch.save({"model": NoiseNet().state_dict()}, checkpoint)
    try:
        export_onnx(checkpoint, output)
    except RuntimeError as exc:
        if "onnx package" not in str(exc):
            raise
        pytest.skip(str(exc))
    assert output.is_file()
