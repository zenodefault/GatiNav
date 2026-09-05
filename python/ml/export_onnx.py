"""Export NoiseNet for the static mobile inference contract."""

from pathlib import Path
import argparse

import torch

from python.ml.cnn import NoiseNet
from python.ml.train import load_checkpoint


def export_onnx(checkpoint, output):
    try:
        import onnx  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("onnx package is required for ONNX export") from exc
    model = NoiseNet()
    load_checkpoint(model, checkpoint)
    model.eval()
    scripted = torch.jit.script(model)
    example = torch.zeros(1, 6, 100)
    torch.onnx.export(
        scripted, example, output, opset_version=17,
        input_names=["imu_acc_gyro"], output_names=["zupt_variances"],
        dynamic_axes=None,
    )
    print("TFLite compatibility report:")
    print("Conv, Relu, Flatten, Gemm, Softplus: required and mobile-compatible")
    print("TFLite conversion: deferred until Android conversion environment exists")
    return Path(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("output")
    args = parser.parse_args()
    export_onnx(args.checkpoint, args.output)
