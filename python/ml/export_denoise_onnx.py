"""Export the residual TCN for on-device inference (Blueprint deployment).

The plan's deployment section requires the denoising network to be
lightweight and quantized for the mobile app (10 Hz) and edge engine
(200 Hz). This script:

1. exports the trained ResidualTCN to ONNX (opset 17, fixed shapes),
2. applies INT8 dynamic quantization via onnxruntime.quantization when the
   package is available (falls back to a printed instruction otherwise),
3. prints a rough CPU latency estimate per inference call.

The exported model takes a (1, 1, W) forward-accel window and outputs the
scalar residual; the same ONNX graph serves the 100 Hz phone path and the
200 Hz edge path (the caller just feeds it at its own stride).
"""

from pathlib import Path
import argparse
import time

import numpy as np
import torch

from python.ml.denoise.dataset import WINDOW_S, FS
from python.ml.denoise.model import ResidualTCN
from python.ml.train import load_checkpoint


def export_onnx(checkpoint, output, quantize=True):
    try:
        import onnx  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("onnx package is required for ONNX export") from exc
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = ResidualTCN()
    model.load_state_dict(state["model"])
    model.eval()
    scripted = torch.jit.script(model)
    example = torch.zeros(1, 1, int(round(WINDOW_S * FS)))
    torch.onnx.export(
        scripted, example, str(output), opset_version=17,
        input_names=["forward_accel_window"],
        output_names=["residual"],
        dynamic_axes=None,
    )
    if quantize:
        _quantize_onnx(output)
    _latency(model, example.numpy())
    print(f"exported {output}")
    return Path(output)


def _quantize_onnx(path):
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
    except ImportError:
        print("onnxruntime not available; INT8 quantization deferred "
              "(install onnxruntime and rerun with --quantize)")
        return path
    out = Path(str(path) + ".int8.onnx")
    quantize_dynamic(str(path), str(out), weight_type=QuantType.QInt8)
    print(f"INT8 dynamic quantization -> {out}")
    return out


def _latency(model, example):
    model.eval()
    with torch.no_grad():
        x = torch.from_numpy(example)
        for _ in range(3):  # warm-up
            model(x)
        start = time.perf_counter()
        n = 20
        for _ in range(n):
            model(x)
        ms = (time.perf_counter() - start) / n * 1000.0
    print(f"CPU latency (CPU-only, no threads): {ms:.2f} ms/inference "
          f"({1000.0 / max(ms, 1e-6):.0f} Hz)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", help="denoise_fold_<driver>.pt")
    parser.add_argument("output", help="output .onnx path")
    parser.add_argument("--no-quantize", action="store_true")
    args = parser.parse_args()
    export_onnx(args.checkpoint, args.output, quantize=not args.no_quantize)