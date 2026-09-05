import numpy as np
import torch

from python.ml.cnn import NoiseNet
from python.ml.train import load_checkpoint, train_noise_net


def test_two_iteration_training_saves_and_loads_checkpoint(tmp_path):
    rng = np.random.default_rng(4)
    train_x = rng.normal(size=(8, 6, 100)).astype(np.float32)
    val_x = rng.normal(size=(4, 6, 100)).astype(np.float32)
    train_y = np.full((8, 3), 0.5, dtype=np.float32)
    val_y = np.full((4, 3), 0.5, dtype=np.float32)
    checkpoint = tmp_path / "weights" / "noisenet.pt"
    curve = tmp_path / "training_curves.png"
    trained, history, path = train_noise_net(
        train_x, train_y, val_x, val_y, checkpoint, curve,
        seed=12, epochs=2, batch_size=8, patience=2,
    )
    assert len(history["train"]) == 2
    assert history["train"][-1] < history["train"][0]
    assert path.is_file()
    assert curve.is_file()
    restored = NoiseNet()
    state = load_checkpoint(restored, path)
    assert state["epoch"] in (0, 1)
    assert torch.allclose(trained(torch.from_numpy(val_x)),
                          restored(torch.from_numpy(val_x)))
