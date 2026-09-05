"""Deterministic lightweight training for NoiseNet."""

from pathlib import Path
import random

import matplotlib.pyplot as plt
import numpy as np
import torch

from python.ml.cnn import NoiseNet


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)


def log_variance_loss(prediction, target):
    target = torch.as_tensor(target, dtype=prediction.dtype, device=prediction.device)
    if prediction.shape != target.shape or prediction.shape[-1] != 3:
        raise ValueError("prediction and target must have shape (batch, 3)")
    return torch.mean((torch.log(prediction) - torch.log(target)) ** 2)


def load_checkpoint(model, path):
    state = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(state["model"])
    return state


def train_noise_net(train_x, train_y, val_x, val_y, checkpoint_path,
                    curve_path=None, seed=0, epochs=100, batch_size=32,
                    learning_rate=1e-3, patience=10, min_delta=1e-6):
    seed_everything(seed)
    model = NoiseNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    train_x = torch.as_tensor(train_x, dtype=torch.float32)
    train_y = torch.as_tensor(train_y, dtype=torch.float32)
    val_x = torch.as_tensor(val_x, dtype=torch.float32)
    val_y = torch.as_tensor(val_y, dtype=torch.float32)
    if train_x.ndim != 3 or train_x.shape[1:] != (6, 100):
        raise ValueError("train_x must have shape (batch, 6, 100)")
    if train_y.shape != (train_x.shape[0], 3) or val_y.shape != (val_x.shape[0], 3):
        raise ValueError("targets must have shape (batch, 3)")
    if torch.any(train_y <= 0) or torch.any(val_y <= 0):
        raise ValueError("variance targets must be positive")
    best = float("inf")
    stale = 0
    history = {"train": [], "validation": []}
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(epochs):
        model.train()
        order = torch.arange(train_x.shape[0])
        train_losses = []
        for start in range(0, train_x.shape[0], batch_size):
            idx = order[start:start + batch_size]
            optimizer.zero_grad()
            loss = log_variance_loss(model(train_x[idx]), train_y[idx])
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach()))
        model.eval()
        with torch.no_grad():
            validation = log_variance_loss(model(val_x), val_y)
        train_loss = float(np.mean(train_losses))
        validation_loss = float(validation)
        history["train"].append(train_loss)
        history["validation"].append(validation_loss)
        if validation_loss < best - min_delta:
            best, stale = validation_loss, 0
            torch.save({"model": model.state_dict(), "epoch": epoch,
                        "validation_drift": validation_loss, "seed": seed},
                       checkpoint_path)
        else:
            stale += 1
            if stale >= patience:
                break
    if curve_path is not None:
        curve_path = Path(curve_path)
        curve_path.parent.mkdir(parents=True, exist_ok=True)
        plt.figure()
        plt.plot(history["train"], label="train")
        plt.plot(history["validation"], label="validation")
        plt.xlabel("epoch")
        plt.ylabel("log-variance drift")
        plt.legend()
        plt.tight_layout()
        plt.savefig(curve_path)
        plt.close()
    return model, history, checkpoint_path
