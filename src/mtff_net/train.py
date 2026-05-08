"""Training utilities for MTFF-Net."""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class ScalerBundle:
    x_scaler: StandardScaler
    y_scaler: StandardScaler


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def fit_scalers(X_train: np.ndarray, y_train: np.ndarray) -> ScalerBundle:
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    x_scaler.fit(X_train.reshape(-1, X_train.shape[-1]))
    y_scaler.fit(y_train)
    return ScalerBundle(x_scaler, y_scaler)


def transform_X(X: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    return scaler.transform(X.reshape(-1, X.shape[-1])).reshape(X.shape).astype(np.float32)


def train_model(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    target_indices: list[int],
    epochs: int = 130,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 12,
    smooth_weight: float = 0.005,
    grad_clip: float = 1.0,
    device: str | None = None,
) -> tuple[nn.Module, list[dict[str, float]]]:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.6, patience=4)
    loss_fn = nn.MSELoss()
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
        batch_size=batch_size,
        shuffle=True,
    )
    val_x = torch.from_numpy(X_val).to(device)
    val_y = torch.from_numpy(y_val).to(device)

    best_state = None
    best_val = float("inf")
    wait = 0
    history: list[dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            mse_loss = loss_fn(pred, yb)
            last_target = xb[:, -1, target_indices]
            smooth_loss = torch.mean((pred - last_target) ** 2)
            loss = mse_loss + smooth_weight * smooth_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))

        model.eval()
        with torch.no_grad():
            val_loss = float(loss_fn(model(val_x), val_y).detach().cpu())
        scheduler.step(val_loss)
        history.append({"epoch": epoch, "train_loss": float(np.mean(train_losses)), "val_loss": val_loss})

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
        if wait >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


def predict(model: nn.Module, X: np.ndarray, batch_size: int = 256, device: str | None = None) -> np.ndarray:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    outputs = []
    loader = DataLoader(TensorDataset(torch.from_numpy(X)), batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for (xb,) in loader:
            outputs.append(model(xb.to(device)).detach().cpu().numpy())
    return np.vstack(outputs)

