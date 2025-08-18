"""
language: python
# AI-Generated Code Header
# **Intent:** Physics-informed band-integrator model for NIR → target prediction with white-box structure.
# **Optimization:** Small torch module; CPU-friendly; minimal parameters; early stopping.
# **Safety:** Non-negative band weights via softplus; bounded Gaussian band widths.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def to_absorbance(reflectance: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    R = np.clip(reflectance, eps, 1.0)
    return -np.log10(R)


def gaussian_masks(wavelengths: np.ndarray, centers_nm: List[float], sigma_nm: float) -> np.ndarray:
    wl = wavelengths.reshape(1, -1)
    centers = np.array(centers_nm, dtype=float).reshape(-1, 1)
    sigma2 = float(sigma_nm) ** 2
    masks = np.exp(-0.5 * (wl - centers) ** 2 / sigma2)
    # Normalize each mask to unit L1 for interpretability
    masks = masks / (np.sum(masks, axis=1, keepdims=True) + 1e-8)
    return masks.astype(np.float32)  # shape: (K, P)


class BandIntegratorPINN(nn.Module):
    def __init__(self, masks: np.ndarray, residual_hidden: int = 32):
        super().__init__()
        K, P = masks.shape
        self.register_buffer("masks", torch.from_numpy(masks))  # (K, P)
        # Non-negative band weights via softplus
        self.alpha_raw = nn.Parameter(torch.zeros(K))
        self.bias = nn.Parameter(torch.zeros(1))
        # Small residual MLP on band integrals
        self.residual = nn.Sequential(
            nn.Linear(K, residual_hidden),
            nn.ReLU(),
            nn.Linear(residual_hidden, 1),
        )

    def forward(self, A: torch.Tensor) -> torch.Tensor:
        # A: (N, P) absorbance
        # band integrals z = A @ masks^T
        z = torch.matmul(A, self.masks.t())  # (N, K)
        alpha = torch.nn.functional.softplus(self.alpha_raw)  # (K,)
        core = self.bias + torch.matmul(z, alpha.unsqueeze(1)).squeeze(1)  # (N,)
        res = self.residual(z).squeeze(1)  # (N,)
        return core + res, z, alpha


@dataclass
class PINNConfig:
    centers_nm: List[float]
    sigma_nm: float = 25.0
    lr: float = 1e-2
    batch_size: int = 64
    max_epochs: int = 200
    patience: int = 20
    residual_l2: float = 1e-5
    monotonicity_weight: float = 0.2  # encourage positive relation with sugar bands


def train_pinn(
    X_reflectance: np.ndarray,
    y: np.ndarray,
    wavelengths: np.ndarray,
    cfg: PINNConfig,
    val_split: float = 0.2,
    seed: int = 42,
) -> Tuple[BandIntegratorPINN, Dict]:
    rng = np.random.RandomState(seed)
    N = X_reflectance.shape[0]
    idx = np.arange(N)
    rng.shuffle(idx)
    n_val = max(1, int(N * val_split))
    val_idx, train_idx = idx[:n_val], idx[n_val:]

    A = to_absorbance(X_reflectance).astype(np.float32)
    y = y.astype(np.float32)
    masks = gaussian_masks(wavelengths, cfg.centers_nm, cfg.sigma_nm)
    model = BandIntegratorPINN(masks=masks, residual_hidden=32)

    device = torch.device("cpu")
    model.to(device)

    ds_train = TensorDataset(torch.from_numpy(A[train_idx]), torch.from_numpy(y[train_idx]))
    ds_val = TensorDataset(torch.from_numpy(A[val_idx]), torch.from_numpy(y[val_idx]))
    tl = DataLoader(ds_train, batch_size=cfg.batch_size, shuffle=True)
    vl = DataLoader(ds_val, batch_size=cfg.batch_size, shuffle=False)

    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    best_rmse = float("inf")
    best_state: Optional[Dict] = None
    epochs_no_improve = 0

    # Determine sugar band indices from centers (approximate sugar bands near ~1200 and ~1700 nm)
    centers_arr = np.array(cfg.centers_nm, dtype=float)
    sugar_idx = np.where(((centers_arr >= 1100.0) & (centers_arr <= 1300.0)) | (centers_arr >= 1650.0))[0]
    sugar_idx_t = torch.from_numpy(sugar_idx).long()

    for epoch in range(cfg.max_epochs):
        model.train()
        for xb, yb in tl:
            xb = xb.to(device)
            yb = yb.to(device)
            yp, z, _ = model(xb)
            loss_data = torch.mean((yp - yb) ** 2)
            l2 = 0.0
            for p in model.residual.parameters():
                l2 = l2 + torch.sum(p ** 2)
            # Monotonicity term: positive correlation between sum of sugar-band integrals and y
            mono = 0.0
            if sugar_idx.size > 0 and cfg.monotonicity_weight > 0:
                z_sugar = torch.sum(z.index_select(1, sugar_idx_t.to(z.device)), dim=1)
                zc = z_sugar - torch.mean(z_sugar)
                yc = yb - torch.mean(yb)
                denom = torch.sqrt(torch.sum(zc ** 2) * torch.sum(yc ** 2) + 1e-8)
                corr = torch.sum(zc * yc) / denom
                mono = torch.relu(-corr)
            loss = loss_data + cfg.residual_l2 * l2 + cfg.monotonicity_weight * mono
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        # Val
        model.eval()
        with torch.no_grad():
            ys, yh = [], []
            for xb, yb in vl:
                yp, _, _ = model(xb.to(device))
                ys.append(yb.numpy())
                yh.append(yp.cpu().numpy())
            ys = np.concatenate(ys)
            yh = np.concatenate(yh)
            rmse = float(np.sqrt(np.mean((yh - ys) ** 2)))
        if rmse + 1e-6 < best_rmse:
            best_rmse = rmse
            best_state = {k: v.cpu().detach().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= cfg.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Export band weights
    with torch.no_grad():
        alpha = torch.nn.functional.softplus(model.alpha_raw).cpu().numpy()
    info = {
        "alpha": alpha.tolist(),
        "centers_nm": cfg.centers_nm,
        "sigma_nm": cfg.sigma_nm,
        "val_rmse": best_rmse,
    }
    return model, info


def predict_pinn(model: BandIntegratorPINN, X_reflectance: np.ndarray) -> np.ndarray:
    model.eval()
    A = to_absorbance(X_reflectance).astype(np.float32)
    with torch.no_grad():
        yp, _, _ = model(torch.from_numpy(A))
    return yp.cpu().numpy()

