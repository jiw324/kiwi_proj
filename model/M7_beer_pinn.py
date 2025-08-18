"""
language: python
# AI-Generated Code Header
# **Intent:** Beer–Lambert physics-informed model: learns analyte spectra eps_k(λ) and per-sample concentrations c_k.
# **Optimization:** Small CPU-friendly Torch model with early stopping.
# **Safety:** Non-negativity via softplus; smoothness penalty on eps_k along wavelength.
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


def init_epsilons_gaussian(wavelengths: np.ndarray, centers_nm: List[float], sigma_nm: float, scale: float = 0.1) -> np.ndarray:
    wl = wavelengths.reshape(1, -1)
    centers = np.array(centers_nm, dtype=float).reshape(-1, 1)
    sigma2 = float(sigma_nm) ** 2
    masks = np.exp(-0.5 * (wl - centers) ** 2 / sigma2)
    masks = masks / (np.sum(masks, axis=1, keepdims=True) + 1e-8)
    # scale to small positive values
    return (scale * masks).astype(np.float32)  # (K, P)


class BeerPINN(nn.Module):
    def __init__(self, init_eps: np.ndarray, hidden_c: int = 64, use_residual: bool = False):
        super().__init__()
        K, P = init_eps.shape
        self.K = K
        self.P = P
        # Epsilon spectra (K, P), non-negative via softplus
        self.epsilon_raw = nn.Parameter(torch.log(torch.expm1(torch.from_numpy(init_eps)) + 1e-6))
        # Map absorbance A (P,) to concentrations c (K,) with nonnegative output
        self.c_layer = nn.Sequential(
            nn.Linear(P, hidden_c), nn.ReLU(), nn.Linear(hidden_c, K)
        )
        self.beta0 = nn.Parameter(torch.zeros(1))
        self.beta_raw = nn.Parameter(torch.zeros(K))
        self.use_residual = use_residual
        if use_residual:
            self.residual = nn.Sequential(nn.Linear(K, 16), nn.ReLU(), nn.Linear(16, 1))

    def forward(self, A: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # A: (N, P) absorbance
        eps = torch.nn.functional.softplus(self.epsilon_raw)  # (K, P)
        # concentrations
        c_raw = self.c_layer(A)  # (N, K)
        c = torch.nn.functional.softplus(c_raw)
        # reconstruct absorbance via Beer–Lambert
        A_hat = torch.matmul(c, eps)  # (N, P)
        # map concentrations to target
        beta = torch.nn.functional.softplus(self.beta_raw)
        y_core = self.beta0 + torch.matmul(c, beta.unsqueeze(1)).squeeze(1)
        if self.use_residual:
            y_pred = y_core + self.residual(c).squeeze(1)
        else:
            y_pred = y_core
        return y_pred, A_hat, c, eps


@dataclass
class BeerConfig:
    centers_nm: List[float]
    sigma_nm: float = 25.0
    lr: float = 5e-3
    batch_size: int = 64
    max_epochs: int = 200
    patience: int = 20
    lambda_rec: float = 0.5
    lambda_smooth: float = 1e-3
    monotonicity_weight: float = 0.2  # encourage positive relation between sugar concentrations and y
    use_residual: bool = False


def train_beer_pinn(
    X_reflectance: np.ndarray,
    y: np.ndarray,
    wavelengths: np.ndarray,
    cfg: BeerConfig,
    val_split: float = 0.2,
    seed: int = 42,
) -> Tuple[BeerPINN, Dict]:
    rng = np.random.RandomState(seed)
    N = X_reflectance.shape[0]
    idx = np.arange(N); rng.shuffle(idx)
    n_val = max(1, int(N * val_split))
    val_idx, train_idx = idx[:n_val], idx[n_val:]

    A = to_absorbance(X_reflectance).astype(np.float32)
    y = y.astype(np.float32)
    init_eps = init_epsilons_gaussian(wavelengths, cfg.centers_nm, cfg.sigma_nm)
    model = BeerPINN(init_eps=init_eps, hidden_c=64, use_residual=cfg.use_residual)
    device = torch.device("cpu"); model.to(device)

    ds_tr = TensorDataset(torch.from_numpy(A[train_idx]), torch.from_numpy(y[train_idx]))
    ds_va = TensorDataset(torch.from_numpy(A[val_idx]), torch.from_numpy(y[val_idx]))
    tl = DataLoader(ds_tr, batch_size=cfg.batch_size, shuffle=True)
    vl = DataLoader(ds_va, batch_size=cfg.batch_size, shuffle=False)

    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    best_rmse = float("inf"); best_state: Optional[Dict] = None; no_improve = 0

    # Identify sugar-related spectral components by center location (~1200, ~1700 nm)
    centers_arr = np.array(cfg.centers_nm, dtype=float)
    sugar_idx = np.where(((centers_arr >= 1100.0) & (centers_arr <= 1300.0)) | (centers_arr >= 1650.0))[0]
    sugar_idx_t = torch.from_numpy(sugar_idx).long()

    wl_t = torch.from_numpy(wavelengths.astype(np.float32))

    for epoch in range(cfg.max_epochs):
        model.train()
        for xb, yb in tl:
            xb = xb.to(device); yb = yb.to(device)
            yp, Ahat, c, eps = model(xb)
            # losses
            loss_y = torch.mean((yp - yb) ** 2)
            loss_rec = torch.mean((Ahat - xb) ** 2)
            # smoothness on eps along wavelength (second difference)
            dif1 = eps[:, 1:] - eps[:, :-1]
            dif2 = dif1[:, 1:] - dif1[:, :-1]
            loss_smooth = torch.mean(dif2 ** 2)
            # monotonicity: encourage positive corr between sum of sugar c and y
            mono = 0.0
            if sugar_idx.size > 0 and cfg.monotonicity_weight > 0:
                z_sugar = torch.sum(c.index_select(1, sugar_idx_t.to(c.device)), dim=1)
                zc = z_sugar - torch.mean(z_sugar)
                yc = yb - torch.mean(yb)
                denom = torch.sqrt(torch.sum(zc ** 2) * torch.sum(yc ** 2) + 1e-8)
                corr = torch.sum(zc * yc) / denom
                mono = torch.relu(-corr)
            loss = loss_y + cfg.lambda_rec * loss_rec + cfg.lambda_smooth * loss_smooth + cfg.monotonicity_weight * mono
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()

        # val rmse on y
        model.eval();
        with torch.no_grad():
            ys = []; yh = []
            for xb, yb in vl:
                yp, _, _, _ = model(xb.to(device))
                ys.append(yb.numpy()); yh.append(yp.cpu().numpy())
            ys = np.concatenate(ys); yh = np.concatenate(yh)
            rmse = float(np.sqrt(np.mean((yh - ys) ** 2)))
        if rmse + 1e-6 < best_rmse:
            best_rmse = rmse
            best_state = {k: v.cpu().detach().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= cfg.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    with torch.no_grad():
        eps = torch.nn.functional.softplus(model.epsilon_raw).cpu().numpy()
    info = {
        "centers_nm": cfg.centers_nm,
        "sigma_nm": cfg.sigma_nm,
        "val_rmse": best_rmse,
        "epsilon": eps.tolist(),  # K x P
    }
    return model, info


def predict_beer_pinn(model: BeerPINN, X_reflectance: np.ndarray) -> np.ndarray:
    model.eval()
    A = to_absorbance(X_reflectance).astype(np.float32)
    with torch.no_grad():
        yp, _, _, _ = model(torch.from_numpy(A))
    return yp.cpu().numpy()

