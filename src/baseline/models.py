"""Probes for the silicon baseline: logistic regression and a small GPU MLP.

Both return a predict_proba(X) -> p(up) callable. Standardization uses train
statistics only.
"""

from __future__ import annotations

import math

import numpy as np


def binomial_margin(n: int, z: float = 1.96) -> float:
    """Half-width of the ~95% CI around 0.5 for n Bernoulli samples."""
    if n <= 0:
        return float("nan")
    return z * math.sqrt(0.25 / n)


def fit_logistic(X_train: np.ndarray, y_train: np.ndarray):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    # No class rebalancing: the gate thresholds predict_proba at 0.5 and
    # scores accuracy against the majority class, both of which assume
    # probabilities calibrated to the true class prior. Balanced weights
    # would shift the decision boundary and manufacture (or hide) edge on
    # drifting slices. If the model just learns the prior, accuracy lands
    # on majority_acc and the gate correctly reads "no signal".
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000),
    )
    model.fit(X_train, y_train)

    def predict_proba(X: np.ndarray) -> np.ndarray:
        return model.predict_proba(X)[:, 1]

    return predict_proba


def fit_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    device: str = "auto",
    hidden: tuple[int, ...] = (64, 64),
    max_epochs: int = 200,
    patience: int = 15,
    batch_size: int = 1024,
    lr: float = 1e-3,
    seed: int = 42,
    verbose: bool = False,
):
    import torch
    from torch import nn

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed)

    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std < 1e-8] = 1.0

    def prep(X: np.ndarray) -> "torch.Tensor":
        return torch.as_tensor((X - mean) / std, dtype=torch.float32)

    Xtr, Xva = prep(X_train).to(device), prep(X_val).to(device)
    ytr = torch.as_tensor(y_train, dtype=torch.float32, device=device)
    yva = torch.as_tensor(y_val, dtype=torch.float32, device=device)

    layers: list[nn.Module] = []
    k = X_train.shape[1]
    for h in hidden:
        layers += [nn.Linear(k, h), nn.ReLU(), nn.Dropout(0.1)]
        k = h
    layers.append(nn.Linear(k, 1))
    net = nn.Sequential(*layers).to(device)

    # Unweighted BCE for the same reason fit_logistic drops class_weight:
    # the 0.5 threshold and majority-class comparison need probabilities
    # calibrated to the true prior, not artificially rebalanced ones.
    loss_fn = nn.BCEWithLogitsLoss()
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4)

    best_val = float("inf")
    best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
    bad_epochs = 0
    n = len(Xtr)
    for epoch in range(max_epochs):
        net.train()
        perm = torch.randperm(n, device=device)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            opt.zero_grad()
            out = net(Xtr[idx]).squeeze(-1)
            loss = loss_fn(out, ytr[idx])
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            val_loss = loss_fn(net(Xva).squeeze(-1), yva).item()
        if val_loss < best_val - 1e-5:
            best_val = val_loss
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break
        if verbose and epoch % 10 == 0:
            print(f"  epoch {epoch}: val_loss={val_loss:.5f}")
    net.load_state_dict(best_state)
    net.eval()

    def predict_proba(X: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            logits = net(prep(X).to(device)).squeeze(-1)
            return torch.sigmoid(logits).cpu().numpy()

    return predict_proba
