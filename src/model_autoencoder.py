import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler
from src.config import DATA_PROC, FIGURES, DEVICE, RANDOM_STATE

torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)


class ConvAE(nn.Module):
    def __init__(self, n_features, window, hidden=16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(n_features, 32, 3, padding=1), nn.ReLU(),
            nn.Conv1d(32, hidden, 3, padding=1), nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Conv1d(hidden, 32, 3, padding=1), nn.ReLU(),
            nn.Conv1d(32, n_features, 3, padding=1),
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        z = self.encoder(x)
        out = self.decoder(z)
        return out.permute(0, 2, 1)


def build_data(window=5):
    df = pd.read_parquet(DATA_PROC / "das_features.parquet")
    feature_cols = [c for c in df.columns if c not in ("timestamp", "channel_id")]
    vals = df[feature_cols].values.astype(np.float32)

    # === RobustScaler ===
    scaler = RobustScaler()
    vals = scaler.fit_transform(vals).astype(np.float32)

    X = []
    for i in range(len(vals) - window):
        X.append(vals[i:i + window])
    X = np.stack(X)
    return X, feature_cols, scaler


def train_ae(epochs=30, batch_size=64, lr=1e-3):
    X, feature_cols, scaler = build_data()
    print(f"[AE] Данные: X={X.shape}")

    Xt = torch.tensor(X).to(DEVICE)
    ds = TensorDataset(Xt, Xt)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

    model = ConvAE(n_features=X.shape[2], window=X.shape[1]).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.MSELoss()

    losses = []
    for ep in range(epochs):
        model.train()
        total = 0
        for xb, _ in dl:
            opt.zero_grad()
            out = model(xb)
            loss = loss_fn(out, xb)
            loss.backward()
            opt.step()
            total += loss.item() * xb.size(0)
        avg = total / len(ds)
        losses.append(avg)
        if (ep + 1) % 5 == 0 or ep == 0:
            print(f"[AE] Эпоха {ep+1}/{epochs}, loss={avg:.5f}")

    torch.save(model.state_dict(), DATA_PROC / "conv_ae_das.pt")

    # Ошибки восстановления
    model.eval()
    with torch.no_grad():
        recon = model(Xt).cpu().numpy()
    errs = ((X - recon) ** 2).mean(axis=(1, 2))

    thr = np.quantile(errs, 0.95)
    anomalies = errs > thr
    print(f"[AE] Порог={thr:.5f}, аномалий={anomalies.sum()}")

    plt.figure(figsize=(14, 5))
    plt.plot(errs, linewidth=0.8, label="Ошибка")
    plt.axhline(thr, color="red", linestyle="--", label="Порог 95%")
    plt.xlabel("Окно")
    plt.ylabel("MSE восстановления")
    plt.title("Conv-AE: ошибки восстановления (пики = аномалии)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES / "ae_errors.png", dpi=150)
    plt.close()

    plt.figure(figsize=(10, 4))
    plt.plot(losses, marker="o")
    plt.xlabel("Эпоха")
    plt.ylabel("Loss")
    plt.title("Conv-AE: кривая обучения")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES / "ae_loss.png", dpi=150)
    plt.close()
    print("[AE] Графики сохранены")


if __name__ == "__main__":
    train_ae()