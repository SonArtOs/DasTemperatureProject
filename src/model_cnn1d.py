import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import RobustScaler
import seaborn as sns
from src.config import DATA_PROC, FIGURES, DEVICE, RANDOM_STATE

torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)


class CNN1D(nn.Module):
    def __init__(self, n_features, n_classes, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(n_features, hidden, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.3),

            nn.Conv1d(hidden, hidden * 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden * 2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(hidden * 2, n_classes),
        )

    def forward(self, x):
        # x: [B, window, features] -> [B, features, window]
        x = x.permute(0, 2, 1)
        x = self.net(x).squeeze(-1)
        return self.head(x)


class FocalLoss(nn.Module):
    """Focal Loss — для дисбаланса классов."""
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce = F.cross_entropy(inputs, targets, weight=self.alpha, reduction="none")
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


def make_weak_labels(df: pd.DataFrame) -> np.ndarray:
    """
    Слабые метки по RMS: 0 — фон, 1 — среднее событие, 2 — сильное событие.
    """
    rms = df["rms"].values
    q1, q2 = np.quantile(rms, [0.7, 0.9])
    labels = np.zeros(len(rms), dtype=np.int64)
    labels[rms > q1] = 1
    labels[rms > q2] = 2
    return labels


def build_dataset(window=5):
    df = pd.read_parquet(DATA_PROC / "das_features.parquet")
    feature_cols = [c for c in df.columns if c not in ("timestamp", "channel_id")]
    labels = make_weak_labels(df)
    vals = df[feature_cols].values.astype(np.float32)

    # === НОРМАЛИЗАЦИЯ (RobustScaler устойчив к выбросам) ===
    scaler = RobustScaler()
    vals = scaler.fit_transform(vals).astype(np.float32)

    X, y = [], []
    for i in range(len(vals) - window):
        X.append(vals[i:i + window])
        y.append(labels[i + window - 1])
    X = np.stack(X).astype(np.float32)
    y = np.array(y)
    return X, y


def train_cnn1d(epochs=50, batch_size=32, lr=1e-3):
    X, y = build_dataset()
    print(f"[CNN1D] Данные: X={X.shape}, классы={np.bincount(y)}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    Xt = torch.tensor(X_train).to(DEVICE)
    yt = torch.tensor(y_train).to(DEVICE)
    Xe = torch.tensor(X_test).to(DEVICE)
    ye = torch.tensor(y_test).to(DEVICE)

    # === WeightedRandomSampler для дисбаланса ===
    class_counts = np.bincount(y_train)
    class_weights = 1.0 / np.maximum(class_counts, 1)
    sample_weights = class_weights[y_train]
    sampler = WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.float32),
        num_samples=len(y_train),
        replacement=True,
    )

    ds = TensorDataset(Xt, yt)
    dl = DataLoader(ds, batch_size=batch_size, sampler=sampler)

    model = CNN1D(n_features=X.shape[2], n_classes=3, hidden=32).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, factor=0.5)

    # === Focal Loss с alpha ===
    alpha = torch.tensor(class_weights / class_weights.sum() * 3, dtype=torch.float32).to(DEVICE)
    loss_fn = FocalLoss(alpha=alpha, gamma=2.0)

    losses = []
    for ep in range(epochs):
        model.train()
        total = 0
        for xb, yb in dl:
            opt.zero_grad()
            out = model(xb)
            loss = loss_fn(out, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item() * xb.size(0)
        avg = total / len(ds)
        losses.append(avg)
        scheduler.step(avg)
        if (ep + 1) % 5 == 0 or ep == 0:
            print(f"[CNN1D] Эпоха {ep+1}/{epochs}, loss={avg:.4f}")

    torch.save(model.state_dict(), DATA_PROC / "cnn1d_das.pt")

    # Метрики
    model.eval()
    with torch.no_grad():
        pred = model(Xe).argmax(dim=1).cpu().numpy()
    print("[CNN1D] Отчёт:")
    print(classification_report(ye.cpu().numpy(), pred, digits=3, zero_division=0))

    cm = confusion_matrix(ye.cpu().numpy(), pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Предсказано")
    plt.ylabel("Истина")
    plt.title("CNN1D: confusion matrix")
    plt.tight_layout()
    plt.savefig(FIGURES / "cnn1d_confusion.png", dpi=150)
    plt.close()

    plt.figure(figsize=(10, 4))
    plt.plot(losses, marker="o")
    plt.xlabel("Эпоха")
    plt.ylabel("Loss")
    plt.title("CNN1D: кривая обучения")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES / "cnn1d_loss.png", dpi=150)
    plt.close()
    print("[CNN1D] Графики сохранены")


if __name__ == "__main__":
    train_cnn1d()