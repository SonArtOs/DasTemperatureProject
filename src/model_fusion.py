"""
Мультимодальная модель: DAS (виброакустика) + DTS (температура).
Context-aware classification: событие + температурный контекст.
"""
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
from sklearn.preprocessing import RobustScaler, StandardScaler
import seaborn as sns
from src.config import DATA_PROC, FIGURES, DEVICE, RANDOM_STATE

torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)


class FusionModel(nn.Module):
    """
    Две ветки:
      - DAS-ветка (1D-CNN) → 32-мерный вектор
      - DTS-ветка (MLP)    →  8-мерный вектор
    Конкатенация → MLP → классификация + детекция аномалии
    """
    def __init__(self, das_channels=5, dts_features=7, n_classes=3):
        super().__init__()

        # DAS-ветка
        self.das_branch = nn.Sequential(
            nn.Conv1d(das_channels, 16, kernel_size=5, padding=2),
            nn.BatchNorm1d(16), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )

        # DTS-ветка
        self.dts_branch = nn.Sequential(
            nn.Linear(dts_features, 16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, 8),
            nn.ReLU(),
        )

        # Объединяющий слой
        self.fusion = nn.Sequential(
            nn.Linear(32 + 8, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 32), nn.ReLU(),
        )

        # Головы
        self.class_head = nn.Linear(32, n_classes)
        self.anomaly_head = nn.Linear(32, 1)

    def forward(self, das, dts):
        # das: [B, channels, time], dts: [B, features]
        h_das = self.das_branch(das).squeeze(-1)   # [B, 32]
        h_dts = self.dts_branch(dts)               # [B, 8]
        h = torch.cat([h_das, h_dts], dim=1)       # [B, 40]
        h = self.fusion(h)                         # [B, 32]
        return self.class_head(h), self.anomaly_head(h).squeeze(-1)


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce = F.cross_entropy(inputs, targets, weight=self.alpha, reduction="none")
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


def make_weak_labels(df):
    rms = df["rms"].values
    q1, q2 = np.quantile(rms, [0.7, 0.9])
    labels = np.zeros(len(rms), dtype=np.int64)
    labels[rms > q1] = 1
    labels[rms > q2] = 2
    return labels


def build_fusion_dataset(window=5):
    """Строит DAS-окна и DTS-фичи, выровненные по времени."""
    das = pd.read_parquet(DATA_PROC / "das_features.parquet")
    das_feature_cols = [c for c in das.columns if c not in ("timestamp", "channel_id")]
    labels = make_weak_labels(das)
    das_vals = das[das_feature_cols].values.astype(np.float32)

    # DAS-нормализация
    das_scaler = RobustScaler()
    das_vals = das_scaler.fit_transform(das_vals).astype(np.float32)

    # DTS-фичи (для каждого временного шага DAS берём ближайшее окно DTS)
    dts = pd.read_parquet(DATA_PROC / "dts_features.parquet")
    dts_cols = ["mean", "std", "min", "max", "slope"]

    # Если DTS короче — ресемплим до длины DAS
    if len(dts) != len(das):
        # Простая интерполяция: каждому DAS-шагу соответствует ближайший DTS
        idx = np.linspace(0, len(dts) - 1, len(das)).astype(int)
        dts_aligned = dts.iloc[idx][dts_cols].values
    else:
        dts_aligned = dts[dts_cols].values

    # Добавляем производные (тренды)
    dts_deriv = np.gradient(dts_aligned, axis=0)
    dts_full = np.hstack([dts_aligned, dts_deriv[:, :2]])  # 5 + 2 = 7 фич

    dts_scaler = StandardScaler()
    dts_full = dts_scaler.fit_transform(dts_full).astype(np.float32)

    # Строим окна
    X_das, X_dts, y = [], [], []
    for i in range(len(das_vals) - window):
        X_das.append(das_vals[i:i + window].T)          # [channels, window]
        X_dts.append(dts_full[i + window - 1])          # [dts_features]
        y.append(labels[i + window - 1])

    X_das = np.stack(X_das).astype(np.float32)
    X_dts = np.stack(X_dts).astype(np.float32)
    y = np.array(y)
    print(f"[FUSION] DAS={X_das.shape}, DTS={X_dts.shape}, y={np.bincount(y)}")
    return X_das, X_dts, y


def train_fusion(epochs=50, batch_size=32, lr=1e-3):
    X_das, X_dts, y = build_fusion_dataset()

    # split
    idx_train, idx_test = train_test_split(
        np.arange(len(y)), test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    X_das_tr, X_das_te = X_das[idx_train], X_das[idx_test]
    X_dts_tr, X_dts_te = X_dts[idx_train], X_dts[idx_test]
    y_tr, y_te = y[idx_train], y[idx_test]

    das_tr = torch.tensor(X_das_tr).to(DEVICE)
    dts_tr = torch.tensor(X_dts_tr).to(DEVICE)
    yt = torch.tensor(y_tr).to(DEVICE)
    das_te = torch.tensor(X_das_te).to(DEVICE)
    dts_te = torch.tensor(X_dts_te).to(DEVICE)
    ye = torch.tensor(y_te).to(DEVICE)

    # WeightedRandomSampler
    class_counts = np.bincount(y_tr)
    class_weights = 1.0 / np.maximum(class_counts, 1)
    sample_weights = class_weights[y_tr]
    sampler = WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.float32),
        num_samples=len(y_tr), replacement=True,
    )

    ds = TensorDataset(das_tr, dts_tr, yt)
    dl = DataLoader(ds, batch_size=batch_size, sampler=sampler)

    model = FusionModel(
        das_channels=X_das.shape[1],
        dts_features=X_dts.shape[1],
        n_classes=3,
    ).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, factor=0.5)

    alpha = torch.tensor(class_weights / class_weights.sum() * 3,
                         dtype=torch.float32).to(DEVICE)
    loss_fn = FocalLoss(alpha=alpha, gamma=2.0)

    losses = []
    for ep in range(epochs):
        model.train()
        total = 0
        for xd, xt, yb in dl:
            opt.zero_grad()
            logits, _ = model(xd, xt)
            loss = loss_fn(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item() * xd.size(0)
        avg = total / len(ds)
        losses.append(avg)
        scheduler.step(avg)
        if (ep + 1) % 5 == 0 or ep == 0:
            print(f"[FUSION] Эпоха {ep+1}/{epochs}, loss={avg:.4f}")

    torch.save(model.state_dict(), DATA_PROC / "fusion_das_dts.pt")

    # Метрики
    model.eval()
    with torch.no_grad():
        logits, _ = model(das_te, dts_te)
        pred = logits.argmax(dim=1).cpu().numpy()
    print("[FUSION] Отчёт:")
    print(classification_report(ye.cpu().numpy(), pred, digits=3, zero_division=0))

    cm = confusion_matrix(ye.cpu().numpy(), pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens")
    plt.xlabel("Предсказано")
    plt.ylabel("Истина")
    plt.title("Fusion (DAS+DTS): confusion matrix")
    plt.tight_layout()
    plt.savefig(FIGURES / "fusion_confusion.png", dpi=150)
    plt.close()

    plt.figure(figsize=(10, 4))
    plt.plot(losses, marker="o")
    plt.xlabel("Эпоха")
    plt.ylabel("Loss")
    plt.title("Fusion (DAS+DTS): кривая обучения")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES / "fusion_loss.png", dpi=150)
    plt.close()
    print("[FUSION] Графики сохранены")


if __name__ == "__main__":
    train_fusion()