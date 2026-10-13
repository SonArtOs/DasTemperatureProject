import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from src.config import (
    DATA_PROC, FIGURES, DEVICE, RANDOM_STATE,
    DTS_WINDOW, DTS_HORIZON,
)
from src.dts_loader import load_processed

torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)


class LSTMForecaster(nn.Module):
    """
    LSTM, которая видит ВСЕ сегменты волокна одновременно.
    Вход:  [B, window, n_lengths]
    Выход: [B] — предсказание средней температуры
    """
    def __init__(self, n_lengths, hidden=32, layers=1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_lengths,   # ВСЕ сегменты сразу — вектор признаков
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            dropout=0.1 if layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.lstm(x)       # [B, window, hidden]
        out = out[:, -1, :]         # [B, hidden]
        out = self.head(out)        # [B, 1]
        return out.squeeze(-1)      # [B]


def make_windows(tensor_agg, window=DTS_WINDOW, horizon=DTS_HORIZON):
    """
    X: [N, window, n_seg] — окно температур всех сегментов
    y: [N]                — средняя температура через horizon шагов
    """
    n_seg, n_time = tensor_agg.shape
    X, y = [], []
    for t in range(0, n_time - window - horizon):
        X.append(tensor_agg[:, t:t + window].T)                  # [window, n_seg]
        y.append(tensor_agg[:, t + window + horizon - 1].mean()) # средняя
    X = np.stack(X).astype(np.float32)
    y = np.array(y, dtype=np.float32)
    print(f"[LSTM] Окна: X={X.shape}, y={y.shape}")
    return X, y


def train_lstm(epochs=100, batch_size=16, lr=1e-3):
    lengths, timestamps, tensor, tensor_agg = load_processed()
    print(f"[LSTM] Агрегированный тензор: {tensor_agg.shape}")

    X, y = make_windows(tensor_agg)
    if len(X) < 20:
        print("[LSTM] Слишком мало окон, обучение пропущено")
        return

    # === НОРМАЛИЗАЦИЯ ===
    n, w, s = X.shape
    X_flat = X.reshape(-1, s)
    scaler_X = StandardScaler().fit(X_flat)
    X = scaler_X.transform(X_flat).reshape(n, w, s).astype(np.float32)

    scaler_y = StandardScaler().fit(y.reshape(-1, 1))
    y_scaled = scaler_y.transform(y.reshape(-1, 1)).flatten().astype(np.float32)

    # разбиение train/test по времени
    n_train = int(n * 0.8)
    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y_scaled[:n_train], y_scaled[n_train:]
    y_test_orig = y[n_train:]

    Xt = torch.tensor(X_train).to(DEVICE)
    yt = torch.tensor(y_train).to(DEVICE)
    Xe = torch.tensor(X_test).to(DEVICE)
    ye = torch.tensor(y_test).to(DEVICE)

    ds = TensorDataset(Xt, yt)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

    model = LSTMForecaster(n_lengths=s, hidden=32, layers=1).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=10, factor=0.5)
    loss_fn = nn.HuberLoss()

    losses = []
    for ep in range(epochs):
        model.train()
        total = 0
        for xb, yb in dl:
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item() * xb.size(0)
        avg = total / len(ds)
        losses.append(avg)
        scheduler.step(avg)
        if (ep + 1) % 10 == 0 or ep == 0:
            print(f"[LSTM] Эпоха {ep+1}/{epochs}, loss={avg:.5f}")

    torch.save(model.state_dict(), DATA_PROC / "lstm_dts.pt")

    # === Метрики в ОРИГИНАЛЬНОМ масштабе ===
    model.eval()
    with torch.no_grad():
        pred_scaled = model(Xe).cpu().numpy().reshape(-1, 1)
    pred_orig = scaler_y.inverse_transform(pred_scaled).flatten()
    mae = float(np.mean(np.abs(pred_orig - y_test_orig)))
    rmse = float(np.sqrt(np.mean((pred_orig - y_test_orig) ** 2)))
    print(f"[LSTM] Test MAE={mae:.3f} °C, RMSE={rmse:.3f} °C")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(losses)
    ax.set_xlabel("Эпоха")
    ax.set_ylabel("Huber loss")
    ax.set_title("LSTM: кривая обучения")
    fig.tight_layout()
    fig.savefig(FIGURES / "10_lstm_loss.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(y_test_orig, label="Факт", linewidth=1.5, marker="o", markersize=3)
    ax.plot(pred_orig, label="Прогноз", linewidth=1.5, marker="s", markersize=3, alpha=0.85)
    ax.set_xlabel("Окно теста")
    ax.set_ylabel("Средняя температура, °C")
    ax.set_title(f"LSTM: прогноз vs факт (MAE={mae:.3f} °C)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES / "11_lstm_prediction.png")
    plt.close(fig)
    print("[PLOT] 11_lstm_prediction.png")


if __name__ == "__main__":
    train_lstm()