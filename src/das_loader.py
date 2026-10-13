import pandas as pd
import numpy as np
import ast
from pathlib import Path
from tqdm import tqdm
from src.config import DAS_RAW, DAS_CHUNK, DATA_PROC, FIGURES
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def find_das_file() -> Path:
    files = list(DAS_RAW.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"Нет CSV в {DAS_RAW}")
    return files[0]


def parse_points(s: str) -> np.ndarray:
    """Парсит строку '[[x,y],[x,y],...]' в массив (N, 2)."""
    try:
        arr = ast.literal_eval(s)
        return np.array(arr, dtype=np.float32)
    except Exception:
        return np.array([], dtype=np.float32)


def load_das_aggregated(chunk_size: int = DAS_CHUNK,
                        n_segments: int = 50) -> pd.DataFrame:
    """
    Читает большой CSV чанками.
    Для каждой записи считает:
      - общие статистики (mean, std, rms, band_*)
      - пространственный профиль по n_segments сегментам волокна
    """
    path = find_das_file()
    print(f"[DAS] Читаю {path}")

    rows = []
    reader = pd.read_csv(path, chunksize=chunk_size)
    for chunk in tqdm(reader, desc="DAS chunks"):
        for _, r in chunk.iterrows():
            pts = parse_points(r["points"])
            if pts.size == 0:
                continue
            x = pts[:, 0]
            y = pts[:, 1]

            # общие статистики
            mean = float(np.mean(y))
            std = float(np.std(y))
            mx = float(np.max(y))
            mn = float(np.min(y))
            rms = float(np.sqrt(np.mean(y ** 2)))
            peak_factor = float(mx / (rms + 1e-9))
            zc = int(np.sum(np.diff(np.sign(y)) != 0))
            zcr = float(zc / len(y))

            # энергии в полосах
            fft = np.abs(np.fft.rfft(y))
            freqs = np.fft.rfftfreq(len(y), d=1.0)
            band_energies = {}
            for lo, hi in [(0, 10), (10, 50), (50, 200), (200, 1000)]:
                mask = (freqs >= lo) & (freqs < hi)
                band_energies[f"band_{lo}_{hi}"] = float(np.sum(fft[mask] ** 2))

            # пространственный профиль
            seg_edges = np.linspace(x.min(), x.max(), n_segments + 1)
            profile = {}
            for s in range(n_segments):
                lo_x, hi_x = seg_edges[s], seg_edges[s + 1]
                mask = (x >= lo_x) & (x < hi_x)
                profile[f"seg_{s:02d}"] = float(np.mean(y[mask])) if mask.any() else 0.0

            row = {
                "timestamp": r["timestamp"],
                "channel_id": r["channel_id"],
                "mean": mean, "std": std, "max": mx, "min": mn,
                "rms": rms, "peak_factor": peak_factor, "zcr": zcr,
                **band_energies,
                **profile,
            }
            rows.append(row)

    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    print(f"[DAS] Агрегировано: {df.shape} "
          f"(строк={df.shape[0]}, колонок={df.shape[1]})")

    # сохраняем полный
    out = DATA_PROC / "das_features.parquet"
    df.to_parquet(out, index=False)
    print(f"[DAS] Сохранено: {out}")

    # отдельно — только пространственный профиль
    seg_cols = [c for c in df.columns if c.startswith("seg_")]
    spatial = df[["timestamp", "channel_id"] + seg_cols].copy()
    spatial.to_parquet(DATA_PROC / "das_spatial.parquet", index=False)
    print(f"[DAS] Пространственный профиль: {spatial.shape}")

    return df


def plot_snapshot():
    """Один снимок DAS: амплитуда по расстоянию."""
    path = find_das_file()
    df = pd.read_csv(path, nrows=1)
    pts = parse_points(df.iloc[0]["points"])
    if pts.size == 0:
        return

    x, y = pts[:, 0], pts[:, 1]
    plt.figure(figsize=(14, 4))
    plt.plot(x, y, linewidth=0.8)
    plt.xlabel("Расстояние (м)")
    plt.ylabel("Амплитуда")
    plt.title(f"DAS: снимок, канал {df.iloc[0]['channel_id']}, "
              f"{df.iloc[0]['timestamp']}")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES / "das_snapshot.png", dpi=150)
    plt.close()
    print(f"[DAS] Снимок: {FIGURES / 'das_snapshot.png'}")


if __name__ == "__main__":
    plot_snapshot()
    load_das_aggregated()