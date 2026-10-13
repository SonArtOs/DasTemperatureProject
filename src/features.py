"""
Извлечение фич из DTS и DAS для моделей.
"""
import numpy as np
import pandas as pd
from src.config import DATA_PROC, LENGTH_STEP, DTS_WINDOW


def add_dts_features(tensor_agg: np.ndarray, lengths: np.ndarray) -> pd.DataFrame:
    """
    Считает простые статистики по каждому сегменту волокна во времени:
    mean, std, min, max, slope (тренд).
    Возвращает DataFrame, где каждая строка = сегмент волокна.
    """
    n_seg, n_time = tensor_agg.shape

    rows = []
    for i in range(n_seg):
        seg = tensor_agg[i]
        # линейный тренд через полифит 1-й степени
        x = np.arange(n_time)
        slope = float(np.polyfit(x, seg, 1)[0]) if n_time > 1 else 0.0

        rows.append({
            "segment": i,
            "length_start": float(lengths[i * LENGTH_STEP]),
            "length_end": float(lengths[min((i + 1) * LENGTH_STEP - 1, len(lengths) - 1)]),
            "mean": float(np.mean(seg)),
            "std": float(np.std(seg)),
            "min": float(np.min(seg)),
            "max": float(np.max(seg)),
            "slope": slope,
        })

    df = pd.DataFrame(rows)
    out = DATA_PROC / "dts_features.parquet"
    df.to_parquet(out, index=False)
    print(f"[FEAT] DTS features: {df.shape} → {out}")
    return df


def main():
    # Загружаем агрегированный тензор
    data = np.load(DATA_PROC / "dts_tensor.npz", allow_pickle=True)
    lengths = data["lengths"]
    tensor_agg = data["tensor_agg"]
    print(f"[FEAT] Загружено: lengths={lengths.shape}, tensor_agg={tensor_agg.shape}")

    add_dts_features(tensor_agg, lengths)


if __name__ == "__main__":
    main()