import pandas as pd
import numpy as np
from pathlib import Path
from src.config import (
    DTS_RAW, DATA_PROC, FIGURES,
    LENGTH_STEP, TIME_STEP,
)


def parse_timestamp(col_name: str):
    """results_2025_10_1612_00_06_dts -> pd.Timestamp('2025-10-16 12:00')"""
    s = col_name.replace("results_", "").replace("_dts", "")
    parts = s.split("_")
    if len(parts) != 5:
        return None
    year, month, dh, m1, m2 = parts
    try:
        return pd.Timestamp(
            year=int(year), month=int(month),
            day=int(dh[:2]), hour=int(dh[2:]), minute=int(m1),
        )
    except Exception:
        return None


def load_single_dts(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path).sort_values("length").reset_index(drop=True)
    return df


def load_all_dts() -> dict:
    files = sorted(DTS_RAW.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"Нет CSV в {DTS_RAW}")
    result = {}
    for f in files:
        print(f"[DTS] Читаю {f.name}")
        result[f.stem] = load_single_dts(f)
    return result


def build_tensor(dfs: dict):
    """Собирает единый длинный временной ряд [n_lengths × n_times]."""
    print("\n[DTS] Диагностика:")
    for name, df in dfs.items():
        print(f"   {name}: {df.shape}")

    # Общие длины (пересечение)
    common_lengths = None
    for df in dfs.values():
        s = set(df["length"].values)
        common_lengths = s if common_lengths is None else (common_lengths & s)
    common_lengths = sorted(common_lengths)
    print(f"[DTS] Общих длин: {len(common_lengths)}")

    # Все timestamps
    all_ts = {}
    for name, df in dfs.items():
        for c in df.columns:
            if c == "length":
                continue
            ts = parse_timestamp(c)
            if ts is not None and ts not in all_ts:
                all_ts[ts] = (name, c)

    sorted_ts = sorted(all_ts.keys())
    print(f"[DTS] Уникальных timestamps: {len(sorted_ts)}")
    if not sorted_ts:
        raise ValueError("Не удалось распарсить timestamps")
    print(f"[DTS] Диапазон: {sorted_ts[0]} → {sorted_ts[-1]}")

    # Сборка сырого тензора
    n_len = len(common_lengths)
    n_time = len(sorted_ts)
    tensor = np.full((n_len, n_time), np.nan, dtype=np.float32)

    ts_to_idx = {ts: i for i, ts in enumerate(sorted_ts)}

    for name, df in dfs.items():
        sub = df[df["length"].isin(common_lengths)].set_index("length").loc[common_lengths]
        for c in df.columns:
            if c == "length":
                continue
            ts = parse_timestamp(c)
            if ts is None:
                continue
            t = ts_to_idx[ts]
            vals = sub[c].values
            for l in range(n_len):
                tensor[l, t] = vals[l]

    # === ФИЛЬТР МУСОРА (-300 = нет данных) ===
    n_before = int((tensor == -300).sum())
    tensor[tensor <= -100] = np.nan
    print(f"[DTS] Заменено -300 → NaN: {n_before}")
    print(f"[DTS] NaN после фильтра: {int(np.isnan(tensor).sum())} "
          f"из {tensor.size} ({100 * np.isnan(tensor).mean():.1f}%)")

    # === ИНТЕРПОЛЯЦИЯ ===
    print("[DTS] Интерполяция пропусков...")
    for l in range(n_len):
        row = tensor[l]
        mask = np.isnan(row)
        if mask.all():
            continue
        if mask.any():
            idx = np.arange(n_time)
            row[mask] = np.interp(idx[mask], idx[~mask], row[~mask])

    tensor = pd.DataFrame(tensor).ffill(axis=1).bfill(axis=1).values

    # === ОТБРАСЫВАЕМ МЁРТВЫЕ СЕГМЕНТЫ ===
    nan_frac = np.isnan(tensor).mean(axis=1)
    valid_rows = nan_frac < 0.8
    print(f"[DTS] Живых сегментов: {int(valid_rows.sum())} из {len(valid_rows)}")
    tensor = tensor[valid_rows]
    common_lengths = np.array(common_lengths)[valid_rows]

    print(f"[DTS] Сырой тензор: {tensor.shape}")
    return common_lengths, sorted_ts, tensor


def aggregate_tensor(tensor: np.ndarray,
                     length_step: int = LENGTH_STEP,
                     time_step: int = TIME_STEP):
    """Усредняет по блокам: длина → сегменты, время → шаги."""
    n_len, n_time = tensor.shape

    n_len_agg = n_len // length_step
    len_trim = n_len_agg * length_step
    t_len = tensor[:len_trim].reshape(n_len_agg, length_step, n_time).mean(axis=1)

    n_time_agg = n_time // time_step
    time_trim = n_time_agg * time_step
    t_agg = t_len[:, :time_trim].reshape(n_len_agg, n_time_agg, time_step).mean(axis=2)

    print(f"[DTS] Агрегированный тензор: {t_agg.shape}")
    return t_agg


def save_processed(lengths, timestamps, tensor, tensor_agg):
    out = DATA_PROC / "dts_tensor.npz"
    np.savez_compressed(
        out,
        lengths=lengths,
        timestamps=np.array([str(t) for t in timestamps], dtype=object),
        tensor=tensor,
        tensor_agg=tensor_agg,
    )
    print(f"[DTS] Сохранено в {out}")


def load_processed():
    data = np.load(DATA_PROC / "dts_tensor.npz", allow_pickle=True)
    return (data["lengths"], data["timestamps"],
            data["tensor"], data["tensor_agg"])


if __name__ == "__main__":
    dfs = load_all_dts()
    lengths, timestamps, tensor = build_tensor(dfs)
    tensor_agg = aggregate_tensor(tensor)
    save_processed(lengths, timestamps, tensor, tensor_agg)
    print("[DTS] Готово")