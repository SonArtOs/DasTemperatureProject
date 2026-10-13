"""
Все графики для DTS и DAS. Только matplotlib + seaborn.
Каждый сохраняется в reports/figures/ как PNG.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from src.config import FIGURES, DATA_PROC, LENGTH_STEP
from src.dts_loader import load_processed


plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 150,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "axes.grid": True,
    "grid.alpha": 0.3,
})


def _seg_label(lengths, i):
    start = int(lengths[i * LENGTH_STEP])
    end_idx = min((i + 1) * LENGTH_STEP - 1, len(lengths) - 1)
    end = int(lengths[end_idx])
    return f"{start}–{end} м"


# ============================================================
# DTS
# ============================================================

def plot_dts_heatmap(tensor_agg: np.ndarray, lengths: np.ndarray):
    n_seg, n_time = tensor_agg.shape

    fig, ax = plt.subplots(figsize=(16, 8))
    im = ax.imshow(
        tensor_agg, aspect="auto", cmap="RdYlBu_r",
        interpolation="nearest",
        vmin=17, vmax=27,        # фиксируем шкалу под реальный диапазон
    )
    fig.colorbar(im, ax=ax, label="Температура, °C")

    ax.set_xlabel("Время (шаги по 1 часу)")
    ax.set_ylabel("Сегмент волокна")

    if n_seg <= 25:
        ax.set_yticks(range(n_seg))
        ax.set_yticklabels([_seg_label(lengths, i) for i in range(n_seg)],
                           fontsize=8)
    else:
        step = max(1, n_seg // 10)
        ax.set_yticks(range(0, n_seg, step))
        ax.set_yticklabels([_seg_label(lengths, i)
                            for i in range(0, n_seg, step)], fontsize=8)

    step_x = max(1, n_time // 12)
    ax.set_xticks(range(0, n_time, step_x))
    ax.set_xticklabels([f"{i}" for i in range(0, n_time, step_x)], fontsize=9)

    ax.set_title("DTS: температура по длине волокна и времени")
    fig.tight_layout()
    fig.savefig(FIGURES / "01_dts_heatmap.png")
    plt.close(fig)
    print("[PLOT] 01_dts_heatmap.png")


def plot_dts_lines(tensor_agg: np.ndarray, lengths: np.ndarray):
    n_seg, n_time = tensor_agg.shape
    sample_idx = [0, n_seg // 4, n_seg // 2, 3 * n_seg // 4, n_seg - 1]

    fig, ax = plt.subplots(figsize=(16, 6))
    for idx in sample_idx:
        ax.plot(tensor_agg[idx], linewidth=1.6, label=_seg_label(lengths, idx))

    ax.set_xlabel("Время (шаги по 1 часу)")
    ax.set_ylabel("Температура, °C")
    ax.set_title("DTS: температура в разных сегментах волокна")
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGURES / "02_dts_lines.png")
    plt.close(fig)
    print("[PLOT] 02_dts_lines.png")


def plot_dts_hist(tensor: np.ndarray):
    vals = tensor.flatten()
    vals = vals[~np.isnan(vals)]
    lo, hi = np.percentile(vals, [1, 99])
    vals_cut = vals[(vals >= lo) & (vals <= hi)]

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.hist(vals_cut, bins=60, color="steelblue",
            edgecolor="black", alpha=0.85)
    ax.set_xlabel("Температура, °C")
    ax.set_ylabel("Частота")
    ax.set_title(f"Распределение температур ({lo:.1f}–{hi:.1f} °C)")
    fig.tight_layout()
    fig.savefig(FIGURES / "03_dts_hist.png")
    plt.close(fig)
    print("[PLOT] 03_dts_hist.png")


def plot_dts_rolling_mean(tensor_agg: np.ndarray):
    """Сглаженная средняя температура во времени + полоса ±std."""
    n_seg, n_time = tensor_agg.shape

    fig, ax = plt.subplots(figsize=(16, 5))
    df = pd.DataFrame(tensor_agg.T).rolling(window=5, min_periods=1).mean()
    mean = df.mean(axis=1)
    std = df.std(axis=1)

    ax.plot(mean, color="crimson", linewidth=2,
            label="Среднее по всем сегментам")
    ax.fill_between(range(n_time), mean - std, mean + std,
                    alpha=0.2, color="crimson", label="±1 std")

    ax.set_xlabel("Время (шаги)")
    ax.set_ylabel("Температура, °C")
    ax.set_title("DTS: сглаженная средняя температура во времени")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(FIGURES / "04_dts_rolling.png")
    plt.close(fig)
    print("[PLOT] 04_dts_rolling.png")


# ============================================================
# DAS
# ============================================================

def plot_das_rms(das: pd.DataFrame):
    df = das.set_index("timestamp").resample("1min")["rms"].mean().reset_index()

    fig, ax = plt.subplots(figsize=(16, 5))
    ax.plot(df["timestamp"], df["rms"], color="crimson", linewidth=0.9)
    ax.set_xlabel("Время")
    ax.set_ylabel("RMS")
    ax.set_title("DAS: RMS амплитуды во времени")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(FIGURES / "05_das_rms.png")
    plt.close(fig)
    print("[PLOT] 05_das_rms.png")


def plot_das_bands(das: pd.DataFrame):
    band_cols = [c for c in das.columns if c.startswith("band_")]
    df = das.set_index("timestamp")[band_cols].resample("1min").mean().reset_index()

    fig, ax = plt.subplots(figsize=(16, 6))
    for c in band_cols:
        ax.plot(df["timestamp"], df[c], label=c, alpha=0.85, linewidth=1)
    ax.set_xlabel("Время")
    ax.set_ylabel("Энергия (лог. шкала)")
    ax.set_yscale("log")
    ax.set_title("DAS: энергии в полосах частот")
    ax.legend(loc="best", fontsize=9)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(FIGURES / "06_das_bands.png")
    plt.close(fig)
    print("[PLOT] 06_das_bands.png")


def plot_das_std(das: pd.DataFrame):
    df = das.set_index("timestamp").resample("1min")["std"].mean().reset_index()

    fig, ax = plt.subplots(figsize=(16, 5))
    ax.plot(df["timestamp"], df["std"], color="darkgreen", linewidth=0.9)
    ax.set_xlabel("Время")
    ax.set_ylabel("STD")
    ax.set_title("DAS: STD амплитуды во времени")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(FIGURES / "07_das_std.png")
    plt.close(fig)
    print("[PLOT] 07_das_std.png")


# ============================================================
# iForest — с осью X = длина волокна
# ============================================================

def plot_iforest(ifr: pd.DataFrame):
    """
    Score аномальности по сегментам волокна.
    По оси X — длина волокна (метры), по оси Y — score.
    Красные точки = аномалии.
    """
    fig, ax = plt.subplots(figsize=(16, 6))

    x = ifr["length_start"].values
    scores = ifr["iforest_score"].values
    anom_mask = ifr["iforest_label"].values == -1

    # Все сегменты
    ax.plot(x, scores, linewidth=1.2, color="steelblue",
            marker="o", markersize=4, label="Score")

    # Порог
    ax.axhline(0, color="red", linestyle="--", alpha=0.6, label="Порог 0")

    # Аномалии — красные точки
    ax.scatter(x[anom_mask], scores[anom_mask],
               color="red", s=140, zorder=5,
               edgecolors="black", linewidth=1.2,
               label=f"Аномалии ({anom_mask.sum()})")

    ax.set_xlabel("Длина волокна, м")
    ax.set_ylabel("Anomaly score")
    ax.set_title(f"Isolation Forest: аномальные сегменты волокна "
                 f"(найдено {anom_mask.sum()} из {len(x)})")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(FIGURES / "08_iforest.png")
    plt.close(fig)
    print("[PLOT] 08_iforest.png")


def plot_das_heatmap():
    """
    Heatmap DAS: время × сегмент волокна.
    Каждая ячейка — средняя амплитуда вибрации в этом сегменте в этот момент.
    """
    spatial = pd.read_parquet(DATA_PROC / "das_spatial.parquet")
    seg_cols = [c for c in spatial.columns if c.startswith("seg_")]
    mat = spatial[seg_cols].values.T  # [n_segments, n_time]

    fig, ax = plt.subplots(figsize=(16, 8))
    im = ax.imshow(mat, aspect="auto", cmap="viridis",
                   interpolation="nearest")
    fig.colorbar(im, ax=ax, label="Средняя амплитуда")

    ax.set_xlabel("Время (снимок)")
    ax.set_ylabel("Сегмент волокна (по длине)")
    ax.set_title("DAS: амплитуда вибрации по длине волокна и времени")

    # подписи по Y — каждые 5 сегментов
    n_seg = len(seg_cols)
    step = max(1, n_seg // 10)
    ax.set_yticks(range(0, n_seg, step))
    ax.set_yticklabels([f"seg {i}" for i in range(0, n_seg, step)], fontsize=8)

    fig.tight_layout()
    fig.savefig(FIGURES / "09_das_heatmap.png")
    plt.close(fig)
    print("[PLOT] 09_das_heatmap.png")


# ============================================================
# Точка входа
# ============================================================

def main():
    print("\n[PLOT] DTS...")
    lengths, timestamps, tensor, tensor_agg = load_processed()

    plot_dts_heatmap(tensor_agg, lengths)
    plot_dts_lines(tensor_agg, lengths)
    plot_dts_hist(tensor)
    plot_dts_rolling_mean(tensor_agg)

    print("\n[PLOT] DAS...")
    das = pd.read_parquet(DATA_PROC / "das_features.parquet")
    plot_das_rms(das)
    plot_das_bands(das)
    plot_das_std(das)

    print("\n[PLOT] iForest...")
    ifr = pd.read_parquet(DATA_PROC / "dts_iforest.parquet")
    plot_iforest(ifr)

    print(f"\n[PLOT] Готово. PNG в {FIGURES}")


if __name__ == "__main__":
    main()
    print("\n[PLOT] DAS heatmap...")
    plot_das_heatmap()