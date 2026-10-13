import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from src.config import DATA_PROC, FIGURES, RANDOM_STATE


def train_iforest():
    """Объединяем DTS-фичи и DAS-фичи для детекции аномалий."""
    dts = pd.read_parquet(DATA_PROC / "dts_features.parquet")
    dts_cols = ["mean", "std", "min", "max", "slope"]
    X_dts = dts[dts_cols].values.astype(np.float32)
    print(f"[IFOREST] DTS: {X_dts.shape}")

    # === DAS-фичи, выровненные по длине DTS ===
    try:
        das = pd.read_parquet(DATA_PROC / "das_features.parquet")
        das_cols = ["rms", "std", "peak_factor", "zcr",
                    "band_0_10", "band_10_50", "band_50_200", "band_200_1000"]
        das_vals = das[das_cols].values.astype(np.float32)

        # Прореживаем DAS до длины DTS через iloc[::step]
        step = max(1, len(das_vals) // len(X_dts))
        das_agg = das_vals[::step][:len(X_dts)]

        # Если получилось меньше строк, чем DTS — дополняем нулями
        if len(das_agg) < len(X_dts):
            pad = np.zeros((len(X_dts) - len(das_agg), das_agg.shape[1]),
                           dtype=np.float32)
            das_agg = np.vstack([das_agg, pad])

        X_combined = np.hstack([X_dts, das_agg]).astype(np.float32)
        print(f"[IFOREST] DAS: {das_agg.shape}, Combined: {X_combined.shape}")
    except Exception as e:
        print(f"[IFOREST] DAS не удалось объединить ({e}), только DTS")
        X_combined = X_dts

    # === Чистка NaN ===
    mask = ~np.isnan(X_combined).any(axis=1)
    X_clean = X_combined[mask]
    print(f"[IFOREST] После чистки: {X_clean.shape}")

    # === Нормализация ===
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_clean)

    # === Модель ===
    model = IsolationForest(
        n_estimators=200,
        contamination=0.05,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(Xs)

    scores = model.decision_function(Xs)
    labels = model.predict(Xs)

    # === Сохранение результатов ===
    dts_result = dts.iloc[mask].copy().reset_index(drop=True)
    dts_result["iforest_score"] = scores
    dts_result["iforest_label"] = labels
    dts_result.to_parquet(DATA_PROC / "dts_iforest.parquet", index=False)

    n_anom = int((labels == -1).sum())
    print(f"[IFOREST] Аномалий: {n_anom} из {len(labels)}")

    # === График ===
    plt.figure(figsize=(14, 5))
    plt.plot(scores, linewidth=0.9, color="darkblue", label="Anomaly score")
    plt.axhline(0, color="red", linestyle="--", alpha=0.6, label="Порог 0")

    anom_mask = labels == -1
    if anom_mask.any():
        x_anom = np.where(anom_mask)[0]
        plt.scatter(x_anom, scores[anom_mask],
                    color="red", s=80, edgecolors="black",
                    zorder=5, label=f"Аномалии ({n_anom})")

    plt.xlabel("Сегмент")
    plt.ylabel("Anomaly score")
    plt.title("Isolation Forest: score аномальности (DTS + DAS)")
    plt.legend(loc="best")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES / "iforest_scores.png", dpi=150)
    plt.close()
    print("[PLOT] iforest_scores.png")


if __name__ == "__main__":
    train_iforest()