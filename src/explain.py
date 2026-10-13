"""
Объяснение событий: что произошло и почему.
- Факты: наблюдаемые паттерны (вибрация + ΔT).
- Бинарный CatBoost: какие признаки важны для детекции события.
- Корреляции: честный анализ (без ML).
- Визуализация: карта событий + важность признаков.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from scipy.stats import pointbiserialr
from src.config import DATA_PROC, FIGURES, RANDOM_STATE

np.random.seed(RANDOM_STATE)


# ============================================================
# 1. Построение таблицы событий
# ============================================================

def build_event_table():
    das = pd.read_parquet(DATA_PROC / "das_features.parquet")
    dts = pd.read_parquet(DATA_PROC / "dts_features.parquet")

    step = max(1, len(das) // len(dts))
    das_aligned = das.iloc[::step].reset_index(drop=True).iloc[:len(dts)]

    dts_vals = dts["mean"].values
    dts_delta = np.diff(dts_vals, prepend=dts_vals[0])
    rms = das_aligned["rms"].values[:len(dts)]

    rms_high = float(np.quantile(rms, 0.75))
    rms_low = float(np.quantile(rms, 0.30))
    d_thr = 0.05

    print(f"[EXPLAIN] Пороги: rms_high={rms_high:.3f}, rms_low={rms_low:.3f}, d_thr={d_thr}")

    events = []
    for i in range(len(dts)):
        r = rms[i]
        d = dts_delta[i]

        if r > rms_high and abs(d) > d_thr:
            label = "vib_and_temp"
        elif r > rms_high and abs(d) <= d_thr:
            label = "vib_no_temp"
        elif r < rms_low and d > d_thr:
            label = "temp_rise_quiet"
        elif r < rms_low and d < -d_thr:
            label = "temp_drop_quiet"
        elif r < rms_low and abs(d) <= d_thr:
            label = "quiet"
        else:
            label = "mixed"

        events.append({
            "segment": i,
            "rms": r,
            "rms_high": rms_high,
            "rms_low": rms_low,
            "dts_mean": dts_vals[i],
            "dts_std": dts["std"].values[i],
            "dts_slope": dts["slope"].values[i],
            "dts_delta": d,
            "das_std": das_aligned["std"].values[i] if i < len(das_aligned) else np.nan,
            "das_zcr": das_aligned["zcr"].values[i] if i < len(das_aligned) else np.nan,
            "label": label,
        })

    df = pd.DataFrame(events)
    df.to_parquet(DATA_PROC / "event_explanations.parquet", index=False)
    print(f"[EXPLAIN] Всего событий: {len(df)}")
    print(df["label"].value_counts())
    return df


# ============================================================
# 2a. Корреляционный анализ (без ML)
# ============================================================

def correlation_analysis(df: pd.DataFrame):
    print("\n[EXPLAIN] Корреляционный анализ (point-biserial):")
    feature_cols = ["rms", "dts_mean", "dts_std", "dts_slope", "dts_delta"]

    results = []
    for label in df["label"].unique():
        y_bin = (df["label"] == label).astype(int).values
        for col in feature_cols:
            x = df[col].fillna(0).values
            if len(np.unique(x)) < 2:
                continue
            r, p = pointbiserialr(y_bin, x)
            results.append({
                "label": label,
                "feature": col,
                "corr": r,
                "p_value": p,
                "significant": p < 0.05,
            })

    res_df = pd.DataFrame(results)
    res_df.to_parquet(DATA_PROC / "event_correlations.parquet", index=False)
    print(res_df.to_string(index=False))
    return res_df


# ============================================================
# 2b. Бинарный CatBoost
# ============================================================

def train_catboost_binary(df: pd.DataFrame):
    """Бинарная классификация: событие vs фон."""
    df = df.copy()
    df["is_event"] = (df["label"].isin(["vib_and_temp", "temp_rise_quiet",
                                        "temp_drop_quiet", "vib_no_temp"])).astype(int)

    feature_cols = ["rms", "dts_mean", "dts_std", "dts_slope", "dts_delta"]
    X = df[feature_cols].fillna(0).values
    y = df["is_event"].values

    print(f"\n[CATBOOST] Бинарная задача: классы={np.bincount(y)}")

    if len(np.unique(y)) < 2 or min(np.bincount(y)) < 2:
        print("[CATBOOST] Слишком мало примеров в одном из классов. Пропускаю.")
        return None, None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=RANDOM_STATE, stratify=y
    )

    model = CatBoostClassifier(
        iterations=100,
        depth=3,
        learning_rate=0.1,
        loss_function="Logloss",
        verbose=False,
        random_seed=RANDOM_STATE,
        auto_class_weights="Balanced",
    )
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    print("[CATBOOST] Отчёт:")
    print(classification_report(y_test, pred, digits=3, zero_division=0))

    imp = model.get_feature_importance()
    imp_df = pd.DataFrame({
        "feature": feature_cols,
        "importance": imp,
    }).sort_values("importance", ascending=False)
    print("\n[CATBOOST] Важность признаков:")
    print(imp_df.to_string(index=False))
    imp_df.to_parquet(DATA_PROC / "catboost_importance.parquet", index=False)

    return model, imp_df, (X_test, y_test, pred)


# ============================================================
# 3. Визуализация
# ============================================================

def plot_events(df: pd.DataFrame):
    fig, ax1 = plt.subplots(figsize=(16, 6))

    ax1.plot(df["segment"], df["rms"], color="steelblue",
             linewidth=1.2, label="DAS RMS")
    ax1.axhline(df["rms_high"].iloc[0], color="steelblue",
                linestyle="--", alpha=0.4, label="Порог high")
    ax1.axhline(df["rms_low"].iloc[0], color="steelblue",
                linestyle=":", alpha=0.4, label="Порог low")
    ax1.set_xlabel("Сегмент волокна")
    ax1.set_ylabel("DAS RMS", color="steelblue")
    ax1.tick_params(axis="y", labelcolor="steelblue")

    ax2 = ax1.twinx()
    ax2.plot(df["segment"], df["dts_delta"], color="crimson",
             linewidth=1.5, alpha=0.85, label="ΔT (DTS)")
    ax2.axhline(0, color="crimson", linestyle="--", alpha=0.3)
    ax2.set_ylabel("ΔT, °C", color="crimson")
    ax2.tick_params(axis="y", labelcolor="crimson")

    colors = {
        "vib_and_temp": "orange",
        "vib_no_temp": "purple",
        "temp_rise_quiet": "red",
        "temp_drop_quiet": "blue",
        "quiet": "green",
        "mixed": "gray",
    }
    for label, color in colors.items():
        sub = df[df["label"] == label]
        if len(sub) == 0:
            continue
        ax1.scatter(sub["segment"], sub["rms"], color=color,
                    s=40, zorder=5, label=label, alpha=0.8)

    ax1.legend(loc="upper left", fontsize=8, ncol=2)
    ax2.legend(loc="upper right", fontsize=8)
    plt.title("Карта событий: DAS (вибрация) + DTS (температура)")
    fig.tight_layout()
    fig.savefig(FIGURES / "explain_events.png", dpi=150)
    plt.close()
    print("[PLOT] explain_events.png")


def plot_importance(imp_df: pd.DataFrame):
    if imp_df is None or len(imp_df) == 0:
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(data=imp_df, x="importance", y="feature",
                hue="feature", palette="viridis", legend=False, ax=ax)
    ax.set_title("CatBoost: важность признаков (бинарная задача)")
    ax.set_xlabel("Importance")
    ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(FIGURES / "catboost_importance.png", dpi=150)
    plt.close()
    print("[PLOT] catboost_importance.png")


def plot_correlations(res_df: pd.DataFrame):
    """Heatmap корреляций признак-класс."""
    pivot = res_df.pivot(index="feature", columns="label", values="corr")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="RdBu_r",
                center=0, vmin=-1, vmax=1, ax=ax)
    ax.set_title("Корреляции признаков с типами событий (point-biserial)")
    fig.tight_layout()
    fig.savefig(FIGURES / "event_correlations.png", dpi=150)
    plt.close()
    print("[PLOT] event_correlations.png")


def plot_catboost_confusion(X_test, y_test, pred):
    """Confusion matrix для бинарной задачи."""
    if y_test is None:
        return
    cm = confusion_matrix(y_test, pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Oranges", ax=ax,
                xticklabels=["Фон", "Событие"],
                yticklabels=["Фон", "Событие"])
    ax.set_xlabel("Предсказано")
    ax.set_ylabel("Истина")
    ax.set_title("CatBoost: confusion matrix (бинарная)")
    fig.tight_layout()
    fig.savefig(FIGURES / "catboost_confusion.png", dpi=150)
    plt.close()
    print("[PLOT] catboost_confusion.png")


# ============================================================
# Точка входа
# ============================================================

def main():
    print("\n[EXPLAIN] Строю таблицу событий...")
    df = build_event_table()

    print("\n[EXPLAIN] Корреляционный анализ...")
    corr_df = correlation_analysis(df)
    plot_correlations(corr_df)

    print("\n[EXPLAIN] Обучаю бинарный CatBoost...")
    try:
        result = train_catboost_binary(df)
        if result[0] is not None:
            model, imp_df, (X_test, y_test, pred) = result
            plot_importance(imp_df)
            plot_catboost_confusion(X_test, y_test, pred)
    except Exception as e:
        print(f"[CATBOOST] Не удалось обучить: {e}")

    print("\n[EXPLAIN] Рисую карту событий...")
    plot_events(df)

    print("\n[EXPLAIN] Готово. Графики в reports/figures/")


if __name__ == "__main__":
    main()