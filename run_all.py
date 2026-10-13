import subprocess, sys

STEPS = [
    ("DTS: загрузка + агрегация", "src.dts_loader"),
    ("DAS: загрузка + агрегация", "src.das_loader"),
    ("Фичи", "src.features"),
    ("Все PNG-графики", "src.eda_plots"),
    ("Isolation Forest", "src.model_iforest"),
    ("LSTM: прогноз температуры", "src.model_lstm"),
    ("CNN1D: классификация DAS", "src.model_cnn1d"),
    ("Conv-AE: аномалии DAS", "src.model_autoencoder"),
]

for name, module in STEPS:
    print(f"\n{'='*70}\n▶ {name}\n{'='*70}")
    r = subprocess.run([sys.executable, "-m", module])
    if r.returncode != 0:
        print(f"❌ Ошибка в {name}")
        break

print(f"\n✅ Пайплайн завершён. PNG в reports/figures/")