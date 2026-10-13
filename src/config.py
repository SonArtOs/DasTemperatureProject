from pathlib import Path

# Пути
ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROC = ROOT / "data" / "processed"
DTS_RAW = DATA_RAW / "dts"
DAS_RAW = DATA_RAW / "das"
FIGURES = ROOT / "reports" / "figures"

DATA_PROC.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)

# --- DTS параметры ---
DTS_WINDOW = 12
DTS_HORIZON = 3
DTS_FS = 1 / 600

LENGTH_STEP = 50     # каждые 50 точек волокна → 1 сегмент
TIME_STEP = 6        # каждые 6 срезов → 1 шаг

# --- DAS параметры ---
DAS_CHUNK = 5000
DAS_FS = 1000
DAS_BANDS = [(0, 10), (10, 50), (50, 200), (200, 1000)]

# --- Общие ---
RANDOM_STATE = 42
DEVICE = "cpu"