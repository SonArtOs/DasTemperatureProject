"""
dts_labeling.py — разметка температурных данных распределённого волоконного сенсора (DTS).

Что делает модуль, по шагам:
  1. ЗАГРУЗКА      — читает файлы двух форматов (длинный: колонка points; широкий: колонки results_..._dts)
                     и приводит их к одной таблице "расстояние x время".
  2. ПОДГОТОВКА    — отрезает калибровочную катушку в начале кабеля и приводит время к равномерной сетке.
  3. ОЧИСТКА       — вычитает у каждого канала его обычный уровень и общий фон здания.
  4. МАСКА         — помечает точки, которые сильно и долго отклоняются от нормы.
  5. РАЗМЕТКА      — нарезает на окна и ставит метку: 1 = аномалия, 0 = норма, -1 = спорное.

Быстрый старт в Jupyter:

    from dts_labeling import load_dts_long, label_dts, plot_overview
    data = load_dts_long('cod_zhivopisnaya_140926.csv')
    result = label_dts(data, skip_start_m=17.5)
    print(result.labels['label'].value_counts())
    result.labels.to_csv('labels.csv', index=False)
    plot_overview(result)

Если в записи есть длинные паузы (label_dts выдаст ошибку), режем на непрерывные куски:

    from dts_labeling import split_continuous
    for part in split_continuous(data):
        result = label_dts(part)

Если запись разрезана на несколько кусков, размечай их с общим базовым уровнем (иначе метки
одного и того же события в разных кусках могут не совпасть):

    parts = [p for p in split_continuous(data) if p.matrix.shape[1] >= 90]
    base = common_baseline(parts, skip_start_m=40)
    results = [label_dts(p, skip_start_m=40, baseline=base) for p in parts]

ВАЖНО: пороги по умолчанию подобраны на одном файле (cod_zhivopisnaya). Для другого файла
их нужно проверить глазами (plot_overview, show_window) и при необходимости поменять.
"""

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import binary_opening


# =====================================================================
# Контейнеры данных
# =====================================================================

@dataclass
class DTSData:
    """Одна запись: температура в виде таблицы 'расстояние x время'."""
    distances: np.ndarray        # расстояние вдоль кабеля, метры, форма (n_каналов,)
    timestamps: pd.DatetimeIndex  # моменты измерений, форма (n_измерений,)
    matrix: np.ndarray           # температура, °C, форма (n_каналов, n_измерений)


@dataclass
class LabelResult:
    """Всё, что получилось после разметки (чтобы можно было строить графики)."""
    data: DTSData          # подготовленные данные (без катушки, с равномерным временем)
    residual: np.ndarray   # T2: отклонение от нормы после очистки
    mask: np.ndarray       # True там, где точка считается аномальной
    labels: pd.DataFrame   # таблица меток по окнам
    params: dict           # какие настройки использовались (для отчёта)


# =====================================================================
# 1. ЗАГРУЗКА
# =====================================================================

def _to_timestamp(x, tz):
    """Приводит строку/время к метке времени в нужном часовом поясе."""
    t = pd.Timestamp(x)
    return t.tz_localize(tz) if t.tzinfo is None else t.tz_convert(tz)


def _parse_times(values, tz=None):
    """
    Разбирает колонку времени. В одном файле могут встречаться записи и с долями секунды
    ('09:33:02.194167+00'), и без ('13:28:51+00'); новые версии pandas по умолчанию придираются
    к такому, поэтому формат указываем явно (с запасными вариантами для старых версий).
    """
    for kwargs in ({'format': 'ISO8601'}, {'format': 'mixed'}, {}):
        try:
            t = pd.to_datetime(values, utc=True, **kwargs)
            break
        except (TypeError, ValueError):
            t = None
    if t is None:
        raise ValueError('Не удалось разобрать колонку timestamp')
    return t.dt.tz_convert(tz) if tz is not None else t


def list_streams(path, nrows=2000):
    """
    Для 'длинного' формата: показывает, какие потоки (dts_id, channel_id) есть в начале файла.
    Нужно, чтобы выбрать, какой именно поток загружать (особенно в огромном cosmos.csv).
    """
    head = pd.read_csv(path, nrows=nrows, usecols=['dts_id', 'channel_id', 'timestamp'])
    head['timestamp'] = _parse_times(head['timestamp'])
    out = (head.groupby(['dts_id', 'channel_id'])['timestamp']
           .agg(['count', 'min', 'max']).reset_index())
    print(out.to_string(index=False))
    return out


def load_dts_long(path, dts_id=None, channel_id=None, start=None, end=None,
                  tz='Europe/Moscow', chunksize=500, max_rows=None):
    """
    Читает 'длинный' формат: колонки dts_id, channel_id, timestamp, points,
    где points — текст вида [[расстояние, температура], ...]. Одна строка = один скан всего кабеля.

    Файл читается кусками, поэтому большие файлы не забивают память целиком.
    dts_id / channel_id — выбрать конкретный поток; start / end — взять только часть времени
    (например start='2026-01-15 09:00', end='2026-01-15 12:00'); max_rows — прочитать не больше строк.
    """
    start = _to_timestamp(start, tz) if start is not None else None
    end = _to_timestamp(end, tz) if end is not None else None

    times, values = [], []
    streams = {}
    dist_ref = np.array([])
    n_taken = 0

    reader = pd.read_csv(path, chunksize=chunksize,
                         usecols=['dts_id', 'channel_id', 'timestamp', 'points'])
    for chunk in reader:
        if dts_id is not None:
            chunk = chunk[chunk['dts_id'] == dts_id]
        if channel_id is not None:
            chunk = chunk[chunk['channel_id'] == channel_id]
        if chunk.empty:
            continue

        ts = _parse_times(chunk['timestamp'], tz)
        keep = np.ones(len(chunk), dtype=bool)
        if start is not None:
            keep &= (ts >= start).to_numpy()
        if end is not None:
            keep &= (ts <= end).to_numpy()
        chunk, ts = chunk[keep], ts[keep]

        for t, s, did, cid in zip(ts, chunk['points'], chunk['dts_id'], chunk['channel_id']):
            arr = np.asarray(json.loads(s), dtype=np.float32)   # форма (n, 2)
            if len(arr) > len(dist_ref):
                dist_ref = arr[:, 0].astype(np.float64)         # самая длинная шкала расстояний
            times.append(t)
            values.append(arr[:, 1])
            streams[(did, cid)] = streams.get((did, cid), 0) + 1
            n_taken += 1
            if max_rows is not None and n_taken >= max_rows:
                break
        if len(streams) > 1:
            raise ValueError('В файле несколько потоков (разные волокна нельзя смешивать в одну таблицу): '
                             + ', '.join(f'dts_id={k[0]}, channel_id={k[1]} ({v} сканов)' for k, v in streams.items())
                             + '. Укажи нужный: load_dts_long(path, channel_id=...).')
        if max_rows is not None and n_taken >= max_rows:
            break

    if not times:
        raise ValueError('Не нашлось ни одной подходящей строки. Проверь dts_id / channel_id / start / end '
                         '(посмотреть доступные потоки: list_streams(path)).')

    # Строки могут быть разной длины (часть сканов короче) — недостающее заполняем NaN
    matrix = np.full((len(dist_ref), len(times)), np.nan, dtype=np.float32)
    for i, v in enumerate(values):
        matrix[:len(v), i] = v

    order = np.argsort(pd.DatetimeIndex(times).asi8)
    return DTSData(distances=dist_ref,
                   timestamps=pd.DatetimeIndex(times)[order],
                   matrix=matrix[:, order])


_WIDE_COL = re.compile(r'results_(\d{4})_(\d{2})_(\d{2})(\d{2})_(\d{2})_(\d{2})_dts')


def load_dts_wide(path_or_folder, tz=None):
    """
    Читает 'широкий' формат (папка dts_data_cosmos_ch02_mapping): колонка length (метры) и по колонке
    results_ГГГГ_ММ_ДДЧЧ_мм_сс_dts на каждый момент времени. Можно указать один файл или целую папку
    (тогда часовые файлы склеиваются по порядку).
    """
    p = Path(path_or_folder)
    files = sorted(p.glob('*.csv')) if p.is_dir() else [p]
    if not files:
        raise ValueError(f'В {p} нет csv-файлов')

    dist_ref, mats, times = None, [], []
    for f in files:
        df = pd.read_csv(f)
        dist = df['length'].to_numpy(dtype=np.float64)
        if dist_ref is None:
            dist_ref = dist
        elif len(dist) != len(dist_ref) or not np.allclose(dist, dist_ref):
            raise ValueError(f'Шкала расстояний в {f.name} отличается от первого файла')
        cols = []
        for c in df.columns:
            m = _WIDE_COL.fullmatch(c)
            if m:
                y, mo, d, h, mi, s = map(int, m.groups())
                times.append(pd.Timestamp(year=y, month=mo, day=d, hour=h, minute=mi, second=s))
                cols.append(c)
        mats.append(df[cols].to_numpy(dtype=np.float32))

    ts = pd.DatetimeIndex(times)
    if tz is not None:
        ts = ts.tz_localize(tz)
    matrix = np.concatenate(mats, axis=1)
    order = np.argsort(ts.asi8)
    return DTSData(distances=dist_ref, timestamps=ts[order], matrix=matrix[:, order])


# =====================================================================
# 2. ПОДГОТОВКА
# =====================================================================

def drop_calibration_coil(data, skip_start_m=17.5):
    """
    Отрезает первые метры кабеля: там обычно калибровочная катушка прибора, а не здание
    (на cod_zhivopisnaya это ~17.5 м). Для других файлов длину катушки нужно проверить глазами:
    нарисуй средний профиль первых ~30 м, катушка видна как регулярная 'гребёнка' и резкий скачок.
    """
    keep = data.distances > skip_start_m
    return DTSData(data.distances[keep], data.timestamps, data.matrix[keep])


def _gaps_minutes(timestamps):
    """Паузы между соседними измерениями в минутах (не зависит от внутренней единицы времени pandas)."""
    d = np.diff(timestamps.to_numpy()).astype('timedelta64[ms]').astype(np.float64)
    return d / 1000.0 / 60.0


def split_continuous(data, max_gap_min=20.0):
    """
    Режет запись на непрерывные куски: новый кусок начинается там, где пауза между измерениями
    больше max_gap_min минут. Так мы не выдумываем данные внутри длинных пропусков.
    Использование: for part in split_continuous(data): result = label_dts(part)
    """
    gaps_min = _gaps_minutes(data.timestamps)
    cuts = np.flatnonzero(gaps_min > max_gap_min) + 1
    bounds = np.concatenate([[0], cuts, [len(data.timestamps)]])
    return [DTSData(data.distances, data.timestamps[a:b], data.matrix[:, a:b])
            for a, b in zip(bounds[:-1], bounds[1:]) if b - a >= 2]


def to_uniform_time(data, step_min=2.0, max_nan_share=0.5, max_gap_min=20.0):
    """
    Приводит время к равномерной сетке (шаг step_min минут). Где измерения чаще — усредняет,
    где реже — достраивает значения между соседними. Нужно, потому что прибор может менять частоту
    измерений прямо по ходу записи. Каналы, где слишком много пропусков, выбрасываются.
    Если в записи есть пауза длиннее max_gap_min минут — выдаёт ошибку (интерполировать через
    длинную паузу значит выдумывать данные): сначала разрежь запись через split_continuous().
    """
    gaps_min = _gaps_minutes(data.timestamps)
    if len(gaps_min) and gaps_min.max() > max_gap_min:
        i = int(np.argmax(gaps_min))
        raise ValueError(f'В записи есть пауза {gaps_min.max():.0f} мин после {data.timestamps[i]}. '
                         f'Разрежь запись на непрерывные куски: split_continuous(data).')
    df = pd.DataFrame(data.matrix.T, index=data.timestamps)
    df = df.resample(f'{int(round(step_min * 60))}s').mean()

    # названия колонок df — это номера исходных каналов, по ним потом восстанавливаем расстояния
    nan_share = df.isna().mean(axis=0).to_numpy()
    df = df.loc[:, nan_share <= max_nan_share].interpolate(limit_direction='both')
    df = df.loc[:, ~df.isna().any(axis=0).to_numpy()]
    kept_idx = df.columns.to_numpy()
    if len(kept_idx) == 0:
        raise ValueError('После очистки не осталось ни одного канала: слишком много пропусков в данных.')
    return DTSData(distances=data.distances[kept_idx],
                   timestamps=df.index,
                   matrix=df.to_numpy(dtype=np.float32).T)


# =====================================================================
# 3-4. ОЧИСТКА И МАСКА
# =====================================================================

def compute_residual(T, baseline=None):
    """
    Убирает из температуры 'нормальное':
      - у каждого канала вычитает его обычный (медианный) уровень -> уходят постоянные тёплые места;
        если передан baseline (общий для нескольких кусков записи), берётся он, а не уровень этого куска;
      - в каждый момент вычитает медиану по всем каналам -> уходит общий суточный фон здания.
    Остаётся отклонение конкретного места от его нормы.
    """
    T = T.astype(np.float64)
    if baseline is None:
        base = np.median(T, axis=1, keepdims=True)
    else:
        base = np.asarray(baseline, dtype=np.float64)[:, None]
    T1 = T - base
    return T1 - np.median(T1, axis=0, keepdims=True)


def anomaly_mask(T2, step_min, z_thr=6.0, min_dev_c=3.0, min_duration_min=6.0):
    """
    Точка считается аномальной, если одновременно:
      - отклонение > z_thr 'обычных шумов' этого канала (шум оценивается по разностям соседних измерений),
      - отклонение по модулю > min_dev_c градусов (защита от шума; этот порог лучше не занижать),
      - состояние держится не меньше min_duration_min минут подряд.
    """
    # 'обычный шум' канала считаем по разностям соседних измерений: так канал, который сам часто
    # переключается между состояниями, не получает огромный 'шум' и не прячет свои события
    d = np.diff(T2, axis=1)
    sigma = 1.4826 * np.median(np.abs(d - np.median(d, axis=1, keepdims=True)), axis=1, keepdims=True) / np.sqrt(2)
    sigma = np.maximum(sigma, 0.05)
    Z = T2 / sigma

    mask = (np.abs(Z) > z_thr) & (np.abs(T2) > min_dev_c)
    min_steps = max(1, int(round(min_duration_min / step_min)))
    return binary_opening(mask, structure=np.ones((1, min_steps)))


# =====================================================================
# 5. РАЗМЕТКА ОКОН
# =====================================================================

# Признаки окна. 'dyn_*' и 'jitter' не используют базовый уровень канала (описывают динамику и контраст),
# 'base_*' считаются от отклонения от нормы, то есть ближе к самому правилу разметки.
FEATURES_DYN = ['dyn_range', 'dyn_std', 'dyn_slope_h', 'dyn_jump_max', 'dyn_jump_med',
                'dyn_spatial_contrast', 'dyn_spatial_std', 'dyn_vs_building', 'dyn_corr_building',
                'dyn_nb_contrast', 'dyn_delta_prev', 'dyn_delta_prev4', 'jitter']
FEATURES_BASE = ['base_mean', 'base_absmax', 'base_p95', 'base_frac3']


def window_features(T, T2, bm, c0, c1, t0, t1, win_ch, win_t, step_min, min_dev_c=3.0):
    """
    Числовые признаки одного окна (каналы c0:c1, измерения t0:t1).
    T — температура, T2 — отклонение от нормы, bm — медиана температуры по всем каналам в каждый момент
    (общий ритм здания). Абсолютное время суток и расстояние в признаки НЕ входят специально,
    чтобы модель не запоминала 'когда' и 'где'.
    """
    block = T[c0:c1, t0:t1].astype(np.float64)
    R = np.abs(T2[c0:c1, t0:t1])
    n_t = block.shape[1]
    m = block.mean(axis=0)                                   # средняя температура окна во времени
    tt = np.arange(n_t) * step_min / 60.0
    slope = float(np.polyfit(tt, m, 1)[0]) if n_t > 1 else 0.0
    d = np.abs(np.diff(block, axis=1)) if n_t > 1 else np.zeros((block.shape[0], 1))
    b = bm[t0:t1]
    corr = float(np.corrcoef(m, b)[0, 1]) if n_t > 2 and m.std() > 1e-9 and b.std() > 1e-9 else 0.0

    nb = []                                                  # соседние участки кабеля в то же время
    if c0 - win_ch >= 0:
        nb.append(float(T[c0 - win_ch:c0, t0:t1].mean()))
    if c1 + win_ch <= T.shape[0]:
        nb.append(float(T[c1:c1 + win_ch, t0:t1].mean()))
    cur = float(m.mean())
    d1 = cur - float(T[c0:c1, t0 - win_t:t0].mean()) if t0 >= win_t else 0.0   # только прошлое
    past = T[c0:c1, max(0, t0 - 4 * win_t):t0]
    d4 = cur - float(past.mean()) if past.shape[1] > 0 else 0.0

    return {
        'dyn_range': float(m.max() - m.min()),
        'dyn_std': float(m.std()),
        'dyn_slope_h': slope,
        'dyn_jump_max': float(d.max()),
        'dyn_jump_med': float(np.median(d, axis=0).max()),
        'dyn_spatial_contrast': float((block.max(axis=0) - np.median(block, axis=0)).mean()),
        'dyn_spatial_std': float(block.std(axis=0).mean()),
        'dyn_vs_building': float((m - b).mean()),
        'dyn_corr_building': corr,
        'dyn_nb_contrast': cur - float(np.mean(nb)) if nb else 0.0,
        'dyn_delta_prev': float(d1),
        'dyn_delta_prev4': float(d4),
        'base_mean': float(T2[c0:c1, t0:t1].mean()),
        'base_absmax': float(R.max()),
        'base_p95': float(np.percentile(R, 95)),
        'base_frac3': float((R > min_dev_c).mean()),
    }


def make_window_labels(T2, mask, distances, timestamps, step_min,
                       win_m=10.0, win_min=30.0, share_thr=0.02, strong_c=4.5, T=None, min_dev_c=3.0):
    """
    Нарезает данные на окна (win_m метров x win_min минут) и ставит метку:
       1  — уверенная аномалия: аномальных точек больше share_thr И их средняя сила >= strong_c °C
       0  — норма
      -1  — спорное: аномальные точки есть, но слабые (в обучение и проверку такие окна не берём)
    Столбцы-подсказки: anomaly_share (доля аномальных точек), strength (средняя сила, °C),
    jitter (шумность окна — чтобы проверять, что норма и аномалии лежат в одном режиме шума).
    """
    dx = float(np.median(np.diff(distances)))
    win_ch = max(1, int(round(win_m / dx)))
    win_t = max(1, int(round(win_min / step_min)))
    absT2 = np.abs(T2)
    bm = np.median(T.astype(np.float64), axis=0) if T is not None else None

    rows = []
    for c0 in range(0, T2.shape[0] - win_ch + 1, win_ch):
        for t0 in range(0, T2.shape[1] - win_t + 1, win_t):
            m = mask[c0:c0 + win_ch, t0:t0 + win_t]
            block = T2[c0:c0 + win_ch, t0:t0 + win_t]
            share = float(m.mean())
            strength = float(absT2[c0:c0 + win_ch, t0:t0 + win_t][m].mean()) if m.any() else 0.0
            jitter = float(np.abs(np.diff(block, axis=1)).mean()) if block.shape[1] > 1 else 0.0
            if share > share_thr:
                label = 1 if strength >= strong_c else -1
            else:
                label = 0
            row = {
                'ch_start': c0, 'ch_end': c0 + win_ch,
                'dist_start_m': round(float(distances[c0]), 1),
                'dist_end_m': round(float(distances[c0 + win_ch - 1]), 1),
                't_start': t0, 't_end': t0 + win_t,
                'time_start': timestamps[t0], 'time_end': timestamps[t0 + win_t - 1],
                'anomaly_share': round(share, 3),
                'strength': round(strength, 2),
                'jitter': round(jitter, 3),
                'label': label,
            }
            if T is not None:
                row.update(window_features(T, T2, bm, c0, c0 + win_ch, t0, t0 + win_t, win_ch, win_t,
                                           step_min, min_dev_c))
            rows.append(row)
    return pd.DataFrame(rows)


@dataclass
class Baseline:
    """Обычный уровень температуры каждого канала, посчитанный по нескольким кускам записи сразу."""
    distances: np.ndarray
    values: np.ndarray


def common_baseline(parts, skip_start_m=17.5, step_min=2.0):
    """
    Считает ОБЩИЙ обычный уровень каналов по всем непрерывным кускам записи. Без него каждый кусок
    сравнивается сам с собой, и событие, которое в одном куске занимает полвремени, а в другом
    четверть, получает разные метки. Передавай в label_dts(part, baseline=...).
    В parts не включай совсем короткие куски (меньше ~3 часов).
    """
    cleans = [to_uniform_time(drop_calibration_coil(p, skip_start_m), step_min) for p in parts]
    common = np.round(cleans[0].distances, 4)
    for c in cleans[1:]:
        common = np.intersect1d(common, np.round(c.distances, 4))
    if len(common) == 0:
        raise ValueError('У кусков записи нет общих каналов')
    stacks = []
    for c in cleans:
        _, idx, _ = np.intersect1d(np.round(c.distances, 4), common, return_indices=True)
        stacks.append(c.matrix[idx])
    return Baseline(common, np.median(np.concatenate(stacks, axis=1), axis=1))


def label_dts(data, skip_start_m=17.5, step_min=2.0,
              z_thr=6.0, min_dev_c=3.0, min_duration_min=6.0,
              win_m=10.0, win_min=30.0, share_thr=0.02, strong_c=4.5, baseline=None, with_features=True):
    """
    Весь конвейер: подготовка -> очистка -> маска -> метки окон.
    baseline — необязательный общий уровень каналов (см. common_baseline) для согласованной
    разметки нескольких кусков одной записи.
    """
    clean = to_uniform_time(drop_calibration_coil(data, skip_start_m), step_min)
    base_vals = None
    if baseline is not None:
        _, i_d, i_b = np.intersect1d(np.round(clean.distances, 4), baseline.distances, return_indices=True)
        if len(i_d) == 0:
            raise ValueError('Каналы этого куска не совпадают с общим базовым уровнем')
        clean = DTSData(clean.distances[i_d], clean.timestamps, clean.matrix[i_d])
        base_vals = baseline.values[i_b]
    T2 = compute_residual(clean.matrix, base_vals)
    mask = anomaly_mask(T2, step_min, z_thr, min_dev_c, min_duration_min)
    labels = make_window_labels(T2, mask, clean.distances, clean.timestamps, step_min,
                                win_m, win_min, share_thr, strong_c,
                                T=clean.matrix if with_features else None, min_dev_c=min_dev_c)
    params = dict(skip_start_m=skip_start_m, step_min=step_min, z_thr=z_thr, min_dev_c=min_dev_c,
                  min_duration_min=min_duration_min, win_m=win_m, win_min=win_min,
                  share_thr=share_thr, strong_c=strong_c,
                  common_baseline=baseline is not None)
    return LabelResult(data=clean, residual=T2, mask=mask, labels=labels, params=params)


def label_record(data, source, skip_start_m=17.5, min_steps=90, baseline=None,
                 return_results=False, **kwargs):
    """
    Размечает ОДНУ запись целиком: режет на непрерывные куски, считает общий базовый уровень
    (или берёт переданный baseline), размечает каждый кусок и склеивает таблицы.
    source — имя источника (например 'N29'); добавляется столбцом, чтобы таблицы разных файлов
    можно было склеить. Столбец group = источник + участок кабеля по 50 м: по нему потом делим
    данные на обучение и проверку (чтобы соседние окна не попадали в обе части).
    kwargs — любые настройки label_dts (step_min, z_thr, min_dev_c, strong_c, win_m, win_min, ...).
    Куски короче min_steps измерений пропускаются (по ним нельзя надёжно оценить норму).
    """
    parts = [p for p in split_continuous(data) if p.matrix.shape[1] >= min_steps]
    if not parts:
        raise ValueError(f'{source}: нет непрерывных кусков длиннее {min_steps} измерений')
    step_min = kwargs.get('step_min', 2.0)
    base = baseline if baseline is not None else common_baseline(parts, skip_start_m, step_min)

    tables, results = [], []
    for i, part in enumerate(parts):
        r = label_dts(part, skip_start_m=skip_start_m, baseline=base, **kwargs)
        t = r.labels.copy()
        t.insert(0, 'part', i)
        t.insert(0, 'source', source)
        t['group'] = source + '_' + (t['dist_start_m'] // 50).astype(int).astype(str)
        tables.append(t)
        results.append(r)
    table = pd.concat(tables, ignore_index=True)
    return (table, results) if return_results else table


def promote_adjacent(table, max_iter=2):
    """
    Спорное окно (-1), которое стоит вплотную (по времени, на том же участке кабеля) к уверенному (1),
    переводит в 1. Так в событие попадают его начало и конец: окно, в котором отклонение только
    нарастает, само не набирает силы, но принадлежит тому же событию. Работает по готовой таблице меток,
    данные перечитывать не нужно. max_iter — на сколько окон вглубь тянется такая цепочка.
    """
    key = ['source', 'part', 'ch_start']
    t = table.copy()
    for _ in range(max_iter):
        t = t.sort_values(key + ['t_start'])
        lab = t['label'].to_numpy()
        same_prev = (t[key].shift(1) == t[key]).all(axis=1).to_numpy()
        same_next = (t[key].shift(-1) == t[key]).all(axis=1).to_numpy()
        prev_pos = np.r_[False, lab[:-1] == 1] & same_prev
        next_pos = np.r_[lab[1:] == 1, False] & same_next
        promote = (lab == -1) & (prev_pos | next_pos)
        if not promote.any():
            break
        t['label'] = np.where(promote, 1, lab)
    return t.sort_index()


def concat_records(a, b):
    """Склеивает две записи одного канала по времени; перекрывающуюся часть берёт из b."""
    keep = a.timestamps < b.timestamps[0]
    return DTSData(distances=a.distances,
                   timestamps=a.timestamps[keep].append(b.timestamps),
                   matrix=np.concatenate([a.matrix[:, keep], b.matrix], axis=1))


# =====================================================================
# Графики для проверки глазами
# =====================================================================

def plot_overview(result):
    """Сверху температура, снизу — что программа посчитала аномалией."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    ax[0].imshow(result.data.matrix, aspect='auto', cmap='coolwarm')
    ax[0].set_title('Температура')
    ax[1].imshow(result.mask, aspect='auto', cmap='gray_r')
    ax[1].set_title('Что программа посчитала аномалией')
    ax[1].set_xlabel('Номер измерения (время)')
    plt.show()


def show_window(result, row, pad=15):
    """
    Показывает, почему окно получило свою метку: отклонение от нормы (после вычитания обычного уровня
    канала и общего фона здания) для самого 'отклонившегося' канала окна и для среднего по окну.
    Пунктир — пороги +-min_dev_c. Среднее по окну может выглядеть спокойным, если событие узкое
    (например, тонкая линия в 3 канала), поэтому смотрим ещё и на самый отклонившийся канал.
    """
    import matplotlib.pyplot as plt
    R = result.residual
    c0, c1 = int(row['ch_start']), int(row['ch_end'])
    t0, t1 = int(row['t_start']), int(row['t_end'])
    a, b = max(0, t0 - pad), min(R.shape[1], t1 + pad)
    best = c0 + int(np.argmax(np.abs(R[c0:c1, t0:t1]).mean(axis=1)))
    thr = result.params['min_dev_c']

    plt.figure(figsize=(7, 3))
    plt.plot(range(a, b), R[c0:c1, a:b].mean(axis=0), color='lightgray', label='среднее по окну')
    plt.plot(range(a, b), R[best, a:b], label=f'самый отклонившийся канал ({result.data.distances[best]:.1f} м)')
    for y in (-thr, thr):
        plt.axhline(y, ls='--', color='gray')
    plt.axvspan(t0, t1, alpha=0.2)
    plt.ylabel('Отклонение от нормы, °C')
    plt.title(f"{row['dist_start_m']}-{row['dist_end_m']} м, метка={row['label']}")
    plt.legend(fontsize=7)
    plt.show()


def show_samples(table, results, label=1, n=4, seed=0):
    """Показывает n случайных окон с данной меткой (results — список из label_record(return_results=True))."""
    sub = table[table['label'] == label]
    if sub.empty:
        print('Окон с меткой', label, 'нет')
        return
    for _, row in sub.sample(min(n, len(sub)), random_state=seed).iterrows():
        show_window(results[int(row['part'])], row)


# =====================================================================
# Запуск из командной строки
# =====================================================================

def _main():
    import argparse
    ap = argparse.ArgumentParser(description='Разметка DTS-данных: аномалия / норма / спорное')
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument('--long', help='csv в длинном формате (колонка points)')
    src.add_argument('--wide', help='csv или папка в широком формате (колонки results_..._dts)')
    ap.add_argument('--dts-id')
    ap.add_argument('--channel-id', type=int)
    ap.add_argument('--start')
    ap.add_argument('--end')
    ap.add_argument('--skip-start-m', type=float, default=17.5)
    ap.add_argument('--out', default='labels.csv')
    args = ap.parse_args()

    if args.long:
        data = load_dts_long(args.long, args.dts_id, args.channel_id, args.start, args.end)
    else:
        data = load_dts_wide(args.wide)
    result = label_dts(data, skip_start_m=args.skip_start_m)
    result.labels.to_csv(args.out, index=False)
    print(result.labels['label'].value_counts())
    print('Настройки:', result.params)


if __name__ == '__main__':
    _main()
