"""
main.py — единая точка входа пайплайна DTS-анализа (Шаг 4).

Полный цикл:
  1. ЗАГРУЗКА     — сырые данные через dts_labeling.load_dts_long / load_dts_wide
                    (или синтетический тестовый фрагмент, если csv не задан/не скачан);
  2. РАЗМЕТКА     — label_record: общий базовый уровень, маска аномалий, окна 10 м x 30 мин,
                    метки label ∈ {1, 0, -1} и колонка group (источник + участок 50 м);
  3. ОБУЧЕНИЕ     — train_eval: сплит по group (без утечки соседних окон), бустинг на признаках
                    окна, метрики Precision/Recall/F1 для класса 1 (label == -1 в обучение не берётся);
  4. ИНФЕРЕНС     — модель применяется к «новому» куску данных (у которого этикетки скрыты),
                    events.predictions_to_events склеивает уверенные окна в физические события;
  5. ОТЧЁТ        — событийная точность (match_events) и человекочитаемый список событий вида
                    "Обнаружено событие: остывание, 119-158 м, с 22:00 22.01 по 00:19 24.01".

Запуск:
    python3 main.py                                   # синтетическая демонстрация
    python3 main.py --long data/cosmos.csv --channel-id 1 --source cosmos_ch1
    python3 main.py --wide data/dts_data_cosmos_ch02_mapping --source cosmos_ch2
"""

import argparse
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

import events as events_mod
import train_eval as te
from dts_labeling import DTSData, LabelResult, label_record, load_dts_long, load_dts_wide

LABEL_PARAMS = dict(skip_start_m=0.0, step_min=2.0, win_m=10.0, win_min=30.0)


# =====================================================================
# Данные: реальные csv или синтетический тестовый фрагмент
# =====================================================================

def _synthetic_record(seed: int, anomaly: bool) -> DTSData:
    """Тестовый фрагмент формата cosmos: суточный фон здания + шум + (опц.) утечка/нагрев."""
    rng = np.random.default_rng(seed)
    n_ch, n_t = 80, 900
    distances = 20.0 + np.arange(n_ch) * 0.5                       # шаг 0.5 м
    timestamps = pd.date_range('2026-01-22 22:00', periods=n_t, freq='2min', tz='Europe/Moscow')
    diurnal = 3.0 * np.sin(2 * np.pi * (np.arange(n_t) % 720) / 720.0)
    T = 30.0 + distances[:, None] * 0.02 + diurnal[None, :] + rng.normal(0, 0.15, (n_ch, n_t))
    if anomaly:
        T[30:50, 300:600] -= 6.0       # холодная утечка ~35–45 м, 8:00–18:00
        T[60:70, 700:800] += 9.0       # узкий локальный нагрев изоляции ~50–55 м, позже утечки
    return DTSData(distances, timestamps, T.astype(np.float32))


def load_source(args: argparse.Namespace, source: str, seed: int,
                with_anomaly: bool) -> Tuple[DTSData, Optional[List[LabelResult]]]:
    """Грузит один источник: сначала пытается csv (long/wide), иначе — синтетику."""
    path = Path(args.long or args.wide) if (args.long or args.wide) else None
    if path is not None and path.exists() and path.stat().st_size > 1000:
        if args.long:
            data = load_dts_long(str(path), dts_id=args.dts_id, channel_id=args.channel_id,
                                 start=args.start, end=args.end)
        else:
            data = load_dts_wide(str(path), tz='Europe/Moscow')
        print(f'[{source}] загружено из {path.name}: '
              f'{data.matrix.shape[0]} каналов x {data.matrix.shape[1]} измерений')
        return data, None
    if path is not None:
        print(f'[{source}] файл {path} недоступен (пустой/LFS?) — беру синтетический фрагмент')
    else:
        print(f'[{source}] csv не задан — беру синтетический тестовый фрагмент')
    return _synthetic_record(seed, with_anomaly), None


# =====================================================================
# Шаги пайплайна
# =====================================================================

def step_labeling(data: DTSData, source: str) -> Tuple[pd.DataFrame, List[LabelResult]]:
    """Шаг 2 пайплайна: разметка одной записи целиком (общий baseline + окна + group)."""
    table, results = label_record(data, source=source, return_results=True, **LABEL_PARAMS)
    counts = table['label'].value_counts().to_dict()
    print(f'[{source}] окон: {len(table)}, метки: '
          f'норма={counts.get(0, 0)}, аномалия={counts.get(1, 0)}, спорное={counts.get(-1, 0)}; '
          f'групп (участков по 50 м): {table["group"].nunique()}')
    return table, results


def step_training(ds: pd.DataFrame, model_path: Optional[str]) -> Tuple[dict, pd.DataFrame]:
    """Шаг 3: сплит по group без утечки, обучение, метрики на валидации (label==-1 исключён из fit)."""
    tr, va = te.group_train_test_split(ds, val_groups=0.3, seed=0)
    overlap = set(tr['group']) & set(va['group'])
    assert not overlap, f'утечка данных: группы {overlap} в train и val одновременно'
    print(f'\n=== Обучение ===\nсплит по group: train={len(tr)} окон ({tr["group"].nunique()} групп), '
          f'val={len(va)} окон ({va["group"].nunique()} групп), пересечение групп: 0')

    if model_path and Path(model_path).exists():
        artifact = te.load_model(model_path)
        print(f'модель загружена из {model_path}')
    else:
        artifact = te.train_model(tr, weighting='class')
        if model_path:
            te.save_model(artifact, model_path)
            print(f'модель сохранена в {model_path}')

    va_eval = va[va['label'] != -1]                     # спорные окна из оценки предсказаний убираем
    proba = te.predict_windows(va_eval, artifact)
    m = te._metrics(va_eval['label'].to_numpy(), proba, artifact['threshold'])
    print('\nМетрики на валидации (класс 1 = аномалия, label==-1 исключён):')
    print(f"  Precision = {m['precision']:.3f}   Recall = {m['recall']:.3f}   F1 = {m['f1']:.3f}"
          f"   PR-AUC = {m['pr_auc']:.3f}   ROC-AUC = {m['roc_auc']:.3f}")
    cv = te.cross_validate_groups(ds, n_splits=min(5, ds['group'].nunique()),
                                  count_disputed_as_positive=False)
    print('\nGroupKFold (по колонке group):')
    print(cv.round(3).to_string(index=False))
    return artifact, va


def step_inference(artifact: dict, new_table: pd.DataFrame, gt_table: Optional[pd.DataFrame],
                   thr: float) -> List[events_mod.Event]:
    """Шаг 4: инференс по «новым» окнам (этикетки модели неизвестны) и склейка в события."""
    print('\n=== Инференс на новом фрагменте и детекция событий ===')
    proba = te.predict_windows(new_table, artifact)
    ev = events_mod.predictions_to_events(new_table, proba, thr=thr,
                                          step_min=LABEL_PARAMS['step_min'])
    if gt_table is not None:
        gt = events_mod.extract_events(gt_table, step_min=LABEL_PARAMS['step_min'])
        scores = events_mod.match_events(gt, ev)
        print('Событийная точность (предсказания vs эталон разметки):',
              {k: round(v, 3) for k, v in scores.items()})
    if not ev:
        print('Событий не обнаружено.')
    for e in ev:
        print(' ', events_mod.format_event(e))
    return ev


# =====================================================================
# Оркестратор
# =====================================================================

def run_pipeline(args: argparse.Namespace) -> None:
    """Прогон всего цикла: загрузка -> разметка -> обучение/оценка -> инференс -> события."""
    if args.long or args.wide:
        # реальный режим: один указанный источник; «новый» фрагмент — вторая половина записи,
        # модель обучается только на первой (сплит по group), этикетки второй скрыты от обучения
        data, _ = load_source(args, args.source, 1, True)
        t, _ = step_labeling(data, args.source)
        half = t['t_start'].median()
        train_part = t[t['t_start'] < half].copy()
        new_table = t[t['t_start'] >= half].copy()          # «новые» данные для инференса
        gt_table = new_table                                # эталон — эвристическая разметка
        ds = te.ensure_feature_table(train_part)
    else:
        # демо-режим: несколько синтетических источников + «новая» запись, которой модель не видела
        tables = []
        for src, anom, seed in [('cosmos_ch1', True, 1), ('N29', True, 2), ('cod_zhivopisnaya', False, 3)]:
            t, _ = step_labeling(_synthetic_record(seed, anom), src)
            tables.append(t)
        new_table, _ = step_labeling(_synthetic_record(11, True), args.source + '_new')
        gt_table = new_table                                # эталон только для отчёта
        ds = te.ensure_feature_table(pd.concat(tables, ignore_index=True))

    artifact, _ = step_training(ds, args.model)

    print('\n=== Инференс на новом фрагменте и детекция событий ===')
    proba = te.predict_windows(new_table, artifact)
    ev = events_mod.predictions_to_events(new_table, proba, thr=args.threshold,
                                          step_min=LABEL_PARAMS['step_min'])
    gt = events_mod.extract_events(gt_table, step_min=LABEL_PARAMS['step_min'])
    scores = events_mod.match_events(gt, ev)
    print('Событийная точность (предсказания vs эталон разметки):',
          {k: round(v, 3) for k, v in scores.items()})
    if not ev:
        print('Событий не обнаружено.')
    for e in ev:
        print(' ', events_mod.format_event(e))


def _argparser() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument('--long', help='csv в длинном формате (колонка points), напр. cosmos.csv')
    ap.add_argument('--wide', help='csv/папка в широком формате (results_..._dts)')
    ap.add_argument('--dts-id')
    ap.add_argument('--channel-id', type=int)
    ap.add_argument('--start')
    ap.add_argument('--end')
    ap.add_argument('--source', default='cosmos_ch1', help='имя источника для колонок source/group')
    ap.add_argument('--model', default=None, help='путь .joblib: загрузить (если есть) и сохранить модель')
    ap.add_argument('--threshold', type=float, default=0.5, help='порог вероятности аномалии для событий')
    return ap.parse_args()


if __name__ == '__main__':
    run_pipeline(_argparser())
