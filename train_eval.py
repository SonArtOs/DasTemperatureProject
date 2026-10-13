"""
train_eval.py — этап обучения/оценки модели «аномалия / норма» поверх разметки dts_labeling.py.

Вход — таблица окон, которую выдаёт ``dts_labeling.label_record`` (или ``label_dts`` +
столбцы ``source``/``part``/``group``):
  * ``label`` ∈ {1 аномалия, 0 норма, -1 спорное};
  * ``group`` = источник + участок кабеля по 50 м — по ней делится train/test,
    чтобы окна одного физического участка никогда не попадали одновременно
    в обучение и проверку (иначе метрика завышается подтекшей информацией);
  * признаки: ``FEATURES_DYN`` (динамика, без базового уровня канала) и
    ``FEATURES_BASE`` (статистики residual T2). Если признаков в таблице нет,
    они извлекаются из LabelResult функцией :func:`features_from_result`.

Что появилось по сравнению со старой версией:
  * :func:`group_train_test_split` (сплит по колонке group) и
    :func:`group_kfold_indices` (GroupKFold) — случайный train_test_split больше не используется;
  * окна с label == -1 исключаются из обучения; при валидации их можно
    считать «консервативными позитивами» (см. ``count_disputed_as_positive``);
  * :func:`train_model` (обучение одной модели) и :func:`predict_windows`
    (инференс по таблице окон) — для оркестратора main.py;
  * старая проверка leave-one-source-out (:func:`evaluate`) сохранена без изменений логики.

Запуск (из командной строки, без csv — генерируются синтетические записи):
    python3 train_eval.py
Использование из кода:
    from train_eval import group_train_test_split, train_model, predict_windows, evaluate
"""

from typing import Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest
from sklearn.metrics import average_precision_score, classification_report, roc_auc_score
from sklearn.model_selection import GroupKFold

from dts_labeling import FEATURES_BASE, FEATURES_DYN, LabelResult, window_features

# модель по умолчанию (лёгкий бустинг, устойчивый к перекосу классов)
DEFAULT_MODEL = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05,
                                               max_depth=4, random_state=0)


def family_of(source: str) -> str:
    """cosmos_ch1 и cosmos_ch2 -> 'cosmos' (один прибор и одно здание — одна семья)."""
    return source.split('_ch')[0]


# =====================================================================
# Стыковка с dts_labeling: признакoвые таблицы и сплиты
# =====================================================================

def ensure_feature_table(table: pd.DataFrame,
                         results: Optional[Sequence[LabelResult]] = None,
                         feature_cols: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """
    Проверяет, что таблица окон совместима с обучением, и при необходимости достраивает признаки.

    Параметры
    ---------
    table : DataFrame
        Выход ``label_record`` (или ``label_dts`` + столбцы source/part/group).
        Обязательные столбцы: ``label``, ``group`` (или ``source``), нужные признаки.
    results : список LabelResult | None
        Если в таблице нет колонок признаков (разметка с ``with_features=False``),
        они считаются заново из ``results[i].data.matrix`` / ``residual`` для строк part == i.
    feature_cols : последовательность str | None
        Какие признаки гарантировать (по умолчанию динамика + базовые).

    Возвращает копию таблицы с нужными столбцами; бросает ValueError, если данных не хватает.
    """
    if 'label' not in table.columns:
        raise ValueError('В таблице нет колонки label (ожидается выход dts_labeling.label_record)')
    if 'group' not in table.columns:
        if 'source' in table.columns:
            table = table.copy()
            table['group'] = table['source'] + '_' + (table['dist_start_m'] // 50).astype(int).astype(str)
        else:
            raise ValueError('Нет ни group, ни source: нельзя построить сплит без утечки данных')

    cols = list(feature_cols) if feature_cols is not None else FEATURES_DYN + FEATURES_BASE
    missing = [c for c in cols if c not in table.columns]
    if missing:
        if results is None:
            raise ValueError(f'В таблице нет признаков {missing[:5]}... и не переданы results '
                             f'(LabelResult) для их расчёта. Используй label_record(..., return_results=True) '
                             f'или размечай с with_features=True.')
        table = features_from_result(table, results)
    return table


def features_from_result(table: pd.DataFrame, results: Sequence[LabelResult]) -> pd.DataFrame:
    """
    Генератор фич (Шаг 2): берёт T (температуру) и T2 (residual) из каждого LabelResult и
    досчитывает ``window_features`` для строк таблицы, у которых ещё нет признаковых колонок.

    Строки сопоставляются окнам по (part, ch_start, t_start); порядок строк сохраняется.
    """
    cols = FEATURES_DYN + FEATURES_BASE
    out = table.copy()
    for i, r in enumerate(results):
        p = r.params
        step_min = p['step_min']
        win_t = max(1, int(round(p['win_min'] / step_min)))
        dx = float(np.median(np.diff(r.data.distances)))
        win_ch = max(1, int(round(p['win_m'] / dx)))
        T, T2 = r.data.matrix, r.residual
        bm = np.median(T.astype(np.float64), axis=0)
        sel = out.index[out['part'] == i] if 'part' in out.columns else out.index
        sub = out.loc[sel]
        vals = [window_features(T, T2, bm, int(c0), int(c0) + win_ch, int(t0), int(t0) + win_t,
                                win_ch, win_t, step_min, p['min_dev_c'])
                for c0, t0 in zip(sub['ch_start'].to_numpy(), sub['t_start'].to_numpy())]
        out.loc[sel, cols] = pd.DataFrame(vals, index=sel)[cols]
    return out


def group_train_test_split(table: pd.DataFrame,
                           val_groups: float = 0.25,
                           seed: int = 0,
                           group_col: str = 'group') -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Сплит по колонке ``group`` (замена случайному train_test_split): все окна одного
    физучастка кабеля (источник + 50 м) целиком попадают либо в train, либо в val.

    Группы, содержащие хотя бы одно окно label==1, распределяются отдельно от чисто
    нормальных групп (стратификация), чтобы в проверке гарантированно остались аномалии.
    """
    rng = np.random.default_rng(seed)
    has_pos = table.groupby(group_col, sort=False)['label'].apply(lambda s: bool((s == 1).any()))
    keys = np.array(has_pos.index)
    pos_keys, neg_keys = keys[has_pos.to_numpy()], keys[~has_pos.to_numpy()]
    rng.shuffle(pos_keys)
    rng.shuffle(neg_keys)
    val = set(pos_keys[:max(1, int(round(len(pos_keys) * val_groups)))]) \
        | set(neg_keys[:int(round(len(neg_keys) * val_groups))])
    in_val = table[group_col].isin(val).to_numpy()
    return table[~in_val].copy(), table[in_val].copy()


def group_kfold_indices(table: pd.DataFrame, n_splits: int = 5,
                        group_col: str = 'group') -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Индексы (train_idx, test_idx) по GroupKFold: окна одной группы не пересекают фолды.
    Окна label == -1 остаются в индексах, но выбрасываются при обучении внутри фолда.
    """
    d = table.reset_index(drop=True)
    gkf = GroupKFold(n_splits=n_splits)
    return list(gkf.split(d, groups=d[group_col]))


# =====================================================================
# Метрики (класс 1 = аномалия)
# =====================================================================

def _metrics(y_te: np.ndarray, s_te: np.ndarray, thr: float,
             disputed_mask: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    Precision/Recall/F1 для класса 1 + PR-AUC/ROC-AUC.
    disputed_mask — булев флаг окон label==-1: если задан, они засчитываются как
    позитивы при подсчёте precision/recall/f1 (консервативная оценка полноты),
    но не влияют на PR-AUC/ROC-AUC (там остаются только уверенные метки).
    """
    pred = s_te >= thr
    y = np.asarray(y_te).astype(bool)
    if disputed_mask is not None:
        y = y | np.asarray(disputed_mask, dtype=bool)
    tp = int((pred & y).sum())
    fp = int((pred & ~y).sum())
    fn = int((~pred & y).sum())
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    bin_y = np.clip(np.asarray(y_te), 0, 1).astype(int)
    m = {'precision': prec, 'recall': rec,
         'f1': 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0}
    if 0 < bin_y.sum() < len(bin_y):
        m['pr_auc'] = float(average_precision_score(bin_y, s_te))
        m['roc_auc'] = float(roc_auc_score(bin_y, s_te))
    else:
        m['pr_auc'] = m['roc_auc'] = float('nan')
    m['chance_pr_auc'] = float(bin_y.mean())
    return m


def evaluate(dataset: pd.DataFrame, feature_sets=None, seed: int = 0, weighting: str = 'class'):
    """
    Старая проверка leave-one-source-out (логика сохранена без изменений).
    dataset — таблица с метками и признаками (label, source, столбцы признаков).
    Окна со спорной меткой (-1) в обучение и проверку не берутся.
    weighting='class' — классы взвешены поровну; 'source' — каждый обучающий источник
    вносит равный вклад. Возвращает таблицу: строка на (проверочный источник, набор признаков, модель).
    """
    d = dataset[dataset['label'].isin([0, 1])].copy()
    d['family'] = d['source'].map(family_of)
    if feature_sets is None:
        feature_sets = {'динамика': FEATURES_DYN, 'динамика + базовый': FEATURES_DYN + FEATURES_BASE}

    rows = []
    for fam in sorted(d['family'].unique()):
        te, tr = d[d['family'] == fam], d[d['family'] != fam]
        if te['label'].sum() == 0 or tr['label'].sum() == 0:
            continue                                  # нечего оценивать
        y_tr, y_te = tr['label'].to_numpy(), te['label'].to_numpy()
        for fs_name, cols in feature_sets.items():
            X_tr, X_te = tr[cols].to_numpy(), te[cols].to_numpy()

            gb = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_depth=4,
                                                random_state=seed)
            w = (_balanced_weights(y_tr) if weighting == 'class'
                 else _source_balanced_weights(tr['source'].to_numpy(), y_tr))
            gb.fit(X_tr, y_tr, sample_weight=w)
            s_te = gb.predict_proba(X_te)[:, 1]
            rows.append({'проверка_на': fam, 'признаки': fs_name, 'модель': 'бустинг',
                         'окон': len(te), 'аномалий': int(y_te.sum()), **_metrics(y_te, s_te, 0.5)})

            iso = IsolationForest(n_estimators=200, random_state=seed).fit(X_tr[y_tr == 0])
            s_tr, s_te = -iso.score_samples(X_tr), -iso.score_samples(X_te)
            thr_iso = float(np.quantile(s_tr, 1 - y_tr.mean()))
            rows.append({'проверка_на': fam, 'признаки': fs_name, 'модель': 'Isolation Forest',
                         'окон': len(te), 'аномалий': int(y_te.sum()), **_metrics(y_te, s_te, thr_iso)})
    return pd.DataFrame(rows)


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    """Средние метрики по проверочным источникам для каждого набора признаков и модели."""
    return (results.groupby(['признаки', 'модель'])[['pr_auc', 'roc_auc', 'f1', 'chance_pr_auc']]
            .mean().round(3).reset_index())


# =====================================================================
# Обучение / инференс одной модели (для пайплайна main.py)
# =====================================================================

def _balanced_weights(y: np.ndarray) -> np.ndarray:
    pos = y.mean()
    return np.where(y == 1, 0.5 / pos, 0.5 / (1 - pos))


def _source_balanced_weights(sources: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Каждый источник вносит в каждый класс одинаковый суммарный вес."""
    w = np.zeros(len(y))
    for _, idx in pd.Series(np.arange(len(y))).groupby([sources, y]):
        w[idx.to_numpy()] = 1.0 / len(idx)
    for lab in (0, 1):
        m = y == lab
        w[m] *= 0.5 / w[m].sum()
    return w


def train_model(train: pd.DataFrame,
                cols: Optional[Sequence[str]] = None,
                model=None,
                weighting: str = 'class',
                seed: int = 0) -> Dict:
    """
    Обучает модель на табличных окнах (label ∈ {0,1}; -1 отбрасывается).

    Возвращает словарь-артефакт {'model', 'cols', 'threshold'} — его можно сохранить
    через :func:`save_model` и позже передать в :func:`predict_windows`.
    """
    cols = list(cols) if cols is not None else FEATURES_DYN + FEATURES_BASE
    d = train[train['label'].isin([0, 1])]
    if d['label'].nunique() < 2:
        raise ValueError('В обучении только один класс — нужны и нормальные окна, и аномалии')
    X, y = d[cols].to_numpy(), d['label'].to_numpy()
    mdl = clone(model) if model is not None else clone(DEFAULT_MODEL)
    w = (_balanced_weights(y) if weighting == 'class'
         else _source_balanced_weights(d['source'].to_numpy(), y))
    mdl.fit(X, y, sample_weight=w)
    return {'model': mdl, 'cols': cols, 'threshold': 0.5}


def cross_validate_groups(table: pd.DataFrame,
                          cols: Optional[Sequence[str]] = None,
                          n_splits: int = 5,
                          model=None,
                          threshold: float = 0.5,
                          count_disputed_as_positive: bool = False,
                          weighting: str = 'class',
                          seed: int = 0) -> pd.DataFrame:
    """
    GroupKFold-валидация: строка метрик на фолд + итоговая средняя строка.
    При count_disputed_as_positive=True окна label==-1 в валидации засчитываются
    как аномалии (консервативная оценка полноты), но в обучение не попадают никогда.
    """
    cols = list(cols) if cols is not None else FEATURES_DYN + FEATURES_BASE
    folds = group_kfold_indices(table, n_splits=n_splits)
    rows = []
    for k, (tr_idx, te_idx) in enumerate(folds):
        tr_fit = table.iloc[tr_idx]
        te = table.iloc[te_idx]
        te_eval = te[te['label'].isin([0, 1, -1])]
        if te_eval['label'].eq(1).sum() == 0:
            continue
        art = train_model(tr_fit, cols=cols, model=model, weighting=weighting, seed=seed)
        s = predict_windows(te_eval, art)
        dis = (te_eval['label'] == -1).to_numpy() if count_disputed_as_positive else None
        rows.append({'fold': k, 'n_val': len(te_eval),
                     'n_anom': int((te_eval['label'] == 1).sum()),
                     **_metrics(te_eval['label'].to_numpy(), s, threshold, dis)})
    res = pd.DataFrame(rows)
    if not res.empty:
        mean = res.drop(columns='fold').mean().to_dict()
        mean['fold'] = 'среднее'
        res = pd.concat([res, pd.DataFrame([mean])], ignore_index=True)
    return res


def predict_windows(table: pd.DataFrame, artifact: Dict) -> np.ndarray:
    """
    Инференс: вероятность класса 1 (аномалия) для каждой строки таблицы окон.
    Порядок строк сохраняется; NaN-признаки бустинг переносит сам.
    """
    X = table[list(artifact['cols'])].to_numpy(dtype=np.float64)
    return artifact['model'].predict_proba(X)[:, 1]


def save_model(artifact: Dict, path: str) -> None:
    """Сохраняет артефакт train_model (модель + список признаков + порог) в файл."""
    joblib.dump(artifact, path)


def load_model(path: str) -> Dict:
    """Загружает артефакт, сохранённый :func:`save_model`."""
    return joblib.load(path)


def report(y_true: np.ndarray, y_pred: np.ndarray) -> str:
    """Человекочитаемый Precision/Recall/F1 для класса 1 (вывод в main.py)."""
    return classification_report(np.clip(np.asarray(y_true), 0, 1), y_pred,
                                 target_names=['норма', 'аномалия'], digits=3, zero_division=0)


# =====================================================================
# Демонстрация без внешних данных: python3 train_eval.py
# =====================================================================

def _demo() -> None:
    """Генерирует две синтетические записи (утечка/норма), размечает, обучает, печатает метрики."""
    from dts_labeling import DTSData, label_record

    def make_record(seed: int, anomaly: bool) -> DTSData:
        r = np.random.default_rng(seed)
        n_ch, n_t = 80, 900
        distances = 20.0 + np.arange(n_ch) * 0.5
        timestamps = pd.date_range('2026-01-22 22:00', periods=n_t, freq='2min', tz='Europe/Moscow')
        diurnal = 3.0 * np.sin(2 * np.pi * (np.arange(n_t) % 720) / 720.0)
        T = 30.0 + distances[:, None] * 0.02 + diurnal[None, :] + r.normal(0, 0.15, (n_ch, n_t))
        if anomaly:
            T[30:50, 300:600] -= 6.0      # холодная утечка ~35-45 м
            T[60:70, 700:800] += 9.0      # узкий локальный нагрев изоляции ~50-55 м
        return DTSData(distances, timestamps, T.astype(np.float32))

    ds = pd.concat([label_record(make_record(1, True), 'cosmos_ch1', skip_start_m=0),
                    label_record(make_record(2, False), 'N29', skip_start_m=0)],
                   ignore_index=True)
    print('Метки:', ds['label'].value_counts().to_dict())
    tr, va = group_train_test_split(ds, val_groups=0.3, seed=0)
    assert set(tr['group']) & set(va['group']) == set(), 'утечка данных между train/val!'
    art = train_model(tr)
    va_eval = va[va['label'] != -1]
    s = predict_windows(va_eval, art)
    m = _metrics(va_eval['label'].to_numpy(), s, 0.5)
    print({k: round(v, 3) for k, v in m.items()})
    print(report(va_eval['label'].to_numpy(), (s >= 0.5).astype(int)))
    print(cross_validate_groups(ds, n_splits=3).round(3).to_string(index=False))
    print(evaluate(ds).round(3).to_string(index=False))


if __name__ == '__main__':
    _demo()
