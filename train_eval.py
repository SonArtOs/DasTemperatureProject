"""
train_eval.py — сравнение моделей для этапа 1 (аномалия / норма).

Идея проверки: обучаемся на одних источниках данных, проверяем на другом, которого модель не видела
(leave-one-source-out). Так проверяется перенос на другое здание/кабель, а не запоминание соседних окон.
Источники cosmos_ch1 и cosmos_ch2 — один прибор и одно здание, поэтому считаются одной семьёй 'cosmos'.

Два набора признаков (см. dts_labeling.py):
  - 'динамика'            — признаки без базового уровня канала: как меняется температура и чем участок
                            отличается от соседей и от общего ритма здания;
  - 'динамика + базовый'  — плюс признаки отклонения от нормы, то есть ближе к самому правилу разметки.
Разница между ними показывает, насколько метки восстанавливаются только по динамике, а насколько
модель просто повторяет правило.

Модели:
  - HistGradientBoosting (с учётом перекоса классов) — основная;
  - IsolationForest — без учителя, учится только на нормальных окнах.

Метрики: PR-AUC (главная при редких аномалиях; для случайной модели она равна доле аномалий),
ROC-AUC и F1: у бустинга при пороге 0.5 (классы взвешены поровну), у Isolation Forest при пороге,
который помечает столько же окон, какова доля аномалий в обучающих данных. PR-AUC и ROC-AUC от порога
не зависят, поэтому им стоит доверять больше, чем F1.

Запуск:
    import pandas as pd
    from train_eval import evaluate
    ds = pd.read_csv('dataset_final.csv')
    res = evaluate(ds)
    print(res.round(3).to_string(index=False))
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest
from sklearn.metrics import average_precision_score, roc_auc_score

from dts_labeling import FEATURES_BASE, FEATURES_DYN


def family_of(source):
    """cosmos_ch1 и cosmos_ch2 -> 'cosmos'."""
    return source.split('_ch')[0]


def _balanced_weights(y):
    pos = y.mean()
    return np.where(y == 1, 0.5 / pos, 0.5 / (1 - pos))


def _source_balanced_weights(sources, y):
    """Каждый источник вносит в каждый класс одинаковый суммарный вес (чтобы N29 не заглушал остальных)."""
    w = np.zeros(len(y))
    for _, idx in pd.Series(np.arange(len(y))).groupby([np.asarray(sources), np.asarray(y)]):
        w[idx.to_numpy()] = 1.0 / len(idx)
    for lab in (0, 1):
        m = np.asarray(y) == lab
        w[m] *= 0.5 / w[m].sum()
    return w


def _metrics(y_te, s_te, thr):
    pred = s_te >= thr
    tp = int(((pred == 1) & (y_te == 1)).sum())
    prec = tp / max(1, int(pred.sum()))
    rec = tp / max(1, int(y_te.sum()))
    return {
        'pr_auc': average_precision_score(y_te, s_te),
        'roc_auc': roc_auc_score(y_te, s_te),
        'precision': prec,
        'recall': rec,
        'f1': 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0,
        'chance_pr_auc': float(y_te.mean()),     # во столько раз надо превзойти случайную модель
    }


def evaluate(dataset, feature_sets=None, seed=0, weighting='class'):
    """
    dataset — таблица с метками и признаками (label, source, столбцы признаков).
    Окна со спорной меткой (-1) в обучение и проверку не берутся.
    weighting='class' — классы взвешены поровну; 'source' — ещё и каждый обучающий источник вносит равный вклад.
    Возвращает таблицу: по строке на (проверочный источник, набор признаков, модель).
    """
    d = dataset[dataset['label'].isin([0, 1])].copy()
    d['family'] = d['source'].map(family_of)
    if feature_sets is None:
        feature_sets = {'динамика': FEATURES_DYN, 'динамика + базовый': FEATURES_DYN + FEATURES_BASE}

    rows = []
    for fam in sorted(d['family'].unique()):
        te, tr = d[d['family'] == fam], d[d['family'] != fam]
        if te['label'].sum() == 0 or tr['label'].sum() == 0:
            continue                                  # нечего оценивать (нет аномалий в проверочной части)
        y_tr, y_te = tr['label'].to_numpy(), te['label'].to_numpy()
        for fs_name, cols in feature_sets.items():
            X_tr, X_te = tr[cols].to_numpy(), te[cols].to_numpy()

            gb = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_depth=4,
                                                random_state=seed)
            w = (_balanced_weights(y_tr) if weighting == 'class'
                 else _source_balanced_weights(tr['source'].to_numpy(), y_tr))
            gb.fit(X_tr, y_tr, sample_weight=w)
            s_tr, s_te = gb.predict_proba(X_tr)[:, 1], gb.predict_proba(X_te)[:, 1]
            rows.append({'проверка_на': fam, 'признаки': fs_name, 'модель': 'бустинг',
                         'окон': len(te), 'аномалий': int(y_te.sum()), **_metrics(y_te, s_te, 0.5)})

            iso = IsolationForest(n_estimators=200, random_state=seed).fit(X_tr[y_tr == 0])
            s_tr, s_te = -iso.score_samples(X_tr), -iso.score_samples(X_te)
            thr_iso = float(np.quantile(s_tr, 1 - y_tr.mean()))   # помечаем столько же окон, сколько аномалий в обучении
            rows.append({'проверка_на': fam, 'признаки': fs_name, 'модель': 'Isolation Forest',
                         'окон': len(te), 'аномалий': int(y_te.sum()), **_metrics(y_te, s_te, thr_iso)})
    return pd.DataFrame(rows)


def summarize(results):
    """Средние метрики по проверочным источникам для каждого набора признаков и модели."""
    return (results.groupby(['признаки', 'модель'])[['pr_auc', 'roc_auc', 'f1', 'chance_pr_auc']]
            .mean().round(3).reset_index())


def importances(dataset, cols=None, weighting='class', seed=0, n_repeats=5):
    """
    Какие признаки реально помогают находить аномалии на 'чужом' источнике: для каждого проверочного
    источника обучаем бустинг на остальных и перемешиваем признаки на проверочном (падение PR-AUC).
    Возвращает таблицу: среднее по источникам и значение для каждого источника.
    """
    from sklearn.inspection import permutation_importance
    cols = cols or FEATURES_DYN
    d = dataset[dataset['label'].isin([0, 1])].copy()
    d['family'] = d['source'].map(family_of)
    out = {}
    for fam in sorted(d['family'].unique()):
        te, tr = d[d['family'] == fam], d[d['family'] != fam]
        if te['label'].sum() == 0 or tr['label'].sum() == 0:
            continue
        y_tr = tr['label'].to_numpy()
        w = (_balanced_weights(y_tr) if weighting == 'class'
             else _source_balanced_weights(tr['source'].to_numpy(), y_tr))
        gb = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_depth=4, random_state=seed)
        gb.fit(tr[cols].to_numpy(), y_tr, sample_weight=w)
        r = permutation_importance(gb, te[cols].to_numpy(), te['label'].to_numpy(),
                                   scoring='average_precision', n_repeats=n_repeats, random_state=seed)
        out[fam] = r.importances_mean
    t = pd.DataFrame(out, index=cols)
    t.insert(0, 'среднее', t.mean(axis=1))
    return t.sort_values('среднее', ascending=False)
