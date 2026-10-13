"""
events.py — этап 2: что за аномалия. Из окон с меткой 1 собираем события и группируем их по типам.

Событие — связная группа уверенных окон (метка 1): соседние по кабелю и по времени окна одного источника
склеиваются. Для каждого события считаем несколько понятных чисел (длительность, ширина, знак отклонения,
сила, 'тонкость', резкость начала). Потом кластеризуем события (k-means) и выбираем число кластеров по
силуэту. Кластеры называет человек: по таблице медиан видно, какой тип события в каждом.

Работает по готовой таблице dataset_final.csv (данные перечитывать не нужно). Окна имеют размер
~10 м x 30 мин, поэтому длительность и ширина событий грубые (кратны размеру окна).

Использование:
    import pandas as pd
    from events import extract_events, cluster_events
    ds = pd.read_csv('dataset_final.csv')
    ev = extract_events(ds)
    ev, summary = cluster_events(ev)
    print(summary.to_string())
"""

import numpy as np
import pandas as pd
from scipy import ndimage

EVENT_FEATURES = ['log_duration_h', 'log_width_m', 'sign', 'log_peak_dev', 'spread', 'log_abruptness']


def extract_events(ds, step_min=2.0, min_windows=2):
    """
    Склеивает окна с меткой 1 в события. min_windows — минимальное число окон в событии
    (одиночные окна обычно не события, а краевые случаи).
    """
    ds = ds.copy()
    ds['_win_ch'] = ds['ch_end'] - ds['ch_start']
    ds['_win_t'] = ds['t_end'] - ds['t_start']
    ds['_ci'] = ds['ch_start'] // ds['_win_ch']
    ds['_ti'] = ds['t_start'] // ds['_win_t']

    rows = []
    for (src, part), g in ds.groupby(['source', 'part']):
        pos = g[g['label'] == 1]
        if pos.empty:
            continue
        win_t = int(g['_win_t'].iloc[0])
        n_ti = int(g['_ti'].max()) + 1
        grid = np.zeros((int(g['_ci'].max()) + 1, n_ti), dtype=bool)
        grid[pos['_ci'].to_numpy(), pos['_ti'].to_numpy()] = True
        comp, n = ndimage.label(grid, structure=np.ones((3, 3)))      # соседи по диагонали тоже склеиваются
        pos = pos.assign(_comp=comp[pos['_ci'].to_numpy(), pos['_ti'].to_numpy()])

        for k, e in pos.groupby('_comp'):
            if len(e) < min_windows:
                continue
            n_steps = e['_ti'].nunique()
            w = e['base_absmax'].clip(lower=1e-6)
            rows.append({
                'source': src, 'part': part, 'n_windows': len(e),
                'dist_start_m': float(e['dist_start_m'].min()), 'dist_end_m': float(e['dist_end_m'].max()),
                'width_m': float(e['dist_end_m'].max() - e['dist_start_m'].min()),
                'duration_h': n_steps * win_t * step_min / 60.0,
                'time_start': e.loc[e['_ti'].idxmin(), 'time_start'],
                'time_end': e.loc[e['_ti'].idxmax(), 'time_end'],
                'sign': 1.0 if np.average(e['base_mean'], weights=w) >= 0 else -1.0,   # +1 нагрев, -1 охлаждение
                'peak_dev': float(e['base_absmax'].max()),
                'spread': float(e['base_frac3'].mean()),           # какая доля окна затронута (тонкая линия -> мало)
                'abruptness': float(e['dyn_delta_prev'].abs().max()),
                'touches_edge': bool(e['_ti'].min() == 0 or e['_ti'].max() == n_ti - 1),
            })
    ev = pd.DataFrame(rows)
    if ev.empty:
        return ev
    ev['log_duration_h'] = np.log10(ev['duration_h'].clip(lower=1e-3))
    ev['log_width_m'] = np.log10(ev['width_m'].clip(lower=1.0))
    ev['log_peak_dev'] = np.log10(ev['peak_dev'].clip(lower=1e-3))
    ev['log_abruptness'] = np.log1p(ev['abruptness'])
    return ev


def cluster_events(ev, k_range=range(2, 7), seed=0, verbose=True):
    """
    Кластеризует события (k-means на стандартизованных признаках), число кластеров выбирает по силуэту.
    Возвращает события со столбцом cluster и таблицу-описание кластеров (медианы признаков).
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import StandardScaler

    ev = ev.copy()
    if len(ev) < 6:
        ev['cluster'] = 0
        print('Событий слишком мало для кластеризации:', len(ev))
        return ev, _describe(ev)

    X = StandardScaler().fit_transform(ev[EVENT_FEATURES].to_numpy())
    best = None
    scores = {}
    for k in k_range:
        if k >= len(ev):
            break
        km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(X)
        sc = silhouette_score(X, km.labels_)
        scores[k] = sc
        if best is None or sc > best[0]:
            best = (sc, k, km.labels_)
    if verbose:
        print('Силуэт по числу кластеров:', {k: round(v, 3) for k, v in scores.items()})
        print('Выбрано k =', best[1])
    ev['cluster'] = best[2]
    return ev, _describe(ev)


def _describe(ev):
    g = ev.groupby('cluster')
    out = g.agg(событий=('n_windows', 'size'),
                длительность_ч=('duration_h', 'median'),
                ширина_м=('width_m', 'median'),
                знак=('sign', 'median'),
                сила_C=('peak_dev', 'median'),
                доля_затронутого=('spread', 'median'),
                резкость=('abruptness', 'median'),
                источники=('source', lambda s: ', '.join(sorted(set(s))))).round(2)
    return out.reset_index()
