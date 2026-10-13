"""
Открывает все PNG из reports/figures в одном окне matplotlib.
Работает только если есть GUI (не в headless-режиме).
"""
import matplotlib
matplotlib.use("TkAgg")  # или "Qt5Agg", если установлен PyQt5
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path
from src.config import FIGURES


def main():
    pngs = sorted(FIGURES.glob("*.png"))
    if not pngs:
        print(f"Нет PNG в {FIGURES}. Запусти src.eda_plots.")
        return

    n = len(pngs)
    cols = 1
    rows = n

    fig, axes = plt.subplots(rows, cols, figsize=(14, 6 * rows))
    if n == 1:
        axes = [axes]

    for ax, path in zip(axes, pngs):
        img = mpimg.imread(path)
        ax.imshow(img)
        ax.set_title(path.name)
        ax.axis("off")

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()