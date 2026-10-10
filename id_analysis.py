"""Проверка: несёт ли числовая часть customer_id сигнал о protection_score?

Иногда ID клиентов идут по порядку выдачи, и в них прячется «время» или группа —
тогда ID помогает предсказывать. Проверяем 4 шага:
  1. Достаём цифры из customer_id (CUST100123 -> 100123) как число.
  2. Рисуем график «цель vs числовой ID» и сохраняем картинку.
  3. Считаем корреляции Пирсона и Спирмена (линейная и ранговая связь).
  4. Обучаем модели ТОЛЬКО на числовом ID и сравниваем с константным средним:
       - обычный 5-fold CV (перемешанный, как в experiments.py);
       - «экстраполяция»: учим на первых 80% ID, проверяем на последних 20% —
         это похоже на реальный тест (его ID лежат ВЫШЕ всех train-ID).

Словарик:
  * Корреляция Пирсона — линейная связь (-1..1); Спирмена — ранговая
    (растёт ли одно, когда растёт другое, пусть и не по прямой).
  * p-value — вероятность увидеть такую связь случайно; маленькое p = связь значима.
  * Экстраполяция — предсказание ВНЕ виденного диапазона (сложнее, чем внутри).

Пример запуска:
    python id_analysis.py --train-csv hard_train.csv --output-dir reports/id_analysis
"""

# Включаем «новые» правила аннотаций типов.
from __future__ import annotations

# Стандартные модули:
import argparse  # аргументы командной строки
import json  # запись id_stats.json
from pathlib import Path  # пути к файлам

# matplotlib в режиме Agg (без окон, только сохранение в файл) — ДО импорта pyplot.
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # графики
import numpy as np  # массивы и математика
import pandas as pd  # таблицы
from scipy.stats import pearsonr, spearmanr  # корреляции Пирсона и Спирмена (+ p-value)
from sklearn.ensemble import GradientBoostingRegressor  # градиентный бустинг sklearn
from sklearn.linear_model import LinearRegression  # обычная линейная регрессия
from sklearn.model_selection import KFold, cross_val_predict  # фолды и кросс-предсказания
from sklearn.neighbors import KNeighborsRegressor  # kNN: предсказание средним соседей
from sklearn.pipeline import make_pipeline  # цепочка шагов обработки
from sklearn.preprocessing import SplineTransformer, StandardScaler  # стандартизация и сплайны

# Наши общие функции/константы.
from model_utils import ID_COLUMN, TARGET_COLUMN, set_global_seed

# Единый seed всего анализа (повторяемость).
SEED = 42


def parse_args() -> argparse.Namespace:
    """Разобрать аргументы командной строки (путь к train и папка для результатов)."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train-csv", default="hard_train.csv")
    parser.add_argument("--output-dir", default="reports/id_analysis")
    return parser.parse_args()


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    """Посчитать RMSE между двумя массивами (предсказания и правда)."""
    # asarray — на случай, если передали списки; дальше обычная формула RMSE.
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def main() -> None:
    """Главная функция: извлечь ID, нарисовать графики, посчитать корреляции и модели."""
    # Разобрать аргументы, зафиксировать случайность, создать папку для результатов.
    args = parse_args()
    set_global_seed(SEED)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Прочитать train.
    data = pd.read_csv(args.train_csv)
    # Достать цифры из ID: .str.extract(r"(\d+)") находит первую группу цифр
    # (r"..." — сырая строка для регулярного выражения; \d+ = «одна или больше цифр»).
    # expand=False — вернуть Series, а не DataFrame.
    digits = data[ID_COLUMN].astype(str).str.extract(r"(\d+)", expand=False)
    # Если где-то цифр нет (NaN) — падаем: такие ID мы не понимаем.
    if digits.isna().any():
        raise ValueError(f"{ID_COLUMN} values without digits were found")
    # Новая колонка id_num — числовой ID (astype(int) — в целые числа).
    data["id_num"] = digits.astype(int)
    # ID должны быть уникальны (дубли = что-то не так с данными).
    if data["id_num"].duplicated().any():
        raise ValueError("Numeric IDs are not unique")

    # X — матрица из одной колонки (моделям нужен 2D-массив: [[id1], [id2], ...]).
    x = data[["id_num"]].to_numpy(dtype=float)
    # y — цели.
    y = data[TARGET_COLUMN].to_numpy(dtype=float)
    # order — индексы строк, сортирующие по id_num (argsort возвращает ПОРЯДОК, а не значения).
    order = np.argsort(data["id_num"].to_numpy())
    # Печатаем диапазон ID и их количество (nunique = число уникальных).
    print(f"Numeric ID range: {data['id_num'].min()} .. {data['id_num'].max()}, unique: {data['id_num'].nunique()}")
    # Проверка «отсортирован ли файл по ID»: все ли разности соседних > 0.
    # np.diff считает разности соседей; .all() — «все ли True».
    print(f"Is the file sorted by ID: {bool((np.diff(data['id_num'].to_numpy()) > 0).all())}")

    # --- 1. Графики: рассеяние + средние по блокам. ---
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))  # 1 строка, 2 столбца
    # Левая панель: каждая точка — клиент (ID по горизонтали, цель по вертикали).
    axes[0].scatter(data["id_num"], y, s=4, alpha=0.4)  # s=размер, alpha=прозрачность
    # Скользящее среднее по 200 соседним ID: y[order] — цели в порядке ID,
    # rolling(200, center=True) — окно 200 с центром в точке, min_periods=50 — минимум точек.
    rolling = pd.Series(y[order], index=data["id_num"].to_numpy()[order]).rolling(200, center=True, min_periods=50).mean()
    # Красная линия тренда поверх точек.
    axes[0].plot(rolling.index, rolling.values, color="crimson", lw=2, label="rolling mean (200 rows)")
    axes[0].set_xlabel("numeric customer_id")
    axes[0].set_ylabel("protection_score")
    axes[0].set_title("protection_score vs numeric ID")
    axes[0].legend()  # показать легенду (подпись красной линии)

    # Правая панель: средние цели по блокам из 500 подряд идущих ID.
    # (id - min) // 500 — номер блока (целочисленное деление): 0, 0, ..., 1, 1, ...
    block = (data["id_num"] - data["id_num"].min()) // 500
    # Группируем по блоку, считаем среднее и std цели в каждом.
    block_means = data.groupby(block)[TARGET_COLUMN].agg(["mean", "std"])
    # errorbar: точки-средние с «усами» ±1 стандартная ошибка (std / sqrt(500)).
    # Плоский профиль = среднего тренда по ID нет.
    axes[1].errorbar(block_means.index * 500, block_means["mean"], yerr=block_means["std"] / np.sqrt(500),
                     fmt="o-", ms=3, capsize=2)  # fmt="o-" — кружки, соединённые линией
    axes[1].set_xlabel("block start (500 consecutive IDs)")
    axes[1].set_ylabel("mean protection_score (±1 s.e.)")
    axes[1].set_title("Mean target by ID block")
    fig.tight_layout()
    fig.savefig(out / "id_vs_protection_score.png", dpi=130)
    plt.close(fig)

    # --- 2. Корреляции ID с целью (значение + p-value). ---
    pearson_r, pearson_p = pearsonr(data["id_num"], y)  # Пирсон: линейная связь
    spearman_r, spearman_p = spearmanr(data["id_num"], y)  # Спирмен: ранговая связь
    # :.3g — компактный формат p-value (например, 0.13 или 1.2e-05).
    print(f"Pearson r = {pearson_r:.4f} (p = {pearson_p:.3g})")
    print(f"Spearman rho = {spearman_r:.4f} (p = {spearman_p:.3g})")

    # Автокорреляция цели с лагом 1 в порядке ID: corr(y[i], y[i+1]).
    # y[:-1] — все кроме последнего, y[1:] — все кроме первого; [0, 1] — внедиагональный элемент.
    # Нужна, чтобы проверить блочную структуру (соседи похожи?).
    lag1 = float(np.corrcoef(y[:-1], y[1:])[0, 1])
    print(f"Lag-1 autocorrelation of target in ID order: {lag1:.4f}")

    # --- 3. Модели ТОЛЬКО на числовом ID. ---
    # Перемешанные 5 фолдов (как в experiments.py).
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    # Словарь «название -> модель» (4 модели разной гибкости):
    models = {
        # Прямая линия по ID (со стандартизацией входа — StandardScaler).
        "linear regression": make_pipeline(StandardScaler(), LinearRegression()),
        # Гибкая кривая: кубические сплайны (8 узлов) + линейная регрессия.
        "spline (cubic, 8 knots) + linear": make_pipeline(
            StandardScaler(), SplineTransformer(n_knots=8, degree=3), LinearRegression()
        ),
        # kNN: предсказание = среднее 200 ближайших по ID соседей (ловит локальные волны).
        "kNN (k=200)": make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=200)),
        # Градиентный бустинг: 300 деревьев глубины 3 (самый гибкий).
        "gradient boosting": GradientBoostingRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=3, subsample=0.8, random_state=SEED
        ),
    }

    # Список словарей под результаты перемешанного CV.
    results = []
    # Сначала бейзлайн «константное среднее»: OOF, где каждому фолду — среднее его train.
    oof_const = np.zeros(len(y))
    for tr, va in kf.split(x):
        oof_const[va] = y[tr].mean()
    results.append({"model": "constant mean", "cv5_rmse": rmse(oof_const, y)})
    # Затем каждая модель: cross_val_predict сам прогоняет 5 фолдов и собирает OOF.
    for name, model in models.items():
        pred = cross_val_predict(model, x, y, cv=kf)
        results.append({"model": name, "cv5_rmse": rmse(pred, y)})

    # Временной срез: train — первые 80% ID, valid — последние 20% (как настоящий тест!).
    cut = int(0.8 * len(data))  # граница среза: 80% строк
    train_idx, valid_idx = order[:cut], order[cut:]  # индексы: нижние 80% и верхние 20% по ID
    # Бейзлайн среза: всем valid — среднее train.
    temporal = [
        {"model": "constant mean", "tail_rmse": rmse(np.full(len(valid_idx), y[train_idx].mean()), y[valid_idx])}
    ]
    # Каждая модель: обучить на нижних 80%, померить на верхних 20%.
    for name, model in models.items():
        model.fit(x[train_idx], y[train_idx])
        temporal.append({"model": name, "tail_rmse": rmse(model.predict(x[valid_idx]), y[valid_idx])})
    temporal_df = pd.DataFrame(temporal)

    # Сводим обе проверки в одну таблицу: merge по колонке model (outer = все строки).
    cv_df = pd.DataFrame(results)
    summary = cv_df.merge(temporal_df, on="model", how="outer")
    print("\nModels on the numeric ID only (RMSE on the 0-100 scale):")
    # Печатаем красиво: округлить до 3 знаков, без индексов строк.
    print(summary.round(3).to_string(index=False))
    # Сохраняем таблицу в CSV.
    summary.to_csv(out / "id_only_models.csv", index=False)

    # Сохраняем ключевые числа в JSON для отчёта.
    stats = {
        "id_min": int(data["id_num"].min()),
        "id_max": int(data["id_num"].max()),
        "pearson_r": float(pearson_r),
        "pearson_p": float(pearson_p),
        "spearman_rho": float(spearman_r),
        "spearman_p": float(spearman_p),
        "lag1_autocorrelation": lag1,
        "target_std": float(y.std()),  # разброс цели — ориентир: RMSE хуже std = модель бесполезна
        "seed": SEED,
    }
    with (out / "id_stats.json").open("w", encoding="utf-8") as file:
        json.dump(stats, file, indent=2)
        file.write("\n")
    print(f"\nSaved: {out / 'id_vs_protection_score.png'}, {out / 'id_only_models.csv'}, {out / 'id_stats.json'}")


# Запуск main(), только если файл запущен напрямую.
if __name__ == "__main__":
    main()
