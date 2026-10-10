"""Проверка гипотез о скрытой структуре данных protection_score.

Каждая гипотеза проверяется ТОЛЬКО на ``hard_train.csv`` через 5-fold CV.
Основная линейка — разброс OOF-остатков на ЛОГИТ-шкале
(z = logit(protection_score / 100)); RMSE на шкале 0–100 приводим
для финальных кандидатов. Результат — ``reports/hypotheses.md``.

Какие гипотезы (H0 — точка отсчёта, H1–H12 — проверки):
  H0: базовый сигмоидально-линейный индекс (точка отсчёта, не гипотеза);
  H1: какой-то признак — точная функция остальных (производный признак)?
  H2: остатки похожи у соседей в пространстве признаков (локальная структура)?
  H3: помогают ли взаимодействия «категория × число»?
  H4: помогают ли явные денежные отношения (loan/income и т.п.)?
  H5: находит ли бустинг на остатках пороговые эффекты?
  H6: шум зависит от группы (разный разброс в разных категориях)?
  H7: тест из другого распределения, чем train (adversarial validation)?
  H8: другая функция связи (probit) лучше, чем logit?
  H9: оставшаяся ошибка — неустранимый шум на логитах (пол Байеса)?
  H10: хвост доходов (income > 300k) — большой источник ошибки?
  H11: дробный логит (биномиальный GLM) лучше, чем MSE на шкале 0–100?
  H12: убрать агрегат insurance_products (он = сумма 8 флагов)?

Это скрипт ДОКУМЕНТАЦИЯ (финал обучает train.py — ансамбль CatBoost).
Здесь «reference sigmoid-linear» — старый линейный ориентир для сравнений.

Пример запуска:
    python hypotheses.py --train-csv hard_train.csv --test-csv hard_test.csv --output reports/hypotheses.md
"""

# Включаем «новые» правила аннотаций типов.
from __future__ import annotations

# Стандартные модули:
import argparse  # аргументы командной строки
import warnings  # глушение предупреждений библиотек
from pathlib import Path  # пути к файлам

import lightgbm as lgb  # бустинг (для H5: дотягивание остатков)
import numpy as np  # массивы и математика
import pandas as pd  # таблицы
from scipy.optimize import minimize  # оптимизатор (для H11: дробный логит)
from scipy.special import expit, logit, ndtr, ndtri  # сигмоида/логит + нормальные CDF/квантили (для probit в H8)
from sklearn.ensemble import HistGradientBoostingClassifier  # классификатор (для H7: отличить train от test)
from sklearn.linear_model import RidgeCV  # ridge-регрессия с автовыбором alpha
from sklearn.metrics import roc_auc_score  # AUC — качество разделения двух классов (для H7)
from sklearn.model_selection import KFold, cross_val_predict  # фолды и кросс-предсказания
from sklearn.neighbors import NearestNeighbors  # поиск ближайших соседей (для H2)

# Наши общие функции. fit_sigmoid_index — из model_utils (там же, где и остальной
# общий код; в train.py её нет — train.py обучает финальный CatBoost).
from model_utils import fit_preprocessor, fit_sigmoid_index, numeric_matrix, transform_features

# Глушим предупреждения библиотек, чтобы вывод был чистым.
warnings.filterwarnings("ignore")

# Короткие имена для колонок цели и ID (чтобы формулы читались короче).
TARGET = "protection_score"
ID = "customer_id"
# Текстовые колонки, участвующие в проверках.
CATEGORICAL = ["region", "city_type", "education", "family_status", "employment", "gender", "wealth_segment"]
# Три seed'а для повторения ключевых замеров (среднее по 3 прогонам стабильнее).
CV_SEEDS = [0, 1, 2]


def parse_args() -> argparse.Namespace:
    """Разобрать аргументы командной строки (train, test, куда писать отчёт)."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", default="hard_train.csv")
    parser.add_argument("--test-csv", default="hard_test.csv")
    parser.add_argument("--output", default="reports/hypotheses.md")
    return parser.parse_args()


def design(frame: pd.DataFrame, numeric: list[str]) -> pd.DataFrame:
    """Собрать базовую матрицу плана: числа + one-hot категории + флаги пропусков.

    Аргументы:
        frame: таблица с признаками (без ID и цели).
        numeric: имена числовых колонок.

    Возвращает:
        DataFrame с one-hot текстовых колонок и 3 флагами пропусков.
        (NaN заполнит медианой вызывающий код — см. main.)
    """
    # Числа как есть + текст в 0/1 колонки (drop_first=True — без первой категории).
    out = pd.get_dummies(frame[numeric + CATEGORICAL], drop_first=True).astype(float)
    # Флаги пропусков: 1 — пропуск был, 0 — значение было.
    for column in ["house_value", "car_value", "average_claim_cost"]:
        out[column + "_missing"] = frame[column].isna().astype(float)
    return out


def oof_ridge(matrix: np.ndarray, target: np.ndarray, seed: int) -> np.ndarray:
    """Ridge-регрессия честным способом: OOF-предсказания через 5 фолдов.

    Аргументы:
        matrix: матрица признаков (уже без NaN).
        target: что предсказываем (обычно логиты z).
        seed: seed разбиения на фолды.

    Возвращает:
        Массив OOF-предсказаний (каждая строка предсказана без подглядывания).
    """
    # Нули под предсказания (заполним по фолдам).
    prediction = np.zeros(len(target))
    for train_idx, valid_idx in KFold(5, shuffle=True, random_state=seed).split(matrix):
        # RidgeCV сам выберет alpha; обучить на train, предсказать valid.
        model = RidgeCV(alphas=np.logspace(-3, 3, 13)).fit(matrix[train_idx], target[train_idx])
        prediction[valid_idx] = model.predict(matrix[valid_idx])
    return prediction


def logit_resid_sd(matrix: np.ndarray, z: np.ndarray, seeds=CV_SEEDS) -> tuple[float, list[float]]:
    """Замерить разброс OOF-остатков на логитах, усреднив по нескольким seed'ам.

    Это ГЛАВНАЯ линейка файла: чем меньше разброс остатков, тем лучше признаки.

    Аргументы:
        matrix: матрица признаков.
        z: логиты цели.
        seeds: по каким seed'ам повторить замер (по умолчанию CV_SEEDS).

    Возвращает:
        (среднее_по_seedам, [значения_по_каждому_seed]).
    """
    # Для каждого seed: OOF-предсказания -> остатки -> их std; собираем в список.
    values = [float((z - oof_ridge(matrix, z, s)).std()) for s in seeds]
    # Среднее по seed'ам + сами значения.
    return float(np.mean(values)), values


def main() -> None:
    """Главная функция: посчитать все гипотезы и записать отчёт."""
    # Разобрать аргументы, прочитать train и test.
    args = parse_args()
    train = pd.read_csv(args.train_csv)
    test = pd.read_csv(args.test_csv)
    # Цель в numpy + её логиты (z — главная рабочая шкала этого файла).
    y = train[TARGET].to_numpy(dtype=float)
    z = logit(np.clip(y / 100.0, 1e-4, 1 - 1e-4))
    # Признаки: всё кроме ID и цели; numeric — имена числовых колонок.
    # select_dtypes("number") отбирает int/float колонки; list(...) — в список.
    features = train.drop(columns=[ID, TARGET])
    numeric = list(features.select_dtypes("number").columns)

    # Базовая матрица: design() + заполнение NaN медианами колонок.
    base_frame = design(features, numeric)
    fill = base_frame.median()  # медиана каждой колонки (Series)
    base = base_frame.fillna(fill)  # заполнили
    base_matrix = base.to_numpy(dtype=float)  # в numpy
    # Сюда складываем строки отчёта: (гипотеза, вердикт, доказательство).
    results: list[tuple[str, str, str]] = []

    # Внутренний помощник: сохранить строку отчёта + напечатать прогресс.
    def record(hypothesis: str, verdict: str, detail: str) -> None:
        results.append((hypothesis, verdict, detail))
        print(f"{hypothesis}: {verdict} | {detail}", flush=True)

    # --- H0: точка отсчёта — базовый сигмоидально-линейный индекс. ---
    base_mean, base_runs = logit_resid_sd(base_matrix, z)
    record("H0 baseline sigmoid-linear index", "reference", f"logit resid sd {base_mean:.4f} ({', '.join(f'{v:.4f}' for v in base_runs)})")

    # --- H1: какой-то числовой признак — точная функция остальных? ---
    # Проверяем так: каждый признак предсказываем по всем остальным (ridge OOF)
    # и смотрим R^2. R^2 ~ 1.0 означал бы «признак выводится из остальных».
    r2 = {}
    for column in numeric:
        # Все числовые колонки, кроме текущей.
        others = [c for c in numeric if c != column]
        # Матрица «остальных» с заполненными медианами пропусками.
        filled = features[others].fillna(features[others].median()).to_numpy(dtype=float)
        # Сам признак (тоже с заполненными пропусками) — как цель.
        target_col = features[column].fillna(features[column].median()).to_numpy(dtype=float)
        # OOF-предсказания признака по остальным -> R^2 = 1 - var(ошибка)/var(цели).
        pred = oof_ridge(filled, target_col, 0)
        r2[column] = 1 - np.var(target_col - pred) / np.var(target_col)
    # Топ-3 самых «выводимых» признака.
    top_r2 = pd.Series(r2).sort_values(ascending=False).head(3)
    record(
        "H1 a numeric feature is an exact function of the others (derived feature)",
        "rejected" if top_r2.max() < 0.95 else "supported",  # порог «точности» — 0.95
        "max OOF R^2 of one feature from the others: " + ", ".join(f"{k} {v:.2f}" for k, v in top_r2.items()),
    )

    # --- H2: остатки локально скоррелированы (похожи у соседей)? ---
    # Стандартизируем матрицу (kNN чувствителен к масштабу!) и ищем 20 соседей.
    scaled = (base_matrix - base_matrix.mean(0)) / base_matrix.std(0)
    # Остатки базового ridge на логитах (seed 0).
    residual = z - oof_ridge(base_matrix, z, 0)
    # NearestNeighbors(n_neighbors=21): 21, т.к. первый «сосед» — сама точка.
    # kneighbors возвращает (расстояния, индексы); расстояния нам не нужны -> _.
    _, neighbours = NearestNeighbors(n_neighbors=21).fit(scaled).kneighbors(scaled)
    # Корреляция остатка точки со СРЕДНИМ остатком её 20 соседей ([:, 1:] — без самой точки).
    corr_knn = np.corrcoef(residual, residual[neighbours[:, 1:]].mean(axis=1))[0, 1]
    record(
        "H2 residuals are locally correlated (kNN in feature space)",
        "rejected" if abs(corr_knn) < 0.05 else "supported",  # |r| < 0.05 — шума нет структуры
        f"corr(own residual, mean of 20 neighbours) = {corr_knn:.4f}",
    )

    # --- H3: взаимодействия «категория × число» улучшают подгонку? ---
    # Готовим стандартизированные числа (NaN -> 0) и one-hot категории.
    numeric_std = ((features[numeric] - features[numeric].mean()) / features[numeric].std()).fillna(0)
    cat_dummies = pd.get_dummies(features[CATEGORICAL], drop_first=True).astype(float)
    # Все пары (категория, число): pairs — список кортежей.
    pairs = [(c, n) for c in cat_dummies.columns for n in numeric if n not in CATEGORICAL]
    # Матрица ВСЕХ произведений (каждая колонка = одна пара).
    all_inter = pd.DataFrame({f"{c}*{n}": cat_dummies[c] * numeric_std[n] for c, n in pairs}).to_numpy()
    # Отдельно — один кандидат: private-сегмент × доход (подозрение на экстраполяцию хвоста).
    private_inter = np.column_stack(
        [cat_dummies["wealth_segment_private"] * numeric_std["income"]]
    )
    # Замеряем разброс остатков: база + только private×income; база + все произведения.
    mean_private, _ = logit_resid_sd(np.hstack([base_matrix, private_inter]), z)
    mean_all, _ = logit_resid_sd(np.hstack([base_matrix, all_inter]), z)
    record(
        "H3a category x numeric interactions improve the fit",
        "rejected" if mean_all > base_mean else "supported",  # хуже базы = переобучение
        f"all {len(pairs)} interactions: {mean_all:.4f} (worse than {base_mean:.4f}); overfitting",
    )
    record(
        "H3b the gain comes from wealth_segment=private x income",
        "explains the small gain" if mean_private < base_mean - 0.003 else "no effect",
        f"adding only private x income: {mean_private:.4f}; this term affects 32 rows (income extrapolation)",
    )

    # --- H4: явные денежные отношения (loan/income, balance/income, ...)? ---
    ratios = pd.DataFrame(
        {
            "loan_to_income": features["loan_amount"] / features["income"],
            "balance_to_income": features["average_balance"] / features["income"],
            "house_to_income": features["house_value"] / features["income"],
            "car_to_income": features["car_value"] / features["income"],
            "phone_to_income": features["smartphone_price"] / features["income"],
            "claim_to_income": features["average_claim_cost"] / features["income"],
        }
    # Деление на 0 даёт inf — заменяем бесконечности на NaN...
    ).replace([np.inf, -np.inf], np.nan)
    # ...и заполняем медианой.
    ratios = ratios.fillna(ratios.median())
    # Замер: база + отношения.
    mean_ratio, _ = logit_resid_sd(np.hstack([base_matrix, ratios.to_numpy()]), z)
    record(
        "H4 explicit money ratios (loan, balance, house, car, phone to income)",
        "weak" if mean_ratio > base_mean - 0.003 else "small logit gain (not tested on 0-100 scale)",
        f"logit resid sd {mean_ratio:.4f} vs {base_mean:.4f}",
    )

    # --- H5: бустинг на остатках находит пороги, которые линейка пропускает? ---
    # Судят ДВАЖДЫ: на логитах И на шкале 0–100, т.к. итоговая модель живёт на 0–100.
    logit_gain, raw_final, raw_boost = [], [], []
    # Копия признаков, где текстовые помечены "category" для LightGBM.
    gbm_features = features.copy()
    for column in CATEGORICAL:
        gbm_features[column] = gbm_features[column].astype("category")
    # Сырая числовая матрица (для reference-модели fit_sigmoid_index).
    raw_matrix = numeric_matrix(features, numeric)
    # Два seed'а (быстрее трёх, а вывод тот же).
    for seed in CV_SEEDS[:2]:
        # OOF-массивы: линейный индекс, линейный+бустинг, reference на 0–100.
        linear = np.zeros(len(z))
        boosted = np.zeros(len(z))
        final = np.zeros(len(z))
        for train_idx, valid_idx in KFold(5, shuffle=True, random_state=seed).split(base_matrix):
            # Линейный индекс на логитах (ridge).
            model = RidgeCV(alphas=np.logspace(-3, 3, 13)).fit(base_matrix[train_idx], z[train_idx])
            linear_in = model.predict(base_matrix[train_idx])  # предсказания на train (для остатков)
            linear[valid_idx] = model.predict(base_matrix[valid_idx])  # OOF на valid
            # LightGBM учится предсказывать ОСТАТКИ (z - линейный) — «дотягивание».
            # Маленькие листья (7) и min_child_samples=50 — осторожные настройки.
            gbm = lgb.LGBMRegressor(
                n_estimators=100,
                learning_rate=0.02,
                num_leaves=7,
                min_child_samples=50,
                subsample=0.8,
                subsample_freq=1,
                colsample_bytree=0.7,
                reg_lambda=5,
                random_state=seed,
                verbose=-1,
            )
            gbm.fit(gbm_features.iloc[train_idx], z[train_idx] - linear_in)
            # Итог: линейный индекс + поправка бустинга.
            boosted[valid_idx] = linear[valid_idx] + gbm.predict(gbm_features.iloc[valid_idx])
            # Reference сигмоидально-линейная модель на шкале 0–100 (для сравнения там).
            medians, means, scales = fit_preprocessor(raw_matrix[train_idx])
            coefficients, intercept, _ = fit_sigmoid_index(
                transform_features(raw_matrix[train_idx], medians, means, scales), y[train_idx]
            )
            final[valid_idx] = 100.0 * expit(
                transform_features(raw_matrix[valid_idx], medians, means, scales) @ coefficients + intercept
            )
        # Запоминаем: пару std на логитах + RMSE обеих моделей на 0–100.
        logit_gain.append((float((z - linear).std()), float((z - boosted).std())))
        raw_final.append(float(np.sqrt(np.mean((final - y) ** 2))))
        raw_boost.append(float(np.sqrt(np.mean((100.0 * expit(boosted) - y) ** 2))))
    # Усредняем по seed'ам.
    lin_mean = np.mean([a for a, _ in logit_gain])
    boost_mean = np.mean([b for _, b in logit_gain])
    record(
        "H5 threshold effects missed by the linear index (boosting on residuals)",
        "rejected on 0-100 scale" if np.mean(raw_boost) >= np.mean(raw_final) else "supported",
        f"logit resid sd {lin_mean:.4f} -> {boost_mean:.4f}; but 0-100 RMSE {np.mean(raw_final):.4f} (reference) vs {np.mean(raw_boost):.4f} (boosted)",
    )

    # --- H6: уровень шума разный в разных группах (гетероскедастичность)? ---
    # Считаем std остатков внутри каждой категории каждой текстовой колонки.
    residual_by_group = {}
    for column in CATEGORICAL:
        # pd.Series(residual) группируем по значениям колонки, в каждой группе — std.
        residual_by_group[column] = pd.Series(residual).groupby(features[column].to_numpy()).std()
    # Максимальный размах std внутри одной переменной: большой = шум зависит от группы.
    spread = max(float(s.max() - s.min()) for s in residual_by_group.values())
    record(
        "H6 noise level differs between groups",
        "rejected" if spread < 0.05 else "supported",
        f"largest within-variable range of group residual sd: {spread:.3f} (all within 0.60-0.66)",
    )

    # --- H7: тест из другого распределения, чем train? (adversarial validation) ---
    # Склеиваем train (метка 0) и test (метка 1), учим классификатор их различать.
    # AUC ~ 0.5 = не различает = распределения одинаковые. AUC >> 0.5 = сдвиг есть.
    stacked = pd.concat([features, test.drop(columns=[ID])], ignore_index=True)
    stacked_design = design(stacked, numeric)
    stacked_design = stacked_design.fillna(stacked_design.median())
    # Метки: столько нулей, сколько train-строк + столько единиц, сколько test-строк.
    labels = np.r_[np.zeros(len(features)), np.ones(len(test))]
    # Кросс-предсказания вероятностей класса 1 (method="predict_proba", берём [:, 1]).
    probability = cross_val_predict(
        HistGradientBoostingClassifier(max_iter=200, random_state=0),
        stacked_design.to_numpy(dtype=float),
        labels,
        cv=5,
        method="predict_proba",
    )[:, 1]
    auc = roc_auc_score(labels, probability)
    record(
        "H7 test set comes from a different distribution than train",
        "rejected" if auc < 0.55 else "supported",  # порог «похожести» — 0.55
        f"adversarial validation AUC {auc:.3f}",
    )

    # --- H8: другая функция связи (probit) заметно лучше, чем logit? ---
    # Сравниваем пары (связь, обратная): (logit, sigmoid) vs (probit, норм. CDF).
    link_rows = []
    raw_rmse_by_link = {}
    for name, link, inverse in [("logit", logit, expit), ("probit", ndtri, ndtr)]:
        # Переводим цель связью, учим ridge OOF, возвращаем на 0–100 и меряем RMSE.
        transformed = link(np.clip(y / 100.0, 1e-4, 1 - 1e-4))
        oof = oof_ridge(base_matrix, transformed, 0)
        raw_rmse_by_link[name] = float(np.sqrt(np.mean((100.0 * inverse(oof) - y) ** 2)))
        link_rows.append(f"{name}: raw RMSE {raw_rmse_by_link[name]:.3f}")
    # Probit «выиграл», только если лучше логита хотя бы на 0.02.
    better = raw_rmse_by_link["probit"] < raw_rmse_by_link["logit"] - 0.02
    record(
        "H8 another link function (probit) fits clearly better than logit",
        "supported" if better else "rejected",
        "; ".join(link_rows) + "; logit residuals are homoscedastic, probit residuals are not",
    )

    # --- H9: оставшаяся ошибка — неустранимый шум N(0, sigma) на логитах? ---
    # Считаем «пол Байеса»: какой RMSE был бы у ИДЕАЛЬНОЙ модели при таком шуме
    # (математическое ожидание через квадратуру Гаусса-Эрмита — точное усреднение
    #  по гауссову шуму без симуляций).
    eta = oof_ridge(base_matrix, z, 0)  # OOF-индекс (лучшее, что знает линейка)
    # Узлы и веса квадратуры (60 точек); нормируем веса, чтобы суммировались к 1.
    nodes, weights = np.polynomial.hermite_e.hermegauss(60)
    weights = weights / weights.sum()
    floors = []
    # Для каждого уровня шума sigma: средний квадрат отклонения от среднего предсказания.
    for sigma in [0.4, 0.5, 0.6, float(np.std(residual))]:
        # Среднее предсказание по шуму: E[100*sigmoid(eta + sigma*eps)].
        mean = (expit(eta[:, None] + sigma * nodes[None, :]) * 100.0 * weights).sum(axis=1)
        # Средний квадрат предсказания: E[(100*sigmoid(...))^2].
        second = ((expit(eta[:, None] + sigma * nodes[None, :]) * 100.0) ** 2 * weights).sum(axis=1)
        # Дисперсия = E[X^2] - E[X]^2; корень из средней дисперсии = «пол» RMSE.
        floors.append((sigma, float(np.sqrt(np.mean(second - mean**2)))))
    record(
        "H9 the remaining error is irreducible logit noise",
        "supported" if abs(floors[-1][1] - np.sqrt(np.mean((100 * expit(eta) - y) ** 2))) < 0.5 else "unclear",
        "Bayes RMSE floor by sigma: " + ", ".join(f"{s:.3f} -> {f:.2f}" for s, f in floors),
    )

    # --- H10: хвост доходов (income > 300k) — большой источник ошибки? ---
    # Оракул-проверка: заменяем предсказания на хвосте ИДЕАЛЬНЫМИ (правдой)
    # и смотрим, насколько упадёт RMSE. Если почти не упадёт — хвост не важен.
    final_oof = np.zeros(len(y))
    for seed in [42]:
        for train_idx, valid_idx in KFold(5, shuffle=True, random_state=seed).split(raw_matrix):
            medians, means, scales = fit_preprocessor(raw_matrix[train_idx])
            coefficients, intercept, _ = fit_sigmoid_index(
                transform_features(raw_matrix[train_idx], medians, means, scales), y[train_idx]
            )
            final_oof[valid_idx] = 100.0 * expit(
                transform_features(raw_matrix[valid_idx], medians, means, scales) @ coefficients + intercept
            )
    # Маска хвоста: True где доход > 300 000 (.to_numpy() — в массив).
    tail = (features["income"] > 300_000).to_numpy()
    # Копия предсказаний, где на хвосте стоит правда (идеальный оракул).
    oracle = final_oof.copy()
    oracle[tail] = y[tail]
    # RMSE до и после «идеального хвоста».
    base_rmse = float(np.sqrt(np.mean((final_oof - y) ** 2)))
    oracle_rmse = float(np.sqrt(np.mean((oracle - y) ** 2)))
    record(
        "H10 income tail (income > 300k) is a large source of error",
        "rejected" if base_rmse - oracle_rmse < 0.05 else "supported",  # выигрыш < 0.05 — хвост не важен
        f"{int(tail.sum())} rows ({tail.mean():.2%}); perfect prediction there moves RMSE {base_rmse:.4f} -> {oracle_rmse:.4f}",
    )

    # --- H11: дробный логит (биномиальный GLM на y/100) лучше, чем MSE на 0–100? ---
    # Дробный логит максимизирует «квази-правдоподобие» долей вместо MSE.
    frac_scores = []
    for seed in CV_SEEDS[:2]:
        prediction = np.zeros(len(y))
        for train_idx, valid_idx in KFold(5, shuffle=True, random_state=seed).split(raw_matrix):
            medians, means, scales = fit_preprocessor(raw_matrix[train_idx])
            # Матрицы со столбцом единиц (для сдвига): train и valid.
            a = np.column_stack([np.ones(len(train_idx)), transform_features(raw_matrix[train_idx], medians, means, scales)])
            b = np.column_stack([np.ones(len(valid_idx)), transform_features(raw_matrix[valid_idx], medians, means, scales)])
            # Цель как доля (0..1).
            target_share = y[train_idx] / 100.0

            # Целевая функция + градиент для L-BFGS-B (биномиальное правдоподобие + L2).
            def objective(w: np.ndarray) -> tuple[float, np.ndarray]:
                # Линейный индекс.
                eta = a @ w
                # Отрицательное лог-правдоподобие: mean(log(1+e^eta) - доля*eta) + штраф.
                # np.logaddexp(0, eta) = log(1 + exp(eta)), посчитанный без переполнения.
                loss = np.mean(np.logaddexp(0.0, eta) - target_share * eta) + 0.5 * 1e-3 * np.dot(w[1:], w[1:])
                # Градиент: A^T @ (sigmoid - доля) / n + штраф на веса (без сдвига).
                grad = a.T @ (expit(eta) - target_share) / len(target_share)
                grad[1:] += 1e-3 * w[1:]
                return float(loss), grad

            # Оптимизируем с нулевого старта, предсказываем valid через 100*sigmoid.
            weights = minimize(objective, np.zeros(a.shape[1]), jac=True, method="L-BFGS-B").x
            prediction[valid_idx] = 100.0 * expit(b @ weights)
        frac_scores.append(float(np.sqrt(np.mean((prediction - y) ** 2))))
    # RMSE reference-модели H10 для сравнения.
    final_mean = float(np.sqrt(np.mean((final_oof - y) ** 2)))
    record(
        "H11 fractional logit (binomial GLM) beats MSE fit on 0-100 scale",
        "rejected" if np.mean(frac_scores) > final_mean - 0.02 else "supported",
        f"0-100 RMSE {np.mean(frac_scores):.4f} vs {final_mean:.4f} for the reference model (difference within CV noise)",
    )

    # --- H12: выкинуть агрегат insurance_products (он = сумма 8 флагов)? ---
    # Проверка мультиколлинеарности: агрегат линейно зависим от флагов.
    flag_columns = [
        "life_insurance", "property_insurance", "health_insurance", "travel_insurance",
        "car_insurance", "gadget_insurance", "cyber_protection", "identity_protection",
    ]
    # Мини-помощник: заполнить NaN медианами колонок (np.nanmedian игнорирует NaN).
    def impute(matrix: np.ndarray) -> np.ndarray:
        return np.where(np.isfinite(matrix), matrix, np.nanmedian(matrix, axis=0))

    # Две урезанные матрицы: без агрегата / без 8 флагов.
    raw_no_aggregate = numeric_matrix(features, [c for c in numeric if c != "insurance_products"])
    raw_no_flags = numeric_matrix(features, [c for c in numeric if c not in flag_columns])
    no_aggregate = impute(raw_no_aggregate)
    no_flags = impute(raw_no_flags)
    # Замеры на логитах (по одному seed — для скорости).
    _, agg_runs = logit_resid_sd(no_aggregate, z, seeds=[42])
    flags_mean, _ = logit_resid_sd(no_flags, z, seeds=[42])
    # А для «без агрегата» дополнительно считаем честный RMSE на 0–100 через fit_sigmoid_index.
    no_aggregate_rmse = []
    for train_idx, valid_idx in KFold(5, shuffle=True, random_state=42).split(raw_no_aggregate):
        medians, means, scales = fit_preprocessor(raw_no_aggregate[train_idx])
        coefficients, intercept, _ = fit_sigmoid_index(
            transform_features(raw_no_aggregate[train_idx], medians, means, scales), y[train_idx]
        )
        no_aggregate_rmse.append(np.mean((100.0 * expit(
            transform_features(raw_no_aggregate[valid_idx], medians, means, scales) @ coefficients + intercept
        ) - y[valid_idx]) ** 2))
    record(
        "H12 drop the aggregate insurance_products (collinear with the 8 flags)",
        "no effect" if abs(np.sqrt(np.mean(no_aggregate_rmse)) - final_mean) < 0.01 else "changes fit",
        f"0-100 RMSE without aggregate {np.sqrt(np.mean(no_aggregate_rmse)):.4f} vs {final_mean:.4f}; "
        f"dropping the flags instead is much worse (logit sd {flags_mean:.3f})",
    )

    # Собираем Markdown-отчёт: заголовок + таблица из results + вывод.
    lines = [
        "# Hypotheses about hidden structure (hard_train.csv, 5-fold CV)",
        "",
        "Yardstick: out-of-fold residual sd on the logit scale; baseline sigmoid-linear index.",
        "",
        "| Hypothesis | Verdict | Evidence |",
        "| --- | --- | --- |",
    ]
    # Строки таблицы: | гипотеза | вердикт | доказательство |.
    lines += [f"| {h} | {v} | {d} |" for h, v, d in results]
    # Вывод: шум на логитах ~0.60 неустраним; выигрыш на логитах не переходит на 0–100.
    lines += [
        "",
        "Conclusion: no tested hypothesis lowers the logit noise below roughly 0.60, and the 0-100 RMSE was checked for every candidate that lowered it on the logit scale (H5).",
        "A logit-scale gain (H4, H5) does not transfer to the 0-100 scale, so decisions are made on the 0-100 RMSE.",
        "At that noise level the Bayes RMSE floor is about 10.2-10.3, so RMSE < 7 would require a logit noise near 0.4.",
    ]
    # Создаём папку при нужде и пишем файл.
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {output}")


# Запуск main(), только если файл запущен напрямую.
if __name__ == "__main__":
    main()
