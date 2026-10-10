"""Обучение финальной модели: ансамбль CatBoost (5 фолдов) на hard_train.csv.

Простыми словами, что делает этот скрипт:
  1. Читает ``hard_train.csv`` (10 000 клиентов с известными оценками).
  2. Создаёт новые признаки (``generate_features``) — их станет 61.
  3. Делит данные на 5 частей (фолдов) и обучает 5 моделей CatBoost:
     каждая учится на 4/5 данных и проверяется на оставшейся 1/5.
     Это называется кросс-валидация (cross-validation).
  4. Сохраняет в ``--model-dir`` (по умолчанию ``artifacts/``):
       * ``catboost_fold_{0..4}.cbm`` — веса 5 моделей;
       * ``fold_preprocessors.npz`` — имена признаков + медианы/средние/масштабы
         каждого фолда + моды текстовых колонок;
       * ``feature_coefficients.csv`` — важности признаков;
       * ``model_metadata.json`` — seed, список признаков, гиперпараметры,
         версии библиотек, метрики (OOF RMSE).

Словарик:
  * CatBoost — градиентный бустинг на деревьях решений от Яндекса:
    строит много маленьких деревьев, каждое исправляет ошибки предыдущих.
  * Ансамбль — несколько моделей, чьи предсказания усредняем (точнее и стабильнее).
  * OOF (out-of-fold) — предсказание для каждой строки от модели, которая
    эту строку НЕ видела при обучении. Честная оценка качества.
  * RMSE — корень из среднего квадрата ошибки: насколько в среднем
    предсказание отличается от правды (в тех же единицах, 0–100).
  * Seed — число для фиксации случайности (см. set_global_seed).

Про seed:
  * ``--seed N`` — фиксированный seed (запишется в метаданные и в имя submission).
  * без ``--seed`` — скрипт сам выберет случайный seed и напечатает его.

Пример запуска:
    python train.py --train-csv hard_train.csv --model-dir artifacts --seed 434089
"""

# Включаем «новые» правила аннотаций типов (ни на что в работе не влияет).
from __future__ import annotations

# Стандартные модули Python:
import argparse  # разбор аргументов командной строки (--seed, --train-csv, ...)
import json  # чтение/запись JSON (для model_metadata.json)
import platform  # узнать версию Python (запишем в метаданные)
import random  # случайные числа (нужен, чтобы ВЫБРАТЬ случайный seed)
from pathlib import Path  # удобная работа с путями к файлам/папкам

# CatBoostRegressor — регрессор CatBoost (предсказывает число, а не класс).
import catboost
from catboost import CatBoostRegressor
import numpy as np  # массивы и математика
import pandas as pd  # таблицы, чтение CSV
import scipy  # нужен для записи версии в метаданные
import sklearn  # нужен для записи версии в метаданные
from sklearn.model_selection import KFold  # разбиение данных на фолды

# Наши общие функции из model_utils.py (единая обработка признаков).
from model_utils import (
    CATEGORICAL_COLUMNS,  # список текстовых колонок (для мод)
    TARGET_COLUMN,  # имя цели "protection_score"
    fit_categorical_modes,  # выучить моды текстовых колонок
    fit_preprocessor,  # выучить медианы/средние/масштабы на train-фолде
    generate_features,  # создать 9 новых признаков
    get_numeric_feature_names,  # отобрать числовые признаки
    numeric_matrix,  # DataFrame -> матрица numpy
    set_global_seed,  # зафиксировать случайность
    transform_features,  # применить препроцессор (заполнить + стандартизировать)
)

# Значения гиперпараметров CatBoost по умолчанию (вынесены в константы,
# чтобы их было видно в одном месте и легко менять).
# Гиперпараметр — настройка обучения, которую задаём МЫ (а не учит модель).
DEFAULT_DEPTH = 5  # глубина каждого дерева (5 — неглубокие деревья = регуляризация)
DEFAULT_LEARNING_RATE = 0.03  # скорость обучения: насколько сильно каждое новое дерево правит ошибки
DEFAULT_L2_LEAF_REG = 30.0  # L2-регуляризация на листьях: большая = сильнее «сглаживает», борется с шумом
DEFAULT_N_ESTIMATORS = 500  # максимум деревьев в одной модели (итераций бустинга)
DEFAULT_EARLY_STOPPING_ROUNDS = 50  # ранняя остановка: если 50 деревьев подряд нет улучшения на валидации — хватит
DEFAULT_LOSS_FUNCTION = "RMSE"  # функция потерь: что минимизируем при обучении (RMSE)


def parse_args() -> argparse.Namespace:
    """Описать и разобрать аргументы командной строки (то, что после `python train.py ...`).

    Возвращает:
        Объект с полями args.train_csv, args.model_dir, args.seed и т.д.
    """
    # Создаём парсер; description=__doc__ покажет текст из шапки файла в --help.
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # Путь к обучающему CSV (по умолчанию hard_train.csv в текущей папке).
    parser.add_argument("--train-csv", default="hard_train.csv", help="Path to labeled training CSV")
    # Папка, куда сохранить веса и метаданные (по умолчанию artifacts/).
    parser.add_argument("--model-dir", default="artifacts", help="Directory for model and metadata")
    # Seed (необязательный): если не передать, выберем случайный.
    parser.add_argument(
        "--seed",
        type=int,  # автоматически превратить текст в целое число
        default=None,  # None = «не передан»
        help="Fixed seed. If omitted, a new random seed is drawn and printed.",
    )
    # Число фолдов кросс-валидации (и число моделей в ансамбле).
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Number of folds for cross-validation and ensemble models",
    )
    # Дальше — гиперпараметры CatBoost (у каждого есть значение по умолчанию выше).
    parser.add_argument("--depth", type=int, default=DEFAULT_DEPTH, help="Tree depth (default: 5)")
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE, help="Learning rate")
    parser.add_argument("--l2-leaf-reg", type=float, default=DEFAULT_L2_LEAF_REG, help="L2 regularization on leaves")
    parser.add_argument("--n-estimators", type=int, default=DEFAULT_N_ESTIMATORS, help="Max iterations/trees")
    # Сколько итераций без улучшения ждать перед ранней остановкой.
    parser.add_argument(
        "--early-stopping-rounds",
        type=int,
        default=DEFAULT_EARLY_STOPPING_ROUNDS,
        help="Early stopping rounds on validation fold",
    )
    # Функция потерь CatBoost (по умолчанию RMSE).
    parser.add_argument(
        "--loss-function",
        default=DEFAULT_LOSS_FUNCTION,
        help="CatBoost loss function (default: RMSE)",
    )
    # Разобрать то, что пользователь написал в командной строке, и вернуть результат.
    return parser.parse_args()


def main() -> None:
    """Главная функция: весь процесс обучения от чтения CSV до сохранения весов."""
    # Шаг 0. Разобрать аргументы командной строки.
    args = parse_args()
    # Меньше 2 фолдов кросс-валидация бессмысленна — сразу падаем с понятной ошибкой.
    if args.cv_folds < 2:
        raise ValueError("--cv-folds must be at least 2")

    # Шаг 1. Определить seed. Запоминаем, задал ли его пользователь явно,
    # чтобы честно записать seed_policy в метаданные.
    seed_was_given = args.seed is not None  # True, если пользователь передал --seed
    if args.seed is None:
        # SystemRandom — криптографически сильный генератор (для выбора seed'а).
        # randrange(1, 1_000_000) — случайное число от 1 до 999999.
        args.seed = random.SystemRandom().randrange(1, 1_000_000)
        print(f"No --seed given; drawn random seed: {args.seed}")
    else:
        print(f"Fixed seed: {args.seed}")
    # Фиксируем все генераторы случайных чисел выбранным seed'ом.
    set_global_seed(args.seed)

    # Шаг 2. Прочитать обучающий CSV в таблицу pandas.
    train_path = Path(args.train_csv)  # превращаем строку пути в объект Path
    # Проверяем, что файл существует, иначе падаем с понятной ошибкой.
    if not train_path.exists():
        raise FileNotFoundError(f"Training CSV not found: {train_path}")

    # Читаем CSV: data — таблица (DataFrame), строки — клиенты, столбцы — признаки + цель.
    data = pd.read_csv(train_path)
    # Проверяем, что в таблице есть колонка цели.
    if TARGET_COLUMN not in data.columns:
        raise ValueError(f"Training CSV must contain the target column {TARGET_COLUMN!r}")
    # Проверяем, что в цели нет пропусков (учиться на пустой правде нельзя).
    if data[TARGET_COLUMN].isna().any():
        raise ValueError(f"Training target {TARGET_COLUMN!r} contains missing values")
    # Проверяем, что все цели в допустимом диапазоне [0, 100].
    # .between(0.0, 100.0) даёт True/False по каждой строке, .all() — «все ли True».
    if not data[TARGET_COLUMN].between(0.0, 100.0).all():
        raise ValueError(f"Training target {TARGET_COLUMN!r} must lie in [0, 100]")

    # Шаг 3. Создать новые признаки (feature engineering): +9 колонок.
    data = generate_features(data)

    # Шаг 4. Отобрать числовые признаки (всё числовое, кроме ID и цели).
    feature_names = get_numeric_feature_names(data)
    # Если числовых признаков нет вообще — учиться не на чем, падаем.
    if not feature_names:
        raise ValueError("No numeric predictor columns were found")
    # Превратить признаки в матрицу numpy (NaN и мусор -> NaN, см. numeric_matrix).
    raw_matrix = numeric_matrix(data, feature_names)
    # Цель — в одномерный numpy-массив float (модели любят numpy).
    target = data[TARGET_COLUMN].astype(float).to_numpy()

    # Шаг 5. Создать папку для весов (parents=True — создать и родителей при нужде,
    # exist_ok=True — не падать, если папка уже есть).
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    # Готовим пустые списки/массивы, куда будем складывать результаты по фолдам:
    fold_medians_list: list[np.ndarray] = []  # медианы каждого фолда
    fold_means_list: list[np.ndarray] = []  # средние каждого фолда
    fold_scales_list: list[np.ndarray] = []  # масштабы каждого фолда
    fold_model_files: list[str] = []  # имена файлов весов ("catboost_fold_0.cbm", ...)
    fold_best_iterations: list[int] = []  # на какой итерации остановился каждый фолд
    fold_rmses: list[float] = []  # RMSE каждого фолда на его валидации
    # Суммарные важности признаков (будем усреднять по фолдам; стартуем с нулей).
    fold_importances = np.zeros(len(feature_names), dtype=float)

    # Массив под OOF-предсказания: для каждой строки — предсказание модели,
    # которая эту строку НЕ видела. Стартуем с NaN, заполним по ходу фолдов.
    oof_prediction = np.full(len(data), np.nan, dtype=float)
    # Создаём разбивальщик на фолды: перемешать (shuffle) с нашим seed и резать на K частей.
    splitter = KFold(n_splits=args.cv_folds, shuffle=True, random_state=args.seed)

    # Шаг 6. Главный цикл кросс-валидации: enumerate(...) даёт (номер, (train_idx, valid_idx)).
    # start=1 — нумерация фолдов с 1 для красивой печати ("fold 1/5").
    # splitter.split(raw_matrix) возвращает индексы строк train/validation каждого фолда.
    for fold_number, (fold_train, fold_validation) in enumerate(splitter.split(raw_matrix), start=1):
        # 6а. Выучить препроцессор ТОЛЬКО на train-части фолда (валидацию не подглядываем!).
        f_medians, f_means, f_scales = fit_preprocessor(raw_matrix[fold_train])
        # 6б. Применить препроцессор к train-части (заполнить пропуски + стандартизировать).
        f_x_train = transform_features(raw_matrix[fold_train], f_medians, f_means, f_scales)
        # 6в. Применить ТОТ ЖЕ препроцессор к validation-части (параметры — с train!).
        f_x_validation = transform_features(raw_matrix[fold_validation], f_medians, f_means, f_scales)

        # 6г. Создать модель CatBoost с нашими гиперпараметрами.
        cb_model = CatBoostRegressor(
            loss_function=args.loss_function,  # что минимизировать (RMSE)
            eval_metric="RMSE",  # что смотреть на валидации для ранней остановки
            depth=args.depth,  # глубина деревьев
            learning_rate=args.learning_rate,  # скорость обучения
            l2_leaf_reg=args.l2_leaf_reg,  # регуляризация листьев
            iterations=args.n_estimators,  # максимум деревьев
            random_seed=args.seed + fold_number,  # свой seed у каждого фолда (разные модели!)
            verbose=False,  # не печатать прогресс каждой итерации
        )
        # 6д. Обучить модель: fit(X_train, y_train), следя за валидацией.
        # eval_set — на чём мерить качество для ранней остановки (обучения на нём НЕТ).
        cb_model.fit(
            f_x_train,  # признаки train-части фолда
            target[fold_train],  # цели train-части фолда
            eval_set=(f_x_validation, target[fold_validation]),  # валидация для early stopping
            early_stopping_rounds=args.early_stopping_rounds,  # терпение ранней остановки
            verbose=False,  # тихий режим
        )

        # 6е. Предсказать валидационную часть фолда и записать в OOF-массив.
        fold_pred = cb_model.predict(f_x_validation)
        # Кладём предсказания на их места (по индексам валидационных строк).
        oof_prediction[fold_validation] = fold_pred

        # 6ж. Сохранить веса модели фолда на диск (имена с нуля: fold_0 ... fold_4).
        model_filename = f"catboost_fold_{fold_number - 1}.cbm"
        # save_model пишет файл; str(...) превращает Path в строку.
        cb_model.save_model(str(model_dir / model_filename))
        # Запоминаем имя файла, чтобы записать в метаданные.
        fold_model_files.append(model_filename)

        # 6з. Узнать номер лучшей итерации (где было лучшее качество на валидации).
        best_iter = int(cb_model.get_best_iteration())
        fold_best_iterations.append(best_iter)
        # 6и. Добавить важности признаков этой модели в общую сумму (сразу делим на
        # число фолдов, чтобы в конце получилось СРЕДНЕЕ по фолдам).
        fold_importances += cb_model.get_feature_importance() / args.cv_folds

        # 6к. Сохранить препроцессор этого фолда (понадобится в predict.py!).
        fold_medians_list.append(f_medians)
        fold_means_list.append(f_means)
        fold_scales_list.append(f_scales)

        # 6л. Посчитать RMSE фолда: sqrt(mean((правда - предсказание)^2)).
        fold_rmse = float(np.sqrt(np.mean((target[fold_validation] - fold_pred) ** 2)))
        fold_rmses.append(fold_rmse)
        # Напечатать прогресс: номер фолда, лучшую итерацию и RMSE (:.4f = 4 знака после точки).
        print(f"Completed CV fold {fold_number}/{args.cv_folds} (best iteration: {best_iter}, RMSE: {fold_rmse:.4f})")

    # Шаг 7. Посчитать ОБЩИЙ OOF RMSE по всем строкам сразу.
    cross_validation_rmse = float(np.sqrt(np.mean((target - oof_prediction) ** 2)))

    # Шаг 8. Выучить моды текстовых колонок (для полноты; моделью не используются,
    # но сохраняются, чтобы будущие модели могли их использовать безопасно).
    categorical_modes = fit_categorical_modes(data, CATEGORICAL_COLUMNS)

    # Шаг 9. Сохранить все параметры препроцессоров в один сжатый .npz файл.
    # np.savez_compressed пишет несколько именованных массивов в один файл.
    np.savez_compressed(
        model_dir / "fold_preprocessors.npz",  # путь к файлу
        feature_names=np.asarray(feature_names, dtype=str),  # имена признаков (порядок важен!)
        fold_medians=np.asarray(fold_medians_list, dtype=float),  # медианы: (фолды, признаки)
        fold_means=np.asarray(fold_means_list, dtype=float),  # средние: (фолды, признаки)
        fold_scales=np.asarray(fold_scales_list, dtype=float),  # масштабы: (фолды, признаки)
        fold_rmses=np.asarray(fold_rmses, dtype=float),  # RMSE фолдов
        categorical_columns=np.asarray(list(categorical_modes.keys()), dtype=str),  # имена текстовых колонок
        categorical_modes=np.asarray(list(categorical_modes.values()), dtype=str),  # их моды
    )

    # Шаг 10. Сохранить таблицу важностей признаков в CSV.
    # (Колонки standardized_coefficient/absolute_coefficient — исторические названия,
    #  оставленные для совместимости; в них те же важности.)
    importance_df = pd.DataFrame(
        {
            "feature": feature_names,  # имя признака
            "importance": fold_importances,  # средняя важность по фолдам
            "standardized_coefficient": fold_importances,  # то же (для совместимости)
            "absolute_coefficient": fold_importances,  # то же (для совместимости)
        }
    # Сортируем по убыванию важности: самые важные сверху.
    ).sort_values("importance", ascending=False)
    # Пишем CSV без колонки индексов (index=False).
    importance_df.to_csv(model_dir / "feature_coefficients.csv", index=False)

    # Шаг 11. Собрать словарь метаданных: всё о прогоне в одном месте.
    metadata = {
        "model_file": "fold_preprocessors.npz",  # главный файл препроцессоров
        "fold_model_files": fold_model_files,  # файлы весов по фолдам
        "model_type": f"CatBoostRegressor {args.cv_folds}-fold ensemble (loss={args.loss_function}, depth={args.depth}, l2_leaf_reg={args.l2_leaf_reg})",
        "target": TARGET_COLUMN,  # что предсказываем
        "prediction_formula": f"Average of {args.cv_folds} CatBoost fold models, clipped to [0, 100]",
        "seed": int(args.seed),  # seed прогона (predict.py возьмёт его для имени файла)
        # Честно пишем, задал ли seed пользователь или он случайный:
        "seed_policy": "fixed with --seed" if seed_was_given else "random per run",
        "training_rows": int(len(data)),  # сколько строк было в обучении
        # Блок кросс-валидации: число фолдов, общий OOF RMSE, RMSE и итерации по фолдам.
        "cross_validation": {
            "folds": int(args.cv_folds),
            "oof_rmse": cross_validation_rmse,
            "fold_rmses": fold_rmses,
            "fold_best_iterations": fold_best_iterations,
        },
        "features": feature_names,  # список признаков в правильном порядке
        # Как заполняли пропуски:
        "imputation": {
            "numeric": "median of training fold",  # числовые — медианой train-фолда
            "categorical": categorical_modes,  # текстовые — модами
            "categorical_used_by_model": False,  # текстовые моделью НЕ используются
        },
        # Гиперпараметры (чтобы точно воспроизвести прогон):
        "hyperparameters": {
            "loss_function": args.loss_function,
            "depth": args.depth,
            "learning_rate": args.learning_rate,
            "l2_leaf_reg": args.l2_leaf_reg,
            "n_estimators": args.n_estimators,
            "early_stopping_rounds": args.early_stopping_rounds,
        },
        # Версии софта (важно для воспроизводимости):
        "versions": {
            "python": platform.python_version(),  # версия Python
            "numpy": np.__version__,  # версия numpy
            "pandas": pd.__version__,  # версия pandas
            "scikit_learn": sklearn.__version__,  # версия sklearn
            "scipy": scipy.__version__,  # версия scipy
            "catboost": catboost.__version__,  # версия catboost
        },
    }
    # Пишем метаданные в JSON: ensure_ascii=False — можно русские буквы,
    # indent=2 — красивые отступы, + перевод строки в конце файла.
    with (model_dir / "model_metadata.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)
        file.write("\n")

    # Шаг 12. Напечатать итоговую сводку в консоль.
    print(f"\n{args.cv_folds}-fold OOF RMSE: {cross_validation_rmse:.4f}")
    print(f"Fold RMSEs: {[round(r, 4) for r in fold_rmses]}")
    print(f"Numeric predictors: {len(feature_names)}")
    print(f"Saved {args.cv_folds} fold models and metadata to: {model_dir}")


# Стандартный приём Python: код ниже выполняется, только если файл запущен
# напрямую (python train.py ...), а НЕ импортирован (import train).
if __name__ == "__main__":
    # Запускаем главную функцию.
    main()
