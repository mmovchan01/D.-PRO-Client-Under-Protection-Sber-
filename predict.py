"""Предсказание protection_score сохранённым ансамблем CatBoost. Обучения здесь НЕТ.

Простыми словами, что делает скрипт:
  1. Загружает из ``--model-dir`` (по умолчанию ``artifacts/``) всё, что сохранил
     ``train.py``: 5 моделей фолдов + препроцессоры + метаданные (включая seed).
  2. Читает тестовый CSV (без колонки цели) и создаёт те же 9 новых признаков.
  3. Аккуратно обрабатывает пропуски и отсутствующие колонки (медианами с train).
  4. Каждой из 5 моделей предсказывает тест, затем УСРЕДНЯЕТ 5 предсказаний.
  5. Обрезает результат в [0, 100] и пишет ``submission_seed_{SEED}.csv``
     (две колонки: customer_id, protection_score).

Важно: НИКАКОГО обучения тут нет — только загрузка весов и предсказание.
Гарантии надёжности: скрипт не падает, если в тесте NaN, текст вместо чисел
или даже целиком отсутствует какая-то колонка, — всё заполнится медианами.

Примеры запуска:
    python predict.py --input-csv hard_test.csv --model-dir artifacts
    python predict.py --input-csv private_test.csv --model-dir artifacts
"""

# Включаем «новые» правила аннотаций типов (на работу не влияет).
from __future__ import annotations

# Стандартные модули Python:
import argparse  # разбор аргументов командной строки
import json  # чтение model_metadata.json
from pathlib import Path  # удобные пути к файлам

# CatBoostRegressor — тот же класс модели, что и в train.py (нужен, чтобы
# создать пустые модели и загрузить в них сохранённые веса).
from catboost import CatBoostRegressor
import numpy as np  # массивы и математика
import pandas as pd  # таблицы, чтение/запись CSV

# Наши общие функции (та же обработка признаков, что при обучении!).
from model_utils import (
    CATEGORICAL_COLUMNS,  # список текстовых колонок
    ID_COLUMN,  # имя колонки ID ("customer_id")
    TARGET_COLUMN,  # имя цели ("protection_score") — так назовём колонку предсказаний
    ensure_columns,  # добавить отсутствующие колонки как NaN
    generate_features,  # создать 9 новых признаков (как в train.py!)
    impute_categorical,  # заполнить текстовые пропуски модами
    numeric_matrix,  # DataFrame -> матрица numpy
    set_global_seed,  # зафиксировать случайность
    transform_features,  # применить сохранённый препроцессор
)


def parse_args() -> argparse.Namespace:
    """Описать и разобрать аргументы командной строки.

    Возвращает:
        Объект с полями args.input_csv, args.model_dir, args.output_csv.
    """
    # Создаём парсер; в --help покажем текст из шапки файла.
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # Путь к тестовому CSV без меток (по умолчанию hard_test.csv).
    parser.add_argument("--input-csv", default="hard_test.csv", help="Unlabeled test CSV")
    # Папка с сохранённой моделью (по умолчанию artifacts/).
    parser.add_argument("--model-dir", default="artifacts", help="Directory containing the saved model")
    # Куда записать submission (по умолчанию submission_seed_<seed модели>.csv).
    parser.add_argument(
        "--output-csv",
        default=None,  # None = «придумать имя автоматически по seed из метаданных»
        help="Submission path (default: submission_seed_<model seed>.csv in the current directory)",
    )
    # Разобрать командную строку и вернуть результат.
    return parser.parse_args()


def load_ensemble(
    model_dir: Path,
) -> tuple[
    dict,
    list[str],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[CatBoostRegressor],
    dict[str, str],
]:
    """Загрузить весь ансамбль: метаданные, признаки, препроцессоры, модели, моды.

    Аргументы:
        model_dir: папка, куда train.py всё сохранил (там model_metadata.json и .cbm файлы).

    Возвращает:
        Кортеж из 7 элементов:
          * metadata — словарь метаданных (там seed, список признаков, имена файлов...);
          * feature_names — имена признаков в правильном порядке;
          * fold_medians — медианы (фолды, признаки);
          * fold_means — средние (фолды, признаки);
          * fold_scales — масштабы (фолды, признаки);
          * models — список из 5 загруженных моделей CatBoost;
          * modes — словарь мод текстовых колонок.
    """
    # Путь к файлу метаданных внутри папки модели.
    metadata_path = model_dir / "model_metadata.json"
    # Если его нет — значит, train.py не запускали; падаем с подсказкой.
    if not metadata_path.exists():
        raise FileNotFoundError(f"Model metadata not found: {metadata_path}; run train.py first")
    # Открываем JSON и читаем в словарь Python.
    with metadata_path.open(encoding="utf-8") as file:
        metadata = json.load(file)

    # Путь к файлу препроцессоров (.npz — несколько массивов в одном файле).
    preprocessor_path = model_dir / "fold_preprocessors.npz"
    # Если его нет — тоже падаем с подсказкой.
    if not preprocessor_path.exists():
        raise FileNotFoundError(f"Fold preprocessors not found: {preprocessor_path}")
    # np.load читает .npz; allow_pickle=False — строгий безопасный режим.
    # with гарантирует закрытие файла; saved работает как словарь массивов.
    with np.load(preprocessor_path, allow_pickle=False) as saved:
        # Имена признаков: из массива строк в обычный список строк Python.
        feature_names = saved["feature_names"].astype(str).tolist()
        # Три массива препроцессоров (каждый размера (фолды, признаки)).
        fold_medians = saved["fold_medians"]
        fold_means = saved["fold_means"]
        fold_scales = saved["fold_scales"]
        # Моды: склеиваем два массива (имена + значения) в словарь через zip.
        modes = dict(zip(saved["categorical_columns"].astype(str), saved["categorical_modes"].astype(str)))

    # Проверка согласованности: список признаков в .npz и в .json должен совпадать.
    # Если нет — артефакты перепутаны (от разных прогонов), так предсказывать нельзя.
    if feature_names != metadata["features"]:
        raise ValueError("Model feature names do not match model_metadata.json")

    # Читаем из метаданных имена файлов весов (["catboost_fold_0.cbm", ...]).
    fold_model_files = metadata["fold_model_files"]
    # Пустой список под загруженные модели.
    models: list[CatBoostRegressor] = []
    # Загружаем модели по одной.
    for model_filename in fold_model_files:
        # Полный путь к файлу весов.
        model_path = model_dir / model_filename
        # Если файла нет — падаем с понятной ошибкой.
        if not model_path.exists():
            raise FileNotFoundError(f"Fold model file not found: {model_path}")
        # Создаём ПУСТУЮ модель CatBoost...
        cb = CatBoostRegressor()
        # ...и загружаем в неё веса из файла. Обучения тут нет — только чтение весов!
        cb.load_model(str(model_path))
        # Кладём готовую модель в список.
        models.append(cb)

    # Возвращаем всё загруженное одним кортежем.
    return metadata, feature_names, fold_medians, fold_means, fold_scales, models, modes


def main() -> None:
    """Главная функция: загрузить модель, предсказать тест, записать submission."""
    # Шаг 0. Разобрать аргументы командной строки.
    args = parse_args()
    # Путь к тестовому CSV.
    input_path = Path(args.input_csv)
    # Проверяем, что тестовый файл существует.
    if not input_path.exists():
        raise FileNotFoundError(f"Test CSV not found: {input_path}")

    # Шаг 1. Загрузить ансамбль (распаковываем кортеж из 7 элементов в 7 переменных).
    model_dir = Path(args.model_dir)
    (
        metadata,  # словарь метаданных
        feature_names,  # имена признаков
        fold_medians,  # медианы по фолдам
        fold_means,  # средние по фолдам
        fold_scales,  # масштабы по фолдам
        models,  # список готовых моделей
        modes,  # моды текстовых колонок
    ) = load_ensemble(model_dir)
    # Достаём seed из метаданных (он же пойдёт в имя submission-файла).
    seed = int(metadata["seed"])
    # Фиксируем случайность этим seed'ом (для полной детерминированности).
    set_global_seed(seed)

    # Шаг 2. Прочитать тестовый CSV в таблицу.
    data = pd.read_csv(input_path)
    # Проверяем, что есть колонка ID (без неё непонятно, чьи предсказания).
    if ID_COLUMN not in data.columns:
        raise ValueError(f"Test CSV must contain the identifier column {ID_COLUMN!r}")
    # Проверяем, что в ID нет пропусков.
    if data[ID_COLUMN].isna().any():
        raise ValueError(f"Identifier column {ID_COLUMN!r} contains missing values")

    # Шаг 3. Создать те же 9 новых признаков, что и при обучении.
    # Порядок важен: сначала новые признаки, потом проверка колонок.
    data = generate_features(data)

    # Шаг 4. Отчёт о том, что пришлось заполнить, — чтобы жюри это видело.
    # Добавляем отсутствующие колонки как NaN; absent — список того, чего не хватало.
    absent = ensure_columns(data, feature_names)
    # Если чего-то не хватало — печатаем ПРЕДУПРЕЖДЕНИЕ со списком.
    if absent:
        print(f"WARNING: {len(absent)} feature column(s) absent from input, filled with training medians: {absent}")
    # Берём только нужные числовые колонки...
    numeric_frame = data[feature_names]
    # ...превращаем в числа (мусор -> NaN), считаем NaN по каждой колонке (.sum() по True=1).
    nan_counts = numeric_frame.apply(pd.to_numeric, errors="coerce").isna().sum()
    # Оставляем только колонки, где пропусков больше нуля.
    nan_counts = nan_counts[nan_counts > 0]
    # Печатаем, что и сколько будет заполнено медианами (прозрачность!).
    if len(nan_counts):
        print("Numeric features filled with training medians (count of missing/invalid values):")
        for column, count in nan_counts.items():
            print(f"  {column}: {int(count)}")
    else:
        print("No missing numeric feature values in input.")

    # Шаг 5. Заполнить текстовые пропуски модами (только известные колонки из CATEGORICAL_COLUMNS).
    # {c: m for ...} — словарь, отфильтрованный по списку известных колонок.
    data = impute_categorical(data, {c: m for c, m in modes.items() if c in CATEGORICAL_COLUMNS})
    # Превратить признаки в матрицу numpy (порядок колонок — как при обучении!).
    raw = numeric_matrix(data, feature_names)

    # Шаг 6. Предсказать тест КАЖДОЙ моделью фолда.
    # Готовим матрицу (n_моделей, n_строк) под предсказания каждого фолда.
    fold_predictions = np.zeros((len(models), len(data)), dtype=float)
    # enumerate даёт (номер_фолда, модель).
    for fold_index, model in enumerate(models):
        # Применяем препроцессор ЭТОГО фолда (у каждого фолда свои медианы/средние/масштабы!).
        transformed = transform_features(raw, fold_medians[fold_index], fold_means[fold_index], fold_scales[fold_index])
        # Предсказываем и кладём в строку матрицы, соответствующую этому фолду.
        fold_predictions[fold_index] = model.predict(transformed)

    # Шаг 7. Усреднить предсказания 5 моделей по столбцам (axis=0 = «схлопнуть фолды»).
    # Это и есть ансамбль: среднее стабильнее любого одиночного предсказания.
    ensemble_prediction = np.mean(fold_predictions, axis=0)
    # Обрезать предсказания в допустимый диапазон [0, 100] (модель могла чуть выйти за края).
    prediction = np.clip(ensemble_prediction, 0.0, 100.0)
    # Страховка: если есть битые предсказания (NaN/inf) — упасть, а не писать мусор в файл.
    if not np.isfinite(prediction).all():
        raise ValueError("Model produced a non-finite prediction")

    # Шаг 8. Собрать submission-таблицу: ID (как строка) + предсказание.
    submission = pd.DataFrame({ID_COLUMN: data[ID_COLUMN].astype(str), TARGET_COLUMN: prediction})
    # Имя выходного файла: либо заданное пользователем, либо submission_seed_<seed>.csv.
    output_path = Path(args.output_csv) if args.output_csv else Path(f"submission_seed_{seed}.csv")
    # Пишем CSV: index=False — без лишней колонки индексов, float_format="%.4f" — 4 знака после точки.
    submission.to_csv(output_path, index=False, float_format="%.4f")

    # Шаг 9. Напечатать сводку: сколько фолдов усреднили, сколько строк, диапазон, куда сохранили.
    print(f"Folds averaged: {len(models)}")
    print(f"Rows written: {len(submission)}")
    print(f"Prediction range: {prediction.min():.4f} .. {prediction.max():.4f}")
    print(f"Saved submission: {output_path}")


# Стандартный приём: main() запускается, только если файл запущен напрямую.
if __name__ == "__main__":
    main()
