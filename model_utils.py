"""Общие помощники (утилиты) для всего проекта.

Этот файл импортируют и ``train.py`` (обучение), и ``predict.py`` (предсказание),
чтобы обработка признаков была ОДИНАКОВОЙ в обучении и в предсказании.
Это очень важно: если при обучении пропуски заполняли медианой, а при
предсказании — нулём, модель будет выдавать мусор.

Что здесь есть (краткая карта файла):
  * Константы ``ID_COLUMN`` / ``TARGET_COLUMN`` — имена колонки ID и цели.
  * ``set_global_seed`` — фиксация случайности (воспроизводимость).
  * ``generate_features`` — создание новых признаков (feature engineering).
  * ``get_numeric_feature_names`` — какие колонки считать числовыми признаками.
  * ``numeric_matrix`` — превратить DataFrame в матрицу чисел (numpy).
  * ``ensure_columns`` — добавить отсутствующие колонки, заполнив NaN.
  * ``fit_preprocessor`` — посчитать медианы/средние/масштабы на train.
  * ``transform_features`` — применить сохранённые медианы/средние/масштабы.
  * ``fit_categorical_modes`` / ``impute_categorical`` — моды категориальных колонок.
  * ``target_to_logit`` / ``logits_to_score`` — переводы между шкалой 0–100 и логитами.
  * ``fit_sigmoid_index`` — старая сигмоидально-линейная модель (L-BFGS-B).
    Финальной моделью она НЕ является (финал — CatBoost в ``train.py``),
    но нужна скриптам ``experiments.py`` и ``hypotheses.py`` для честного
    сравнения и проверки гипотез.

Словарик для новичка:
  * Признак (feature) — входной столбец, по которому предсказываем (возраст, доход...).
  * Цель (target) — то, что предсказываем (``protection_score`` от 0 до 100).
  * NaN — «пустое значение», пропуск в данных (Not a Number).
  * Медиана — среднее по порядку значение (устойчиво к выбросам).
  * Стандартизация — вычитаем среднее и делим на разброс, чтобы признаки
    были в одном масштабе: (x - mean) / std.
"""

# Эта строка включает «новые» правила аннотаций типов даже на старых Python.
# Она ни на что не влияет в работе кода, только помогает проверке типов.
from __future__ import annotations

# Импортируем стандартные модули Python:
import os  # работа с переменными окружения (нужно для PYTHONHASHSEED)
import random  # встроенный генератор случайных чисел Python

# Импортируем научные библиотеки:
import numpy as np  # numpy — массивы и математика (математика проекта)
import pandas as pd  # pandas — таблицы (DataFrame), чтение CSV
from scipy.optimize import minimize  # minimize — оптимизатор (подбор чисел, минимизирующих ошибку)
from scipy.special import expit  # expit — сигмоида: expit(x) = 1 / (1 + exp(-x)), переводит любое число в (0, 1)

# Имя колонки с идентификатором клиента. Это НЕ признак и НЕ цель,
# а просто «паспорт» строки, чтобы сопоставить предсказания с клиентами.
ID_COLUMN = "customer_id"
# Имя колонки-цели: то число (0–100), которое мы учимся предсказывать.
TARGET_COLUMN = "protection_score"

# Список категориальных (текстовых) колонок из описания задачи.
# В CSV они имеют тип object (строки), например region = "Moscow".
# Текущая финальная модель их НЕ использует (числовых признаков хватило),
# но мы всё равно умеем аккуратно заполнять в них пропуски — на будущее.
CATEGORICAL_COLUMNS = [
    "region",  # регион проживания
    "city_type",  # тип города (metro / large_city / town / rural)
    "gender",  # пол
    "education",  # образование
    "family_status",  # семейное положение
    "employment",  # занятость
    "wealth_segment",  # сегмент богатства (mass / affluent / private)
]


def set_global_seed(seed: int) -> None:
    """Зафиксировать ВСЕ генераторы случайных чисел одним числом seed.

    Зачем: многие алгоритмы используют случайность (разбиение на фолды,
    обучение CatBoost). Без фиксации каждый запуск давал бы чуть другой
    результат. С фиксацией — результат можно повторить побитно.

    Аргументы:
        seed: любое целое число, например 434089.
    """
    # PYTHONHASHSEED влияет на порядок обхода множеств/словарей в Python.
    # Фиксируем его, чтобы всё было детерминировано (повторяемо).
    os.environ["PYTHONHASHSEED"] = str(seed)
    # Фиксируем встроенный генератор случайных чисел Python.
    random.seed(seed)
    # Фиксируем генератор случайных чисел numpy.
    np.random.seed(seed)


def generate_features(data: pd.DataFrame) -> pd.DataFrame:
    """Создать новые признаки из старых (feature engineering).

    Идея: иногда модели легче учиться, если подсказать ей готовые комбинации,
    например «риск умножить на отсутствие страховки». Мы добавляем 9 признаков:

    1. ``property_exposure`` = crime_rate * (1 - property_insurance):
       «незащищённость жилья» — высокий криминал И нет страховки = плохо.
    2. ``cyber_exposure`` = cyber_risk * (1 - cyber_protection):
       то же для кибер-риска.
    3. ``cyber_vulnerability`` = digital_behavior_score * (3 - сумма 3 защит):
       активный в цифре человек БЕЗ защит (менеджер паролей, 2FA, обучение).
    4. ``income_to_balance_ratio`` = average_balance / (income + 1):
       доля среднемесячного остатка от дохода.
    5. ``credit_load_ratio`` = loan_amount / (income + 1):
       кредитная нагрузка относительно дохода.
    6. ``digital_channel_ratio`` = website_visits / (mobile_sessions + 1):
       каким каналом пользуется чаще: сайт или приложение.
    7. ``net_active_policies`` = active_policies - expired_policies:
       «чистые» активные полисы (активные минус истёкшие).
    8. ``age_squared`` = age ** 2: квадрат возраста (ловит нелинейность по возрасту).
    9. ``is_young`` = 1 если age < 25 иначе 0: флаг «молодой клиент».

    (+1 к доходам/сессиям добавляем, чтобы никогда не делить на ноль.)

    Аргументы:
        data: таблица с исходными колонками (train или test).

    Возвращает:
        НОВУЮ таблицу (копию) с добавленными колонками. Входная не меняется.
    """
    # Делаем копию таблицы, чтобы не менять входные данные «по месту».
    # (Функция без побочных эффектов — её безопасно вызывать где угодно.)
    df = data.copy()

    # Внутренняя мини-функция: достать колонку как числа.
    def _col(name: str) -> pd.Series | None:
        # Если такой колонки вообще нет в таблице — вернуть None (признак пропустим).
        if name not in df.columns:
            return None
        # pd.to_numeric превращает значения в числа; errors="coerce" означает:
        # всё, что не число (например текст "n/a"), станет NaN, а не упадёт с ошибкой.
        return pd.to_numeric(df[name], errors="coerce")

    # --- 1. Перекрёстные взаимодействия рисков и страховок ---
    crime_rate = _col("crime_rate")  # уровень преступности в регионе (число)
    property_insurance = _col("property_insurance")  # есть ли страховка жилья (0/1)
    # Создаём признак, только если ОБЕ исходные колонки на месте.
    if crime_rate is not None and property_insurance is not None:
        # (1 - страховка): если страховки нет (0), множитель = 1, риск остаётся;
        # если есть (1), множитель = 0, риск «гасится».
        df["property_exposure"] = crime_rate * (1 - property_insurance)

    cyber_risk = _col("cyber_risk")  # уровень кибер-риска клиента
    cyber_protection = _col("cyber_protection")  # есть ли кибер-защита (0/1)
    if cyber_risk is not None and cyber_protection is not None:
        df["cyber_exposure"] = cyber_risk * (1 - cyber_protection)

    # --- 2. Индекс цифровой уязвимости ---
    digital_behavior_score = _col("digital_behavior_score")  # насколько активен в цифре (0–100)
    password_manager = _col("password_manager")  # пользуется ли менеджером паролей (0/1)
    two_factor_auth = _col("two_factor_auth")  # включена ли двухфакторка (0/1)
    security_training = _col("security_training")  # проходил ли обучение безопасности (0/1)
    # Проверяем, что все 4 колонки существуют, и только тогда считаем.
    if (
        digital_behavior_score is not None
        and password_manager is not None
        and two_factor_auth is not None
        and security_training is not None
    ):
        # (password_manager + two_factor_auth + security_training) — число включённых защит (0–3).
        # (3 - защиты) — число ОТСУТСТВУЮЩИХ защит: чем больше, тем уязвимее.
        # Умножаем на цифровую активность: активный + беззащитный = максимальная уязвимость.
        df["cyber_vulnerability"] = digital_behavior_score * (
            3 - (password_manager + two_factor_auth + security_training)
        )

    # --- 3. Финансовое плечо и кредитная нагрузка ---
    average_balance = _col("average_balance")  # средний остаток на счетах
    income = _col("income")  # доход
    loan_amount = _col("loan_amount")  # сумма кредита
    if average_balance is not None and income is not None:
        # Остаток относительно дохода. +1 в знаменателе — защита от деления на 0.
        df["income_to_balance_ratio"] = average_balance / (income + 1)
    if loan_amount is not None and income is not None:
        # Кредит относительно дохода — классическая «кредитная нагрузка».
        df["credit_load_ratio"] = loan_amount / (income + 1)

    # --- 4. Цифровые каналы и полисы ---
    website_visits = _col("website_visits")  # визиты на сайт
    mobile_sessions = _col("mobile_sessions")  # сессии в мобильном приложении
    active_policies = _col("active_policies")  # число активных полисов
    expired_policies = _col("expired_policies")  # число истёкших полисов
    if website_visits is not None and mobile_sessions is not None:
        # Соотношение каналов: много сайта и мало приложения (или наоборот).
        df["digital_channel_ratio"] = website_visits / (mobile_sessions + 1)
    if active_policies is not None and expired_policies is not None:
        # Чистые активные полисы: активные минус истёкшие.
        df["net_active_policies"] = active_policies - expired_policies

    # --- 5. Возрастные нелинейности ---
    age = _col("age")  # возраст клиента
    if age is not None:
        # Квадрат возраста: позволяет модели учить «параболу» по возрасту
        # (например, защита сначала растёт, потом падает), а не только прямую.
        df["age_squared"] = age**2
        # Флаг молодости: (age < 25) даёт True/False, .astype(int) превращает в 1/0.
        df["is_young"] = (age < 25).astype(int)

    # Возвращаем новую таблицу со всеми добавленными колонками.
    return df


# Псевдонимы (другие имена) той же функции — для удобства.
# Все 4 имени вызывают один и тот же код выше; используйте любое.
engineer_features = generate_features
create_features = generate_features
add_features = generate_features
add_engineered_features = generate_features


def get_numeric_feature_names(data: pd.DataFrame) -> list[str]:
    """Вернуть имена ЧИСЛОВЫХ колонок-признаков (кроме ID и цели).

    Аргументы:
        data: таблица (обычно ПОСЛЕ generate_features, чтобы включить новые признаки).

    Возвращает:
        Список имён колонок, например ["age", "income", ...].
    """
    # Множество колонок, которые признаками НЕ являются и их надо исключить.
    excluded = {ID_COLUMN, TARGET_COLUMN}
    # Проходим по всем колонкам и берём те, что: не ID/цель И числового типа.
    # pd.api.types.is_numeric_dtype проверяет тип: int/float — да, строки — нет.
    return [
        column
        for column in data.columns
        if column not in excluded and pd.api.types.is_numeric_dtype(data[column])
    ]


def numeric_matrix(data: pd.DataFrame, feature_names: list[str]) -> np.ndarray:
    """Превратить именованные признаки в матрицу чисел (numpy массив).

    Модели (CatBoost, sklearn) не умеют работать с DataFrame напрямую так же удобно —
    им нужна обычная матрица чисел размера (n_строк, n_признаков).

    Значения, которые нельзя прочитать как числа (например текст "n/a"),
    и бесконечности (inf) превращаются в NaN — их потом заполнит препроцессор.

    Аргументы:
        data: таблица, где есть все нужные колонки.
        feature_names: какие колонки взять и в каком порядке.

    Возвращает:
        Двумерный numpy-массив float размера (строки, признаки).
    """
    # Берём нужные колонки в нужном порядке и каждую превращаем в числа.
    # errors="coerce": нечисловой текст -> NaN (без падения).
    frame = data[feature_names].apply(pd.to_numeric, errors="coerce")
    # Переводим таблицу в numpy-массив вещественных чисел (float).
    matrix = frame.to_numpy(dtype=float)
    # np.isfinite = True только для обычных чисел (не NaN, не inf, не -inf).
    # ~ — отрицание: выбираем НЕ-обычные значения и записываем туда NaN.
    matrix[~np.isfinite(matrix)] = np.nan
    return matrix


def ensure_columns(
    data: pd.DataFrame, columns: list[str], *, fill_value: float = np.nan
) -> list[str]:
    """Добавить в таблицу ОТСУТСТВУЮЩИЕ колонки, заполнив их NaN.

    Зачем: закрытый тест может приехать без какой-то колонки (например, удалили
    ``credit_score``). Вместо падения с ошибкой мы создаём колонку из NaN —
    и дальше все её значения заполнятся сохранёнными медианами с train.

    Аргументы:
        data: таблица (меняется ПО МЕСТУ — новые колонки появятся в ней).
        columns: какие колонки обязаны быть.
        fill_value: чем заполнить (по умолчанию NaN).

    Возвращает:
        Список имён колонок, которых не хватало (чтобы напечатать предупреждение).
    """
    # Собираем имена колонок, которых нет в таблице.
    absent = [column for column in columns if column not in data.columns]
    # Каждую отсутствующую колонку создаём и заполняем fill_value.
    for column in absent:
        data[column] = fill_value
    # Возвращаем список «что добавили», чтобы вызывающий код мог предупредить.
    return absent


def fit_preprocessor(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Выучить параметры предобработки на TRAIN-матрице: медианы, средние, масштабы.

    Это «обучение препроцессора» (fit = подогнать):
      * медианы — чем заполнять пропуски (по каждой колонке своя медиана);
      * средние (means) и разбросы (scales) — для стандартизации:
        standardized = (значение - среднее) / разброс.

    ВАЖНО: считать эти числа можно ТОЛЬКО на обучающих данных.
    На тесте мы их лишь ПРИМЕНЯЕМ (см. transform_features).

    Аргументы:
        matrix: train-матрица (строки, признаки), может содержать NaN.

    Возвращает:
        Кортеж (medians, means, scales) — три массива длиной n_признаков.
    """
    # Приводим вход к numpy-массиву float (на случай, если передали список).
    values = np.asarray(matrix, dtype=float)
    # Проверяем, что матрица непустая и двумерная; иначе дальше всё сломается.
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError("Expected a non-empty two-dimensional feature matrix")

    # Готовим массив под медианы: нули длиной = числу признаков (столбцов).
    medians = np.zeros(values.shape[1], dtype=float)
    # Идём по каждому столбцу отдельно (у каждого своя медиана).
    for column_index in range(values.shape[1]):
        # Берём один столбец целиком (все строки, column_index-й столбец).
        column = values[:, column_index]
        # Оставляем только конечные значения (выкидываем NaN и бесконечности).
        finite = column[np.isfinite(column)]
        # Если конечных значений нет вообще (столбец весь пустой) — медиана 0,
        # иначе обычная медиана numpy. if/else в одну строку.
        medians[column_index] = np.median(finite) if finite.size else 0.0

    # Временно заполняем пропуски медианами, чтобы посчитать среднее и разброс.
    # np.where(условие, если_да, если_нет): где конечное — оставить, где нет — взять медиану.
    # medians здесь «размножается» по строкам автоматически (broadcasting).
    imputed = np.where(np.isfinite(values), values, medians)
    # Среднее по каждому столбцу (axis=0 означает «схлопнуть строки»).
    means = imputed.mean(axis=0)
    # Стандартное отклонение по каждому столбцу — мера разброса.
    scales = imputed.std(axis=0)
    # Защита: если разброс нулевой/битый (константный столбец), ставим 1.0,
    # чтобы при делении не получить inf/nan. 1e-12 — крошечный порог «почти ноль».
    # | означает «или»: (не конечное) ИЛИ (меньше порога).
    scales[~np.isfinite(scales) | (scales < 1e-12)] = 1.0
    # Возвращаем все три массива разом.
    return medians, means, scales


def transform_features(
    matrix: np.ndarray,
    medians: np.ndarray,
    means: np.ndarray,
    scales: np.ndarray,
) -> np.ndarray:
    """Применить СОХРАНЁННЫЕ параметры: заполнить пропуски и стандартизировать.

    Это «применение препроцессора» (transform = преобразовать).
    Используется и на train-фолдах, и на тесте — всегда с параметрами,
    выученными на train (иначе будет утечка информации из теста!).

    Аргументы:
        matrix: матрица для преобразования (может содержать NaN).
        medians: сохранённые медианы (чем заполнять пропуски).
        means: сохранённые средние (что вычитать).
        scales: сохранённые масштабы (на что делить).

    Возвращает:
        Новую матрицу без NaN, стандартизированную.
    """
    # Приводим вход к numpy-массиву float.
    values = np.asarray(matrix, dtype=float)
    # Заполняем пропуски сохранёнными медианами (по каждому столбцу своей).
    imputed = np.where(np.isfinite(values), values, medians)
    # Стандартизация: (значение - среднее) / масштаб — по каждому столбцу.
    transformed = (imputed - means) / scales
    # Страховка: если после всех операций остались битые значения — упасть с ошибкой,
    # а не молча кормить модель мусором. .all() = «все ли True».
    if not np.isfinite(transformed).all():
        raise ValueError("Feature preprocessing produced non-finite values")
    return transformed


def fit_categorical_modes(data: pd.DataFrame, columns: list[str]) -> dict[str, str]:
    """Выучить моды категориальных колонок (самые частые значения).

    Мода — самое частое значение. Ею мы заполняем пропуски в текстовых колонках.
    При равенстве частот побеждает первое по алфавиту (чтобы было детерминировано).

    Аргументы:
        data: train-таблица.
        columns: какие текстовые колонки обработать.

    Возвращает:
        Словарь {имя_колонки: самое_частое_значение}, например {"region": "Moscow"}.
    """
    # Пустой словарь под результат.
    modes: dict[str, str] = {}
    # Идём по каждой запрошенной колонке.
    for column in columns:
        # Если колонки нет в таблице — пропускаем (continue = «следующая итерация»).
        if column not in data.columns:
            continue
        # Считаем частоты: убрать NaN (dropna), всё в строки (astype(str)),
        # подсчитать каждое значение (value_counts) — уже отсортировано по убыванию.
        counts = data[column].dropna().astype(str).value_counts()
        # Если после удаления NaN ничего не осталось — пропускаем колонку.
        if counts.empty:
            continue
        # Берём все значения с максимальной частотой (их может быть несколько при равенстве),
        # сортируем по алфавиту (sort_values) и берём первое — детерминированный выбор.
        top = counts[counts == counts.max()].index.sort_values()
        # Сохраняем моду в словарь (str(...) на всякий случай приводит к строке).
        modes[column] = str(top[0])
    return modes


def impute_categorical(data: pd.DataFrame, modes: dict[str, str]) -> pd.DataFrame:
    """Заполнить пропуски в текстовых колонках сохранёнными модами.

    Аргументы:
        data: таблица ( НЕ меняется — работаем с копией).
        modes: словарь {колонка: мода} с train (см. fit_categorical_modes).

    Возвращает:
        НОВУЮ таблицу с заполненными текстовыми пропусками.
    """
    # Копируем таблицу, чтобы не менять оригинал.
    filled = data.copy()
    # Идём по парам (колонка, мода) из словаря.
    for column, mode in modes.items():
        # Если колонки нет вообще — создаём её целиком из моды.
        if column not in filled.columns:
            filled[column] = mode
        else:
            # .where(условие, замена): где условие True — оставить как есть,
            # где False (т.е. NaN, т.к. notna()=False) — поставить моду.
            filled[column] = filled[column].where(filled[column].notna(), mode)
    return filled


def target_to_logit(target: np.ndarray) -> np.ndarray:
    """Перевести цель 0–100 в логиты (только для СТАРТА оптимизации).

    Логит(p) = ln(p / (1 - p)) — «растягивает» шкалу 0–100 в (-inf, +inf).
    Используется лишь как хорошая начальная точка для fit_sigmoid_index:
    сначала решаем простую линейную задачу на логитах, потом уточняем.

    Аргументы:
        target: массив целей в диапазоне [0, 100].

    Возвращает:
        Массив логитов.
    """
    # Приводим к float-массиву.
    values = np.asarray(target, dtype=float)
    # Делим на 100 -> доля (0..1) и «подрезаем» края: p в [1e-4, 1-1e-4].
    # Это нужно, т.к. logit(0) = -inf и logit(1) = +inf (логарифм нуля!).
    probability = np.clip(values / 100.0, 1e-4, 1.0 - 1e-4)
    # Сама формула логита: ln(p / (1 - p)).
    return np.log(probability / (1.0 - probability))


def logits_to_score(logits: np.ndarray) -> np.ndarray:
    """Перевести логиты модели обратно в проценты 0–100.

    Это обратная операция к target_to_logit: score = 100 * sigmoid(logit).
    Сигмоида expit «сжимает» любое число в (0, 1), умножение на 100 — в (0, 100).

    Аргументы:
        logits: массив выходов линейной модели (любые числа).

    Возвращает:
        Массив предсказаний в диапазоне (0, 100).
    """
    # expit = сигмоида; asarray на случай, если передали список.
    return 100.0 * expit(np.asarray(logits, dtype=float))


def fit_sigmoid_index(
    matrix: np.ndarray,
    target: np.ndarray,
    l2_alpha: float = 10_000.0,
) -> tuple[np.ndarray, float, object]:
    """Подобрать старую сигмоидально-линейную модель (НЕ финальная!).

    Модель: prediction = 100 * sigmoid(X @ w + b), где w — веса признаков,
    b — сдвиг (intercept). Подбираем w и b так, чтобы минимизировать:

        MSE = mean((prediction - target)^2) + (l2_alpha / n) * sum(w^2)

    Первое слагаемое — среднеквадратичная ошибка на шкале 0–100 (чем меньше,
    тем точнее). Второе — L2-штраф за большие веса (регуляризация: не даёт
    модели переобучиться, «прижимает» веса к нулю). Деление на n (число строк)
    делает силу штрафа независимой от размера выборки.

    Оптимизация — метод L-BFGS-B из scipy (умный итеративный поиск минимума),
    старт — из решения линейной задачи на логитах (см. target_to_logit).

    Зачем эта функция живёт в проекте: её используют ``experiments.py``
    (честное сравнение «старая линейная модель vs CatBoost») и
    ``hypotheses.py`` (проверки гипотез H5/H10/H12). Финальную модель
    обучает ``train.py`` (ансамбль CatBoost), эта функция там НЕ используется.

    Аргументы:
        matrix: матрица признаков (обычно уже стандартизированная).
        target: цели 0–100.
        l2_alpha: сила L2-регуляризации (по умолчанию 10000, оптимум ~1e4
            по ``reports/extra_hypotheses/RESULTS.md``).

    Возвращает:
        Кортеж (coefficients, intercept, result):
          * coefficients — веса w (по одному на признак);
          * intercept — сдвиг b (одно число);
          * result — служебный объект scipy с деталями оптимизации.
    """
    # Приводим входы к float-массивам numpy.
    design = np.asarray(matrix, dtype=float)
    values = np.asarray(target, dtype=float)
    # Запоминаем размеры: n_rows — число строк, n_features — число признаков.
    n_rows, n_features = design.shape
    # Сила штрафа на один вес: делим на число строк (как в отчётах: L2 = 1e4 / n).
    penalty = l2_alpha / n_rows

    # Стартовые веса: решаем ОБЫЧНУЮ линейную регрессию на логитах цели.
    # np.column_stack склеивает матрицу X и столбец из единиц (для сдвига b):
    # последняя колонка даст нам стартовое b, остальные — стартовые w.
    start_matrix = np.column_stack([design, np.ones(n_rows)])
    # np.linalg.lstsq решает задачу наименьших квадратов: найти w, минимизирующие
    # ||A @ w - z||^2, где z — логиты цели. Берём только первое из 4 возвращаемых
    # значений (само решение), остальное игнорируем через *_.
    start_weights, *_ = np.linalg.lstsq(start_matrix, target_to_logit(values), rcond=None)

    # Внутренняя функция: по вектору весов w считает (ошибка, градиент).
    # Оптимизатор будет вызывать её много раз, каждый раз с новыми w.
    def loss_and_gradient(weights: np.ndarray) -> tuple[float, np.ndarray]:
        # Линейный индекс z = X @ w + b (последний элемент weights — это b,
        # т.к. последний столбец start_matrix — единицы).
        linear_index = start_matrix @ weights
        # Сигмоида индекса: доля в (0, 1).
        probability = expit(linear_index)
        # Предсказание на шкале 0–100.
        prediction = 100.0 * probability
        # Ошибки по каждой строке: насколько предсказание отличается от правды.
        error = prediction - values
        # MSE (средний квадрат ошибки) + L2-штраф (только на веса, БЕЗ сдвига:
        # сдвиг штрафовать не принято, weights[:-1] = всё кроме последнего).
        loss = float(np.mean(error**2)) + penalty * float(np.dot(weights[:-1], weights[:-1]))
        # Производная предсказания по индексу: d(100*sigmoid)/dz = 100*s*(1-s).
        d_prediction = 100.0 * probability * (1.0 - probability)
        # Градиент MSE по весам (цепное правило): 2/n * A^T @ (ошибка * производная).
        gradient = (2.0 / n_rows) * (start_matrix.T @ (error * d_prediction))
        # Добавляем градиент штрафа: d(penalty*sum(w^2))/dw = 2*penalty*w (сдвиг не трогаем).
        gradient[:-1] += 2.0 * penalty * weights[:-1]
        # Возвращаем ошибку и градиент (jac=True ниже обещает оптимизатору градиент).
        return loss, gradient

    # Запускаем оптимизатор L-BFGS-B: итеративно улучшает веса, пока ошибка падает.
    # jac=True означает «градиент нам даёт сама функция» (быстрее и точнее).
    result = minimize(loss_and_gradient, start_weights, jac=True, method="L-BFGS-B")
    # Разбираем найденный вектор: всё кроме последнего — веса, последнее — сдвиг.
    coefficients = result.x[:-1]
    intercept = float(result.x[-1])
    # Возвращаем веса, сдвиг и служебный объект оптимизации.
    return coefficients, intercept, result
