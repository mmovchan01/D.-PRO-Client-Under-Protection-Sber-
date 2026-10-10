"""Разведочный анализ данных (EDA) для задачи protection_score.

EDA (Exploratory Data Analysis) — это «осмотр данных глазами» ДО обучения:
смотрим распределения, пропуски, связи признаков с целью, проверяем
согласованность данных и структуру шума в цели.

Скрипт читает ТОЛЬКО ``hard_train.csv`` (с метками) и ``hard_test.csv`` (без меток,
для сравнения пропусков/распределений) и пишет в ``reports/eda``:
  * ``eda_summary.md`` — текстовый отчёт с таблицами;
  * ``target_distribution.png`` — распределение цели (обычное и на логит-шкале);
  * ``top_correlations.png`` — топ-20 корреляций признаков с целью;
  * ``target_by_groups.png`` — цель в разрезе групп (boxplot-ы);
  * ``noise_logit_vs_raw.png`` — остатки модели на логит-шкале и на шкале 0–100.

Словарик:
  * Корреляция Пирсона — число от -1 до 1: насколько линейно связаны две величины.
  * Логит — ln(p/(1-p)); «растягивает» шкалу 0–100, шум на ней проще изучать.
  * Ridge-регрессия — линейная регрессия с L2-штрафом (устойчива к шуму).
  * R^2 — доля объяснённой дисперсии (1.0 = идеал, 0 = не лучше среднего).
  * OOF — предсказания «честным» способом: каждая строка предсказана моделью,
    которая её не видела (см. train.py).

Пример запуска:
    python eda.py --train-csv hard_train.csv --test-csv hard_test.csv --output-dir reports/eda
"""

# Включаем «новые» правила аннотаций типов.
from __future__ import annotations

# Стандартные модули:
import argparse  # аргументы командной строки
from pathlib import Path  # пути к файлам

# matplotlib — библиотека графиков. Agg — режим «без окна»: только сохранение в файл.
# Важно вызывать matplotlib.use("Agg") ДО импорта pyplot, иначе не сработает.
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  # pyplot — построение графиков
import numpy as np  # noqa: E402  # массивы и математика
import pandas as pd  # noqa: E402  # таблицы
from scipy import stats  # noqa: E402  # статистики: асимметрия (skew), эксцесс (kurtosis)
from scipy.special import expit, logit  # noqa: E402  # сигмоида и логит (переходы между шкалами)
from sklearn.linear_model import RidgeCV  # noqa: E402  # ridge-регрессия с автовыбором силы штрафа
from sklearn.model_selection import KFold  # noqa: E402  # разбиение на фолды

# Наши константы: имена колонок ID и цели.
from model_utils import ID_COLUMN, TARGET_COLUMN  # noqa: E402

# (noqa: E402 означает «не ругаться, что импорты после кода» — тут это нужно
#  из-за matplotlib.use("Agg"), который обязан идти первым.)

# Список текстовых колонок — по ним посмотрим средние цели по группам.
CATEGORICAL = ["gender", "region", "city_type", "education", "family_status", "employment", "wealth_segment"]
# 8 страховых флагов 0/1 — их сумма должна равняться insurance_products (проверим!).
INSURANCE_FLAGS = [
    "life_insurance",
    "property_insurance",
    "health_insurance",
    "travel_insurance",
    "car_insurance",
    "gadget_insurance",
    "cyber_protection",
    "identity_protection",
]


def parse_args() -> argparse.Namespace:
    """Разобрать аргументы командной строки.

    Возвращает:
        Объект с полями args.train_csv, args.test_csv, args.output_dir, args.seed.
    """
    # Создаём парсер с описанием из шапки файла.
    parser = argparse.ArgumentParser(description=__doc__)
    # Путь к train (с метками).
    parser.add_argument("--train-csv", default="hard_train.csv")
    # Путь к открытому тесту (без меток) — для сравнения пропусков.
    parser.add_argument("--test-csv", default="hard_test.csv")
    # Куда писать отчёт и картинки.
    parser.add_argument("--output-dir", default="reports/eda")
    # Seed для фолдов в диагностике шума (чтобы результат повторялся).
    parser.add_argument("--seed", type=int, default=0, help="Seed for the CV folds used in the noise diagnostics")
    return parser.parse_args()


def missing_table(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """Сравнить доли пропусков в train и test по каждой колонке.

    Аргументы:
        train: train-таблица (без цели — её уберёт вызывающий код).
        test: test-таблица.

    Возвращает:
        Табличка только с теми колонками, где пропуски вообще есть.
    """
    # Строим таблицу из двух колонок: доля NaN в train и доля NaN в test.
    # .isna() — True где пропуск; .mean() — доля True (True считается за 1).
    # test.reindex(columns=train.columns) — выравниваем колонки теста под train
    # (у теста нет цели; лишние колонки отбросятся, недостающие станут NaN).
    table = pd.DataFrame(
        {
            "train_missing_share": train.isna().mean(),
            "test_missing_share": test.reindex(columns=train.columns).isna().mean(),
        }
    )
    # Оставляем строки, где ХОТЯ БЫ в одной колонке значение > 0
    # ((table > 0).any(axis=1) — «есть ли True в строке»), и округляем до 4 знаков.
    return table[(table > 0).any(axis=1)].round(4)


def consistency_checks(train: pd.DataFrame) -> list[str]:
    """Проверить внутреннюю согласованность данных (совпадают ли связанные колонки).

    Аргументы:
        train: train-таблица.

    Возвращает:
        Список строк Markdown для отчёта (каждая — один факт с процентом).
    """
    # Пустой список под строки отчёта.
    lines = []
    # Сумма 8 страховых флагов по каждой строке (axis=1 = «сложить по строке»).
    flags_sum = train[INSURANCE_FLAGS].sum(axis=1)
    # Сравниваем с insurance_products: (равенство).mean() = доля совпавших строк.
    # :.2% форматирует долю как проценты с 2 знаками (0.99 -> "99.00%").
    lines.append(f"- `insurance_products` equals the sum of the 8 insurance flags: {(flags_sum == train['insurance_products']).mean():.2%} of rows")
    # Проверка claim_frequency = number_of_claims / years_with_company:
    years = train["years_with_company"]  # стаж в компании
    # Считаем ожидаемую частоту: где стаж > 0 — делим, иначе 0.
    # .clip(lower=1) — годы меньше 1 заменить на 1 (защита от деления на 0).
    freq = np.where(years > 0, train["number_of_claims"] / years.clip(lower=1), 0.0)
    # np.isclose(a, b, atol=6e-5) — «почти равны» с допуском (данные округлены до 4 знаков).
    lines.append(
        "- `claim_frequency` equals `number_of_claims / years_with_company` (4-decimal rounding): "
        f"{np.isclose(freq, train['claim_frequency'], atol=6e-5).mean():.2%} of rows"
    )
    # Проверка: average_claim_cost отсутствует РОВНО когда number_of_claims == 0
    # (нет обращений — нечего усреднять). Сравниваем два True/False массива.
    lines.append(
        "- `average_claim_cost` is missing exactly when `number_of_claims == 0`: "
        f"{((train['average_claim_cost'].isna()) == (train['number_of_claims'] == 0)).mean():.2%} of rows"
    )
    # Фактические доли пропусков house_value/car_value + сравнение с описанием задачи.
    lines.append(
        "- `house_value` is missing for "
        f"{train['house_value'].isna().mean():.1%} of rows (dataset card: ~62%); `car_value` for "
        f"{train['car_value'].isna().mean():.1%} (~58%)"
    )
    # Диапазоны дохода по сегментам богатства: groupby делит строки по wealth_segment,
    # ["income"].agg(["min", "max"]) считает мин/макс дохода в каждой группе.
    segments = train.groupby("wealth_segment")["income"].agg(["min", "max"]).round(0)
    # .to_markdown() превращает таблицу в Markdown-текст для отчёта.
    lines.append("- `wealth_segment` vs `income` ranges:\n\n" + segments.to_markdown())
    # Число полных дублей строк признаков (без ID и цели): .duplicated() помечает
    # повторы, .sum() считает их.
    lines.append(f"- Duplicated feature rows in train: {train.drop(columns=[ID_COLUMN, TARGET_COLUMN]).duplicated().sum()}")
    return lines


def logit_noise_diagnostics(train: pd.DataFrame, seed: int, output_dir: Path) -> list[str]:
    """Проверить, похожа ли цель на 100*sigmoid(линейный_индекс + гауссов шум).

    Идея: переводим цель в логиты z = logit(y/100), обучаем ridge-регрессию
    предсказывать z по признакам (честно, через 5-fold OOF) и изучаем остатки:
    их разброс, форму распределения и однородность по децилям предсказания.
    Если остатки — однородный гауссов шум, то оставшаяся ошибка почти
    неустранима («потолок» качества) — и мы считаем этот «пол» RMSE.

    Аргументы:
        train: train-таблица.
        seed: seed для разбиения на фолды.
        output_dir: куда сохранить график остатков.

    Возвращает:
        Список строк Markdown для отчёта.
    """
    # Цель в numpy-массив float.
    y = train[TARGET_COLUMN].to_numpy(dtype=float)
    # Логиты цели: делим на 100 (-> 0..1), подрезаем края (защита от inf),
    # применяем logit. 1 - 1e-4 = 0.9999.
    z = logit(np.clip(y / 100.0, 1e-4, 1 - 1e-4))
    # Матрица признаков: убираем ID и цель, текст превращаем в 0/1 колонки
    # (pd.get_dummies = one-hot; drop_first=True = первую категорию выкинуть,
    #  чтобы не было линейной зависимости), всё в float.
    features = pd.get_dummies(train.drop(columns=[ID_COLUMN, TARGET_COLUMN]), drop_first=True).astype(float)
    # Заполняем пропуски медианами колонок (медиана считается по всей колонке).
    features = features.fillna(features.median())

    # OOF-предсказания логитов: нули, заполним по фолдам.
    oof_logit = np.zeros(len(y))
    # 5 фолдов: обучаем ridge на 4/5, предсказываем 1/5.
    for train_idx, valid_idx in KFold(5, shuffle=True, random_state=seed).split(features):
        # RidgeCV сам выбирает лучший alpha из сетки (логарифмическая сетка 1e-3..1e3).
        # .fit(X_train, z_train) — обучить; .predict(X_valid) — предсказать.
        # .iloc[...] — доступ к строкам по позициям (numpy-индексы фолдов).
        model = RidgeCV(alphas=np.logspace(-3, 3, 13)).fit(features.iloc[train_idx], z[train_idx])
        oof_logit[valid_idx] = model.predict(features.iloc[valid_idx])
    # Остатки на логит-шкале: правда минус предсказание.
    residual = z - oof_logit
    # R^2 на логитах: 1 - дисперсия_остатков / дисперсия_цели.
    r2_logit = 1.0 - residual.var() / z.var()
    # RMSE той же модели, но на шкале 0–100: переводим OOF-логиты в проценты и сравниваем.
    rmse_raw = float(np.sqrt(np.mean((100.0 * expit(oof_logit) - y) ** 2)))

    # Делим предсказания на 10 равных групп (децили): pd.qcut режет по квантилям,
    # labels=False — вернуть номера групп 0..9 вместо интервалов.
    bins = pd.qcut(oof_logit, 10, labels=False)
    # Остатки на шкале 0–100.
    raw_residual = y - 100.0 * expit(oof_logit)
    # Собираем табличку и считаем СТАНДАРТНОЕ ОТКЛОНЕНИЕ остатков в каждом дециле:
    # если на логитах оно постоянно, а на 0–100 — нет, шум живёт на логитах.
    by_bin = pd.DataFrame(
        {
            "decile": bins,
            "logit_resid": residual,
            "raw_resid": raw_residual,
        }
    ).groupby("decile").agg(logit_resid_std=("logit_resid", "std"), raw_resid_std=("raw_resid", "std"))

    # Ожидаемый RMSE, если шум — это N(0, sigma) на логитах вокруг найденного индекса.
    # N(0, sigma) = нормальное (гауссово) распределение со средним 0 и разбросом sigma.
    rng = np.random.default_rng(seed)  # новый генератор случайных чисел numpy с нашим seed
    sigma = float(residual.std())  # наблюдаемый разброс остатков на логитах
    reference = 100.0 * expit(oof_logit)  # «чистые» предсказания без шума (точка отсчёта)

    # Внутренняя функция: оценить RMSE, который давал бы шум с разбросом noise_sd.
    def expected_rmse(noise_sd: float, draws: int = 20) -> float:
        # Список RMSE по повторениям (усредним для стабильности).
        values = []
        # Повторяем draws раз: добавляем случайный шум и меряем отклонение от reference.
        for _ in range(draws):
            # Шумные предсказания: к логитам добавляем N(0, noise_sd), переводим в 0–100.
            noisy = 100.0 * expit(oof_logit + rng.normal(0.0, noise_sd, len(y)))
            # RMSE шумных предсказаний относительно «чистых».
            values.append(np.sqrt(np.mean((noisy - reference) ** 2)))
        # Среднее по повторениям.
        return float(np.mean(values))

    # «Пол» RMSE при наблюдаемом sigma — ниже не прыгнуть, даже зная идеальный индекс.
    floor = expected_rmse(sigma)
    # Сетка уровней шума для таблицы: 0.2..0.5 + наблюдаемый sigma.
    sigma_grid = [0.2, 0.3, 0.4, 0.5, sigma]
    # Таблица «уровень шума -> ожидаемый RMSE», округлённая до 3 знаков.
    floor_table = pd.DataFrame(
        {"logit_noise_sd": sigma_grid, "expected_rmse": [expected_rmse(v) for v in sigma_grid]}
    ).round(3)

    # Рисуем 2 графика рядом: остатки на логитах и остатки на шкале 0–100.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))  # 1 строка, 2 столбца, размер в дюймах
    # Левый: рассеяние (каждая точка — клиент): остатки должны быть «облаком» вокруг нуля.
    axes[0].scatter(oof_logit, residual, s=3, alpha=0.4)  # s=размер точки, alpha=прозрачность
    axes[0].axhline(0, color="k", lw=0.8)  # горизонтальная линия нуля (k=чёрный, lw=толщина)
    axes[0].set_title("Logit scale: residual vs fitted index")
    axes[0].set_xlabel("out-of-fold linear index")
    axes[0].set_ylabel("logit(target) residual")
    # Правый: то же на шкале 0–100 (цвет tab:orange — оранжевый).
    axes[1].scatter(100.0 * expit(oof_logit), raw_residual, s=3, alpha=0.4, color="tab:orange")
    axes[1].axhline(0, color="k", lw=0.8)
    axes[1].set_title("Raw 0-100 scale: residual vs prediction")
    axes[1].set_xlabel("prediction")
    axes[1].set_ylabel("target residual")
    fig.tight_layout()  # аккуратно разместить подписи, чтобы не налезали
    fig.savefig(output_dir / "noise_logit_vs_raw.png", dpi=120)  # сохранить в файл (dpi=чёткость)
    plt.close(fig)  # закрыть фигуру и освободить память

    # Собираем строки отчёта: статистики остатков + таблицы.
    # stats.skew — асимметрия (0 = симметрично), stats.kurtosis — эксцесс
    # (0 = «хвосты» как у нормального распределения).
    lines = [
        f"- Logit target `z = logit(protection_score/100)`: std {z.std():.3f}, range [{z.min():.2f}, {z.max():.2f}]",
        f"- Out-of-fold ridge R^2 on the logit scale: {r2_logit:.3f}",
        f"- Logit residual: std {residual.std():.3f}, skew {stats.skew(residual):.3f}, excess kurtosis {stats.kurtosis(residual):.3f}",
        "- Residual std by decile of the fitted index (logit scale is roughly constant, raw scale is not):",
        "",
        by_bin.round(3).to_markdown(),
        "",
        f"- Out-of-fold RMSE of the linear logit model on the 0-100 scale: {rmse_raw:.3f}",
        f"- Expected RMSE from the logit-scale noise alone (sigma={sigma:.3f}): {floor:.2f}",
        "- Expected RMSE for other logit-noise levels (a model that recovers the true index would still be at this level):",
        "",
        floor_table.to_markdown(index=False),
        "",
        "Reaching RMSE < 7 on this data would require the logit-scale noise to be well below the observed ~0.6,",
        "i.e. the remaining signal must be explained by information that is not in the given features.",
    ]
    return lines


def make_plots(train: pd.DataFrame, output_dir: Path) -> None:
    """Нарисовать и сохранить 3 картинки EDA: распределение цели, корреляции, группы.

    Аргументы:
        train: train-таблица.
        output_dir: куда сохранять PNG-файлы.
    """
    # Цель как Series (один столбец).
    y = train[TARGET_COLUMN]

    # Картинка 1: два распределения рядом — обычное и на логит-шкале.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    # Гистограмма цели: 60 столбиков (bins), синий цвет.
    axes[0].hist(y, bins=60, color="tab:blue")
    axes[0].set_title("protection_score distribution")
    axes[0].set_xlabel("protection_score")
    # Гистограмма логитов цели (зелёная).
    axes[1].hist(logit(np.clip(y / 100.0, 1e-4, 1 - 1e-4)), bins=60, color="tab:green")
    axes[1].set_title("logit(protection_score / 100) distribution")
    axes[1].set_xlabel("logit")
    fig.tight_layout()
    fig.savefig(output_dir / "target_distribution.png", dpi=120)
    plt.close(fig)

    # Готовим данные для картинки 2: только числовые колонки без цели.
    # select_dtypes("number") — отобрать колонки числовых типов.
    numeric = train.select_dtypes("number").drop(columns=[TARGET_COLUMN])
    # Корреляция каждого признака с целью; сортируем по МОДУЛЮ (|r|), берём топ-20.
    # key=np.abs — сортировать по абсолютному значению; ascending=False — по убыванию.
    correlations = numeric.corrwith(y).sort_values(key=np.abs, ascending=False).head(20)
    # Горизонтальные столбики: [::-1] переворачивает (лучшие сверху), красным — отрицательные.
    fig, ax = plt.subplots(figsize=(8, 6))
    correlations[::-1].plot.barh(ax=ax, color=["tab:red" if v < 0 else "tab:blue" for v in correlations[::-1]])
    ax.set_title("Top-20 Pearson correlations with protection_score")
    fig.tight_layout()
    fig.savefig(output_dir / "top_correlations.png", dpi=120)
    plt.close(fig)

    # Картинка 3: boxplot-ы цели по группам (ящик = квартили, линия = медиана).
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    # Распределение цели для каждого значения insurance_products (0..8).
    train.boxplot(column=TARGET_COLUMN, by="insurance_products", ax=axes[0])
    axes[0].set_title("protection_score by insurance_products")
    axes[0].set_xlabel("insurance_products")
    # То же по сегментам богатства.
    train.boxplot(column=TARGET_COLUMN, by="wealth_segment", ax=axes[1])
    axes[1].set_title("protection_score by wealth_segment")
    fig.suptitle("")  # убрать автоматический общий заголовок pandas (он лишний)
    fig.tight_layout()
    fig.savefig(output_dir / "target_by_groups.png", dpi=120)
    plt.close(fig)


def main() -> None:
    """Главная функция: прочитать данные, собрать отчёт, нарисовать графики."""
    # Разобрать аргументы командной строки.
    args = parse_args()
    # Папка для отчёта; создать, если её нет.
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Прочитать train и test.
    train = pd.read_csv(args.train_csv)
    test = pd.read_csv(args.test_csv)
    # Цель отдельно (пригодится много раз).
    y = train[TARGET_COLUMN]

    # Начинаем собирать Markdown-отчёт построчно: заголовок + пустая строка.
    lines: list[str] = ["# EDA: protection_score", ""]
    # Добавляем разделы: размеры, описание цели, пропуски, заголовок категориальных.
    # train.shape = (строки, столбцы); y.describe() — статистики (среднее, квартили...).
    lines += [
        "## Shapes",
        f"- train: {train.shape[0]} rows x {train.shape[1]} columns",
        f"- open test: {test.shape[0]} rows x {test.shape[1]} columns",
        "",
        "## Target",
        y.describe().round(3).to_markdown(),
        "",
        "## Missing values (share of rows)",
        # Сравниваем пропуски train (без цели) и test.
        missing_table(train.drop(columns=[TARGET_COLUMN]), test).to_markdown(),
        "",
        "## Categorical features: target mean by level",
    ]
    # Для каждой текстовой колонки: средняя цель/разброс/число строк по каждому уровню.
    for column in CATEGORICAL:
        # groupby(column)[цель].agg([...]) — сгруппировать и посчитать 3 статистики.
        group = train.groupby(column)[TARGET_COLUMN].agg(["mean", "std", "count"]).round(2)
        # Добавляем подзаголовок + таблицу + пустую строку.
        lines += [f"### {column}", group.to_markdown(), ""]
    # Раздел проверок согласованности (функция вернёт готовые строки).
    lines += ["## Consistency checks", *consistency_checks(train), ""]
    # Раздел топ-корреляций: считаем Пирсона каждого числового признака с целью.
    lines += ["## Top correlations with the target (Pearson)"]
    numeric = train.select_dtypes("number").drop(columns=[TARGET_COLUMN])
    # corrwith — корреляция каждого столбца с целью; сортируем по |r|; переименовываем в pearson_r.
    correlations = numeric.corrwith(y).sort_values(key=np.abs, ascending=False).round(3).rename("pearson_r")
    # Топ-20 в таблицу (.to_frame() превращает Series в DataFrame с одной колонкой).
    lines += [correlations.head(20).to_frame().to_markdown(), ""]
    # Раздел про шум цели (функция заодно сохранит график остатков).
    lines += ["## Target noise structure"]
    lines += logit_noise_diagnostics(train, args.seed, output_dir)
    # Раздел про выбросы: правило 1.5*IQR, только подсчёт (строки НЕ удаляем!).
    # IQR (межквартильный размах) = Q3 - Q1; выброс — дальше 1.5*IQR от ящика.
    lines += [
        "",
        "## Outliers",
        "Non-binary numeric features were checked with the 1.5*IQR rule; the counts are informational only, "
        "no rows were removed because the residuals on the logit scale are close to Gaussian.",
    ]
    # Считаем выбросы по каждой не-бинарной числовой колонке.
    outlier_counts = {}
    for column in numeric.columns:
        # Бинарные флаги (2 уникальных значения) пропускаем — у них выбросов не бывает.
        if numeric[column].nunique() <= 2:  # binary flags have no meaningful IQR outliers
            continue
        # Квартили Q1 (25%) и Q3 (75%).
        q1, q3 = numeric[column].quantile([0.25, 0.75])
        # Межквартильный размах.
        iqr = q3 - q1
        # Считаем значения вне [Q1 - 1.5*IQR, Q3 + 1.5*IQR]; int(...) — в целое число.
        count = int(((numeric[column] < q1 - 1.5 * iqr) | (numeric[column] > q3 + 1.5 * iqr)).sum())
        # В словарь пишем только колонки, где выбросы нашлись (count > 0).
        if count:
            outlier_counts[column] = count
    # Табличка выбросов: Series -> DataFrame, сортировка по убыванию.
    lines += [pd.Series(outlier_counts, name="outliers").sort_values(ascending=False).to_frame().to_markdown(), ""]

    # Рисуем 3 картинки EDA.
    make_plots(train, output_dir)
    # Пишем весь отчёт в файл: склеиваем строки через "\n" + перевод строки в конце.
    (output_dir / "eda_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {output_dir / 'eda_summary.md'} and plots to {output_dir}")


# Запуск main(), только если файл запущен напрямую.
if __name__ == "__main__":
    main()
