"""Обучение модели уровня страховой защиты (protection_score).

Запуск целиком:
    python train.py                 # SEED выбирается случайно и печатается
    python train.py --seed 12345    # воспроизведение с фиксированным SEED

Результат: папка weights/ с весами модели, препроцессингом и SEED.
Обучение использует только hard_train.csv. Тестовые выборки не используются.
Флаг --cv дополнительно печатает 5-fold CV RMSE (только для отчёта, на веса не влияет).
"""
import argparse
import json
import os
import secrets

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

from model import Preprocessor, fit_bagged, predict_bagged

HERE = os.path.dirname(os.path.abspath(__file__))
TRAIN_PATH = os.path.join(HERE, 'hard_train.csv')
WEIGHTS_DIR = os.path.join(HERE, 'weights')
N_BAGS = 30
LAMBDA = 1e-3


def rmse(a, b):
    return float(np.sqrt(mean_squared_error(a, b)))


def run_cv(df, y, seed):
    kf = KFold(5, shuffle=True, random_state=seed)
    oof = np.zeros(len(y))
    for tri, vai in kf.split(df):
        pp = Preprocessor().fit(df.iloc[tri])
        W = fit_bagged(pp.transform(df.iloc[tri]), y[tri], seed=seed, n_bags=N_BAGS, lam=LAMBDA)
        oof[vai] = predict_bagged(pp.transform(df.iloc[vai]), W)
    return rmse(y, oof)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=None,
                        help='Seed; если не задан — выбирается случайно')
    parser.add_argument('--cv', action='store_true', help='Посчитать 5-fold CV RMSE')
    args = parser.parse_args()

    seed = args.seed if args.seed is not None else secrets.randbelow(10 ** 6)
    np.random.seed(seed)
    print(f'SEED = {seed}')

    df = pd.read_csv(TRAIN_PATH)
    y = df['protection_score'].values

    if args.cv:
        print(f'5-fold CV RMSE: {run_cv(df, y, seed):.4f}')

    # Финальное обучение на всей обучающей выборке
    pp = Preprocessor().fit(df)
    A = pp.transform(df)
    W = fit_bagged(A, y, seed=seed, n_bags=N_BAGS, lam=LAMBDA)
    train_pred = predict_bagged(A, W)
    print(f'Train RMSE (in-sample, справочно): {rmse(y, train_pred):.4f}')

    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    np.save(os.path.join(WEIGHTS_DIR, 'bagged_weights.npy'), W)
    with open(os.path.join(WEIGHTS_DIR, 'preprocessing.json'), 'w', encoding='utf-8') as f:
        json.dump(pp.to_dict(), f, ensure_ascii=False)
    with open(os.path.join(WEIGHTS_DIR, 'meta.json'), 'w', encoding='utf-8') as f:
        json.dump({
            'seed': seed,
            'model': 'bagged sigmoid-linear: 100 * sigmoid(Xw + b), MSE loss',
            'n_bags': N_BAGS,
            'l2_lambda': LAMBDA,
            'n_features': int(A.shape[1]),
            'train_file': 'hard_train.csv',
        }, f, ensure_ascii=False, indent=2)
    print(f'Веса сохранены в {WEIGHTS_DIR}/. Для предсказаний: python predict.py')


if __name__ == '__main__':
    main()
