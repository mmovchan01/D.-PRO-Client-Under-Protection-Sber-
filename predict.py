"""Предсказания на тестовой выборке без обучения.

Запуск целиком:
    python predict.py                  # веса из weights/, SEED берётся из meta.json
    python predict.py --test hard_test.csv

Результат: submission_seed_{SEED}.csv с колонками customer_id, protection_score.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

from model import Preprocessor, predict_bagged

HERE = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_DIR = os.path.join(HERE, 'weights')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', default=os.path.join(HERE, 'hard_test.csv'))
    parser.add_argument('--out', default=None, help='Путь к файлу; по умолчанию submission_seed_{SEED}.csv')
    args = parser.parse_args()

    with open(os.path.join(WEIGHTS_DIR, 'meta.json'), encoding='utf-8') as f:
        meta = json.load(f)
    with open(os.path.join(WEIGHTS_DIR, 'preprocessing.json'), encoding='utf-8') as f:
        pp = Preprocessor.from_dict(json.load(f))
    W = np.load(os.path.join(WEIGHTS_DIR, 'bagged_weights.npy'))

    seed = meta['seed']
    df = pd.read_csv(args.test)
    pred = np.clip(predict_bagged(pp.transform(df), W), 0.0, 100.0)

    out = args.out or os.path.join(HERE, f'submission_seed_{seed}.csv')
    pd.DataFrame({'customer_id': df['customer_id'], 'protection_score': pred}).to_csv(out, index=False)
    print(f'SEED = {seed}; записано {len(df)} строк в {out}')


if __name__ == '__main__':
    main()
