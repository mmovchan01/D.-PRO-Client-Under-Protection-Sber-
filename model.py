"""Общие функции: предобработка признаков и модель.

Модель: «линейная комбинация признаков -> сигмоида -> умножение на 100».
Обучается напрямую по MSE (RMSE — целевая метрика) с L2-регуляризацией.
Для устойчивости используется бэггинг: K моделей, каждая обучена на
bootstrap-выборке, предсказания усредняются. Все случайные выборы
управляются одним SEED.
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize

CATS = ['gender', 'region', 'city_type', 'education', 'family_status',
        'employment', 'wealth_segment']
CAT_LEVELS = {
    'gender': ['F', 'M'],
    'region': ['FarEast', 'Moscow', 'NorthWest', 'Siberia', 'SPb', 'South', 'Ural', 'Volga'],
    'city_type': ['large_city', 'metro', 'rural', 'town'],
    'education': ['bachelor', 'master', 'phd', 'secondary'],
    'family_status': ['divorced', 'married', 'single', 'widowed'],
    'employment': ['business_owner', 'employed', 'retired', 'self_employed', 'student', 'unemployed'],
    'wealth_segment': ['affluent', 'mass', 'premium', 'private'],
}
NA_COLS = ['house_value', 'car_value', 'smartphone_age', 'days_since_last_login',
           'average_claim_cost', 'credit_score', 'customer_loyalty',
           'marketing_response', 'health_index']
INSURANCE_COLS = ['life_insurance', 'property_insurance', 'health_insurance',
                  'travel_insurance', 'car_insurance', 'gadget_insurance']


def build_raw_features(df):
    """Индикаторы пропусков, сумма страховых продуктов и one-hot категорий."""
    out = df.drop(columns=[c for c in ['customer_id', 'protection_score'] if c in df.columns]).copy()
    for c in NA_COLS:
        out[c + '_na'] = out[c].isna().astype(int) if c in out else 0
    out['ins_sum'] = df[INSURANCE_COLS].sum(axis=1)
    for c in CATS:
        out[c] = pd.Categorical(out[c], categories=CAT_LEVELS[c])
    return pd.get_dummies(out, columns=CATS, dtype=float)


class Preprocessor:
    """Запоминает медианы, список колонок и статистики стандартизации с обучения."""

    def fit(self, df):
        X = build_raw_features(df)
        medians = X.median()
        X = X.fillna(medians)
        self.columns_ = [c for c in X.columns if X[c].nunique() > 1]
        self.medians_ = medians[self.columns_]
        X = X[self.columns_]
        self.mean_ = X.mean()
        self.std_ = X.std().replace(0, 1.0)
        return self

    def transform(self, df):
        X = build_raw_features(df).reindex(columns=self.columns_, fill_value=0.0)
        X = X.fillna(self.medians_)
        X = (X - self.mean_) / self.std_
        return X.values.astype(float)

    def to_dict(self):
        return {
            'columns': self.columns_,
            'medians': self.medians_.tolist(),
            'mean': self.mean_.tolist(),
            'std': self.std_.tolist(),
        }

    @classmethod
    def from_dict(cls, d):
        p = cls()
        p.columns_ = d['columns']
        p.medians_ = pd.Series(d['medians'], index=d['columns'])
        p.mean_ = pd.Series(d['mean'], index=d['columns'])
        p.std_ = pd.Series(d['std'], index=d['columns'])
        return p


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def fit_sigmoid_linear(A, t, lam=1e-3):
    """Подбор весов для pred = 100 * sigmoid(A @ w + b) по MSE."""
    n, p = A.shape
    A1 = np.hstack([A, np.ones((n, 1))])

    def loss_grad(w):
        s = _sigmoid(A1 @ w)
        r = 100.0 * s - t
        loss = np.mean(r ** 2) + lam * np.sum(w[:-1] ** 2)
        grad = A1.T @ (2.0 * r / n * 100.0 * s * (1.0 - s))
        grad[:-1] += 2.0 * lam * w[:-1]
        return loss, grad

    res = minimize(loss_grad, np.zeros(p + 1), jac=True, method='L-BFGS-B',
                   options={'maxiter': 3000})
    return res.x


def fit_bagged(A, t, seed, n_bags=30, lam=1e-3):
    """Бэггинг: n_bags моделей на bootstrap-выборках, сид фиксирует выборки."""
    rng = np.random.default_rng(seed)
    W = []
    for _ in range(n_bags):
        idx = rng.integers(0, len(A), size=len(A))
        W.append(fit_sigmoid_linear(A[idx], t[idx], lam=lam))
    return np.array(W)


def predict_bagged(A, W):
    A1 = np.hstack([A, np.ones((len(A), 1))])
    preds = 100.0 * _sigmoid(A1 @ W.T)  # (n, n_bags)
    return preds.mean(axis=1)
