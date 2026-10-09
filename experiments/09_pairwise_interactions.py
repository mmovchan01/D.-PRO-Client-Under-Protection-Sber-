import sys, numpy as np, pandas as pd, itertools
sys.path.insert(0, '/home/user/D.-PRO-Client-Under-Protection-Sber-')
from model import Preprocessor, fit_sigmoid_linear
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
D = '/home/user/D.-PRO-Client-Under-Protection-Sber-/'
tr = pd.read_csv(D + 'hard_train.csv')
y = tr['protection_score'].values
kf = KFold(5, shuffle=True, random_state=42)
top = ['insurance_products', 'active_policies', 'cyber_protection', 'digital_behavior_score',
       'two_factor_auth', 'password_manager', 'internet_activity', 'mobile_app_usage',
       'late_payments', 'regional_risk', 'occupation_risk', 'security_training']
oof = np.zeros(len(y)); oof2 = np.zeros(len(y))
for tri, vai in kf.split(tr):
    pp = Preprocessor().fit(tr.iloc[tri])
    A = pp.transform(tr.iloc[tri]); B = pp.transform(tr.iloc[vai])
    w = fit_sigmoid_linear(A, y[tri], lam=1e-3)
    sig = lambda Z, w: 100 / (1 + np.exp(-(np.hstack([Z, np.ones((len(Z), 1))]) @ w)))
    oof[vai] = sig(B, w)
    # interactions among top raw features (standardized values from the same preprocessing)
    cols = pp.columns_
    idx = [cols.index(c) for c in top]
    def addint(Z):
        extra = [Z[:, i] * Z[:, j] for i, j in itertools.combinations(idx, 2)]
        return np.hstack([Z, np.column_stack(extra)])
    A2 = addint(A); B2 = addint(B)
    w2 = fit_sigmoid_linear(A2, y[tri], lam=1e-2)
    oof2[vai] = sig(B2, w2)
print('base', np.sqrt(mean_squared_error(y, oof)), 'with interactions', np.sqrt(mean_squared_error(y, oof2)))
