import sys, numpy as np, pandas as pd
sys.path.insert(0, '/home/user/D.-PRO-Client-Under-Protection-Sber-')
from model import Preprocessor, fit_bagged, predict_bagged, fit_sigmoid_linear
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
D = '/home/user/D.-PRO-Client-Under-Protection-Sber-/'
tr = pd.read_csv(D + 'hard_train.csv')
y = tr['protection_score'].values
kf = KFold(5, shuffle=True, random_state=42)
oof1 = np.zeros(len(y)); oofB = np.zeros(len(y))
for tri, vai in kf.split(tr):
    pp = Preprocessor().fit(tr.iloc[tri])
    A = pp.transform(tr.iloc[tri]); B = pp.transform(tr.iloc[vai])
    w = fit_sigmoid_linear(A, y[tri])
    oof1[vai] = predict_bagged(B, w[None, :])
    W = fit_bagged(A, y[tri], seed=42, n_bags=30)
    oofB[vai] = predict_bagged(B, W)
    print('fold p', A.shape, flush=True)
print('single RMSE', np.sqrt(mean_squared_error(y, oof1)))
print('bagged RMSE', np.sqrt(mean_squared_error(y, oofB)))
print('clipped bagged', np.sqrt(mean_squared_error(y, np.clip(oofB, 0, 100))))
