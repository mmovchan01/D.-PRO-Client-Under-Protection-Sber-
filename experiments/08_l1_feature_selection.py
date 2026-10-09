import sys, numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, '/home/user/D.-PRO-Client-Under-Protection-Sber-')
from model import Preprocessor, fit_sigmoid_linear
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from scipy.optimize import minimize
D = '/home/user/D.-PRO-Client-Under-Protection-Sber-/'
tr = pd.read_csv(D + 'hard_train.csv')
y = tr['protection_score'].values
kf = KFold(5, shuffle=True, random_state=42)
sig = lambda z: 1 / (1 + np.exp(-z))
R = lambda p: float(np.sqrt(mean_squared_error(y, p)))

def l1_fit(A, t, l1):
    n, p = A.shape; A1 = np.hstack([A, np.ones((n, 1))])
    def f(w):
        s = sig(A1 @ w); r = 100 * s - t
        ww = w[:-1]; sq = np.sqrt(ww ** 2 + 1e-10)
        loss = np.mean(r ** 2) + l1 * np.sum(sq)
        g = A1.T @ (2 * r / n * 100 * s * (1 - s))
        g[:-1] += l1 * ww / sq
        return loss, g
    return minimize(f, np.zeros(p + 1), jac=True, method='L-BFGS-B', options={'maxiter': 3000}).x

# sparsity profile on full data (diagnostic)
pp = Preprocessor().fit(tr); A = pp.transform(tr)
for l1 in [1e-4, 3e-4, 1e-3, 3e-3]:
    w = l1_fit(A, y, l1)
    nz = np.sum(np.abs(w[:-1]) > 1e-2)
    print('L1', l1, 'nonzero-ish coefs', nz, 'of', A.shape[1], flush=True)

# CV: L1 selection of features (chosen inside each fold) followed by unpenalized-ish refit
def cv_l1(l1_sel, k=None):
    oof = np.zeros(len(y))
    for tri, vai in kf.split(tr):
        pp = Preprocessor().fit(tr.iloc[tri]); Atr = pp.transform(tr.iloc[tri]); Ava = pp.transform(tr.iloc[vai])
        w = l1_fit(Atr, y[tri], l1_sel)
        keep = np.where(np.abs(w[:-1]) > 1e-2)[0]
        if len(keep) == 0: keep = np.arange(Atr.shape[1])
        wf = fit_sigmoid_linear(Atr[:, keep], y[tri], lam=1e-3)
        oof[vai] = sig(np.hstack([Ava[:, keep], np.ones((len(vai), 1))]) @ wf) * 100
    return R(oof)
for l1 in [1e-3, 3e-3]:
    print('CV L1-select', l1, cv_l1(l1), flush=True)
