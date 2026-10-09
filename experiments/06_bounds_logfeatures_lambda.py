import sys, numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, '/home/user/D.-PRO-Client-Under-Protection-Sber-')
from model import Preprocessor, fit_sigmoid_linear, CATS, NA_COLS, INSURANCE_COLS
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from scipy.optimize import minimize
D = '/home/user/D.-PRO-Client-Under-Protection-Sber-/'
tr = pd.read_csv(D + 'hard_train.csv')
y = tr['protection_score'].values
kf = KFold(5, shuffle=True, random_state=42)
R = lambda p: float(np.sqrt(mean_squared_error(y, p)))
sig = lambda z: 1 / (1 + np.exp(-z))

def cv(fn):
    oof = np.zeros(len(y))
    for tri, vai in kf.split(tr):
        oof[vai] = fn(tr.iloc[tri], y[tri], tr.iloc[vai])
    return R(oof), oof

# H1: learnable output bounds lo + (hi-lo)*sigmoid
def fit_bounds(A, t, lam=1e-3):
    n, p = A.shape; A1 = np.hstack([A, np.ones((n, 1))])
    def f(th):
        w = th[:-2]; lo, hi = th[-2], th[-1]
        s = sig(A1 @ w); pred = lo + (hi - lo) * s; r = pred - t
        loss = np.mean(r ** 2) + lam * np.sum(w[:-1] ** 2)
        gw = A1.T @ (2 * r / n * (hi - lo) * s * (1 - s)); gw[:-1] += 2 * lam * w[:-1]
        glo = np.mean(2 * r * (1 - s)); ghi = np.mean(2 * r * s)
        return loss, np.concatenate([gw, [glo, ghi]])
    th0 = np.concatenate([np.zeros(p + 1), [0.0, 100.0]])
    th = minimize(f, th0, jac=True, method='L-BFGS-B', options={'maxiter': 5000}).x
    return th
def h1(trd, yt, ted):
    pp = Preprocessor().fit(trd); A = pp.transform(trd); B = pp.transform(ted)
    th = fit_bounds(A, yt)
    print('   bounds', np.round(th[-2:], 2))
    return th[-2] + (th[-1] - th[-2]) * sig(np.hstack([B, np.ones((len(B), 1))]) @ th[:-2])

# H2: log-transform skewed monetary features before the sigmoid model
SKEW = ['income', 'average_balance', 'loan_amount', 'house_value', 'car_value',
        'smartphone_price', 'average_claim_cost']
def add_logs(df):
    df = df.copy()
    for c in SKEW:
        df['log_' + c] = np.log1p(df[c])
    return df
def h2(trd, yt, ted):
    trd = add_logs(trd); ted = add_logs(ted)
    pp = Preprocessor().fit(trd); A = pp.transform(trd); B = pp.transform(ted)
    w = fit_sigmoid_linear(A, yt, lam=1e-3)
    return sig(np.hstack([B, np.ones((len(B), 1))]) @ w) * 100

# H3: lambda sweep for base sigmoid-linear
def base(lam):
    def f(trd, yt, ted):
        pp = Preprocessor().fit(trd); A = pp.transform(trd); B = pp.transform(ted)
        w = fit_sigmoid_linear(A, yt, lam=lam)
        return sig(np.hstack([B, np.ones((len(B), 1))]) @ w) * 100
    return f

# H4: ID-order signal (diagnostic only; NOT used in the final model)
def id_autocorr():
    idn = tr['customer_id'].str.replace('CUST', '').astype(int).values
    o = np.argsort(idn); ys = y[o]
    print('   lag-1 autocorr of target over id order:', np.corrcoef(ys[:-1], ys[1:])[0, 1].round(4))

if __name__ == '__main__':
    print('base lam=1e-3', cv(base(1e-3))[0], flush=True)
    for lam in [1e-2, 1e-1, 1.0]:
        print('lam', lam, cv(base(lam))[0], flush=True)
    print('H1 learnable bounds', cv(h1)[0], flush=True)
    print('H2 log features', cv(h2)[0], flush=True)
    id_autocorr()
