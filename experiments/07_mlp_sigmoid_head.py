import sys, numpy as np, pandas as pd, warnings, time
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
R = lambda p: float(np.sqrt(mean_squared_error(y, p)))
sig = lambda z: 1 / (1 + np.exp(-z))

def fit_mlp(A, t, H=8, lam=1e-2, seed=0, restarts=2):
    n, p = A.shape
    rng = np.random.default_rng(seed)
    best = None
    def unpack(th):
        W1 = th[:p * H].reshape(p, H); b1 = th[p * H:p * H + H]
        w2 = th[p * H + H:p * H + 2 * H]; b2 = th[-1]
        return W1, b1, w2, b2
    def f(th):
        W1, b1, w2, b2 = unpack(th)
        Hh = np.tanh(A @ W1 + b1); s = sig(Hh @ w2 + b2); r = 100 * s - t
        loss = np.mean(r ** 2) + lam * (np.sum(W1 ** 2) + np.sum(w2 ** 2))
        g = 2 * r / n * 100 * s * (1 - s)
        gw2 = Hh.T @ g; gb2 = g.sum()
        gH = np.outer(g, w2) * (1 - Hh ** 2)
        gW1 = A.T @ gH + 2 * lam * W1; gb1 = gH.sum(0)
        gw2 += 2 * lam * w2
        return loss, np.concatenate([gW1.ravel(), gb1, gw2, [gb2]])
    for r_ in range(restarts):
        th0 = rng.normal(0, 0.1, p * H + 2 * H + 1)
        res = minimize(f, th0, jac=True, method='L-BFGS-B', options={'maxiter': 2000})
        if best is None or res.fun < best.fun:
            best = res
    W1, b1, w2, b2 = unpack(best.x)
    return lambda B: 100 * sig(np.tanh(B @ W1 + b1) @ w2 + b2)

def run(model_fn):
    oof = np.zeros(len(y))
    for tri, vai in kf.split(tr):
        pp = Preprocessor().fit(tr.iloc[tri])
        A = pp.transform(tr.iloc[tri]); B = pp.transform(tr.iloc[vai])
        oof[vai] = model_fn(A, y[tri], B)
    return R(oof), oof

def mlp_factory(H, lam):
    return lambda A, t, B: fit_mlp(A, t, H=H, lam=lam)(B)

def l1_sig(lam_l1):
    # sigmoid-linear with L1 penalty (sparsity check)
    def m(A, t, B):
        n, p = A.shape; A1 = np.hstack([A, np.ones((n, 1))])
        def f(w):
            s = sig(A1 @ w); r = 100 * s - t
            loss = np.mean(r ** 2) + lam_l1 * np.sum(np.sqrt(w[:-1] ** 2 + 1e-8))
            g = A1.T @ (2 * r / n * 100 * s * (1 - s))
            g[:-1] += lam_l1 * w[:-1] / np.sqrt(w[:-1] ** 2 + 1e-8)
            return loss, g
        w = minimize(f, np.zeros(p + 1), jac=True, method='L-BFGS-B', options={'maxiter': 3000}).x
        m.w = w
        return sig(np.hstack([B, np.ones((len(B), 1))]) @ w) * 100
    return m

if __name__ == '__main__':
    t0 = time.time()
    for lam in [1e-2, 1e-1]:
        for H in [4, 16]:
            s, _ = run(mlp_factory(H, lam)); print('MLP H', H, 'lam', lam, 'RMSE', round(s, 4), round(time.time() - t0), 's', flush=True)
    for l1 in [1e-3, 1e-2]:
        s, _ = run(l1_sig(l1)); print('L1 sig', l1, 'RMSE', round(s, 4), flush=True)
