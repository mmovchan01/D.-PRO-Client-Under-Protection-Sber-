import sys, numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, '/home/user/D.-PRO-Client-Under-Protection-Sber-')
from model import Preprocessor, fit_sigmoid_linear
import lightgbm as lgb, xgboost as xgb, catboost as cb
from category_encoders import TargetEncoder
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from scipy.optimize import nnls
D = '/home/user/D.-PRO-Client-Under-Protection-Sber-/'
tr = pd.read_csv(D + 'hard_train.csv')
y = tr['protection_score'].values
kf = KFold(5, shuffle=True, random_state=42)
R = lambda p: float(np.sqrt(mean_squared_error(y, p)))
sig = lambda z: 1 / (1 + np.exp(-z))
folds = list(kf.split(tr))

def add_ratios(df):
    df = df.copy()
    df['income_to_credit'] = df['income'] / (df['credit_score'] + 1)
    df['risk_per_loyalty'] = df['regional_risk'] / (df['customer_loyalty'] + 1)
    df['claim_to_income'] = df['average_claim_cost'] / (df['income'] + 1)
    return df

def sl_predict(trd, yt, ted):
    pp = Preprocessor().fit(trd); A = pp.transform(trd); B = pp.transform(ted)
    w = fit_sigmoid_linear(A, yt, lam=1e-3)
    return sig(np.hstack([B, np.ones((len(B), 1))]) @ w) * 100

# --- Idea 1: clipping (diagnostic)
oof_sl = np.zeros(len(y))
for tri, vai in folds:
    oof_sl[vai] = sl_predict(tr.iloc[tri], y[tri], tr.iloc[vai])
lo, hi = y.min(), y.max()
print('I1 sigmoid base', round(R(oof_sl), 4), 'clipped to train min/max', round(R(np.clip(oof_sl, lo, hi)), 4), flush=True)

# --- Idea 2: ratio features (names corrected: regional_risk, not total_regional_risk)
oof_r = np.zeros(len(y))
for tri, vai in folds:
    oof_r[vai] = sl_predict(add_ratios(tr.iloc[tri]), y[tri], add_ratios(tr.iloc[vai]))
print('I2 sigmoid + ratio features', round(R(oof_r), 4), flush=True)

# --- Idea 3: out-of-fold target encoding of region (inside each training fold)
oof_te = np.zeros(len(y))
for tri, vai in folds:
    trd = tr.iloc[tri].copy(); ted = tr.iloc[vai].copy()
    enc = TargetEncoder(cols=['region', 'employment', 'city_type'], smoothing=20)
    enc.fit(trd[['region', 'employment', 'city_type']], y[tri])
    trd_te = enc.transform(trd[['region', 'employment', 'city_type']]).add_suffix('_te')
    ted_te = enc.transform(ted[['region', 'employment', 'city_type']]).add_suffix('_te')
    trd = pd.concat([trd, trd_te], axis=1)
    ted = pd.concat([ted, ted_te], axis=1)
    oof_te[vai] = sl_predict(trd, y[tri], ted)
print('I3 sigmoid + target encoding (region/employment/city) ', round(R(oof_te), 4), flush=True)

# --- Idea 4: stacking of diverse models
def lgb_predict(trd, yt, ted):
    pp = Preprocessor().fit(trd)
    m = lgb.LGBMRegressor(n_estimators=1500, learning_rate=0.02, num_leaves=15, min_child_samples=40,
                          subsample=0.8, subsample_freq=1, colsample_bytree=0.7, verbose=-1, random_state=42)
    m.fit(pp.transform(trd), yt); return m.predict(pp.transform(ted))
def xgb_predict(trd, yt, ted):
    pp = Preprocessor().fit(trd)
    m = xgb.XGBRegressor(n_estimators=1500, learning_rate=0.02, max_depth=4, subsample=0.8, colsample_bytree=0.7,
                         min_child_weight=5, random_state=42)
    m.fit(pp.transform(trd), yt); return m.predict(pp.transform(ted))
cats = ['gender', 'region', 'city_type', 'education', 'family_status', 'employment', 'wealth_segment']
def cb_predict(trd, yt, ted):
    m = cb.CatBoostRegressor(iterations=1000, depth=5, learning_rate=0.03, random_seed=42, verbose=0, cat_features=cats)
    m.fit(trd, yt); return m.predict(ted)

oofs = {'sigmoid': oof_sl, 'lgb': np.zeros(len(y)), 'xgb': np.zeros(len(y)), 'cat': np.zeros(len(y))}
for tri, vai in folds:
    oofs['lgb'][vai] = lgb_predict(tr.iloc[tri], y[tri], tr.iloc[vai])
    oofs['xgb'][vai] = xgb_predict(tr.iloc[tri], y[tri], tr.iloc[vai])
    drop = ['customer_id', 'protection_score']
    oofs['cat'][vai] = cb_predict(tr.iloc[tri].drop(columns=drop), y[tri], tr.iloc[vai].drop(columns=drop))
for k, v in oofs.items():
    print('  base', k, round(R(v), 4), flush=True)
names = list(oofs); M = np.column_stack([oofs[k] for k in names])
# honest stacking estimate: weights fitted on 4 folds of OOF, evaluated on the 5th
st = np.zeros(len(y))
for tri, vai in KFold(5, shuffle=True, random_state=7).split(M):
    w, _ = nnls(M[tri], y[tri]); st[vai] = M[vai] @ w
w_full, _ = nnls(M, y)
print('I4 nested nnls stacking', round(R(st), 4), 'weights', dict(zip(names, np.round(w_full, 3))), flush=True)
print('I4 simple avg of all', round(R(M.mean(1)), 4), flush=True)
print('I4 sigmoid + cat 0.7/0.3', round(R(0.7 * oof_sl + 0.3 * oofs['cat']), 4), flush=True)
