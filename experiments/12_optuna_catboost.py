import numpy as np, pandas as pd, optuna, json, warnings
warnings.filterwarnings('ignore')
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
optuna.logging.set_verbosity(optuna.logging.WARNING)
D = '/home/user/D.-PRO-Client-Under-Protection-Sber-/'
tr = pd.read_csv(D + 'hard_train.csv')
y = tr.pop('protection_score').values; tr = tr.drop(columns='customer_id')
cats = ['gender', 'region', 'city_type', 'education', 'family_status', 'employment', 'wealth_segment']
X = tr.copy()
for c in ['house_value', 'car_value', 'smartphone_age', 'days_since_last_login', 'average_claim_cost', 'credit_score']:
    X[c + '_na'] = tr[c].isna().astype(int)
X['ins_sum'] = tr[['life_insurance', 'property_insurance', 'health_insurance', 'travel_insurance', 'car_insurance', 'gadget_insurance']].sum(axis=1)
kf = KFold(5, shuffle=True, random_state=42)

def objective(trial):
    p = dict(
        depth=trial.suggest_int('depth', 4, 10),
        l2_leaf_reg=trial.suggest_float('l2_leaf_reg', 1, 10, log=True),
        learning_rate=trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        random_strength=trial.suggest_float('random_strength', 0, 10),
        bagging_temperature=trial.suggest_float('bagging_temperature', 0, 2),
    )
    oof = np.zeros(len(y))
    for tri, vai in kf.split(X):
        m = CatBoostRegressor(iterations=1000, random_seed=42, verbose=0, cat_features=cats, **p)
        m.fit(X.iloc[tri], y[tri])
        oof[vai] = m.predict(X.iloc[vai])
    return float(np.sqrt(mean_squared_error(y, oof)))

study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(objective, n_trials=20, show_progress_bar=False)
print('best', study.best_value, study.best_params, flush=True)
json.dump({'best_rmse': study.best_value, 'best_params': study.best_params},
          open('/home/user/scratch/optuna_cb_best.json', 'w'), indent=2)
for t in study.trials:
    print(t.number, round(t.value, 4), t.params, flush=True)
