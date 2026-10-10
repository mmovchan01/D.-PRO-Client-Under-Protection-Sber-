"""Черновик: линейные/полиномиальные модели на подмножествах признаков. Запуск из корня.

Сравнивает линейную регрессию и полином 2-й степени на топ-5 и топ-15 признаках,
затем LightGBM на всех числовых и на топ-5. Вывод: сигнал распределён по многим
признакам, топ-5 мало (RMSE ~12.7-13.4). См. RESULTS.md.
"""
import pandas as pd, numpy as np, lightgbm as lgb
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.linear_model import RidgeCV, LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
df=pd.read_csv('hard_train.csv')
y=df.protection_score.values
kf=KFold(5,shuffle=True,random_state=0)
def rm(p): return np.sqrt(np.mean((p-y)**2))
X=df.drop(columns=['customer_id','protection_score'])
num=X.select_dtypes('number').fillna(-999)
# linear on 5 top feats
top=['digital_behavior_score','cyber_protection','insurance_products','two_factor_auth','active_policies']
for feats in [top, top+['mobile_app_usage','internet_activity','identity_protection','mobile_sessions','password_manager','income','website_visits','marketing_response','number_of_claims','late_payments']]:
    Xs=X[feats].fillna(X[feats].median())
    p=cross_val_predict(make_pipeline(StandardScaler(),LinearRegression()),Xs,y,cv=kf)
    p2=cross_val_predict(make_pipeline(StandardScaler(),PolynomialFeatures(2),RidgeCV(alphas=np.logspace(-3,3,13))),Xs,y,cv=kf)
    print(len(feats),'lin',rm(p).round(3),'poly2',rm(p2).round(3))
# lgb on all
p=cross_val_predict(lgb.LGBMRegressor(n_estimators=600,learning_rate=0.02,num_leaves=15,min_child_samples=40,subsample=0.8,subsample_freq=1,colsample_bytree=0.8,verbose=-1),num,y,cv=kf)
print('lgb all',rm(p))
p=cross_val_predict(lgb.LGBMRegressor(n_estimators=600,learning_rate=0.02,num_leaves=15,min_child_samples=40,subsample=0.8,subsample_freq=1,colsample_bytree=0.8,verbose=-1),num[top],y,cv=kf)
print('lgb top5',rm(p))
