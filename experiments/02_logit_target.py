import pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import KFold
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
D='/home/user/D.-PRO-Client-Under-Protection-Sber-/'
tr=pd.read_csv(D+'hard_train.csv')
y=tr.pop('protection_score').values; tr=tr.drop(columns='customer_id')
cats=['gender','region','city_type','education','family_status','employment','wealth_segment']
X=pd.get_dummies(tr,columns=cats,dtype=float)
X=X.fillna(X.median())
def rmse(a,b): return np.sqrt(mean_squared_error(a,b))
kf=KFold(5,shuffle=True,random_state=42)
def logit(p): p=np.clip(p/100,1e-3,1-1e-3); return np.log(p/(1-p))
def sig(z): return 100/(1+np.exp(-z))
z=logit(y)
print('logit target quantiles',np.percentile(z,[0,5,50,95,100]).round(2))
oofL=np.zeros(len(y)); oofR=np.zeros(len(y)); oofLG=np.zeros(len(y))
for tri,vai in kf.split(X):
    m=make_pipeline(StandardScaler(),RidgeCV(alphas=np.logspace(-3,3,13))).fit(X.iloc[tri],z[tri])
    oofL[vai]=sig(m.predict(X.iloc[vai]))
    m2=lgb.LGBMRegressor(n_estimators=4000,learning_rate=0.01,num_leaves=15,min_child_samples=20,subsample=0.8,subsample_freq=1,colsample_bytree=0.7,verbose=-1,random_state=42)
    m2.fit(X.iloc[tri],z[tri],eval_set=[(X.iloc[vai],z[vai])],callbacks=[lgb.early_stopping(150,verbose=False)])
    oofLG[vai]=sig(m2.predict(X.iloc[vai]))
print('ridge on logit -> sigmoid RMSE',rmse(y,oofL))
print('lgb on logit -> sigmoid RMSE',rmse(y,oofLG))
