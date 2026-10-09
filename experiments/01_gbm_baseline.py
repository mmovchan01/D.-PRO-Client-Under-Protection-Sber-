import pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import lightgbm as lgb, catboost as cb
D='/home/user/D.-PRO-Client-Under-Protection-Sber-/'
tr=pd.read_csv(D+'hard_train.csv'); te=pd.read_csv(D+'hard_test.csv')
y=tr.pop('protection_score'); tr=tr.drop(columns='customer_id'); te=te.drop(columns='customer_id')
cats=['gender','region','city_type','education','family_status','employment','wealth_segment']
def fe(df):
    df=df.copy()
    for c in ['house_value','car_value','smartphone_age','days_since_last_login','average_claim_cost']:
        df[c+'_na']=df[c].isna().astype(int)
    df['ins_sum']=df[['life_insurance','property_insurance','health_insurance','travel_insurance','car_insurance','gadget_insurance']].sum(1)
    df['cyber_sum']=df[['cyber_protection','identity_protection']].sum(1)
    df['sec_sum']=df[['password_manager','two_factor_auth','security_training']].sum(1)
    df['log_income']=np.log1p(df.income)
    df['log_bal']=np.log1p(df.average_balance)
    return df
X=fe(tr); XT=fe(te)
for c in cats:
    X[c]=X[c].astype('category'); XT[c]=pd.Categorical(XT[c],categories=X[c].cat.categories)
def rmse(a,b): return np.sqrt(mean_squared_error(a,b))
kf=KFold(5,shuffle=True,random_state=42)
res={}
for name in ['lgb','cat']:
    oof=np.zeros(len(X))
    for tri,vai in kf.split(X):
        if name=='lgb':
            m=lgb.LGBMRegressor(n_estimators=3000,learning_rate=0.02,num_leaves=31,min_child_samples=20,subsample=0.8,subsample_freq=1,colsample_bytree=0.7,reg_lambda=1,verbose=-1,random_state=42)
            m.fit(X.iloc[tri],y.iloc[tri],eval_set=[(X.iloc[vai],y.iloc[vai])],callbacks=[lgb.early_stopping(100,verbose=False)])
            oof[vai]=m.predict(X.iloc[vai])
        else:
            m=cb.CatBoostRegressor(iterations=3000,learning_rate=0.03,depth=6,loss_function='RMSE',random_seed=42,verbose=0,cat_features=cats,early_stopping_rounds=150)
            m.fit(X.iloc[tri],y.iloc[tri],eval_set=(X.iloc[vai],y.iloc[vai]))
            oof[vai]=m.predict(X.iloc[vai])
    res[name]=oof
    print(name,'CV RMSE',rmse(y,oof),flush=True)
b=(res['lgb']+res['cat'])/2
print('blend',rmse(y,b),'clipped',rmse(y,np.clip(b,0,100)),flush=True)
