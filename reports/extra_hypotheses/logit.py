import pandas as pd, numpy as np, lightgbm as lgb
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_ridge import KernelRidge
df=pd.read_csv('hard_train.csv')
y=df.protection_score.values/100
z=np.log(y/(1-y))
X=df.drop(columns=['customer_id','protection_score'])
num=pd.get_dummies(X,columns=['region','city_type','education','family_status','employment','gender','wealth_segment'],dtype=float)
Xf=num.copy()
for c in Xf.columns:
    if Xf[c].isna().any(): Xf[c+'_na']=Xf[c].isna().astype(float); Xf[c]=Xf[c].fillna(Xf[c].median())
kf=KFold(5,shuffle=True,random_state=0)
sig=lambda t:100/(1+np.exp(-t))
def rep(name,pz):
    print(name,'logit sd',np.sqrt(np.mean((pz-z)**2)).round(4),'raw rmse',np.sqrt(np.mean((sig(pz)-100*y)**2)).round(3),flush=True)
rep('ridge',cross_val_predict(make_pipeline(StandardScaler(),RidgeCV(alphas=np.logspace(-2,4,20))),Xf,z,cv=kf))
rep('lgb logit',cross_val_predict(lgb.LGBMRegressor(n_estimators=1500,learning_rate=0.01,num_leaves=8,min_child_samples=50,subsample=0.7,subsample_freq=1,colsample_bytree=0.7,verbose=-1),Xf,z,cv=kf))
Xs=StandardScaler().fit_transform(Xf)
for a in [1,3,10]:
  for g in [1e-3,3e-3]:
    rep(f'krr a{a} g{g}',cross_val_predict(KernelRidge(alpha=a,kernel='rbf',gamma=g),Xs,z-z.mean(),cv=kf)+z.mean())
