import pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from scipy.optimize import minimize
import lightgbm as lgb
D='/home/user/D.-PRO-Client-Under-Protection-Sber-/'
tr=pd.read_csv(D+'hard_train.csv')
y=tr.pop('protection_score').values; tr=tr.drop(columns='customer_id')
cats=['gender','region','city_type','education','family_status','employment','wealth_segment']
X=pd.get_dummies(tr,columns=cats,dtype=float)
X=X.fillna(X.median())
X['na_hv']=tr.house_value.isna().astype(float); X['na_car']=tr.car_value.isna().astype(float)
X['na_cl']=tr.credit_score.isna().astype(float)
X=X.fillna(X.median())
X=X.drop(columns=[c for c in X.columns if X[c].nunique()<=1])
Xv=X.values.astype(float)
mu=Xv.mean(0); sd=Xv.std(0)+1e-9
Xs=(Xv-mu)/sd
def rmse(a,b): return np.sqrt(mean_squared_error(a,b))
kf=KFold(5,shuffle=True,random_state=42)

def fit_sig(A,t,lam):
    n,p=A.shape
    A1=np.hstack([A,np.ones((n,1))])
    def f(w):
        s=1/(1+np.exp(-(A1@w)))
        pred=100*s
        r=pred-t
        loss=(r**2).mean()+lam*(w[:-1]**2).sum()
        g=(2*r/n*100*s*(1-s))
        grad=A1.T@g
        grad[:-1]+=2*lam*w[:-1]
        return loss,grad
    w0=np.zeros(p+1)
    res=minimize(f,w0,jac=True,method='L-BFGS-B',options={'maxiter':2000})
    return res.x
def pred_sig(A,w):
    A1=np.hstack([A,np.ones((len(A),1))]); return 100/(1+np.exp(-(A1@w)))

for lam in [1e-4,1e-3,1e-2]:
    oof=np.zeros(len(y)); oofb=np.zeros(len(y))
    for tri,vai in kf.split(Xs):
        w=fit_sig(Xs[tri],y[tri],lam)
        oof[vai]=pred_sig(Xs[vai],w)
        # residual boosting on top of sigmoid fit (in-fold residuals)
        ptr=pred_sig(Xs[tri],w)
        m=lgb.LGBMRegressor(n_estimators=2000,learning_rate=0.01,num_leaves=15,min_child_samples=40,subsample=0.8,subsample_freq=1,colsample_bytree=0.7,verbose=-1,random_state=42)
        Xdf=X.iloc[tri]; Xvdf=X.iloc[vai]
        m.fit(Xdf,y[tri]-ptr,eval_set=[(X.iloc[vai],y[vai]-oof[vai])],callbacks=[lgb.early_stopping(100,verbose=False)])
        oofb[vai]=oof[vai]+m.predict(Xvdf)
    print('lam',lam,'sigmoid-linear RMSE',rmse(y,oof),'+lgb residual',rmse(y,oofb),flush=True)
