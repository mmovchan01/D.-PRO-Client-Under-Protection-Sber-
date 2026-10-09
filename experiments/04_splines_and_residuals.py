import pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import SplineTransformer, StandardScaler
import sys
sys.path.insert(0,'/home/user/scratch')
from scipy.optimize import minimize
D='/home/user/D.-PRO-Client-Under-Protection-Sber-/'
tr=pd.read_csv(D+'hard_train.csv')
y=tr.pop('protection_score').values; tr=tr.drop(columns='customer_id')
cats=['gender','region','city_type','education','family_status','employment','wealth_segment']
num=[c for c in tr.columns if c not in cats]
Xbase=pd.get_dummies(tr,columns=cats,dtype=float)
Xbase['na_hv']=tr.house_value.isna().astype(float); Xbase['na_car']=tr.car_value.isna().astype(float)
Xbase['na_cl']=tr.credit_score.isna().astype(float); Xbase['na_dl']=tr.days_since_last_login.isna().astype(float)
Xbase['na_sa']=tr.smartphone_age.isna().astype(float); Xbase['na_cac']=tr.average_claim_cost.isna().astype(float)
Xbase=Xbase.fillna(Xbase.median())
Xbase=Xbase.drop(columns=[c for c in Xbase.columns if Xbase[c].nunique()<=1])
def rmse(a,b): return np.sqrt(mean_squared_error(a,b))
kf=KFold(5,shuffle=True,random_state=42)
def fit_sig(A,t,lam):
    n,p=A.shape
    A1=np.hstack([A,np.ones((n,1))])
    def f(w):
        s=1/(1+np.exp(-(A1@w))); r=100*s-t
        loss=(r**2).mean()+lam*(w[:-1]**2).sum()
        grad=A1.T@(2*r/n*100*s*(1-s)); grad[:-1]+=2*lam*w[:-1]
        return loss,grad
    return minimize(f,np.zeros(p+1),jac=True,method='L-BFGS-B',options={'maxiter':3000}).x
def pred_sig(A,w):
    return 100/(1+np.exp(-(np.hstack([A,np.ones((len(A),1))])@w)))

# spline expansion of numeric columns (not dummies / binary)
numcols=[c for c in num if tr[c].nunique()>2 and c in Xbase.columns]
for nk in [4,6]:
  oof=np.zeros(len(y))
  for tri,vai in kf.split(Xbase):
    sp=SplineTransformer(n_knots=nk,degree=3,extrapolation='constant')
    Ftr=sp.fit_transform(Xbase[numcols].iloc[tri]); Fva=sp.transform(Xbase[numcols].iloc[vai])
    rest_tr=Xbase.drop(columns=numcols).iloc[tri].values; rest_va=Xbase.drop(columns=numcols).iloc[vai].values
    A=np.hstack([rest_tr,Ftr]); B=np.hstack([rest_va,Fva])
    ss=StandardScaler().fit(A); A=ss.transform(A); B=ss.transform(B)
    w=fit_sig(A,y[tri],1e-3)
    oof[vai]=pred_sig(B,w)
  print('splines nk',nk,'RMSE',rmse(y,oof),flush=True)
# residual diagnostics for plain sigmoid-linear
oof=np.zeros(len(y))
Xs=StandardScaler().fit_transform(Xbase.values)
for tri,vai in kf.split(Xs):
    w=fit_sig(Xs[tri],y[tri],1e-3); oof[vai]=pred_sig(Xs[vai],w)
res=y-oof
print('resid by pred decile', pd.Series(res).groupby(pd.qcut(oof,10,labels=False)).std().round(2).tolist())
print('resid mean by pred decile', pd.Series(res).groupby(pd.qcut(oof,10,labels=False)).mean().round(2).tolist())
