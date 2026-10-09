import pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize
import catboost as cb
D='/home/user/D.-PRO-Client-Under-Protection-Sber-/'
tr=pd.read_csv(D+'hard_train.csv')
y=tr.pop('protection_score').values; tr=tr.drop(columns='customer_id')
cats=['gender','region','city_type','education','family_status','employment','wealth_segment']
Xc=tr.copy()
for c in ['house_value','car_value','smartphone_age','days_since_last_login','average_claim_cost','credit_score','customer_loyalty','marketing_response','health_index']:
    Xc[c+'_na']=tr[c].isna().astype(int)
Xc['ins_sum']=tr[['life_insurance','property_insurance','health_insurance','travel_insurance','car_insurance','gadget_insurance']].sum(1)
Xl=pd.get_dummies(Xc,columns=cats,dtype=float)
Xl=Xl.fillna(Xl.median())
Xl=Xl.drop(columns=[c for c in Xl.columns if Xl[c].nunique()<=1])
def rmse(a,b): return np.sqrt(mean_squared_error(a,b))
kf=KFold(5,shuffle=True,random_state=42)
def fit_sig(A,t,lam):
    n,p=A.shape; A1=np.hstack([A,np.ones((n,1))])
    def f(w):
        s=1/(1+np.exp(-(A1@w))); r=100*s-t
        loss=(r**2).mean()+lam*(w[:-1]**2).sum()
        grad=A1.T@(2*r/n*100*s*(1-s)); grad[:-1]+=2*lam*w[:-1]
        return loss,grad
    return minimize(f,np.zeros(p+1),jac=True,method='L-BFGS-B',options={'maxiter':3000}).x
def pred_sig(A,w): return 100/(1+np.exp(-(np.hstack([A,np.ones((len(A),1))])@w)))
oofS=np.zeros(len(y)); oofC=np.zeros(len(y))
Xs=StandardScaler().fit_transform(Xl.values)
for tri,vai in kf.split(Xs):
    w=fit_sig(Xs[tri],y[tri],1e-3); oofS[vai]=pred_sig(Xs[vai],w)
    m=cb.CatBoostRegressor(iterations=1500,learning_rate=0.03,depth=6,random_seed=42,verbose=0,cat_features=cats)
    m.fit(Xc.iloc[tri],y[tri]); oofC[vai]=m.predict(Xc.iloc[vai])
print('sig',rmse(y,oofS),'cat',rmse(y,oofC),flush=True)
for a in [0.3,0.5,0.7,0.8]:
    print('blend w_sig',a,rmse(y,a*oofS+(1-a)*oofC))
