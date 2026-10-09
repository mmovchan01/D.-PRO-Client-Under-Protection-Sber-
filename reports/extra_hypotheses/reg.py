import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import KFold
from sklearn.linear_model import RidgeCV
from catboost import CatBoostRegressor
import train as T
from model_utils import fit_preprocessor, numeric_matrix, transform_features, get_numeric_feature_names, logits_to_score, target_to_logit
df=pd.read_csv('hard_train.csv'); y=df.protection_score.values
feats=get_numeric_feature_names(df); R=numeric_matrix(df,feats)
kf=KFold(5,shuffle=True,random_state=434089)
def oof(fn):
    p=np.zeros(len(y))
    for a,b in kf.split(R):
        p[b]=fn(a,b)
    return np.sqrt(np.mean((p-y)**2))
def sig_fit(l2):
    def f(a,b):
        med,mu,sc=fit_preprocessor(R[a]); xa=transform_features(R[a],med,mu,sc); xb=transform_features(R[b],med,mu,sc)
        c,i,_=T.fit_sigmoid_index(xa,y[a],l2_alpha=l2)
        return logits_to_score(xb@c+i)
    return f
for l2 in []:
    print('sigmoid L2=%g  OOF RMSE %.4f'%(l2,oof(sig_fit(l2))),flush=True)
def ridge(a,b):
    med,mu,sc=fit_preprocessor(R[a]); xa=transform_features(R[a],med,mu,sc); xb=transform_features(R[b],med,mu,sc)
    m=RidgeCV(alphas=np.logspace(-2,5,30)).fit(xa,target_to_logit(y[a]))
    return logits_to_score(m.predict(xb))
print('ridge (logit target, RidgeCV) OOF RMSE %.4f'%oof(ridge),flush=True)
X=pd.DataFrame(R,columns=feats)
for depth in [3,4]:
    for l2r in [30]:
        def cb(a,b):
            m=CatBoostRegressor(iterations=800,learning_rate=0.03,depth=depth,l2_leaf_reg=l2r,loss_function='RMSE',verbose=0,random_seed=434089,thread_count=4)
            m.fit(X.iloc[a],y[a]); return m.predict(X.iloc[b])
        print('catboost depth=%d l2_leaf_reg=%d (800 it, no early stop) OOF RMSE %.4f'%(depth,l2r,oof(cb)),flush=True)
