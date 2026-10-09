import pandas as pd, numpy as np
from sklearn.model_selection import KFold
from catboost import CatBoostRegressor
df=pd.read_csv('hard_train.csv')
y=df.protection_score.values
X=df.drop(columns=['customer_id','protection_score'])
cats=[c for c in X.columns if X[c].dtype==object]
kf=KFold(5,shuffle=True,random_state=0)
oof=np.zeros(len(y))
for tr,va in kf.split(X):
    m=CatBoostRegressor(iterations=3000,learning_rate=0.03,depth=6,loss_function='RMSE',verbose=0,random_seed=0,thread_count=-1)
    m.fit(X.iloc[tr],y[tr],cat_features=cats,eval_set=(X.iloc[va],y[va]),early_stopping_rounds=200)
    oof[va]=m.predict(X.iloc[va])
    print('fold rmse',np.sqrt(np.mean((oof[va]-y[va])**2)),m.get_best_iteration(),flush=True)
print('OOF RMSE',np.sqrt(np.mean((oof-y)**2)))
m=CatBoostRegressor(iterations=1,verbose=0).fit(X,y,cat_features=cats)
fi=pd.Series(m.get_feature_importance(),index=X.columns).sort_values(ascending=False)
print(fi.head(25).round(2))
