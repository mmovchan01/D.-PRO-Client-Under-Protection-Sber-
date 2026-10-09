import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
from gplearn.genetic import SymbolicRegressor
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
df=pd.read_csv('hard_train.csv'); y=df.protection_score.values
top=['cyber_protection','insurance_products','digital_behavior_score','two_factor_auth','internet_activity']
X=df[top].astype(float).fillna(df[top].median()).values
kf=KFold(5,shuffle=True,random_state=42)
rm=lambda p: np.sqrt(np.mean((p-y)**2))
lin=np.zeros(len(y)); sr=np.zeros(len(y))
for a,b in kf.split(X):
    m=make_pipeline(StandardScaler(),LinearRegression()).fit(X[a],y[a]); lin[b]=m.predict(X[b])
    s=SymbolicRegressor(population_size=2000,generations=20,stopping_criteria=0.0,p_crossover=0.6,p_subtree_mutation=0.1,p_hoist_mutation=0.05,p_point_mutation=0.1,max_samples=0.9,parsimony_coefficient=0.001,function_set=('add','sub','mul','div'),feature_names=top,random_state=42,n_jobs=1)
    sc=StandardScaler().fit(X[a]); s.fit(sc.transform(X[a]),y[a]); sr[b]=s.predict(sc.transform(X[b]))
print('linear on top5 CV RMSE: %.3f'%rm(lin))
print('symbolic (gplearn) on top5 CV RMSE: %.3f'%rm(sr))
sc=StandardScaler().fit(X); s=SymbolicRegressor(population_size=2000,generations=20,parsimony_coefficient=0.001,function_set=('add','sub','mul','div'),feature_names=top,random_state=42,n_jobs=1).fit(sc.transform(X),y)
print('best formula (standardized inputs):', s._program)
