"""Черновой скрипт: подбор регуляризации (сигмоида / ridge / CatBoost). Запуск из корня репозитория.

Что делает (по строкам ниже):
  1. Читает hard_train.csv, собирает матрицу числовых признаков.
  2. oof(fn) — считает OOF RMSE для любой модели fn(train_idx, valid_idx).
  3. sig_fit(l2) — сигмоидально-линейная модель с заданным L2 (цикл по L2 сейчас
     пуст — значения 1e2..1e6 уже проверены, итог в RESULTS.md).
  4. ridge — RidgeCV на логитах цели (OOF RMSE ~10.25).
  5. CatBoost depth 3/4 с сильной регуляризацией (проверка «а вдруг малая глубина лучше»).

ИСПРАВЛЕНИЕ: раньше здесь было `import train as T` + `T.fit_sigmoid_index`,
но train.py теперь обучает CatBoost и такой функции в нём нет — скрипт падал.
Теперь fit_sigmoid_index импортируется из model_utils (та же математика).
"""

import numpy as np, pandas as pd, warnings  # массивы, таблицы, управление предупреждениями
warnings.filterwarnings("ignore")  # глушим предупреждения библиотек для чистого вывода
from sklearn.model_selection import KFold  # разбиение на фолды
from sklearn.linear_model import RidgeCV  # ridge с автовыбором alpha
from catboost import CatBoostRegressor  # бустинг CatBoost
from model_utils import fit_preprocessor, numeric_matrix, transform_features, get_numeric_feature_names, logits_to_score, target_to_logit, fit_sigmoid_index  # наши общие функции (fit_sigmoid_index — из model_utils, НЕ из train!)
df=pd.read_csv('hard_train.csv'); y=df.protection_score.values  # читаем train; y — цели в numpy
feats=get_numeric_feature_names(df); R=numeric_matrix(df,feats)  # имена числовых признаков + сырая матрица
kf=KFold(5,shuffle=True,random_state=434089)  # единые перемешанные фолды
def oof(fn):  # OOF RMSE для модели fn(a,b): обучить на a, предсказать b
    p=np.zeros(len(y))  # нули под предсказания
    for a,b in kf.split(R):  # по всем фолдам
        p[b]=fn(a,b)  # предсказать valid-часть и положить на место
    return np.sqrt(np.mean((p-y)**2))  # RMSE по всем строкам
def sig_fit(l2):  # фабрика: вернуть модель-сигмоиду с силой L2 = l2
    def f(a,b):  # обучить на a, предсказать b
        med,mu,sc=fit_preprocessor(R[a]); xa=transform_features(R[a],med,mu,sc); xb=transform_features(R[b],med,mu,sc)  # препроцессор на train, применить к обеим частям
        c,i,_=fit_sigmoid_index(xa,y[a],l2_alpha=l2)  # подобрать веса (w, b); третье значение не нужно
        return logits_to_score(xb@c+i)  # 100*sigmoid(Xb@w + b)
    return f
for l2 in []:  # ПУСТОЙ список: перебор L2 уже сделан (см. RESULTS.md), повторять не надо
    print('sigmoid L2=%g  OOF RMSE %.4f'%(l2,oof(sig_fit(l2))),flush=True)
def ridge(a,b):  # RidgeCV на логитах: обучить на a, предсказать b
    med,mu,sc=fit_preprocessor(R[a]); xa=transform_features(R[a],med,mu,sc); xb=transform_features(R[b],med,mu,sc)  # препроцессор внутри фолда
    m=RidgeCV(alphas=np.logspace(-2,5,30)).fit(xa,target_to_logit(y[a]))  # учимся предсказывать ЛОГИТЫ цели
    return logits_to_score(m.predict(xb))  # возвращаем проценты 0-100
print('ridge (logit target, RidgeCV) OOF RMSE %.4f'%oof(ridge),flush=True)  # замер ridge
X=pd.DataFrame(R,columns=feats)  # та же матрица как DataFrame (CatBoost любит .iloc)
for depth in [3,4]:  # перебор глубины деревьев
    for l2r in [30]:  # перебор L2 на листьях (одно значение — 30)
        def cb(a,b):  # CatBoost: обучить на a (без ранней остановки!), предсказать b
            m=CatBoostRegressor(iterations=800,learning_rate=0.03,depth=depth,l2_leaf_reg=l2r,loss_function='RMSE',verbose=0,random_seed=434089,thread_count=4)  # 800 деревьев, сильная регуляризация
            m.fit(X.iloc[a],y[a]); return m.predict(X.iloc[b])  # обучить и предсказать
        print('catboost depth=%d l2_leaf_reg=%d (800 it, no early stop) OOF RMSE %.4f'%(depth,l2r,oof(cb)),flush=True)  # замер
