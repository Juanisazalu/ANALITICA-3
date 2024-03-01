from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import Lasso
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
import pandas as pd
#Carga de taablas
tabla=pd.read_csv("tabla_exploración.csv")
tabla2=pd.read_csv("tabla2.csv")

dfx=tabla.iloc[:,:-1]
dfy=tabla.iloc[:,-1]
variables_categorias = list(dfx.select_dtypes(include="object"))
variables_continuas = list(dfx.select_dtypes(exclude="object"))
categorical_transformer=Pipeline(steps=[("onehot", OneHotEncoder(handle_unknown='ignore'))])
numerical_transformer=Pipeline(steps=[("standarscaler", StandardScaler())])
preprocesador=ColumnTransformer(transformers=[("num",numerical_transformer, variables_continuas),
                                ("cat",categorical_transformer,variables_categorias)])
pipeline=Pipeline(steps=[("transformacion",preprocesador)])
dfxx=pipeline.fit_transform(dfx)
#Metodo integrado
select=SelectFromModel(Lasso(alpha = 0.001, max_iter=10000),max_features=30)
select.fit(dfxx,dfy)
#Coeficientes del estimador, los mas cercanos a cero son eliminados
select.estimator_.coef_

xnew=select.get_support()
xtrain=dfxx[:,xnew]
#falta para el xtest
dftestx=pipeline.transform(tabla2)
xtest=dftestx[:,xnew]



