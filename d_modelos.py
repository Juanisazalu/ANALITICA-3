from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import Lasso
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
import pandas as pd
import a_funciones as funciones
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import seaborn as sns
import matplotlib.pyplot as plt


import numpy as np
from sklearn.model_selection import cross_val_predict, cross_val_score, cross_validate


#Carga de taablas
tabla=pd.read_csv("tabla_exploración.csv")
tabla2=pd.read_csv("tabla2.csv")

dfx=tabla.iloc[:,:-1]
dfy=tabla.iloc[:,-1]
"""
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


# Pasar el array a DataFrame
columnas_numericas = variables_continuas
categorical_transformer = preprocesador.named_transformers_['cat']
categorias = categorical_transformer.named_steps['onehot'].get_feature_names_out(variables_categorias)
nuevas_columnas = columnas_numericas + list(categorias)

X_df = pd.DataFrame(dfxx, columns=nuevas_columnas)

#Selección de variables 
mcla = LogisticRegression()
mdtc= DecisionTreeClassifier()
mrfc= RandomForestClassifier()
mgbc=GradientBoostingClassifier()
modelos= [ mcla, mdtc, mrfc, mgbc]

var_names=funciones.sel_variables1(modelos,X_df,dfy,threshold=0.25)
var_names.shape

dfx2=X_df[var_names] ### matriz con variables seleccionadas
X_df.info()
dfx2.info()"""
#Como el profesor
cat=dfx.select_dtypes(include="object").columns
tabla[cat]
num=dfx.select_dtypes(exclude="object").columns
tabla[num]

df_dum=pd.get_dummies(dfx,columns=cat)
df_dum.info()

scaler=StandardScaler()
scaler.fit(df_dum)
xnor=scaler.transform(df_dum)
x=pd.DataFrame(xnor,columns=df_dum.columns)
x.columns
#Selección de variables 
mcla = LogisticRegression()
mdtc= DecisionTreeClassifier()
mrfc= RandomForestClassifier()
mgbc=GradientBoostingClassifier()
modelos= [ mcla, mdtc, mrfc, mgbc]
var_names=funciones.sel_variables(modelos, x, dfy, threshold="1.25*mean")
var_names.shape

xtrain=x[var_names]

#Medir los modelos 
accu_x=funciones.medir_modelos(modelos,"accuracy",x,dfy,20) ## base con todas las variables 
accu_xtrain=funciones.medir_modelos(modelos,"accuracy",xtrain,dfy,20) ### base con variables seleccionadas


accu=pd.concat([accu_x,accu_xtrain],axis=1)
accu.columns=['rl', 'dt', 'rf', 'gb',
       'rl_Sel', 'dt_sel', 'rf_sel', 'gb_Sel']


sns.boxplot(data=accu_x, palette="Set3")
sns.boxplot(data=accu_xtrain, palette="Set3")
sns.boxplot(data=accu, palette="Set3")

df_resultado = pd.DataFrame()
thres=0.5
for i in range(30):
    df_actual=0
    var_names=funciones.sel_variables(modelos, x, dfy, threshold="{}*mean".format(thres))
    xtrain=x[var_names]
    accu_xtrain=funciones.medir_modelos(modelos,"accuracy",xtrain,dfy,5)
    df=accu_xtrain.mean(axis=0)
    df_actual = pd.DataFrame(df, columns=['threshold {}'.format(thres)])
    df_resultado = pd.concat([df_resultado, df_actual], axis=1)
    thres+=0.15
    thres=round(thres,2)


df=df_resultado.T
plt.figure(figsize=(10,10))
sns.lineplot(data=df)
plt.xticks(rotation=45, ha="right")
plt.tight_layout()

df.idxmax(axis=0)
#Los dos modelos a tunear son random_forest y decision_tree