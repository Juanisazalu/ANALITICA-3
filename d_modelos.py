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

# Convierte el array de NumPy a un DataFrame de pandas
columnas_numericas = variables_continuas
categorical_transformer = preprocesador.named_transformers_['cat']
categorias = categorical_transformer.named_steps['onehot'].get_feature_names_out(variables_categorias)

nuevas_columnas = columnas_numericas + list(categorias)

X_df = pd.DataFrame(dfxx, columns=nuevas_columnas)

X_df.shape

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
dfx2.info()



