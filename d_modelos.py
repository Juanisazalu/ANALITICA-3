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
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import f1_score
from sklearn.model_selection import RandomizedSearchCV
import joblib  ### para guardar modelos
import numpy as np
from sklearn.model_selection import cross_val_predict, cross_val_score, cross_validate
from sklearn import metrics
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

#Carga de taablas
tabla=pd.read_csv("tabla_exploración.csv")
tabla2=pd.read_csv("tabla2.csv")

dfx=tabla.iloc[:,:-1]
dfy=tabla.iloc[:,-1]

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
var_names=funciones.sel_variables(modelos, x, dfy, threshold="2.2*mean")
var_names.shape

xtrain=x[var_names]

#Medir los modelos 
accu_x=funciones.medir_modelos(modelos,"f1",x,dfy,20) ## base con todas las variables 
accu_xtrain=funciones.medir_modelos(modelos,"f1",xtrain,dfy,20) ### base con variables seleccionadas


accu=pd.concat([accu_x,accu_xtrain],axis=1)
accu.columns=['rl', 'dt', 'rf', 'gb',
       'rl_Sel', 'dt_sel', 'rf_sel', 'gb_Sel']
np.mean(accu, axis=0)

sns.boxplot(data=accu_x, palette="Set3")
sns.boxplot(data=accu_xtrain, palette="Set3")
sns.boxplot(data=accu, palette="Set3")

# funcion para buscar el mejor threshold para cada modelo.
df_resultado = pd.DataFrame()
thres=0.5
for i in range(30):
    df_actual=0
    var_names=funciones.sel_variables(modelos, x, dfy, threshold="{}*mean".format(thres))
    xtrain=x[var_names]
    accu_xtrain=funciones.medir_modelos(modelos,"f1",xtrain,dfy,10)
    df=accu_xtrain.mean(axis=0)
    df_actual = pd.DataFrame(df, columns=['threshold {}'.format(thres)])
    df_resultado = pd.concat([df_resultado, df_actual], axis=1)
    thres+=0.15
    thres=round(thres,2)

#Grafica 
df=df_resultado.T
plt.figure(figsize=(10,10))
sns.lineplot(data=df)
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
#threshol para cada modelo.
df.idxmax(axis=0)
#Los dos modelos a tunear son random_forest y decision_tree

modelos= [ mcla, mdtc, mrfc, mgbc]
var_names=funciones.sel_variables(modelos, x, dfy, threshold="2.2*mean")
var_names.shape
xtrainf=x[var_names]
#al final se deja este valor que da como resultado 7 variables

#Tuning para DTC
parameters0 = {'class_weight': ['balanced'],
              'max_depth': [ None, 7, 10, 12],
              'max_features': [None,0.4, 1],
              'max_leaf_nodes': [None, 15, 17, 20],
              'min_samples_leaf': [ 1,3,5, 7]}

dtctuning=DecisionTreeClassifier()
grid_search0=GridSearchCV(dtctuning, parameters0, scoring="f1",cv=10, n_jobs=-1)
grid_result0=grid_search0.fit(xtrainf, dfy)

pd.set_option('display.max_colwidth', 100)
resultados0=grid_result0.cv_results_
grid_result0.best_params_
pd_resultados0=pd.DataFrame(resultados0)
pd_resultados0[["params","mean_test_score"]].sort_values(by="mean_test_score", ascending=False)

dtc_final=grid_result0.best_estimator_ ### Guardar el modelo con hyperparameter tunning

"""
parameters1 = {'class_weight': ['balanced'],
              'max_depth': [ None, 5, 7, 10],
              'max_features': ['sqrt',0.05,0.4],
              'max_leaf_nodes': [None, 9, 15, 17],
              'min_samples_leaf': [ 1,3,5, 7],
              'n_estimators': [ 100, 150, 200]}
"""

#Tunnig para RFC
parameters1 = {
              'max_depth': [5, 7, 10],
              'max_features': ['sqrt',0.05,0.4],
              'max_leaf_nodes': [ 9, 15, 17],
              'min_samples_leaf': [ 1,3,5, 7],
              'n_estimators': [ 100, 150, 200]}

rfctuning=RandomForestClassifier()
grid_search1=GridSearchCV(rfctuning, parameters1, scoring="f1",cv=10, n_jobs=-1)
grid_result1=grid_search1.fit(xtrainf, dfy)

pd.set_option('display.max_colwidth', 100)
resultados1=grid_result1.cv_results_
grid_result1.best_params_
pd_resultados1=pd.DataFrame(resultados1)
pd_resultados1[["params","mean_test_score"]].sort_values(by="mean_test_score", ascending=False)

rfc_final=grid_result1.best_estimator_ ### Guardar el modelo con hyperparameter tunning

#Testear el modelo
eval=cross_validate(rfc_final,xtrainf,dfy,cv=10,scoring="f1",return_train_score=True)
eval2=cross_validate(dtc_final,xtrainf,dfy,cv=10,scoring="f1",return_train_score=True)


#### convertir resultado de evaluacion entrenamiento y evaluacion en data frame para RF
train_rfc=pd.DataFrame(eval['train_score'])
test_rfc=pd.DataFrame(eval['test_score'])
train_test_rfc=pd.concat([train_rfc, test_rfc],axis=1)
train_test_rfc.columns=['train_score','test_score']

#### convertir resultado de evaluacion entrenamiento y evaluacion en data frame para RL
train_dtc=pd.DataFrame(eval2['train_score'])
test_dtc=pd.DataFrame(eval2['test_score'])
train_test_dtc=pd.concat([train_dtc, test_dtc],axis=1)
train_test_dtc.columns=['train_score','test_score']

train_test_rfc["test_score"].mean()
train_test_dtc["test_score"].mean()

#Se escoge el modelo de arbol de decision 
#Analisis modelos para random forest
print ("Train - Accuracy :", metrics.accuracy_score(dfy, dtc_final.predict(xtrainf)))
print ("Train - classification report:\n", metrics.classification_report(dfy, dtc_final.predict(xtrainf)))

# Matriz de confusion
cm= confusion_matrix(dfy, dtc_final.predict(xtrainf))
# Visualización de la matriz de confusion
cm_display = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels=['No renuncia', 'renuncia'])
cm_display.plot()
plt.show()

##### Despliegue: Mirar importancia de variables para tomar acciones ###
importances = dtc_final.feature_importances_

# Crear un DataFrame con las importancias y los nombres de las variables
feature_importances_df = pd.DataFrame({'Feature': xtrainf.columns, 'Importance': importances})

# Ordenar el DataFrame por importancia en orden descendente
feature_importances_df = feature_importances_df.sort_values(by='Importance', ascending=False)

# Visualizar las importancias de las variables
plt.figure(figsize=(10, 6))
plt.barh(feature_importances_df['Feature'], feature_importances_df['Importance'], color='skyblue')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Feature Importances')
plt.gca().invert_yaxis()  # Invertir el eje y para mostrar la importancia más alta arriba
plt.show()

# Supongamos que ya tienes un modelo entrenado 'clf' y tus datos de prueba 'X_test'
predictions = dtc_final.predict(xtrainf)

# Para obtener el camino de decisión para una instancia específica (por ejemplo, la primera)
instance_path = dtc_final.decision_path(xtrainf.iloc[[0]])

# Imprimir el camino de decisión
print(f"Camino de decisión para la instancia 0: {instance_path}")

feature_names = xtrainf.columns.tolist()
# También puedes visualizar el árbol para entender mejor las decisiones
from sklearn.tree import plot_tree
plt.figure(figsize=(20, 10))
plot_tree(dtc_final, feature_names=feature_names, filled=True)
plt.show()

joblib.dump(dtc_final, "salidas\\dtc_final.pkl") ## 
joblib.dump(rfc_final, "salidas\\m_lreg.pkl") ## 
#joblib.dump(cat, "salidas\\list_cat.pkl") ### para realizar imputacion
joblib.dump(cat, "salidas\\list_dummies.pkl")  ### para convertir a dummies
joblib.dump(var_names, "salidas\\var_names.pkl")  ### para variables con que se entrena modelo
joblib.dump(scaler, "salidas\\scaler.pkl") ## 
