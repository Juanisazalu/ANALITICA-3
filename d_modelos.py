#Carga de librerias
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
from sklearn import tree ###para ajustar arboles de decisión

#Carga de tabla
tabla=pd.read_csv("tabla_exploración.csv")# se llama igual a la tabla cargada en exploración ya que no se modifico ninguna fila o columna en la exploración

#Separación de variables explicativas con variable objetivo.
dfx=tabla.iloc[:,:-1]
dfy=tabla.iloc[:,-1]

#Separación de variables para dummizar, en esta parte se inspeccionaron todas las variables en la separación para confirmar que las variables a dummizar si fueran las correctas
cat=dfx.select_dtypes(include="object").columns
tabla[cat]
num=dfx.select_dtypes(exclude="object").columns
tabla[num]

#get_dummies
df_dum=pd.get_dummies(dfx,columns=cat)
df_dum.info()

#Escalamos las variables
scaler=StandardScaler()
scaler.fit(df_dum)
xnor=scaler.transform(df_dum)
x=pd.DataFrame(xnor,columns=df_dum.columns)
x.columns

#x columns contiene el dataframe ya listo para ser entrenado
#Selección de variables a traves de los modelos descritos a continuación 
mcla = LogisticRegression()
mdtc= DecisionTreeClassifier()
mrfc= RandomForestClassifier()
mgbc=GradientBoostingClassifier()
modelos= [ mcla, mdtc, mrfc, mgbc]
var_names=funciones.sel_variables(modelos, x, dfy, threshold="2.2*mean")
var_names.shape

#Variables elegidas incialmente con el threshold 2.2 
xtrain=x[var_names] #8 variables


#Medir los modelos 
accu_x=funciones.medir_modelos(modelos,"f1",x,dfy,20) ## base con todas las variables 
accu_xtrain=funciones.medir_modelos(modelos,"f1",xtrain,dfy,20) ### base con variables seleccionadas

#Dataframe con los resultados
accu=pd.concat([accu_x,accu_xtrain],axis=1)
accu.columns=['rl', 'dt', 'rf', 'gb',
       'rl_Sel', 'dt_sel', 'rf_sel', 'gb_Sel']

#Promedio para cada modelo
np.mean(accu, axis=0)

#Gráfico de F1 score para modelos con todas las variables y modelos con variables seleccionadas
sns.boxplot(data=accu_x, palette="Set3")
sns.boxplot(data=accu_xtrain, palette="Set3")
sns.boxplot(data=accu, palette="Set3")
#en esta validación cruzada que incluye todas la variables y las variables con threshold de 2.2*mean
#se observa que los modelos DCT y RFC sostienen la misma metrica F1 score, desde aqui se sospecha
#que pueden ser los modelos elegidos

# "función" para buscar el mejor threshold que seleccina las variables para cada modelo.------------------
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

#Gráfica de los resultados __________________________________
df=df_resultado.T
plt.figure(figsize=(10,10))
sns.lineplot(data=df)
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.ylabel("F1 score")
plt.title("Variacion threshold")
#Como se observa los modelos DCT Y RFC no se ven afectados por la cantidad de variables selecionados por los estimadores,
#sin embargo los otros dos modelos al incrementar la exigencia en el threshold dejan caer su rendimiento

#Mejor threshold para cada modelo
df.idxmax(axis=0)
#Sin embargo los threshold con los mejores rendimientos en los modelos DTC Y RFC 
#estan exigiendo tanto a la seleccion de variables que solo aparece una o dos variables.

#Los dos modelos a tunear son random_forest y decision_tree con un trheshold de 2.2 por la media,
modelos= [ mcla, mdtc, mrfc, mgbc]
var_names=funciones.sel_variables(modelos, x, dfy, threshold="2.5*mean")
var_names.shape
#Finalmente se escogen 5 variables para entrenar el modelo, se determino este número
#ya que según la gráfica presentan un desempeño casi igual al threshold con mayor rendimiento

#tabla final
xtrainf=x[var_names] 
#Al final se deja este threshold que da como resultado 5 variables

#Volvemos a medir el modelo pero con las 5 variables y todas las variables
accu_x=funciones.medir_modelos(modelos,"f1",x,dfy,20) ## base con todas las variables 
accu_xtrainf=funciones.medir_modelos(modelos,"f1",xtrainf,dfy,20) ### base con variables seleccionadas

#Dataframe con los resultados
accu=pd.concat([accu_x,accu_xtrainf],axis=1)
accu.columns=['rl', 'dt', 'rf', 'gb',
       'rl_Sel_f', 'dt_sel_f', 'rf_sel_f', 'gb_Sel_f']

#Promedio para cada modelo
np.mean(accu, axis=0)

#Gráfico de F1 score para modelos con todas las variables y modelos con variables seleccionadas
sns.boxplot(data=accu_x, palette="Set3")
sns.boxplot(data=accu_xtrainf, palette="Set3")
sns.boxplot(data=accu, palette="Set3")

#Comparación de modelo con variables completas, con 8 variables y con 5 variables
comp_final=pd.concat([accu, accu_xtrain],axis=1)
comp_final.columns=['rl', 'dt', 'rf', 'gb',
       'rl_Sel_f', 'dt_sel_f', 'rf_sel_f', 'gb_Sel_f','rl_Sel', 'dt_sel', 'rf_sel', 'gb_Sel']
comp_final.mean()
#Gráfico para comparar
sns.boxplot(data=comp_final, palette="Set3")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.ylabel("F1 score")
plt.title("Comparación de seleccion de variables")
# Finalmente al unir los tres grupos de datos, todas las variables, las 8 variables y las 5 variables 
# se obtienen resultados practicamente iguales, por lo que para buscar una explicabilidad mayor se decidio
# la base con las 5 variables elegidas por los estimadores

#Tuning para DTC--------------------------------------------------------
parameters0 = {'class_weight': ['balanced'],
              'max_depth': [ None, 7, 10, 12],
              'max_features': [None,0.4, 1],
              'max_leaf_nodes': [None, 15, 17, 20],
              'min_samples_leaf': [ 1,3,5, 7]}

dtctuning=DecisionTreeClassifier()
grid_search0=GridSearchCV(dtctuning, parameters0, scoring="f1",cv=10, n_jobs=-1)
grid_result0=grid_search0.fit(xtrainf, dfy)

#Resultados
pd.set_option('display.max_colwidth', 100)
resultados0=grid_result0.cv_results_
grid_result0.best_params_
pd_resultados0=pd.DataFrame(resultados0)
#Se mejora la metrica de 0,84 a 0,93, por lo que el tunnig funciono adecuadamente

#Se guarda el modelo
dtc_final=grid_result0.best_estimator_ ### Guardar el modelo con hyperparameter tunning


#Tunning para RFC-------------------------------------------
parameters1 = {'class_weight': ['balanced'],
              'max_depth': [ None, 5, 7, 10],
              'max_features': ['sqrt',0.05,0.4],
              'max_leaf_nodes': [None, 9, 15, 17],
              'min_samples_leaf': [ 1,3, 7],
              'n_estimators': [ 100, 150, 200]}

"""parameters1 = {
              'max_depth': [5, 7, 10],
              'max_features': ['sqrt',0.05,0.4],
              'max_leaf_nodes': [ 9, 15, 17],
              'min_samples_leaf': [ 1,3,5, 7],
              'n_estimators': [ 100, 150, 200]}
"""
rfctuning=RandomForestClassifier()
grid_search1=GridSearchCV(rfctuning, parameters1, scoring="f1",cv=10, n_jobs=-1)
grid_result1=grid_search1.fit(xtrainf, dfy)

#Resultados
pd.set_option('display.max_colwidth', 100)
resultados1=grid_result1.cv_results_
grid_result1.best_params_
pd_resultados1=pd.DataFrame(resultados1)
pd_resultados1[["params","mean_test_score"]].sort_values(by="mean_test_score", ascending=False)
# El modelo mejoro de 0,85 a 0,93 en la métrica de evaluación

#Se guarda el modelo
rfc_final=grid_result1.best_estimator_ ### Guardar el modelo con hyperparameter tunning

#Testear el modelo ambos modelos-----------------------------------------------------
eval=cross_validate(rfc_final,xtrainf,dfy,cv=20,scoring="f1",return_train_score=True)
eval2=cross_validate(dtc_final,xtrainf,dfy,cv=20,scoring="f1",return_train_score=True)

#Para RFC
#convertir resultado de evaluación entrenamiento y evaluación en data frame para RFC
train_rfc=pd.DataFrame(eval['train_score'])
test_rfc=pd.DataFrame(eval['test_score'])
train_test_rfc=pd.concat([train_rfc, test_rfc],axis=1)
train_test_rfc.columns=['train_score','test_score']
ax=train_test_rfc.plot()
ax.set_ylim([0, 1])
#El modelo generaliza de forma correcta los datos

#PARA DTC
# convertir resultado de evaluación entrenamiento y evaluación en data frame para DTC
train_dtc=pd.DataFrame(eval2['train_score'])
test_dtc=pd.DataFrame(eval2['test_score'])
train_test_dtc=pd.concat([train_dtc, test_dtc],axis=1)
train_test_dtc.columns=['train_score','test_score']
ax=train_test_dtc.plot()
ax.set_ylim([0, 1])
ax.set_ylabel("F1 score")
ax.set_title("Evaluacion de generalización del modelo")
#El modelo generaliza de forma correcta los datos

#Cálculo de las medias
train_test_rfc["test_score"].mean()
train_test_dtc["test_score"].mean()

#Se escoge el modelo de árbol de decisión en el informe se explica por que----------------------------------------------------------
# #Análisis modelos para  DTC
print ("Train - Accuracy :", metrics.accuracy_score(dfy, dtc_final.predict(xtrainf)))
print ("Train - classification report:\n", metrics.classification_report(dfy, dtc_final.predict(xtrainf)))
#El informe se explican las metricas

# Matriz de confusión
cm= confusion_matrix(dfy, dtc_final.predict(xtrainf))
# Visualización de la matriz de confusión
cm_display = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels=['No renuncia', 'renuncia'])
cm_display.plot()
plt.title("Matriz de confusion")
plt.show()

#Gráfica del árbol
plt.figure(figsize=(200,100))
tree.plot_tree(dtc_final,fontsize=9,impurity=False,filled=True, feature_names=xtrainf.columns)
plt.show()

#Guardado de objetos para el despliegue
#------------------------------------------------------
joblib.dump(dtc_final, "salidas\\dtc_final.pkl") ## 
joblib.dump(rfc_final, "salidas\\rfc_final.pkl") ## 
#joblib.dump(cat, "salidas\\list_cat.pkl") ### para realizar imputacion
joblib.dump(cat, "salidas\\list_dummies.pkl")  ### para convertir a dummies
joblib.dump(var_names, "salidas\\var_names.pkl")  ### para variables con que se entrena modelo
joblib.dump(scaler, "salidas\\scaler.pkl") ## 


