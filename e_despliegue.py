import joblib  ### para guardar modelos
import a_funciones as funciones  ###archivo de funciones propias
import pandas as pd ### para manejo de datos
import sqlite3 as sql
import joblib
import openpyxl ## para exportar a excel
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
# Conectar a la base de datos
con=sql.connect("data\\db_basedatos")
cur=con.cursor()

###### el despliegue consiste en dejar todo el código listo para una ejecucion automática en el periodo definido:
###### en este caso se ejecutara el proceso de entrenamiento y prediccion anualmente.
if __name__=="__main__":
    #Carga de tablas
    tabla2=pd.read_csv("tabla2.csv")
    xtest=tabla2.drop("employeeid",axis=1)
    ####Otras transformaciones en python (imputación, dummies y seleccion de variables)
    df_t= funciones.preparar_datos(xtest)
    
    ##Cargar modelo y predecir
    m_dtc = joblib.load("salidas\\dtc_final.pkl")
    predicciones=m_dtc.predict(df_t)
    pd_pred=pd.DataFrame(predicciones, columns=['renuncia'])
    #Crear base con predicciones 
    perf_pred = pd.concat([tabla2['employeeid'], pd_pred], axis=1)
    # Filtrar solo los empleados que van a renunciar (renuncia == 1)
    condicion2 = perf_pred[perf_pred["renuncia"] == 1]
    # Traer de 'tabla2' los valores de las variables originales para los empleados que van a renunciar
    variables_desertores = tabla2.loc[condicion2.index, df_t.columns.tolist() + ['employeeid']]    # Concatenar los valores de las variables originales y la condición de renuncia en un nuevo DataFrame 'tabla_final'
    #LLevar a BD para despliegue 
    variables_desertores.to_sql("prediccion_renuncias",con,if_exists="replace") ## llevar predicciones a BD con ID Empleados
    
    #Importancia de las variables del modelo
    importances = m_dtc.feature_importances_
    feature_importances_df = pd.DataFrame({'Feature': df_t.columns, 'Importance': importances})
    feature_importances_df = feature_importances_df.sort_values(by='Importance', ascending=False)
    #Guardado en BD
    feature_importances_df.to_sql("importancia_variables",con,if_exists="replace") ## llevar predicciones a BD con ID Empleados

    #Guardar en excel
    variables_desertores.to_excel("salidas\\prediccion.xlsx")   #### exportar predicciones mas bajas y variables explicativas
    feature_importances_df.to_excel("salidas\\importancia_variables.xlsx") ### exportar coeficientes para analizar predicciones
    #Ruta de la decisión
    m_dtc.tree_.node_count
    dense_matrix = pd.DataFrame(m_dtc.decision_path(df_t).todense())
    ruta= pd.concat([perf_pred, dense_matrix,], axis=1)
    ruta=ruta[ruta["renuncia"]==1]
    ruta.to_sql("Ruta_decision",con,if_exists="replace") ## llevar predicciones a BD con ID Empleados
    #Al tener un arbol de este tamaño se hace complejo observar la ruta de decision
    #Guardado en excel
    ruta=ruta.head()
    ruta.to_excel("salidas\\Ruta_decisión.xlsx") ### exportar coeficientes para analizar predicciones
    
    #Tabla estandarizada para interpretacion
    estandar=pd.concat([perf_pred, df_t], axis=1)
    estandar=estandar.head()
    #Guardado en excel
    estandar.to_excel("salidas\\valores_estandarizados.xlsx")
#consultas para interpretar la ruta de decision del empleado dos en el documento
#En este caso observamos el nodo 1
m_dtc.tree_.feature[1]  
m_dtc.tree_.threshold[1]
m_dtc.tree_.n_node_samples[1]
m_dtc.tree_.value[1]
#Graficos relacionados a la clasificacion, se buscaria automatizar para que aparezcan automaticamente segun el numero de variables seleccionadas
tabla_graficos=pd.concat([pd_pred, tabla2], axis=1)
tablagraf=tabla_graficos.loc[:, df_t.columns.tolist() + ['employeeid',"renuncia"]]

sns.boxplot(y=tablagraf.age, x=tablagraf.renuncia)
plt.ylabel("Edad")
plt.xlabel("Renuncia")
plt.title("Boxplot edad y renuncia")
sns.boxplot(y=tablagraf.monthlyincome, x=tablagraf.renuncia)
plt.ylabel("Ingresos mensuales")
plt.xlabel("Renuncia")
plt.title("Boxplot ingresos mensuales y renuncia")
sns.boxplot(y=tablagraf.jobsatisfaction, x=tablagraf.renuncia)
plt.ylabel("Satisfacción laboral")
plt.xlabel("Renuncia")
plt.title("Boxplot satisfacción laboral y renuncia")
sns.boxplot(y=tablagraf.yearssincelastpromotion, x=tablagraf.renuncia)
plt.ylabel("Años desde el ultimo ascenso")
plt.xlabel("Renuncia")
plt.title("Boxplot años desde el ultimo ascenso y renuncia")
sns.boxplot(y=tablagraf.yearswithcurrmanager, x=tablagraf.renuncia)
plt.ylabel("Años con el mismo jefe")
plt.xlabel("Renuncia")
plt.title("Boxplot años con el mismo jefe y renuncia")
#La interpretacion de los graficos esta ligado a las estrategias propuestas en el documento
