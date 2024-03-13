import joblib  ### para guardar modelos
import a_funciones as funciones  ###archivo de funciones propias
import pandas as pd ### para manejo de datos
import sqlite3 as sql
import joblib
import openpyxl ## para exportar a excel
import numpy as np

con=sql.connect("data\\db_basedatos")
cur=con.cursor()

###### el despliegue consiste en dejar todo el código listo para una ejecucion automática en el periodo definido:
###### en este caso se ejecutara el proceso de entrenamiento y prediccion anualmente.
if __name__=="__main__":
    
    tabla2=pd.read_csv("tabla2.csv")
    xtest=tabla2.drop("employeeid",axis=1)
    ####Otras transformaciones en python (imputación, dummies y seleccion de variables)
    df_t= funciones.preparar_datos(xtest)
    
    ##Cargar modelo y predecir
    m_dtc = joblib.load("salidas\\dtc_final.pkl")
    predicciones=m_dtc.predict(df_t)
    pd_pred=pd.DataFrame(predicciones, columns=['renuncia'])
    pd_pred.value_counts()
    ###Crear base con predicciones ####
    perf_pred = pd.concat([tabla2['employeeid'], pd_pred], axis=1)
    # Filtrar solo los empleados que van a renunciar (renuncia == 1)
    condicion2 = perf_pred[perf_pred["renuncia"] == 1]
    # Traer de 'tabla2' los valores de las variables originales para los empleados que van a renunciar
    variables_desertores = tabla2.loc[condicion2.index, df_t.columns.tolist() + ['employeeid']]    # Concatenar los valores de las variables originales y la condición de renuncia en un nuevo DataFrame 'tabla_final'
    ####LLevar a BD para despliegue 
    variables_desertores.to_sql("prediccion_renuncias",con,if_exists="replace") ## llevar predicciones a BD con ID Empleados
    
    importances = m_dtc.feature_importances_
    feature_importances_df = pd.DataFrame({'Feature': df_t.columns, 'Importance': importances})
    feature_importances_df = feature_importances_df.sort_values(by='Importance', ascending=False)
    
    feature_importances_df.to_sql("importancia_variables",con,if_exists="replace") ## llevar predicciones a BD con ID Empleados

    #Guardar en excel
    variables_desertores.to_excel("salidas\\prediccion.xlsx")   #### exportar predicciones mas bajas y variables explicativas
    feature_importances_df.to_excel("salidas\\importancia_variables.xlsx") ### exportar coeficientes para analizar predicciones

    