import joblib  ### para guardar modelos

f_final = joblib.load("salidas\\rf_final.pkl")
m_lreg = joblib.load("salidas\\m_lreg.pkl")
list_cat=joblib.load("salidas\\list_cat.pkl")
list_dummies=joblib.load("salidas\\list_dummies.pkl")
var_names=joblib.load("salidas\\var_names.pkl")
scaler=joblib.load("salidas\\scaler.pkl") 

import a_funciones as funciones  ###archivo de funciones propias
import pandas as pd ### para manejo de datos
import sqlite3 as sql
import joblib
import openpyxl ## para exportar a excel
import numpy as np


###### el despliegue consiste en dejar todo el código listo para una ejecucion automática en el periodo definido:
###### en este caso se ejecutara el proceso de entrenamiento y prediccion anualmente.
if __name__=="__main__":


    ### conectarse a la base de datos ###
    conn=sql.connect("data\\db_empleados")
    cur=conn.cursor()

    ### Ejecutar sql de preprocesamiento inicial y juntarlo 
    #### con base de preprocesamiento con la que se entrenó para evitar perdida de variables por conversión a dummies

    funciones.ejecutar_sql('Preprocesamiento.sql',cur) ### con las fechas actualizadas explicativas 2023- predecir 2024
    df=pd.read_sql('''select  * from tabla_completa2''',conn)

  
    ####Otras transformaciones en python (imputación, dummies y seleccion de variables)
    df_t= funciones.preparar_datos(df)


    ##Cargar modelo y predecir
    m_dtc = joblib.load("salidas\\dtc_final.pkl")
    predicciones=m_dtc.predict(df_t)
    pd_pred=pd.DataFrame(predicciones, columns=['renuncia'])


    ###Crear base con predicciones ####

    perf_pred=pd.concat([df['EmployeeID'],df_t,pd_pred],axis=1)
   
    ####LLevar a BD para despliegue 
    perf_pred.loc[:,['EmployeeID', 'renuncia']].to_sql("renuncia_pred",conn,if_exists="replace") ## llevar predicciones a BD con ID Empleados
    

    ####ver_predicciones_bajas ###
    emp_pred_bajo=perf_pred.sort_values(by=["renuncia"],ascending=True).head(10)
    
    emp_pred_bajo.set_index('EmpID2', inplace=True) 
    pred=emp_pred_bajo.T
    
    coeficientes=pd.DataFrame( np.append(m_lreg.intercept_,m_lreg.coef_) , columns=['coeficientes'])  ### agregar coeficientes
   
    pred.to_excel("salidas\\prediccion.xlsx")   #### exportar predicciones mas bajas y variables explicativas
    coeficientes.to_excel("salidas\\coeficientes.xlsx") ### exportar coeficientes para analizar predicciones
    
    