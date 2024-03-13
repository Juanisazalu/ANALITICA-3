import joblib  ### para guardar modelos

"""f_final = joblib.load("salidas\\dtc_final.pkl")
rfc_final = joblib.load("salidas\\rfc_final.pkl")
#list_cat=joblib.load("salidas\\list_cat.pkl")
list_dummies=joblib.load("salidas\\list_dummies.pkl")
var_names=joblib.load("salidas\\var_names.pkl")
scaler=joblib.load("salidas\\scaler.pkl") 
"""
import a_funciones as funciones  ###archivo de funciones propias
import pandas as pd ### para manejo de datos
import sqlite3 as sql
import joblib
import openpyxl ## para exportar a excel
import numpy as np
"""
tabla2=pd.read_csv("tabla2.csv")


conn=sql.connect("data\\db_basedatos")
cur=conn.cursor()
funciones.ejecutar_sql('Preprocesamiento.sql',cur) ### con las fechas actualizadas explicativas 2023- predecir 2024
df1=pd.read_sql('''select  * from tabla_exploracion''',conn)
df2=pd.read_sql('''select  * from tabla2''',conn)
"""


###### el despliegue consiste en dejar todo el código listo para una ejecucion automática en el periodo definido:
###### en este caso se ejecutara el proceso de entrenamiento y prediccion anualmente.
if __name__=="__main__":


    ### conectarse a la base de datos ###
    """conn=sql.connect("data\\db_basedatos")
    cur=conn.cursor()

    ### Ejecutar sql de preprocesamiento inicial y juntarlo 
    #### con base de preprocesamiento con la que se entrenó para evitar perdida de variables por conversión a dummies

    funciones.ejecutar_sql('Preprocesamiento.sql',cur) ### con las fechas actualizadas explicativas 2023- predecir 2024
    df=pd.read_sql('''select  * from tabla_completa2''',conn)
    """
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

    perf_pred=pd.concat([tabla2['employeeid'],df_t,pd_pred],axis=1)
   
    ####LLevar a BD para despliegue 
    perf_pred.loc[:,['employeeid', 'renuncia']].to_sql("renuncia_pred",conn,if_exists="replace") ## llevar predicciones a BD con ID Empleados
    

    ####ver_predicciones_bajas ###
    emp_pred_bajo=perf_pred.sort_values(by=["renuncia"],ascending=True).head(10)
    
    emp_pred_bajo.set_index('EmpID2', inplace=True) 
    pred=emp_pred_bajo.T
    
    coeficientes=pd.DataFrame( np.append(m_lreg.intercept_,m_lreg.coef_) , columns=['coeficientes'])  ### agregar coeficientes
   
    pred.to_excel("salidas\\prediccion.xlsx")   #### exportar predicciones mas bajas y variables explicativas
    coeficientes.to_excel("salidas\\coeficientes.xlsx") ### exportar coeficientes para analizar predicciones

    