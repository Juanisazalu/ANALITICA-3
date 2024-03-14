#Importación de librerias
import pandas as pd
import sqlite3 as sql #### para bases de datos sql
import a_funciones as funciones
import sys
import numpy as np
from matplotlib.pyplot import figure
import seaborn as sns
from itertools import product
#Ruta 
sys.path
#sys.path.append('c:\\cod\\LEA3_HR\\data') ## este comanda agrega una ruta

#Lectura de archivos
encuesta_empleado=pd.read_csv("data\employee_survey_data.csv")
general_data=pd.read_csv("data\general_data.csv")
encuesta_gerente=pd.read_csv("data\manager_survey.csv")
info_retiros=pd.read_csv("data/retirement_info.csv")

#Vista general
encuesta_empleado.head()
general_data.head()
encuesta_gerente.head()
info_retiros.head()

#Información
encuesta_empleado.info() #describir que variables tienen nulos
general_data.info()
encuesta_gerente.info() #no tiene nulos
info_retiros.info()

#Cambio de formato de fecha
encuesta_empleado["DateSurvey"]
encuesta_empleado["DateSurvey"] = pd.to_datetime(encuesta_empleado["DateSurvey"])

general_data["InfoDate"]
general_data["InfoDate"]=pd.to_datetime(general_data["InfoDate"])
                              
encuesta_gerente["SurveyDate"]
encuesta_gerente["SurveyDate"]=pd.to_datetime(encuesta_gerente["SurveyDate"])

info_retiros["retirementDate"]
info_retiros["retirementDate"]=pd.to_datetime(info_retiros["retirementDate"])

#Valores unicos para las variables exploración inicial 
for tabla in [general_data,encuesta_empleado,encuesta_gerente,info_retiros]:
    print("------------------------------------") #cambio de tabla
    for col in tabla.columns:
        funciones.cat_summary(tabla, col) 
#La exploración anterior arroja que las variables  EmployeeCount, Over18, StandardHours
#son constantes y por lo tanto no aportaran nada al modelo, ademas algunas otras variables
#que no tienen un sentido claro.
    
#Eliminacion de columnas segun la exploración
encuesta_empleado.drop("Unnamed: 0", axis=1, inplace=True)
general_data.drop("Unnamed: 0", axis=1, inplace=True)
encuesta_gerente.drop("Unnamed: 0", axis=1, inplace=True)
info_retiros.drop("Unnamed: 0.1", axis=1, inplace=True)
info_retiros.drop("Unnamed: 0", axis=1, inplace=True)
general_data.drop(["EmployeeCount","Over18","StandardHours"],axis=1, inplace=True)


#Crear base de datos
con=sql.connect("data\\db_basedatos")
cur=con.cursor()

#Cargar bases al DB
encuesta_empleado.to_sql('encuesta_empleado', con, if_exists ="replace")
general_data.to_sql('general_data', con, if_exists ="replace")
encuesta_gerente.to_sql('encuesta_gerente', con, if_exists ="replace")
info_retiros.to_sql('info_retiros', con, if_exists ="replace")

#Cursor
cur.execute("Select name from sqlite_master where type='table'") ### consultar bases de datos
cur.fetchall()

#Ejecutar consulta a la base de datos
funciones.ejecutar_sql('Preprocesamiento.sql',cur)
tabla=pd.read_sql("""select *  from tabla_completa """ , con)

#Relleno de nulos
tabla.isnull().sum()

#Manualmente
tabla["NumCompaniesWorked"]=tabla["NumCompaniesWorked"].apply(lambda x: x if not pd.isnull(x) else int(tabla["NumCompaniesWorked"].median()))
tabla["TotalWorkingYears"]=tabla["TotalWorkingYears"].apply(lambda x: x if not pd.isnull(x) else int(tabla["TotalWorkingYears"].median()))
tabla["EnvironmentSatisfaction"]=tabla["EnvironmentSatisfaction"].apply(lambda x: x if not pd.isnull(x) else int(tabla["EnvironmentSatisfaction"].median()))
tabla["JobSatisfaction"]=tabla["JobSatisfaction"].apply(lambda x: x if not pd.isnull(x) else int(tabla["JobSatisfaction"].median()))
tabla["WorkLifeBalance"]=tabla["WorkLifeBalance"].apply(lambda x: x if not pd.isnull(x) else int(tabla["WorkLifeBalance"].median()))

tabla.isnull().sum()
#Convertir a minusculas el nombre de las columnas
tabla.columns=tabla.columns.str.lower()

#Analisis relacion entre variables numericas como matriz de correlación
figure(figsize=(20,6))
sns.heatmap(tabla.corr(),cmap = sns.cubehelix_palette(as_cmap=True), annot = True, fmt = ".2f")

#Segun la matriz de correlación las variables que tienen mayor correlación con otras variables son:
#yearsatcompany
#totalworkingyears
#performancerating
tabla.drop("yearsatcompany", axis=1, inplace=True)
tabla.drop("totalworkingyears", axis=1, inplace=True)
tabla.drop("performancerating", axis=1, inplace=True)

#Resultado final
figure(figsize=(20,6))
sns.heatmap(tabla.corr(),cmap = sns.cubehelix_palette(as_cmap=True), annot = True, fmt = ".2f")

#Analisis relacion entre variables categoricas
funciones.prueba_chicuadrado(tabla)
#Se elimina 
#Genero
tabla.drop("gender", axis=1, inplace=True)

#Conteo de empleados que renunciaron
len(tabla[tabla["v_objetivo"]==1])

#Eliminación de la columna employeeid
tabla.drop("employeeid", axis=1, inplace=True)

#Se gurda en excel para la exploración y luego para la seleccion de variables
tabla.to_csv('tabla_exploración.csv', index=False) #Carga base para exploración
tabla.info()

#-----------------------------------------------------------------------------------------------
#Preprocesamiento para el test 
#Carga de tabla 
funciones.ejecutar_sql('Preprocesamiento.sql',cur)
tabla2=pd.read_sql("""select *  from tabla_completa2 """ , con)

#Relleno de nulos
len(tabla2.columns)
tabla2.isnull().sum()
tabla2["NumCompaniesWorked"]=tabla2["NumCompaniesWorked"].apply(lambda x: x if not pd.isnull(x) else int(tabla2["NumCompaniesWorked"].median()))
tabla2["EnvironmentSatisfaction"]=tabla2["EnvironmentSatisfaction"].apply(lambda x: x if not pd.isnull(x) else int(tabla2["EnvironmentSatisfaction"].median()))
tabla2["JobSatisfaction"]=tabla2["JobSatisfaction"].apply(lambda x: x if not pd.isnull(x) else int(tabla2["JobSatisfaction"].median()))
tabla2["WorkLifeBalance"]=tabla2["WorkLifeBalance"].apply(lambda x: x if not pd.isnull(x) else int(tabla2["WorkLifeBalance"].median()))
tabla2.isnull().sum()

#Convertir a minúscula 
tabla2.columns=tabla2.columns.str.lower()

#Guardado de tabla para el despliegue
tabla2.to_csv('tabla2.csv', index=False)

