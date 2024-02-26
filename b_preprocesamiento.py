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
for col in general_data.columns:
    funciones.cat_summary(general_data, col)
 
for col in encuesta_empleado.columns:
    funciones.cat_summary(encuesta_empleado, col)

for col in encuesta_gerente.columns:
    funciones.cat_summary(encuesta_gerente, col)

for col in info_retiros.columns:
    funciones.cat_summary(info_retiros, col)
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

#Cargar bases
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

tabla.info()

#Relleno de nulos
tabla.isnull().sum()
#Con la funcion no da
cat= [x for x in tabla.columns if tabla[x].dtypes =="O"]
funciones.imputar_f(tabla,cat)

#Manualmente
tabla["NumCompaniesWorked"]=tabla["NumCompaniesWorked"].apply(lambda x: x if not pd.isnull(x) else int(tabla["NumCompaniesWorked"].median()))
tabla["TotalWorkingYears"]=tabla["TotalWorkingYears"].apply(lambda x: x if not pd.isnull(x) else int(tabla["TotalWorkingYears"].median()))
tabla["EnvironmentSatisfaction"]=tabla["EnvironmentSatisfaction"].apply(lambda x: x if not pd.isnull(x) else int(tabla["EnvironmentSatisfaction"].median()))
tabla["JobSatisfaction"]=tabla["JobSatisfaction"].apply(lambda x: x if not pd.isnull(x) else int(tabla["JobSatisfaction"].median()))
tabla["WorkLifeBalance"]=tabla["WorkLifeBalance"].apply(lambda x: x if not pd.isnull(x) else int(tabla["WorkLifeBalance"].median()))

tabla.isnull().sum()
#Convertir a minusculas el nombre de las columnas
tabla.columns=tabla.columns.str.lower()

#Analisis relacion entre variables numericas
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
tabla_cat = tabla.select_dtypes(include=['object']).copy()
print(tabla.select_dtypes(include=['object']).columns)

cat_var1 = ('businesstravel', 'department', 'educationfield', 'gender', 'jobrole','maritalstatus')
cat_var2 = ('businesstravel', 'department', 'educationfield', 'gender', 'jobrole','maritalstatus')

cat_var_prod = list(product(cat_var1,cat_var2, repeat = 1))

import scipy.stats as ss
result = []
for i in cat_var_prod:
  if i[0] != i[1]:
      result.append((i[0],i[1],list(ss.chi2_contingency(pd.crosstab(
                            tabla_cat[i[0]], tabla_cat[i[1]])))[1]))
resultados_filtrados = [tupla for tupla in result if tupla[2] > 0.05]

# Imprimir los resultados filtrados
for tupla in resultados_filtrados:
    print(tupla)
    
#Se elimina 
#Genero
tabla.drop("gender", axis=1, inplace=True)
tabla.to_csv('tabla_exploración1.csv', index=False)
#Seleccion de modelos
