import pandas as pd
import sqlite3 as sql #### para bases de datos sql
import a_funciones as funciones
import sys
import numpy as np
from matplotlib.pyplot import figure
import seaborn as sns
from itertools import product

sys.path
#sys.path.append('c:\\cod\\LEA3_HR\\data') ## este comanda agrega una ruta

encuesta_empleado=pd.read_csv("data\employee_survey_data.csv")
general_data=pd.read_csv("data\general_data.csv")
encuesta_gerente=pd.read_csv("data\manager_survey.csv")
info_retiros=pd.read_csv("data/retirement_info.csv")

encuesta_empleado
general_data
encuesta_gerente
info_retiros

#.info()
encuesta_empleado.info() #describir que variables tienen nulos
general_data.info()
encuesta_gerente.info() #no tiene nulos
info_retiros.info()

#los codigos de acciones se pasan a object generalmente
#Cambio de formato de fecha
encuesta_empleado["DateSurvey"]
encuesta_empleado["DateSurvey"] = pd.to_datetime(encuesta_empleado["DateSurvey"])

general_data["InfoDate"]
general_data["InfoDate"]=pd.to_datetime(general_data["InfoDate"])
general_data["EmployeeCount"].value_counts()
general_data["Over18"].value_counts()
#justificar
#variables a eliminar de entrada:EmployeeCount, Over18                               
encuesta_gerente["SurveyDate"]
encuesta_gerente["SurveyDate"]=pd.to_datetime(encuesta_gerente["SurveyDate"])

info_retiros["retirementDate"]
info_retiros["retirementDate"]=pd.to_datetime(info_retiros["retirementDate"])

#Eliminacion de columnas
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

cur.execute("Select name from sqlite_master where type='table'") ### consultar bases de datos
cur.fetchall()

#REALIZAR EXPLORACIONES INICIALES
for col in general_data.columns:
    funciones.cat_summary(general_data, col)
    #jobrole tiene 7 categorias
#Analizar que se puede categorizar
for col in encuesta_empleado.columns:
    funciones.cat_summary(encuesta_empleado, col)
#jobsatisfactio ??? tiene niveles de 1,3, 

for col in encuesta_gerente.columns:
    funciones.cat_summary(encuesta_gerente, col)

for col in info_retiros.columns:
    funciones.cat_summary(info_retiros, col)

#Hacer consultas
pd.read_sql("""select *  
                            from info_retiros 
                            WHERE strftime('%Y',retirementDate) = '2015' """, con)


funciones.ejecutar_sql('Preprocesamiento.sql',cur)

tabla=pd.read_sql("""select *  
                     from tabla_completa """ , con)

tabla.info()
#Relleno de nulos
#Con la funcion no da
cat= [x for x in tabla.columns if tabla[x].dtypes =="O"]
funciones.imputar_f(tabla,cat)
tabla.isnull().sum()

#Manualmente
tabla["NumCompaniesWorked"]=tabla["NumCompaniesWorked"].apply(lambda x: x if not pd.isnull(x) else int(tabla["NumCompaniesWorked"].median()))
tabla["TotalWorkingYears"]=tabla["TotalWorkingYears"].apply(lambda x: x if not pd.isnull(x) else int(tabla["TotalWorkingYears"].median()))
tabla["EnvironmentSatisfaction"]=tabla["EnvironmentSatisfaction"].apply(lambda x: x if not pd.isnull(x) else int(tabla["EnvironmentSatisfaction"].median()))
tabla["JobSatisfaction"]=tabla["JobSatisfaction"].apply(lambda x: x if not pd.isnull(x) else int(tabla["JobSatisfaction"].median()))
tabla["WorkLifeBalance"]=tabla["WorkLifeBalance"].apply(lambda x: x if not pd.isnull(x) else int(tabla["WorkLifeBalance"].median()))

tabla.columns=tabla.columns.str.lower()

#Analisis relacion entre variables numericas

figure(figsize=(20,6))
sns.heatmap(tabla.corr(),cmap = sns.cubehelix_palette(as_cmap=True), annot = True, fmt = ".2f")



#Se elimina
#yearsatcompany
#totalworkingyears
#performancerating
tabla.drop("yearsatcompany", axis=1, inplace=True)
tabla.drop("totalworkingyears", axis=1, inplace=True)
tabla.drop("performancerating", axis=1, inplace=True)

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

#Seleccion de modelos
