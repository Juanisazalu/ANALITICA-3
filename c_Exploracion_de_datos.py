#### Cargar paquetes siempre al inicio
import pandas as pd ### para manejo de datos
import sqlite3 as sql #### para bases de datos sql
import a_funciones as funciones ### archivo de funciones propias
import matplotlib as mpl ## gráficos
import matplotlib.pyplot as plt ### gráficos
from pandas.plotting import scatter_matrix  ## para matriz de correlaciones
from sklearn import tree ###para ajustar arboles de decisión
from sklearn.tree import export_text ## para exportar reglas del árbol
import seaborn as sns

## Visualization
import matplotlib.pyplot as plt
import seaborn as sns
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff


tabla=pd.read_csv("tabla_exploración.csv")
tabla.columns

cat= tabla.select_dtypes(include='object').columns
continuas= tabla.select_dtypes(exclude='object').columns

### explorar variable respuesta ###
# se crea el dataset y se muestra en un grafico de barras su comportamiento
fig=tabla.v_objetivo.hist(bins=20,ec='black') ## no hay atípicos
fig.grid(False)
plt.show()

### Dimensiones del dataset
tabla.shape

# Número de datos ausentes por variable (no se encuentran datos nulos)
tabla.isna().sum().sort_values()

tabla.describe()

tabla.info()


#Exploración variables categóricas en un diagrama de tortas
plt.figure(figsize=(25, 10))
#Se puede observar en el siguiente gráfico mayor que cerca del 70 % de los empleados viajan poco,
# el 20% frecuentemente y el 10% no viaja
plt.subplot(2,3,1)
tabla['businesstravel'].value_counts().plot(kind='pie',autopct='%.2f')
#Se observa que la mayoría de los empleados pertenecen a investigación y desarrollo, el 30 % a ventas 
# y un 4% a recursos humanos
plt.subplot(2,3,2)
tabla['department'].value_counts().plot(kind='pie',autopct='%.2f')
# Los puestos de trabajo en su mayoria están invertigador cientifico, tecnico de laboratorio y ejecutivos de ventas
plt.subplot(2,3,3)
tabla['jobrole'].value_counts().plot(kind='pie',autopct='%.2f')
# Su educación con un 40% está en el campo de las ciencias, un 30% en un campo medico, 
# y esto puede explicar su gran porcentaje que se tiene en investigación y desarrollo.
plt.subplot(2,3,4)
tabla['educationfield'].value_counts().plot(kind='pie',autopct='%.2f')
# Los empleados en su mayoría están casados con un 46%, solteros un 31% y divorciados con un 22%
plt.subplot(2,3,5)
tabla['maritalstatus'].value_counts().plot(kind='pie',autopct='%.2f')


###Se exploran las variables numéricas y se observan sus respectivas distribuciones
# Cómo era de esperar muchas de las variables numéricas estudiadas presetan un sesgo hacia la izquierda, 
# tales como la distancia a casa, la edad, el salario, el número de compañias, entre otras
# Las encuestas de satisfación si no mostraban una distribución clara aparente en las gráficas, 
# por lo que no se deduce nada al respecto en estas variables, como la satifacción en el trabajo, 
# el ambiente laboral, entre otras.
fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(9, 5))
axes = axes.flat
columnas_numeric = tabla.select_dtypes(include=['float64', 'int']).columns
columnas_numeric = columnas_numeric.drop('v_objetivo')

for i, colum in enumerate(columnas_numeric):
    sns.histplot(
        data     = tabla,
        x        = colum,
        stat     = "count",
        kde      = True,
        color    = (list(plt.rcParams['axes.prop_cycle'])*2)[i]["color"],
        line_kws = {'linewidth': 2},
        alpha    = 0.3,
        ax       = axes[i]
    )
    axes[i].set_title(colum, fontsize = 7, fontweight = "bold")
    axes[i].tick_params(labelsize = 6)
    axes[i].set_xlabel("")
    
    
fig.tight_layout()
plt.subplots_adjust(top = 0.9)
fig.suptitle('Distribución variables numéricas', fontsize = 10, fontweight = "bold")



### Relación de variables continuas con variable objetivo
sns.boxplot(data=tabla, x="v_objetivo", y="age")#Los desertores aparetemente registran una menor edad frente a los que no
sns.boxplot(data=tabla, x="v_objetivo", y="distancefromhome")#No hay una diferencia significativa
sns.boxplot(data=tabla, x="v_objetivo", y="education")#No hay una diferencia significativa
sns.boxplot(data=tabla, x="v_objetivo", y="joblevel")#No hay una diferencia significativa
sns.boxplot(data=tabla, x="v_objetivo", y="monthlyincome")#No hay una diferencia significativa
sns.boxplot(data=tabla, x="v_objetivo", y="numcompaniesworked")#No hay una diferencia significativa
sns.boxplot(data=tabla, x="v_objetivo", y="percentsalaryhike")#No hay una diferencia significativa
sns.boxplot(data=tabla, x="v_objetivo", y="stockoptionlevel")#No hay una diferencia significativa
sns.boxplot(data=tabla, x="v_objetivo", y="trainingtimeslastyear")#No hay una diferencia significativa
sns.boxplot(data=tabla, x="v_objetivo", y="yearssincelastpromotion")#No hay una diferencia significativa
sns.boxplot(data=tabla, x="v_objetivo", y="yearswithcurrmanager")#No hay una diferencia significativa
sns.boxplot(data=tabla, x="v_objetivo", y="environmentsatisfaction") # Se encuentran que el satifación del ambiente si es mucho mejor en los no desertores
sns.boxplot(data=tabla, x="v_objetivo", y="jobsatisfaction") #La satifación labor si era mucho mayor en los que no desertaron
sns.boxplot(data=tabla, x="v_objetivo", y="worklifebalance")#No hay una diferencia significativa
sns.boxplot(data=tabla, x="v_objetivo", y="jobinvolvement")#No hay una diferencia significativa

###Relación de variables categórica con variable respuesta

# El siguiente gráfico se puede observar como hay una menor proporción de desertores en las personas que no viajan 
cross_tab = pd.crosstab(tabla['businesstravel'], tabla['v_objetivo'])
plt.figure(figsize=(8, 6))
cross_tab.plot(kind='bar', stacked=True, colormap='coolwarm')
plt.title('Gráfico de Barras Apiladas: Deserción por viajante')
plt.xlabel('Viaja o no viaja')
plt.ylabel('Número de Personas')
plt.legend(title='Deserción')
plt.show()

# En el gráfico no se observa una distinción aparente entre el tipo de departamento
# en el que se encuentran las personas 
cross_tab = pd.crosstab(tabla['department'], tabla['v_objetivo'])
plt.figure(figsize=(8, 6))
cross_tab.plot(kind='bar', stacked=True, colormap='coolwarm')
plt.title('Gráfico de Barras Apiladas: Deserción por departamento al que pertenece')
plt.xlabel('Departamento')
plt.ylabel('Número de Personas')
plt.legend(title='Deserción')
plt.show()

# Para los médicos y los científicos se puede encontrar mayor proporción de desertores 
# con respecto a los otros campos de estudios
cross_tab = pd.crosstab(tabla['educationfield'], tabla['v_objetivo'])
plt.figure(figsize=(8, 6))
cross_tab.plot(kind='bar', stacked=True, colormap='coolwarm')
plt.title('Gráfico de Barras Apiladas: Deserción por la educación que tiene el empleado')
plt.xlabel('Campo en el que estudió')
plt.ylabel('Número de Personas')
plt.legend(title='Deserción')
plt.show()

# Los técnicos de laboratorio, ejucutivos en venta y los investigadores científicos 
# muestran una mayor proporción de deserciones con respecto a los otros cargos
cross_tab = pd.crosstab(tabla['jobrole'], tabla['v_objetivo'])
plt.figure(figsize=(8, 6))
cross_tab.plot(kind='bar', stacked=True, colormap='coolwarm')
plt.title('Gráfico de Barras Apiladas: Deserción por cargo que tiene en el trabajo')
plt.xlabel('Cargo que desempeña')
plt.ylabel('Número de Personas')
plt.legend(title='Deserción')
plt.show()

# Aunque la diferencia proporcional no es mucha, los solteros si suelen ser más desertores según la graficas, 
# seguidos por los casados y posteriormete los divorciados
cross_tab = pd.crosstab(tabla['maritalstatus'], tabla['v_objetivo'])
plt.figure(figsize=(8, 6))
cross_tab.plot(kind='bar', stacked=True, colormap='coolwarm')
plt.title('Gráfico de Barras Apiladas: Deserción por Estado civil')
plt.xlabel('Estado civil')
plt.ylabel('Número de Personas')
plt.legend(title='Deserción')
plt.show()
