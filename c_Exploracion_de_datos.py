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
# crear dataset
fig=tabla.v_objetivo.hist(bins=20,ec='black') ## no hay atípicos
fig.grid(False)
plt.show()

### Dimensiones del dataset
tabla.shape
# Número de datos ausentes por variable
tabla.isna().sum().sort_values()

tabla.describe()

tabla.info()


#Exploracion variables numericas
plt.figure(figsize=(25, 10))
plt.subplot(2,3,1)
tabla['businesstravel'].value_counts().plot(kind='pie',autopct='%.2f')
plt.subplot(2,3,2)
tabla['department'].value_counts().plot(kind='pie',autopct='%.2f')
plt.subplot(2,3,3)
tabla['jobrole'].value_counts().plot(kind='pie',autopct='%.2f')
plt.subplot(2,3,4)
tabla['educationfield'].value_counts().plot(kind='pie',autopct='%.2f')
plt.subplot(2,3,5)
tabla['maritalstatus'].value_counts().plot(kind='pie',autopct='%.2f')


#explorar variables numéricas 
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



### relación de variables continuas con variable objetivo
sns.boxplot(data=tabla, x="v_objetivo", y="age")
sns.boxplot(data=tabla, x="v_objetivo", y="distancefromhome")
sns.boxplot(data=tabla, x="v_objetivo", y="education")
sns.boxplot(data=tabla, x="v_objetivo", y="joblevel")
sns.boxplot(data=tabla, x="v_objetivo", y="monthlyincome")
sns.boxplot(data=tabla, x="v_objetivo", y="numcompaniesworked")
sns.boxplot(data=tabla, x="v_objetivo", y="percentsalaryhike")
sns.boxplot(data=tabla, x="v_objetivo", y="stockoptionlevel")
sns.boxplot(data=tabla, x="v_objetivo", y="trainingtimeslastyear")
sns.boxplot(data=tabla, x="v_objetivo", y="yearssincelastpromotion")
sns.boxplot(data=tabla, x="v_objetivo", y="yearswithcurrmanager")
sns.boxplot(data=tabla, x="v_objetivo", y="environmentsatisfaction")
sns.boxplot(data=tabla, x="v_objetivo", y="jobsatisfaction")
sns.boxplot(data=tabla, x="v_objetivo", y="worklifebalance")
sns.boxplot(data=tabla, x="v_objetivo", y="jobinvolvement")

###Relacion de variables categorica con variable respuesta
tabla[cat].columns

cross_tab = pd.crosstab(tabla['businesstravel'], tabla['v_objetivo'])
# Crear el gráfico de barras apiladas
plt.figure(figsize=(8, 6))
cross_tab.plot(kind='bar', stacked=True, colormap='coolwarm')
plt.title('Gráfico de Barras Apiladas: Deserción por Estado civil')
plt.xlabel('Estado civil')
plt.ylabel('Número de Personas')
plt.legend(title='Deserción')
plt.show()


cross_tab = pd.crosstab(tabla['department'], tabla['v_objetivo'])
# Crear el gráfico de barras apiladas
plt.figure(figsize=(8, 6))
cross_tab.plot(kind='bar', stacked=True, colormap='coolwarm')
plt.title('Gráfico de Barras Apiladas: Deserción por Estado civil')
plt.xlabel('Estado civil')
plt.ylabel('Número de Personas')
plt.legend(title='Deserción')
plt.show()

cross_tab = pd.crosstab(tabla['educationfield'], tabla['v_objetivo'])
# Crear el gráfico de barras apiladas
plt.figure(figsize=(8, 6))
cross_tab.plot(kind='bar', stacked=True, colormap='coolwarm')
plt.title('Gráfico de Barras Apiladas: Deserción por Estado civil')
plt.xlabel('Estado civil')
plt.ylabel('Número de Personas')
plt.legend(title='Deserción')
plt.show()

cross_tab = pd.crosstab(tabla['jobrole'], tabla['v_objetivo'])
# Crear el gráfico de barras apiladas
plt.figure(figsize=(8, 6))
cross_tab.plot(kind='bar', stacked=True, colormap='coolwarm')
plt.title('Gráfico de Barras Apiladas: Deserción por Estado civil')
plt.xlabel('Estado civil')
plt.ylabel('Número de Personas')
plt.legend(title='Deserción')
plt.show()

cross_tab = pd.crosstab(tabla['maritalstatus'], tabla['v_objetivo'])
# Crear el gráfico de barras apiladas
plt.figure(figsize=(8, 6))
cross_tab.plot(kind='bar', stacked=True, colormap='coolwarm')
plt.title('Gráfico de Barras Apiladas: Deserción por Estado civil')
plt.xlabel('Estado civil')
plt.ylabel('Número de Personas')
plt.legend(title='Deserción')
plt.show()
