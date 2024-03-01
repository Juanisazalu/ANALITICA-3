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

cat= [x for x in tabla.columns if tabla[x].dtypes =="O"]
continuas= tabla.select_dtypes(exclude='object')
continuas=continuas.columns
### explorar variable respuesta ###
# crear dataset
fig=tabla.v_objetivo.hist(bins=20,ec='black') ## no hay atípicos
fig.grid(False)
plt.show()

"""boxprops = dict(linestyle='-', color='black')
medianprops = dict(linestyle='-',  color='black')
fig=tabla.boxplot("v_objetivo",patch_artist=True,
                boxprops=boxprops,
                medianprops=medianprops,
                whiskerprops=dict(color='black'),
                showmeans=True)
fig.grid(False)
plt.show()"""

### Dimensiones del dataset
tabla.shape
# Número de datos ausentes por variable
tabla.isna().sum().sort_values()

tabla.describe()

tabla.info()

#explorar variables numéricas con histograma
fig=tabla[continuas].hist(bins=50, figsize=(40,30),grid=False,ec='black')
plt.show() 

scatter_matrix(tabla[continuas], figsize=(100, 100))
plt.show()

cont=tabla[continuas]
corr_matrix = cont.corr()

### Gráfico de distribución de las variables númericas
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


##### Variable categóricas ####


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

sns.pairplot(tabla, hue='v_objetivo', size=2.5)
sns.boxplot(data=tabla, x="v_objetivo", y="jobsatisfaction")

### relación de variables continuas con variable objetivo
tabla[continuas].columns
# Añade etiquetas y título
fig, ax = plt.subplots(3, 5, sharex='col', sharey='row', figsize=(12, 10))
sns.boxplot(data=tabla, x="v_objetivo", y="age", ax=ax[0, 0])
sns.boxplot(data=tabla, x="v_objetivo", y="distancefromhome", ax=ax[0, 1])
sns.boxplot(data=tabla, x="v_objetivo", y="education", ax=ax[0, 2])
sns.boxplot(data=tabla, x="v_objetivo", y="joblevel", ax=ax[0, 3])
sns.boxplot(data=tabla, x="v_objetivo", y="monthlyincome", ax=ax[0, 4])
sns.boxplot(data=tabla, x="v_objetivo", y="numcompaniesworked", ax=ax[1, 0])
sns.boxplot(data=tabla, x="v_objetivo", y="percentsalaryhike", ax=ax[1, 1])
sns.boxplot(data=tabla, x="v_objetivo", y="stockoptionlevel", ax=ax[1, 2])
sns.boxplot(data=tabla, x="v_objetivo", y="trainingtimeslastyear", ax=ax[1, 3])
sns.boxplot(data=tabla, x="v_objetivo", y="yearssincelastpromotion", ax=ax[1, 4])
sns.boxplot(data=tabla, x="v_objetivo", y="yearswithcurrmanager", ax=ax[2, 0])
sns.boxplot(data=tabla, x="v_objetivo", y="environmentsatisfaction", ax=ax[2, 1])
sns.boxplot(data=tabla, x="v_objetivo", y="jobsatisfaction", ax=ax[2, 2])
sns.boxplot(data=tabla, x="v_objetivo", y="worklifebalance", ax=ax[2, 3])
sns.boxplot(data=tabla, x="v_objetivo", y="jobinvolvement", ax=ax[2, 4])
# Añade títulos y etiquetas si es necesario
plt.suptitle("Análisis de variables continuas con la variable objetivo", y=1.02)
ax[0, 0].set_title("Edad")
ax[0, 1].set_title("Distancia desde casa")
ax[1, 0].set_title("Nivel de educación")
ax[1, 1].set_title("Nivel de trabajo")

plt.tight_layout()
plt.show()

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
