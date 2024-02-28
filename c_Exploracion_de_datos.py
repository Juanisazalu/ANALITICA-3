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


tabla=pd.read_csv("tabla_exploración1.csv")
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

### relación con variable objetivo


tabla[continuas].columns

fig, ax=plt.subplots(figsize=(10,10),nrows=4, ncols=4)

plt.figure(figsize=(15, 8))  # Ajusta el tamaño de la figura según sea necesario

for variable in continuas:
    sns.boxplot(x='v_objetivo', y=variable, data=tabla, width=0.5, notch=True)

# Añade etiquetas y título

len(continuas)
for i in range(len(continuas)):
 plt.subplot(2, 3, i)
 sns.catplot(data=tabla, x="v_objetivo", y=tabla[i], kind="box")
 #plt.text(0.5, 0.5, str((2, 3, i)),
 #fontsize=18, ha='center')

sns.catplot(data=tabla, x="v_objetivo", y="age", kind="box")

tabla.boxplot("v_objetivo","jobrole",figsize=(5,5),grid=False)
tabla.barplot("v_objetivo","businesstravel",figsize=(5,5),grid=False)

sns.barplot(x = "v_objetivo", y = "businesstravel", data = tabla)

fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(15, 10), sharey=True)

# Aplanar la matriz de subgráficos para facilitar el acceso
axes = axes.flatten()

# Iterar sobre las variables continuas y crear un boxplot en cada subgráfico
for i, variable in enumerate(continuas):
    sns.boxplot(x='v_objetivo', y=variable, data=tabla, ax=axes[i], width=0.5, notch=True)
    axes[i].set_title(variable)


fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(15, 10), sharey=True)

# Aplanar la matriz de subgráficos para facilitar el acceso
axes = axes.flatten()

# Iterar sobre las variables continuas y crear un boxplot en cada subgráfico
for i, variable in enumerate(continuas):
    sns.boxplot(x='v_objetivo', y=variable, data=tabla, ax=axes[i], width=0.5, notch=True)
    axes[i].set_title(variable)
    
    # Ajustar los límites del eje y para evitar que la caja esté pegada a la parte inferior
    ylim = axes[i].get_ylim()
    axes[i].set_ylim(ylim[0] - 0.1 * (ylim[1] - ylim[0]), ylim[1])

# Ajustar el diseño de los subgráficos
plt.tight_layout()

# Muestra el gráfico
plt.show()