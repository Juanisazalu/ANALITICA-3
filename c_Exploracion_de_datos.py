#### Cargar paquetes siempre al inicio
import pandas as pd ### para manejo de datos
import sqlite3 as sql #### para bases de datos sql
import a_funciones as funciones ### archivo de funciones propias
import matplotlib as mpl ## gráficos
import matplotlib.pyplot as plt ### gráficos
from pandas.plotting import scatter_matrix  ## para matriz de correlaciones
from sklearn import tree ###para ajustar arboles de decisión
from sklearn.tree import export_text ## para exportar reglas del árbol

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

tabla.describe()

tabla.info()

#explorar variables numéricas con histograma
fig=tabla[continuas].hist(bins=50, figsize=(40,30),grid=False,ec='black')
plt.show() 

scatter_matrix(tabla[continuas], figsize=(12, 8))
plt.show()

cont=tabla[continuas]
corr_matrix = cont.corr()




##### analizar relación con categóricas ####

tabla.boxplot("v_objetivo","jobrole",figsize=(5,5),grid=False)
tabla.barplot("v_objetivo","businesstravel",figsize=(5,5),grid=False)


plt.figure(figsize=(20, 10))
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


