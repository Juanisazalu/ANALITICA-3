#### Cargar paquetes siempre al inicio
import pandas as pd ### para manejo de datos
import sqlite3 as sql #### para bases de datos sql
import a_funciones as funciones ### archivo de funciones propias
import matplotlib as mpl ## gráficos
import matplotlib.pyplot as plt ### gráficos
from pandas.plotting import scatter_matrix  ## para matriz de correlaciones
from sklearn import tree ###para ajustar arboles de decisión
from sklearn.tree import export_text ## para exportar reglas del árbol

con=sql.connect("data\\db_basedatos")
cur=con.cursor()

tabla=pd.read_sql("""select *  
                     from tabla_completa """ , con)

### explorar variable respuesta ###
fig=tabla.v_objetivo.hist(bins=20,ec='black') ## no hay atípicos
fig.grid(False)
plt.show()

boxprops = dict(linestyle='-', color='black')
medianprops = dict(linestyle='-',  color='black')
fig=tabla.boxplot("v_objetivo",patch_artist=True,
                boxprops=boxprops,
                medianprops=medianprops,
                whiskerprops=dict(color='black'),
                showmeans=True)
fig.grid(False)
plt.show()

####explorar variables numéricas con histograma
fig=tabla.hist(bins=50, figsize=(40,30),grid=False,ec='black')
plt.show()

continuas = ['v_objetivo',
             'EnvironmentSatisfaction',
             'YearsAtCompany',
             'WorkLifeBalance',
             'JobInvolvement',
             ]
scatter_matrix(tabla[continuas], figsize=(12, 8))
plt.show()
cont=tabla[continuas]
corr_matrix = cont.corr()
corr_matrix["v_objetivo"].sort_values(ascending=False)


##### analizar relación con categóricas ####

tabla.boxplot("v_objetivo","JobRole",figsize=(5,5),grid=False)
tabla.boxplot("v_objetivo","BusinessTravel",figsize=(5,5),grid=False)
