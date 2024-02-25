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
fig=tabla.v_objetivo.hist(bins=50,ec='black') ## no hay atípicos
fig.grid(False)
plt.show()