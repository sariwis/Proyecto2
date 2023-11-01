import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

datos = 'Datos_Iniciales.xlsx'

# Dataframe
df = pd.read_excel(datos)

# Cambiamos la última columna 
df['Target'] = df['Target'].apply(lambda x: 0 if x == "Dropout" else 1)

# Seleccionamos solo los que son numéricos
df_numeric = df.select_dtypes(include=['number'])

# Calculamos la matriz de correlación
correlation_matrix = df_numeric.corr()

# Configuramos los tamaños
plt.figure(figsize=(12, 10))

# Creamos el mapa de calor
sns.heatmap(correlation_matrix, annot=True, cmap="Spectral", linewidths=.5, vmin=-1, vmax=1)

# Título del diagrama
plt.title("Mapa de calor de la matriz de correlación")

# Muestra el mapa de calor
plt.show()