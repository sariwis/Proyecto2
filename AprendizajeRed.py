import pandas as pd
import numpy as np
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.sampling import BayesianModelSampling
from pgmpy.estimators import BayesianEstimator
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import plotly.express as px
from pgmpy.inference import VariableElimination
from sklearn.metrics import accuracy_score, confusion_matrix

#holaaa

#Métodos

#Restricciones
datos ='Datos_P2.xlsx'
df = pd.read_excel(datos)

print(df.head())
print(df.describe())
print(df.columns)

import pgmpy
from pgmpy.estimators import PC
est = PC(data=df)

estimated_model = est.estimate(variant="stable", max_cond_vars=4)
print(estimated_model)
print(estimated_model.nodes())
print(estimated_model.edges())

from pgmpy . models import BayesianNetwork
from pgmpy . estimators import MaximumLikelihoodEstimator
estimated_model = BayesianNetwork ( estimated_model )
estimated_model .fit( data =df , estimator = MaximumLikelihoodEstimator
)
for i in estimated_model . nodes () :
    print( estimated_model . get_cpds (i) )

import networkx as nx
import matplotlib.pyplot as plt

# Crear un gráfico de networkx a partir del modelo estimado
nx_graph = nx.DiGraph()

# Agregar nodos al gráfico
nx_graph.add_nodes_from(estimated_model.nodes())

# Agregar bordes al gráfico
nx_graph.add_edges_from(estimated_model.edges())

# Dibujar el gráfico
pos = nx.spring_layout(nx_graph)  # Layout del gráfico
nx.draw(nx_graph, pos, with_labels=True, node_size=1000, node_color="skyblue")
labels = {node: node for node in estimated_model.nodes()}
nx.draw_networkx_labels(nx_graph, pos, labels=labels)
plt.title("Red Bayesiana Estimada")
plt.show()

#SCORE PUNTAJE RESTRICCIÓN

modelo = BayesianNetwork([('D', 'target'), ('target', 'AE'), ('target', 'C'), ('AE', 'MS'), ('AE', 'C'), ('G', 'C'), ('G', 'target'), ('MS', 'C'), ('AG', 'C'), ('AO', 'C'), ('PQ', 'AG')])

sample_train, sample_test = train_test_split(df, test_size=0.2, random_state=777)
emv = MaximumLikelihoodEstimator(model = modelo, data = sample_train)

modelo.fit(data=sample_train, estimator=MaximumLikelihoodEstimator)

infer = VariableElimination(modelo)

# Se extraen los valores reales
y_real = sample_test["target"].values

df2 = sample_test.drop(columns=['target'])
y_p = modelo.predict(df2)

accuracy = accuracy_score(y_real, y_p)
print(accuracy)

conf = confusion_matrix(y_real, y_p)

print(conf)





#Puntaje


from pgmpy.estimators import HillClimbSearch
from pgmpy.estimators import K2Score

scoring_method = K2Score(data=df)
esth = HillClimbSearch(data=df)
estimated_modelh = esth.estimate(
    scoring_method=scoring_method, max_indegree=4, max_iter=int(1e4)
)
print(estimated_modelh)
print(estimated_modelh.nodes())
print(estimated_modelh.edges())

#Gráfico
import networkx as nx
import matplotlib.pyplot as plt

# Crear un gráfico de networkx a partir del modelo estimado
nx_graph = nx.DiGraph()

# Agregar nodos al gráfico
nx_graph.add_nodes_from(estimated_modelh.nodes())

# Agregar bordes al gráfico
nx_graph.add_edges_from(estimated_modelh.edges())

# Dibujar el gráfico
pos = nx.spring_layout(nx_graph)  # Layout del gráfico
nx.draw(nx_graph, pos, with_labels=True, node_size=3000, node_color="skyblue")
labels = {node: node for node in estimated_modelh.nodes()}
nx.draw_networkx_labels(nx_graph, pos, labels=labels)
plt.title("Red Bayesiana Estimada (HillClimbSearch)")
plt.show()

#Se imprime el puntaje obtenido
print(scoring_method.score(estimated_modelh))

#SCORE PUNTAJE K2SCore

modeloK2Score = BayesianNetwork([('MS', 'AE'), ('MS', 'target'), ('C', 'target'), ('C', 'AO'), ('C', 'G'), ('PQ', 'AG'), ('AG', 'C'), ('G', 'target'), ('AE', 'C'), ('AE', 'target'), ('AE', 'G'), ('AE', 'AG'), ('target', 'D')])

sample_train, sample_test = train_test_split(df, test_size=0.2, random_state=777)
emv = MaximumLikelihoodEstimator(model = modeloK2Score, data = sample_train)

modeloK2Score.fit(data=sample_train, estimator=MaximumLikelihoodEstimator)

infer = VariableElimination(modeloK2Score)

# Se extraen los valores reales
y_real = sample_test["target"].values

df2 = sample_test.drop(columns=['target'])
y_p = modeloK2Score.predict(df2)

accuracy = accuracy_score(y_real, y_p)
print(accuracy)

conf = confusion_matrix(y_real, y_p)

# Extraer verdaderos positivos, falsos positivos, verdaderos negativos y falsos negativos
VerdPos =conf[1, 1]
FalsoPos = conf[0, 1]
VerdNeg = conf[0, 0]
FalsoNeg = conf[1, 0]

print(conf)




#Mismo procedimiento con BicScore
from pgmpy.estimators import BicScore

scoring_method = BicScore(data=df)
esth = HillClimbSearch(data=df)
estimated_modelBicScore = esth.estimate(
    scoring_method=scoring_method, max_indegree=4, max_iter=int(1e4)
)
print(estimated_modelBicScore)
print(estimated_modelBicScore.nodes())
print(estimated_modelBicScore.edges())

#Gráfico

# Crear un gráfico de networkx a partir del modelo estimado
nx_graph = nx.DiGraph()

# Agregar nodos al gráfico
nx_graph.add_nodes_from(estimated_modelBicScore.nodes())

# Agregar bordes al gráfico
nx_graph.add_edges_from(estimated_modelBicScore.edges())

# Dibujar el gráfico
pos = nx.spring_layout(nx_graph)  # Layout del gráfico
nx.draw(nx_graph, pos, with_labels=True, node_size=3000, node_color="skyblue")
labels = {node: node for node in estimated_modelBicScore.nodes()}
nx.draw_networkx_labels(nx_graph, pos, labels=labels)
plt.title("Red Bayesiana Estimada (HillClimbSearch)")
plt.show()

#Se imprime el puntaje obtenido
print(scoring_method.score(estimated_modelBicScore))    
print(scoring_method.score(estimated_modelh))
#SCORE PUNTAJE BicScore

modeloBic = BayesianNetwork([('C', 'AE'), ('C', 'target'), ('AG', 'PQ'), ('G', 'C'), ('AE', 'MS'), ('AE', 'AO'), ('AE', 'AG'), ('target', 'D')])

sample_train, sample_test = train_test_split(df, test_size=0.2, random_state=777)
emv = MaximumLikelihoodEstimator(model = modeloBic, data = sample_train)

modeloBic.fit(data=sample_train, estimator=MaximumLikelihoodEstimator)

infer = VariableElimination(modeloBic)

# Se extraen los valores reales
y_real = sample_test["target"].values

df2 = sample_test.drop(columns=['target'])
y_p = modeloBic.predict(df2)

accuracy = accuracy_score(y_real, y_p)
print(accuracy)

conf = confusion_matrix(y_real, y_p)

print(conf)