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
#intento
datos = 'Datos_FINALP1.xlsx'
df = pd.read_excel(datos)

# Categorizar la variable target
df['target'] = df['target'].apply(lambda x: 0 if x == "Dropout" else 1)
df['AG'] = round(df['AG'])
# Categorizar las calificaciones previas y de ingreso
lim_Notas = [-1, 49, 99, 149, 201]
labels = ['Malo', 'Regular', 'Buena', 'Excelente']
df['PQ'] = pd.cut(df['PQ'], bins=lim_Notas, labels=labels)
df['PQ'] = df['PQ'].replace({'Malo':1,'Regular':2,'Buena':3, 'Excelente':4})

df['AG'] = pd.cut(df['AG'], bins=lim_Notas, labels=labels)
df['AG'] = df['AG'].replace({'Malo':1,'Regular':2,'Buena':3, 'Excelente':4})

# Categorizar edad
lim_Edad = [16, 30, 45, 63]
labels_Edad = ['Joven', 'Adulto', 'Mayor']
df['AE'] = pd.cut(df['AE'], bins=lim_Edad, labels=labels_Edad)
df['AE'] = df['AE'].replace({'Joven':1,'Adulto':2,'Mayor':3})

# Categorizar los cursos
df['C'] = df['C'].replace({"Biofuel Production Technologies":1,"Animation and Multimedia Design":2,"Social Service (evening attendance)":3,"Agronomy":4,"Communication Design":5,"Veterinary Nursing":6,"Informatics Engineering":7,"Equinculture":8,"Management":9,"Social Service":10,"Tourism":11,"Nursing":12,"Oral Hygiene":13,"Advertising and Marketing Management":14,"Journalism and Communication":15,"Basic Education":16, "Management (evening attendance)":17})

modelo = BayesianNetwork([('MS', 'target'), ('C', 'target'), ('MS', 'AO'), ('AG', 'C'), ('AG', 'PQ'), ('G', 'C'), ('C', 'PQ'), ('PQ', 'D'), ('PQ', 'AE'), ('AE', 'target'), ('D', 'target')])

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

# Extraer verdaderos positivos, falsos positivos, verdaderos negativos y falsos negativos
VerdPos =conf[1, 1]
FalsoPos = conf[0, 1]
VerdNeg = conf[0, 0]
FalsoNeg = conf[1, 0]

print(VerdPos)
print(FalsoPos)
print(VerdNeg)
print(FalsoNeg)

print(conf)

# Calcular F1 Score
accuracy = VerdPos / (VerdPos + FalsoPos)
recall = VerdPos / (VerdPos + FalsoNeg)
f1_score = 2 * (accuracy* recall) / (accuracy + recall)

# Imprimir las métricas
print("Precisión:", accuracy)
print("Recall:", recall)
print("F1 Score:", f1_score)
