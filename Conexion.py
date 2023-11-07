import psycopg2
import pandas as pd
import matplotlib.pyplot as plt

# Conectarse a la base de datos PostgreSQL
engine = psycopg2.connect(
    dbname="p2",
    user="postgres",
    password="proyecto2",
    host="proyecto2.c9pexl84mjtw.us-east-1.rds.amazonaws.com",
    port="5432"
)

# Consulta SQL para obtener los datos de graduados
query = """
SELECT G, target
FROM grado;
"""
# Ejecutar la consulta y cargar los resultados en un DataFrame
df = pd.read_sql_query(query, engine)

# Cerrar la conexión a la base de datos
engine.close()

# Mapear los valores numéricos de género a "Mujer" y "Hombre"
gender_mapping = {0: 'Mujer', 1: 'Hombre'}
df['G'] = df['G'].map(gender_mapping)

# Filtrar los datos para obtener solo los graduados (donde 'target' es igual a 1)
graduados = df[df['target'] == 1]

# Calcular el porcentaje de graduados por género
gender_percentage = (graduados['G'].value_counts() / len(graduados)) * 100

# Define tus colores personalizados en formato hexadecimal
custom_colors = ['#1B4D2C', '#66A550']

# Crear un gráfico de barras para comparar los porcentajes de graduados por género
ax = gender_percentage.plot(kind='bar', color=custom_colors, ylim=(0, 100), width=0.3)

# Personalizar el título
plt.title('Porcentaje de estudiantes graduados por género', fontsize=18, fontweight='bold')

# Eliminar la etiqueta del eje x
plt.xlabel('')

# Aumentar el tamaño de fuente y poner en negrita los valores de porcentaje
for index, value in enumerate(gender_percentage):
    plt.text(index, value, f'{value:.2f}%', ha='center', va='bottom', fontsize=14, fontweight='bold')

# Ajustar los márgenes
plt.margins(x=0.2)

# Eliminar el eje y y el marco cuadrado
ax.get_yaxis().set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)

# Ajustar la rotación de las etiquetas en el eje x
plt.xticks(rotation=0)

plt.show()




#########################################################################################


import psycopg2
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# Conectarse a la base de datos PostgreSQL
engine = psycopg2.connect(
    dbname="p2",
    user="postgres",
    password="proyecto2",
    host="proyecto2.c9pexl84mjtw.us-east-1.rds.amazonaws.com",
    port="5432"
)

# Consulta SQL para obtener los datos de graduados con información de edad
query = """
SELECT AE AS age, target
FROM grado;
"""

# Ejecutar la consulta y cargar los resultados en un DataFrame
df = pd.read_sql_query(query, engine)

# Cerrar la conexión a la base de datos
engine.close()

# Filtrar los datos para obtener solo los graduados y los que no se graduaron
graduados = df[df['target'] == 1]
no_graduados = df[df['target'] == 0]

# Calcular el porcentaje de graduados y no graduados por categoría de edad
age_categories = df['age'].value_counts().sort_index()
total_students = age_categories.sum()
graduates = graduados['age'].value_counts().sort_index()
non_graduates = no_graduados['age'].value_counts().sort_index()
graduates_percentage = (graduates / total_students) * 100
non_graduates_percentage = (non_graduates / total_students) * 100

# Crear un gráfico de barras apiladas que compare los porcentajes de graduados y no graduados por categoría de edad
plt.figure(figsize=(8, 6))

# Etiquetas de las categorías con rangos de edad
labels = ['Joven\n(16-30)', 'Adulto\n(31-45)', 'Mayor\n(46-63)']

# Colores personalizados
custom_colors = ['#1B4D2C', '#66A550']

# Crear el gráfico de barras apiladas con los colores personalizados
bars = plt.bar(labels, graduates_percentage, color=custom_colors[0], label='Graduados')
plt.bar(labels, non_graduates_percentage, color=custom_colors[1], label='No Graduados', bottom=graduates_percentage)

plt.title('Porcentaje de Graduados y No Graduados por edades')

# Eliminar los porcentajes de las barras
for bar in bars:
    bar.set_label("")

# Eliminar la etiqueta del eje y (pero dejar los valores del eje)
plt.gca().get_yaxis().set_visible(True)

# Eliminar el eje x y su etiqueta
plt.xticks([])

# Agregar etiquetas en la parte inferior de cada barra
for index, label in enumerate(labels):
    plt.text(index, -8, label, ha='center')

# Formatear el eje y con el signo de porcentaje
plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.0f}%'))

# Agregar la leyenda
plt.legend()

plt.show()



######################################################################################################################


import psycopg2
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# Conectarse a la base de datos PostgreSQL
engine = psycopg2.connect(
    dbname="p2",
    user="postgres",
    password="proyecto2",
    host="proyecto2.c9pexl84mjtw.us-east-1.rds.amazonaws.com",
    port="5432"
)

# Consulta SQL para obtener los datos de graduados con información de edad
query = """
SELECT C AS curso, target
FROM grado;
"""

df = pd.read_sql_query(query, engine)

# Cerrar la conexión a la base de datos
engine.close()

# Filtrar los datos para obtener solo los graduados y los que no se graduaron
graduados = df[df['target'] == 1]
no_graduados = df[df['target'] == 0]

# Calcular el porcentaje de graduados y no graduados por categoría de curso
course_categories = df['curso'].value_counts().sort_index()
total_students = course_categories.sum()
graduates = graduados['curso'].value_counts().sort_index()
non_graduates = no_graduados['curso'].value_counts().sort_index()
graduates_percentage = (graduates / total_students) * 100
non_graduates_percentage = (non_graduates / total_students) * 100

# Crear un gráfico de barras apiladas que compare los porcentajes de graduados y no graduados por categoría de curso
plt.figure(figsize=(10, 6))

# Etiquetas de las categorías de curso (puedes ajustarlas según tus necesidades)
course_labels = [
    "Biofuel Production Technologies", "Animation and Multimedia Design",
    "Social Service (evening attendance)", "Agronomy",
    "Communication Design", "Veterinary Nursing",
    "Informatics Engineering", "Equinculture",
    "Management", "Social Service",
    "Tourism", "Nursing",
    "Oral Hygiene", "Advertising and Marketing Management",
    "Journalism and Communication", "Basic Education",
    "Management (evening attendance)"
]

# Colores personalizados
custom_colors = ['#1B4D2C', '#66A550']

# Crear el gráfico de barras apiladas con los colores personalizados
bars = plt.bar(course_labels, graduates_percentage, color=custom_colors[0], label='Graduados')
plt.bar(course_labels, non_graduates_percentage, color=custom_colors[1], label='No Graduados', bottom=graduates_percentage)

plt.title('Porcentaje de Graduados y No Graduados por curso')

# Eliminar las etiquetas del eje x (puedes ajustar el formato de las etiquetas)
plt.xticks(rotation=90)

# Formatear el eje y con el signo de porcentaje
plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.0f}%'))

# Agregar la leyenda
plt.legend()

plt.show()


############################################################################################################





































































import psycopg2
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# Conectarse a la base de datos PostgreSQL
engine = psycopg2.connect(
    dbname="p2",
    user="postgres",
    password="proyecto2",
    host="proyecto2.c9pexl84mjtw.us-east-1.rds.amazonaws.com",
    port="5432"
)

# Consulta SQL para obtener los datos de graduados con información de edad
query = """
SELECT AE AS age, target
FROM grado;
"""

# Ejecutar la consulta y cargar los resultados en un DataFrame
df = pd.read_sql_query(query, engine)

# Cerrar la conexión a la base de datos
engine.close()

# Filtrar los datos para obtener solo los graduados y los que no se graduaron
graduados = df[df['target'] == 1]
no_graduados = df[df['target'] == 0]

# Calcular el porcentaje de graduados y no graduados por categoría de edad
age_categories = df['age'].value_counts().sort_index()
total_students = age_categories.sum()
graduates = graduados['age'].value_counts().sort_index()
non_graduates = no_graduados['age'].value_counts().sort_index()
graduates_percentage = (graduates / total_students) * 100
non_graduates_percentage = (non_graduates / total_students) * 100

# Crear un gráfico de barras apiladas que compare los porcentajes de graduados y no graduados por categoría de edad
plt.figure(figsize=(8, 6))

# Escala logarítmica en el eje y
plt.yscale('log')

# Etiquetas de las categorías con rangos de edad
labels = ['Joven\n(16-30)', 'Adulto\n(31-45)', 'Mayor\n(46-63)']

# Colores personalizados
custom_colors = ['#1B4D2C', '#66A550']

# Crear el gráfico de barras apiladas con los colores personalizados
bars = plt.bar(labels, graduates_percentage, color=custom_colors[0], label='Graduados')
plt.bar(labels, non_graduates_percentage, color=custom_colors[1], label='No Graduados', bottom=graduates_percentage)

plt.title('Porcentaje de Graduados y No Graduados por edades')

# Eliminar los porcentajes de las barras
for bar in bars:
    bar.set_label("")

# Etiquetas del eje x
plt.xticks(labels)

# Agregar etiquetas en la parte superior de cada barra
for index, label in enumerate(labels):
    plt.text(index, 1e-6, label, ha='center')  # Ajusta la posición de las etiquetas para la escala logarítmica

# Formatear el eje y con el signo de porcentaje
plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.0e}%'))

# Agregar la leyenda
plt.legend()

plt.show()





#########################################################################
import psycopg2
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# Conectarse a la base de datos PostgreSQL
engine = psycopg2.connect(
    dbname="p2",
    user="postgres",
    password="proyecto2",
    host="proyecto2.c9pexl84mjtw.us-east-1.rds.amazonaws.com",
    port="5432"
)

# Consulta SQL para obtener los datos de graduados con información de edad
query = """
SELECT C AS curso, target
FROM grado;
"""

df = pd.read_sql_query(query, engine)

# Cerrar la conexión a la base de datos
engine.close()

# Filtrar los datos para obtener solo los graduados y los que no se graduaron
graduados = df[df['target'] == 1]
no_graduados = df[df['target'] == 0]

# Calcular el porcentaje de graduados y no graduados por categoría de curso
course_categories = df['curso'].value_counts().sort_index()
total_students = course_categories.sum()
graduates = graduados['curso'].value_counts().sort_index()
non_graduates = no_graduados['curso'].value_counts().sort_index()
graduates_percentage = (graduates / total_students) * 100
non_graduates_percentage = (non_graduates / total_students) * 100

# Crear un gráfico de barras apiladas que compare los porcentajes de graduados y no graduados por categoría de curso
plt.figure(figsize=(10, 6))

# Etiquetas de las categorías de curso (puedes ajustarlas según tus necesidades)
course_labels = [
    "Biofuel Production Technologies", "Animation and Multimedia Design",
    "Social Service (evening attendance)", "Agronomy",
    "Communication Design", "Veterinary Nursing",
    "Informatics Engineering", "Equinculture",
    "Management", "Social Service",
    "Tourism", "Nursing",
    "Oral Hygiene", "Advertising and Marketing Management",
    "Journalism and Communication", "Basic Education",
    "Management (evening attendance)"
]

# Colores personalizados
custom_colors = ['#1B4D2C', '#66A550']

# Crear el gráfico de barras apiladas con los colores personalizados
bars = plt.bar(range(len(course_labels)), graduates_percentage, color=custom_colors[0], label='Graduados')
plt.bar(range(len(course_labels)), non_graduates_percentage, color=custom_colors[1], label='No Graduados', bottom=graduates_percentage)

plt.title('Porcentaje de Graduados y No Graduados por curso')

# Etiquetas personalizadas en el eje X
plt.xticks(range(len(course_labels)), [str(i+1) for i in range(len(course_labels))])

# Formatear el eje y con el signo de porcentaje
plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.0f}%'))

# Agregar una leyenda para los colores
legend_colors = [
    plt.Line2D([0], [0], marker='o', color='w', label='Graduados', markerfacecolor=custom_colors[0], markersize=10),
    plt.Line2D([0], [0], marker='o', color='w', label='No Graduados', markerfacecolor=custom_colors[1], markersize=10)
]
plt.legend(handles=legend_colors, title='Leyenda Colores')

# Agregar una segunda leyenda para descripción de categorías
category_legend = [
    "1: Biofuel Production Technologies",
    "2: Animation and Multimedia Design",
    "3: Social Service (evening attendance)",
    "4: Agronomy",
    "5: Communication Design",
    "6: Veterinary Nursing",
    "7: Informatics Engineering",
    "8: Equinculture",
    "9: Management",
    "10: Social Service",
    "11: Tourism",
    "12: Nursing",
    "13: Oral Hygiene",
    "14: Advertising and Marketing Management",
    "15: Journalism and Communication",
    "16: Basic Education",
    "17: Management (evening attendance)"
]

plt.legend(handles=[plt.Line2D([0], [0], color='white', label=desc) for desc in category_legend], title="Leyenda Categorías")

plt.show()
